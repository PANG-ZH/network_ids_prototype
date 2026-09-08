#!/usr/bin/env python3
"""
CodeBERT 原型验证 - HTTP 恶意 Payload 文本分类

使用预训练 CodeBERT (microsoft/codebert-base) 对 HTTP 请求 Payload 做二分类：
0 = BENIGN（正常请求），1 = ATTACK（恶意请求，如 SQLi / XSS / 命令注入等）。

前置依赖（额外安装，体积较大）：
    pip install transformers torch

用法：
    # 方式一：默认合成演示数据，快速跑通流程
    python codebert_http_payload.py

    # 方式二：真实标注数据（CSV，需含文本列 + Label 列，Label 除 BENIGN 外视为 ATTACK）
    python codebert_http_payload.py --data path/to/payloads.csv

    # 可选参数：--model 更换预训练模型 / --epochs / --batch-size / --max-length 等

注意：
- 首次运行需联网下载 CodeBERT 权重（约 500MB），离线会报错。
- 本脚本为可行性原型验证，未做大规模训练与调参。
  若机器资源紧张，可换成更小的模型，如 --model distilbert-base-uncased。
"""

import argparse
import random
import sys

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)

# ==================== 配置 ====================
RANDOM_STATE = 42
MODEL_NAME = "microsoft/codebert-base"
EPOCHS = 2
BATCH_SIZE = 8
MAX_LENGTH = 128
LEARNING_RATE = 2e-5
# 合成演示数据总样本数（恶意 / 正常各一半）
N_SYNTHETIC = 1000
# 测试集占比
TEST_SIZE = 0.2

# 恶意 / 正常 Payload 模板池（真实数据替换方式见函数 generate_synthetic_payloads 说明）
ATTACK_TEMPLATES = [
    "GET /login?user=admin&pass=' OR '1'='1'-- HTTP/1.1",
    "POST /search HTTP/1.1\nquery=1; DROP TABLE users--",
    "GET /item?id=<script>alert(document.cookie)</script> HTTP/1.1",
    "POST /comment HTTP/1.1\nbody=<img src=x onerror=alert(1)>",
    "GET /file?path=../../../../etc/passwd HTTP/1.1",
    "GET /download?f=/etc/shadow%00.html HTTP/1.1",
    "POST /upload HTTP/1.1\ncmd=cat+/etc/passwd",
    "GET /admin?debug=true;id HTTP/1.1",
    "GET /profile?uid=1 UNION SELECT username,password FROM users HTTP/1.1",
    "POST /order HTTP/1.1\nprice=0&qty=999999",
]

BENIGN_TEMPLATES = [
    "GET /index.html HTTP/1.1\nHost: www.example.com\nUser-Agent: Mozilla/5.0",
    "GET /css/main.css HTTP/1.1\nAccept: text/css",
    "POST /login HTTP/1.1\nusername=alice&password=secret123",
    "GET /api/users?page=2 HTTP/1.1\nAccept: application/json",
    "POST /api/order HTTP/1.1\n{\"name\":\"Bob\",\"items\":[1,2,3]}",
    "GET /favicon.ico HTTP/1.1\nReferer: https://example.com/",
    "GET /images/logo.png HTTP/1.1\nAccept-Encoding: gzip",
    "GET /product/1024 HTTP/1.1\nConnection: keep-alive",
    "POST /feedback HTTP/1.1\ncontent=Great%20service&rating=5",
    "GET /robots.txt HTTP/1.1\nIf-Modified-Since: Tue, 01 Sep 2026 00:00:00 GMT",
]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_synthetic_payloads(n_samples: int) -> pd.DataFrame:
    """
    生成模拟的 HTTP 恶意/正常 Payload（用于快速跑通原型流程）。
    真实项目请替换为带标注的 HTTP 请求日志/Payload 数据集，CSV 格式：
    text,Label   （text 为原始请求 Payload，Label 为 BENIGN 或攻击类型）
    """
    rng = random.Random(RANDOM_STATE)
    n_half = n_samples // 2
    texts, labels = [], []

    # 通过拼接随机编号与模板产生多样性样本
    for _ in range(n_half):
        t = BENIGN_TEMPLATES[rng.randrange(len(BENIGN_TEMPLATES))]
        texts.append(f"{t}\nX-Request-Id: {rng.randint(1000, 9999)}")
        labels.append("BENIGN")

    for _ in range(n_samples - n_half):
        t = ATTACK_TEMPLATES[rng.randrange(len(ATTACK_TEMPLATES))]
        texts.append(f"{t}\nX-Request-Id: {rng.randint(1000, 9999)}")
        labels.append("ATTACK")

    df = pd.DataFrame({"text": texts, "Label": labels})
    return df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def load_real_payloads(data_path: str) -> pd.DataFrame:
    """加载真实标注 CSV：自动识别文本列与 Label 列。"""
    df = pd.read_csv(data_path, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    # 文本列：优先列名含 text/payload
    text_col = None
    for cand in ["text", "payload", "request", "content", "data"]:
        if cand in df.columns:
            text_col = cand
            break
    if text_col is None:
        # 兜底取第一列
        text_col = df.columns[0]
    df = df.rename(columns={text_col: "text"})

    # 标签列：优先 Label / label，否则取最后一列
    label_col = "Label" if "Label" in df.columns else df.columns[-1]
    df = df.rename(columns={label_col: "Label"})

    # 二分类标签：BENIGN -> 0，其余 -> 1
    df = df[["text", "Label"]].dropna()
    df["Label"] = df["Label"].apply(
        lambda x: "BENIGN" if str(x).strip().upper() == "BENIGN" else "ATTACK"
    )
    return df


def tokenize(tokenizer, texts):
    return tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )


def train_and_evaluate(model, tokenizer, train_enc, y_train, val_enc, y_val, device):
    train_ds = TensorDataset(
        train_enc["input_ids"], train_enc["attention_mask"], y_train
    )
    val_ds = TensorDataset(val_enc["input_ids"], val_enc["attention_mask"], y_val)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    model.to(device)
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_loader):
            input_ids, attention_mask, labels = (b.to(device) for b in batch)
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            if (step + 1) % 20 == 0:
                print(f"  epoch {epoch + 1}/{EPOCHS} step {step + 1}/{len(train_loader)} loss {loss.item():.4f}")
        print(f"epoch {epoch + 1}/{EPOCHS} 平均 loss: {total_loss / len(train_loader):.4f}")

    # 评估
    model.eval()
    all_preds, all_probs = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, _ = (b.to(device) for b in batch)
            logits = model(input_ids, attention_mask=attention_mask).logits
            probs = torch.softmax(logits, dim=1)[:, 1]  # 攻击概率
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)
    y_true = y_val.cpu().numpy()

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    return metrics, y_true, y_pred, y_prob


def main():
    global EPOCHS, BATCH_SIZE, MAX_LENGTH
    parser = argparse.ArgumentParser(description="CodeBERT HTTP Payload 恶意文本分类原型验证")
    parser.add_argument("--data", type=str, default=None,
                        help="真实标注 CSV 路径；不提供则使用合成演示数据")
    parser.add_argument("--model", type=str, default=MODEL_NAME, help="预训练模型名")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="最多使用多少样本（用于快速验证，默认全部/合成默认值）")
    args = parser.parse_args()

    EPOCHS, BATCH_SIZE, MAX_LENGTH = args.epochs, args.batch_size, args.max_length
    set_seed(RANDOM_STATE)

    print("=" * 60)
    print("CodeBERT HTTP 恶意 Payload 分类 - 原型验证")
    print("=" * 60)

    # ---------- 1. 数据 ----------
    if args.data:
        print(f"加载真实数据: {args.data}")
        df = load_real_payloads(args.data)
    else:
        print(f"使用合成演示数据（{N_SYNTHETIC} 条）。真实数据请用 --data 指定 CSV。")
        df = generate_synthetic_payloads(N_SYNTHETIC)

    if args.max_samples and len(df) > args.max_samples:
        df = df.sample(args.max_samples, random_state=RANDOM_STATE)

    print(f"样本数: {len(df)}, 攻击占比: {(df['Label'] == 'ATTACK').mean():.2%}")
    y = (df["Label"] == "ATTACK").astype(int).values
    X = df["text"].tolist()
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # ---------- 2. 加载 CodeBERT ----------
    print(f"\n加载预训练模型: {args.model}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model, num_labels=2
        )
    except Exception as e:
        print(f"模型加载失败（请检查网络 / 模型名是否正确）: {e}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"计算设备: {device}")

    # ---------- 3. 训练与评估 ----------
    print("\nTokenizing ...")
    train_enc = tokenize(tokenizer, X_train)
    val_enc = tokenize(tokenizer, X_val)
    y_train_t = torch.tensor(y_train)
    y_val_t = torch.tensor(y_val)

    print("训练中 ...")
    metrics, y_true, y_pred, y_prob = train_and_evaluate(
        model, tokenizer, train_enc, y_train_t, val_enc, y_val_t, device
    )

    # ---------- 4. 结果 ----------
    print("\n" + "=" * 60)
    print("原型验证结果")
    print("=" * 60)
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-Score : {metrics['f1']:.4f}")
    print("\n混淆矩阵:\n", confusion_matrix(y_true, y_pred))

    # 示例预测（展示前 5 条测试样本）
    print("\n示例预测:")
    shown = 0
    for text, t, p, prob in zip(X_val, y_true, y_pred, y_prob):
        if shown >= 5:
            break
        truth = "ATTACK" if t == 1 else "BENIGN"
        pred = "ATTACK" if p == 1 else "BENIGN"
        flag = "✓" if t == p else "✗"
        print(f"  [{flag}] 真实={truth:<6} 预测={pred:<6} 攻击概率={prob:.3f} | {text[:60]}")
        shown += 1

    print("\n结论: CodeBERT 在该验证集上的表现如上，可作为 HTTP 恶意 Payload")
    print("文本分类的可行性参考；实际落地需更大规模真实标注数据与调参。")


if __name__ == "__main__":
    main()
