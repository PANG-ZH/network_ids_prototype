#!/usr/bin/env python3
"""
基于机器学习的网络入侵检测原型系统 - 训练脚本
使用 CICIDS2017 风格数据，训练 Random Forest 与 XGBoost 二分类模型
准确率目标：99%+
"""

import os
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from xgboost import XGBClassifier
import joblib

warnings.filterwarnings("ignore")

# ==================== 配置 ====================
DATA_DIR = Path(__file__).parent / "data"
MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2
# 为了演示可快速跑通，默认使用合成数据；真实使用时把 USE_SYNTHETIC=False 并放入 CICIDS2017 CSV
USE_SYNTHETIC = True


def generate_synthetic_cicids(n_samples: int = 50000) -> pd.DataFrame:
    """
    生成模拟 CICIDS2017 风格的流量特征数据（约 20 个关键特征）
    真实项目中请替换为真实 CICIDS2017 CSV
    """
    rng = np.random.default_rng(RANDOM_STATE)
    n_benign = int(n_samples * 0.7)
    n_attack = n_samples - n_benign

    def make_flows(n, is_attack=False):
        data = {
            "Destination Port": rng.integers(1, 65535, n),
            "Flow Duration": rng.integers(1, 120000000, n),
            "Total Fwd Packets": rng.integers(1, 500, n),
            "Total Backward Packets": rng.integers(0, 500, n),
            "Total Length of Fwd Packets": rng.integers(0, 100000, n),
            "Total Length of Bwd Packets": rng.integers(0, 100000, n),
            "Fwd Packet Length Max": rng.integers(0, 1500, n),
            "Fwd Packet Length Min": rng.integers(0, 100, n),
            "Fwd Packet Length Mean": rng.uniform(0, 800, n),
            "Bwd Packet Length Max": rng.integers(0, 1500, n),
            "Bwd Packet Length Mean": rng.uniform(0, 800, n),
            "Flow Bytes/s": rng.uniform(0, 1e7, n),
            "Flow Packets/s": rng.uniform(0, 1e5, n),
            "Flow IAT Mean": rng.uniform(0, 1e6, n),
            "Flow IAT Std": rng.uniform(0, 1e6, n),
            "Fwd IAT Mean": rng.uniform(0, 1e6, n),
            "Bwd IAT Mean": rng.uniform(0, 1e6, n),
            "Packet Length Mean": rng.uniform(0, 1000, n),
            "Packet Length Std": rng.uniform(0, 500, n),
            "Average Packet Size": rng.uniform(0, 1000, n),
        }
        df = pd.DataFrame(data)
        if is_attack:
            # 攻击流量特征偏移（模拟 DoS / PortScan / BruteForce 等）
            df["Flow Duration"] = rng.integers(1, 5000000, n)
            df["Flow Packets/s"] = rng.uniform(1000, 5e5, n)
            df["Fwd Packet Length Max"] = rng.integers(100, 1500, n)
            df["Destination Port"] = rng.choice([21, 22, 80, 443, 3389, 445], n)
        return df

    benign = make_flows(n_benign, is_attack=False)
    attack = make_flows(n_attack, is_attack=True)
    benign["Label"] = "BENIGN"
    attack["Label"] = "ATTACK"
    df = pd.concat([benign, attack], ignore_index=True)
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    return df


def load_real_cicids(data_dir: Path) -> pd.DataFrame:
    """
    加载真实 CICIDS2017 MachineLearningCSV 中的一个或多个 CSV
    用户需要自行下载并放到 data/ 目录
    推荐文件：Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv 等
    """
    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"未在 {data_dir} 找到 CSV 文件。\n"
            "请从 https://www.unb.ca/cic/datasets/ids-2017.html 下载 MachineLearningCSV，\n"
            "或从 Kaggle 搜索 CICIDS2017 后把 CSV 放入 data/ 目录。"
        )
    dfs = []
    for f in csv_files:
        print(f"加载: {f.name}")
        df = pd.read_csv(f, low_memory=False)
        # 统一列名（去掉空格）
        df.columns = [c.strip() for c in df.columns]
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)
    # 二分类标签
    if "Label" in data.columns:
        data["Label"] = data["Label"].apply(
            lambda x: "BENIGN" if str(x).strip().upper() == "BENIGN" else "ATTACK"
        )
    return data


def preprocess(df: pd.DataFrame):
    """清洗、特征选择、标准化"""
    # 删除无用列
    drop_cols = [
        "Flow ID", "Source IP", "Source Port", "Destination IP",
        "Timestamp", "Protocol"
    ]
    for c in drop_cols:
        if c in df.columns:
            df = df.drop(columns=c)

    # 处理 inf / nan
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    # 标签
    y = (df["Label"] != "BENIGN").astype(int)  # 0=BENIGN, 1=ATTACK
    X = df.drop(columns=["Label"])

    # 只保留数值特征
    X = X.select_dtypes(include=[np.number])

    # 简单特征选择：去掉方差极低的列
    variances = X.var()
    keep = variances[variances > 1e-6].index
    X = X[keep]

    print(f"特征数量: {X.shape[1]}, 样本数量: {X.shape[0]}")
    print(f"攻击占比: {y.mean():.2%}")

    return X, y


def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    # ---------- Random Forest ----------
    print("\n[1/2] 训练 Random Forest ...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        class_weight="balanced"
    )
    rf.fit(X_train_scaled, y_train)
    y_pred_rf = rf.predict(X_test_scaled)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    results["RandomForest"] = {
        "accuracy": acc_rf,
        "precision": precision_score(y_test, y_pred_rf),
        "recall": recall_score(y_test, y_pred_rf),
        "f1": f1_score(y_test, y_pred_rf),
        "model": rf
    }
    print(f"Random Forest 准确率: {acc_rf:.4f}")

    # ---------- XGBoost ----------
    print("\n[2/2] 训练 XGBoost ...")
    xgb = XGBClassifier(
        n_estimators=150,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    xgb.fit(X_train_scaled, y_train)
    y_pred_xgb = xgb.predict(X_test_scaled)
    acc_xgb = accuracy_score(y_test, y_pred_xgb)
    results["XGBoost"] = {
        "accuracy": acc_xgb,
        "precision": precision_score(y_test, y_pred_xgb),
        "recall": recall_score(y_test, y_pred_xgb),
        "f1": f1_score(y_test, y_pred_xgb),
        "model": xgb
    }
    print(f"XGBoost 准确率: {acc_xgb:.4f}")

    # 打印详细报告
    print("\n===== XGBoost 详细报告 =====")
    print(classification_report(y_test, y_pred_xgb, target_names=["BENIGN", "ATTACK"]))
    print("混淆矩阵:\n", confusion_matrix(y_test, y_pred_xgb))

    # 保存最优模型（通常 XGBoost 或 RF 都 >99%）
    best_name = max(results, key=lambda k: results[k]["accuracy"])
    best_model = results[best_name]["model"]
    print(f"\n最优模型: {best_name} (准确率 {results[best_name]['accuracy']:.4f})")

    joblib.dump(best_model, MODEL_DIR / "ids_model.joblib")
    joblib.dump(scaler, MODEL_DIR / "scaler.joblib")
    joblib.dump(list(X.columns), MODEL_DIR / "feature_names.joblib")

    # 同时保存两个模型方便对比
    joblib.dump(results["RandomForest"]["model"], MODEL_DIR / "rf_model.joblib")
    joblib.dump(results["XGBoost"]["model"], MODEL_DIR / "xgb_model.joblib")

    print(f"\n模型已保存到: {MODEL_DIR}")
    return results


def main():
    print("=" * 60)
    print("网络入侵检测原型系统 - 模型训练")
    print("=" * 60)

    if USE_SYNTHETIC:
        print("使用合成数据（演示用）。真实项目请设置 USE_SYNTHETIC=False 并放入 CICIDS2017 CSV。")
        df = generate_synthetic_cicids(60000)
    else:
        df = load_real_cicids(DATA_DIR)

    X, y = preprocess(df)
    results = train_and_evaluate(X, y)

    print("\n训练完成！可运行 python app.py 启动 API 服务。")


if __name__ == "__main__":
    main()
