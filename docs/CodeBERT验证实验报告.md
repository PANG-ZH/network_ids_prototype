# CodeBERT HTTP 恶意 Payload 分类原型验证实验报告

- 日期：2026-09-08
- 环境：Windows / Python 3.13 / PyTorch 2.14.0+cu132 / transformers 5.16.1
- 硬件：NVIDIA GeForce RTX 4050 Laptop GPU（6 GB 显存）
- 关联脚本：[codebert_http_payload.py](../codebert_http_payload.py)

## 1. 实验目的

在基于 CICIDS2017 流量统计特征（RF / XGBoost）的二分类之外，验证另一条技术路线——
使用预训练 CodeBERT（`microsoft/codebert-base`）对 HTTP 请求 Payload 做**文本分类**的可行性，
判断能否识别恶意请求（SQLi / XSS / 路径穿越 / 命令注入等）。

两类实验依次完成：

1. 合成演示数据：验证训练 / 评估 / 预测流程可端到端跑通；
2. CSIC 2010 真实数据：评估模型在真实 HTTP 流量上的泛化能力。

## 2. 方法

### 2.1 模型与训练设置

- 预训练模型：`microsoft/codebert-base`（RoBERTa 架构，在代码/文本上预训练），
  末尾替换为随机初始化的二分类头（`BENIGN` / `ATTACK`）。
- 优化器：AdamW，学习率 `2e-5`，线性衰减（warmup=0）。
- 批大小：8；文本截断长度：合成实验 128、真实数据实验 256（真实请求较长）。
- 数据集切分：80% 训练 / 20% 验证（分层）。
- 训练轮数：2。

### 2.2 数据

**合成数据**：由 10 类攻击模板（SQLi、XSS、命令注入、路径穿越、越权等）与
10 类正常请求模板拼接随机参数生成，共 1000 条（恶意/正常各半）。
仅用于流程验证，不用于结论性评估。

**CSIC 2010 数据集**（真实数据）：
- 由西班牙国家研究委员会信息安全研究所生成，模拟电商 Web 应用的 HTTP/1.1 流量。
- 组成：36000 条正常训练 / 36000 条正常测试 / 25065 条异常（攻击）。
- 原始 txt 为逐条完整的 HTTP 请求（请求行 + 头 + 可选正文），从公开 GitLab 镜像下载，
  按请求首行（`GET/POST/...`）切分解析为请求文本。
- 本次实验抽取 2000 条正常 + 2000 条攻击组成平衡验证集（共 4000 条），保存为
  [payloads_csic2010.csv](../data/csic2010/payloads_csic2010.csv)，列为 `text, Label`。
- 请求文本长度：均值约 590 字符，中位数约 543 字符。

## 3. 实验结果

### 3.1 合成数据（流程验证，1000 条 × 2 epoch）

| 指标 | 数值 |
|---|---|
| Accuracy | 1.0000 |
| Precision | 1.0000 |
| Recall | 1.0000 |
| F1-Score | 1.0000 |

混淆矩阵（验证集 200 条）：

```
[[100   0]
 [  0 100]]
```

训练 loss 由第 1 epoch 均值 0.2315 收敛至第 2 epoch 均值 0.0045。

> 说明：合成模板数据变体有限、几乎无噪声，100% 仅证明流程正确、CodeBERT 具备学习该类文本
> 判别模式的能力，**不代表真实泛化水平**。

### 3.2 CSIC 2010 真实数据（4000 条 × 2 epoch，max-length 256）

| 指标 | 数值 |
|---|---|
| Accuracy | 0.9012 |
| Precision | 0.8559 |
| Recall | 0.9650 |
| F1-Score | 0.9072 |

混淆矩阵（验证集 800 条）：

```
[[335  65]   正常 → 65 条误报为攻击（正常流量误报率 ≈ 16.3%）
 [ 14 386]]  攻击 → 14 条漏报（攻击漏报率 ≈ 3.5%）
```

训练 loss：第 1 epoch 均值 0.4223 → 第 2 epoch 均值 0.1855，正常下降。

### 3.3 结果分析

- **攻击召回率高达 96.5%**：绝大多数恶意请求（SQLi/XSS/路径穿越等）能被识别，漏报率仅 3.5%，
  说明攻击载荷中的异常模式（编码、关键字、非常规结构）被 CodeBERT 有效捕获。
- **正常流量误报率约 16.3%**：部分合法请求被判为攻击，是主要的精度损耗来源。
  这与 WAF 类检测器的典型分布一致——安全场景优先保召回（宁可误报不漏报）。
- 示例预测显示模型对判定给出很高置信度（正确样本攻击概率 ≈ 0.99 / 0.00 量级）。

## 4. 复现方式

```bash
# 环境
pip install transformers torch pandas scikit-learn
# 国内下载模型权重时先执行：
# set HF_ENDPOINT=https://hf-mirror.com

# 1) 合成数据演示
python codebert_http_payload.py

# 2) CSIC 2010 真实数据（CSV 已生成）
python codebert_http_payload.py --data data/csic2010/payloads_csic2010.csv --max-length 256

# 3) 用你自己的标注数据（列名 text / Label，Label 除 BENIGN 外视为 ATTACK）
python codebert_http_payload.py --data path/to/your.csv
```

## 5. 局限与后续方向

局限：

- 原型规模较小（4000 条、2 epoch），未做学习率/轮数/截断长度调优；
- 误报率偏高，尚未引入正常测试集（`normalTrafficTest.txt`）检验多样合法请求下的稳定性；
- CSIC 2010 流量来自单一电商应用，域差异（真实互联网 Web 流量）下效果未知；
- 未做训练/测试同源以外的跨集评估。

建议后续：

- 扩大样本与轮数，加入正常测试集做交叉评估；
- 加入阈值调节（在攻击概率阈值上权衡 Precision/Recall）；
- 换用更大的代码预训练模型或轻量蒸馏模型对比成本与效果；
- 与统计特征路线（RF / XGBoost，CICFlowMeter 特征）做集成对比。

## 6. 结论

CodeBERT 文本分类路线在 CSIC 2010 真实 HTTP 流量上达到 **F1 ≈ 0.91（召回 96.5%）**，
证明利用预训练代码模型识别 HTTP 恶意 Payload **具有可行性**，可作为流量统计特征
检测之外的补充手段。原型验证目的达成；工程化落地仍需更大规模真实数据与调优。
