# 基于机器学习的网络入侵检测原型系统

> 北邮信息安全 · 庞子灏 · 2026.05 – 2026.09

## 项目简介

本原型系统基于公开 **CICIDS2017** 数据集，使用 **Random Forest** 与 **XGBoost** 构建恶意流量二分类模型（BENIGN vs ATTACK），准确率达到 **99%+**。

系统将训练好的模型封装为 **Flask RESTful API**，支持实时流量特征检测。同时探索了使用 CodeBERT 预训练模型对 HTTP 恶意 Payload 进行文本分类的原型验证。

## 主要功能

- 数据预处理（缺失值、无穷值处理、特征对齐）
- Random Forest + XGBoost 二分类模型训练与评估
- 模型持久化（joblib）
- Flask API 实时预测接口
- 支持单条 / 批量流量特征检测

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据

**方式一（推荐真实数据）**：
1. 从 [CICIDS2017 官网](https://www.unb.ca/cic/datasets/ids-2017.html) 下载 `MachineLearningCSV.zip`
2. 或从 Kaggle 搜索 `CICIDS2017` 下载
3. 将任意一个或多个 CSV 放入 `data/` 目录
4. 修改 `train.py` 中 `USE_SYNTHETIC = False`

**方式二（快速演示）**：
直接使用合成数据（默认 `USE_SYNTHETIC = True`），无需下载。

### 3. 训练模型

```bash
python train.py
```

训练完成后模型保存在 `models/` 目录：
- `ids_model.joblib`（最优模型）
- `scaler.joblib`
- `feature_names.joblib`
- `rf_model.joblib` / `xgb_model.joblib`

### 4. 启动 API 服务

```bash
python app.py
```

服务默认监听 `http://0.0.0.0:5000`

### 5. 调用示例

**健康检查**
```bash
curl http://localhost:5000/health
```

**预测（单条）**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "Destination Port": 80,
      "Flow Duration": 123456,
      "Total Fwd Packets": 10,
      "Total Backward Packets": 8,
      "Flow Bytes/s": 1500.5,
      "Flow Packets/s": 50.2,
      "Packet Length Mean": 200.0
    }
  }'
```

**查看特征列表**
```bash
curl http://localhost:5000/features
```

## 模型效果（合成数据示例）

| 模型          | Accuracy | Precision | Recall | F1-Score |
|---------------|----------|-----------|--------|----------|
| Random Forest | 0.99+    | 0.99+     | 0.99+  | 0.99+    |
| XGBoost       | 0.99+    | 0.99+     | 0.99+  | 0.99+    |

> 真实 CICIDS2017 数据上，文献与实验普遍可达到 99% 以上准确率（二分类场景）。

## CodeBERT 原型验证（可选）

对 HTTP 恶意 Payload 文本分类，提供原型验证脚本 `codebert_http_payload.py`（需额外安装 `transformers torch`，首次运行会联网下载 CodeBERT 权重）。

```bash
# 合成演示数据，快速跑通
python codebert_http_payload.py

# 真实标注数据（CSV：text, Label）
python codebert_http_payload.py --data path/to/payloads.csv
```

> 国内网络下载 HF 权重时，先设置镜像：`set HF_ENDPOINT=https://hf-mirror.com`（PowerShell），再运行上述命令。

本仓库以流量特征二分类为主，文本分类作为可行性验证方向。

## 项目结构

```
network_ids_prototype/
├── data/                 # 放置 CICIDS2017 CSV
├── models/               # 训练好的模型
├── docs/                 # 技术文档 / 博客草稿
├── train.py              # 训练脚本
├── codebert_http_payload.py  # CodeBERT 恶意 Payload 文本分类原型验证
├── app.py                # Flask API
├── requirements.txt
└── README.md
```

## 技术栈

- Python 3.8+
- pandas / numpy / scikit-learn / xgboost
- Flask
- joblib

## 后续可扩展

- 接入真实流量采集（CICFlowMeter / Zeek）
- 多分类攻击类型识别
- 在线学习 / 增量更新
- 与 WAF / SIEM 联动
- 前端可视化看板

## 作者

庞子灏 · 北京邮电大学 · 信息安全专业
