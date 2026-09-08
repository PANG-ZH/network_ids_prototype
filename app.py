#!/usr/bin/env python3
"""
基于机器学习的网络入侵检测原型系统 - Flask RESTful API
支持实时流量特征检测
"""

import os
from pathlib import Path
from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

MODEL_DIR = Path(__file__).parent / "models"
model = None
scaler = None
feature_names = None


def load_artifacts():
    global model, scaler, feature_names
    model_path = MODEL_DIR / "ids_model.joblib"
    scaler_path = MODEL_DIR / "scaler.joblib"
    feat_path = MODEL_DIR / "feature_names.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            "模型文件不存在，请先运行 python train.py 训练模型。"
        )
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_names = joblib.load(feat_path)
    print(f"模型加载成功，特征数: {len(feature_names)}")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


@app.route("/predict", methods=["POST"])
def predict():
    """
    接收单条或批量流量特征，返回是否为攻击
    请求体示例（JSON）:
    {
      "features": {
        "Destination Port": 80,
        "Flow Duration": 12345,
        ...
      }
    }
    或
    {
      "features": [ {...}, {...} ]   # 批量
    }
    """
    if model is None:
        return jsonify({"error": "模型未加载"}), 500

    data = request.get_json(force=True)
    if not data or "features" not in data:
        return jsonify({"error": "请提供 features 字段"}), 400

    feats = data["features"]
    # 统一成 list of dict
    if isinstance(feats, dict):
        feats = [feats]

    try:
        df = pd.DataFrame(feats)
        # 对齐特征顺序，缺失补 0
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0.0
        df = df[feature_names]
        X = scaler.transform(df.values)
        preds = model.predict(X)
        probs = model.predict_proba(X)[:, 1]  # 攻击概率

        results = []
        for i, (p, prob) in enumerate(zip(preds, probs)):
            results.append({
                "index": i,
                "prediction": "ATTACK" if p == 1 else "BENIGN",
                "attack_probability": float(round(prob, 4)),
                "is_attack": bool(p == 1)
            })
        return jsonify({"results": results, "count": len(results)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/features", methods=["GET"])
def list_features():
    """返回模型期望的特征列表"""
    return jsonify({"features": feature_names, "count": len(feature_names)})


if __name__ == "__main__":
    load_artifacts()
    # 开发模式
    app.run(host="0.0.0.0", port=5000, debug=False)
