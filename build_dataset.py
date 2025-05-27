import json
import os
import numpy as np
from data_utils import parse_chart_json, mirror_chart
import random

DATA_DIR = "charts"  # 你的谱面文件夹路径
SAVE_PATH = "dataset/training_data.npz"

X_data = []
y_data = []

for file in os.listdir(DATA_DIR):
    if not file.endswith(".json"):
        continue
    try:
        if file.split("_")[0].split("-")[1] == '0' and random.random() < 0.8:
            continue
        if file.split("_")[0].split("-")[1] == '1' and random.random() < 0.7:
            continue
        if file.split("_")[0].split("-")[1] == '2' and int(file.split("_")[1].split(".")[0]) < 20 and random.random() < 0.6:
            continue
        if file.split("_")[0].split("-")[1] == '2' and int(file.split("_")[1].split(".")[0]) >= 20:
            continue

        label = int(file.split("_")[1].split(".")[0])  # 从文件名提取参考难度
    except ValueError:
        print(f"跳过非法文件名: {file}")
        continue

    chart_path = os.path.join(DATA_DIR, file)
    try:
        with open(chart_path, 'r') as f:
            chart = json.load(f)
        features = parse_chart_json(chart)
        mirror_chart(chart)
        features_mirror = parse_chart_json(chart)
        if len(features) == 0:
            print(f"跳过音符过少的谱面: {file}")
            continue
        for i in range(25, max(26, label)):
            X_data.append(features)
            y_data.append(label)
            if file.split("_")[0].split("-")[1] == '3' or file.split("_")[0].split("-")[1] == '4':
                X_data.append(features_mirror)
                y_data.append(label)
        print(f"处理 {file}，参考难度 {label}")
    except Exception as e:
        print(f"解析失败 {file}: {e}")

# 转换为 numpy 对象数组
X_data = np.array(X_data, dtype=object)
y_data = np.array(y_data, dtype=np.int32)

# 保存为 npz
np.savez_compressed(SAVE_PATH, X=X_data, y=y_data)
print(f"共处理谱面 {len(X_data)} 份，已保存至 {SAVE_PATH}")
