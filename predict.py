import json
import sys

import torch
import torch.nn as nn
from train import DifficultyRegressor

from data_utils import parse_chart_json  # 替代旧的特征提取函数

# 加载模型
def load_model(input_dim, model_path="model/difficulty_model.pt"):
    model = DifficultyRegressor(input_dim)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict_difficulty(chart):
    feat = parse_chart_json(chart)
    if len(feat) == 0:
        print("谱面音符过少，无法预测。")
        return None

    model = load_model(input_dim=feat.shape[1])
    with torch.no_grad():
        input_tensor = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)
        pred = model(input_tensor).item()
        return round(pred, 2)

# 预测谱面难度
if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("请指定谱面路径：")
        chart_path = input()
    else:
        chart_path = sys.argv[1]
    with open(chart_path, 'r') as f:
        chart = json.load(f)
    difficulty = predict_difficulty(chart)
    if difficulty is None:
        exit(1)
    print(f"该谱面的难度为 {difficulty}。")