import json
from predict import predict_difficulty
import os
import csv

DATA_DIR = "charts"
OUTPUT_FILE = "difficulty_predictions.csv"

# 准备保存结果的列表
results = []

# 遍历目录中的文件
for file in os.listdir(DATA_DIR):
    try:
        if not file.endswith(".json"):
            continue
        with open(os.path.join(DATA_DIR, file), 'r', encoding='utf-8') as f:
            chart = json.load(f)
        
        if int(file.split("_")[0].split("-")[1]) < 3:
            continue

        difficulty = predict_difficulty(chart)
        results.append([file, difficulty, file.split("_")[1].split(".")[0]])
    except Exception as e:
        print(f"预测出错：{file}，{e}")

# 将结果写入CSV文件
with open(OUTPUT_FILE, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    # 写入表头
    writer.writerow(['File', 'Predict', 'Level'])
    # 写入结果数据
    writer.writerows(results)

print(f"所有预测结果已保存到 {OUTPUT_FILE}")
