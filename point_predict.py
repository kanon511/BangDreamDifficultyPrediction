import json
import numpy as np
import matplotlib.pyplot as plt
from predict import predict_difficulty
from data_utils import annotate_with_time

plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

def generate_realtime_difficulty_curve(json_path, window_size=4.0, step=0.5):
    # 读取并注入 time
    with open(json_path, 'r', encoding='utf-8') as f:
        chart_data = json.load(f)
    chart_with_time = annotate_with_time(chart_data)

    # 拆分音符
    notes = [n for n in chart_with_time if n['type'] != 'BPM']
    bpm_events = [n for n in chart_with_time if n['type'] == 'BPM']
    all_times = sorted([n['time'] for n in notes])
    if not all_times:
        print("⚠️ 没有音符数据。")
        return

    max_time = all_times[-1]

    times = []
    difficulties = []

    t = 0
    while t + window_size <= max_time:
        start_time = t
        end_time = t + window_size
        center_time = (start_time + end_time) / 2

        # 当前窗口内所有音符
        segment = [b for b in bpm_events] + [
            n for n in notes if start_time <= n['time'] < end_time
        ]

        # 跳过无音符段
        if len(segment) == len(bpm_events):
            t += step
            continue

        difficulty = predict_difficulty(segment)
        times.append(center_time)
        difficulties.append(difficulty)

        t += step

    # 绘图
    plt.figure(figsize=(12, 5))
    plt.plot(times, difficulties, color='darkred', linewidth=2, marker=' ')
    plt.title("谱面实时难度趋势曲线（不是准确难度）")
    plt.xlabel("时间 (秒)")
    plt.ylabel("局部预测难度趋势")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 示例调用
generate_realtime_difficulty_curve("charts/243-3_28.json")
