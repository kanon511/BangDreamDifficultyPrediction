import json
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from data_utils import parse_chart_json
from train import DifficultyRegressor
import os

plt.rcParams['font.family'] = 'SimHei'  # 使用黑体支持中文
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# ===== 参数设置 =====
MODEL_PATH = "model/difficulty_model.pt"
INPUT_JSON = "charts/128-4_29.json"  # 替换为你的谱面路径
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== 载入模型 =====
def load_model(input_dim):
    model = DifficultyRegressor(input_dim).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

# ===== 按时间切割谱面 JSON 为多个片段 =====
def segment_chart_by_time(json_data, num_segments=30):
    """
    将谱面按照时间划分为多个片段，支持 Slide（根据 connection 的 time 切割）。
    要求输入数据已通过 annotate_with_time_preserve_slide 处理，注入 time 字段。
    """
    def extract_times(event):
        if event["type"] == "Slide":
            return [conn["time"] for conn in event["connections"] if "time" in conn]
        elif event["type"] == "BPM":
            return []
        else:
            return [event.get("time", 0)]

    # 获取所有音符（非 BPM）的时间列表
    all_times = []
    for note in json_data:
        all_times.extend(extract_times(note))
    if not all_times:
        return []

    max_time = max(all_times)
    segment_points = np.linspace(0.5, max_time, num_segments)
    segments = []

    for t in segment_points:
        segment = []

        for event in json_data:
            if event["type"] == "BPM":
                segment.append(event)
            elif event["type"] == "Slide":
                # 检查是否有至少一个 connection 在当前时间段内
                keep_connections = [c for c in event["connections"] if "time" in c and c["time"] <= t]
                if len(keep_connections) >= 2:
                    # 至少保留两个连接才能形成有效 Slide
                    new_slide = event.copy()
                    new_slide["connections"] = keep_connections
                    segment.append(new_slide)
            else:
                if event.get("time", 0) <= t:
                    segment.append(event)

        segments.append((t, segment))

    return segments


# ===== 主流程 =====
def main(input_json = None):
    if input_json is None:
        input_json = INPUT_JSON
    else:
        input_json = os.path.join("charts", input_json)

    # 加载 JSON
    with open(input_json, 'r') as f:
        raw_data = json.load(f)

    # 添加时间属性（用于切分）
    from data_utils import annotate_with_time
    raw_data = annotate_with_time(raw_data)

    # 切分谱面
    segments = segment_chart_by_time(raw_data, num_segments=150)

    predictions = []
    input_dim = 12
    model = load_model(input_dim)

    for t, seg_data in segments:
        try:
            features = parse_chart_json(seg_data)
            if len(features) == 0:
                predictions.append((t, None))
                continue
            x = torch.tensor([features], dtype=torch.float32).to(DEVICE)
            with torch.no_grad():
                pred = model(x).item()
            predictions.append((t, pred))
        except Exception as e:
            predictions.append((t, None))
            continue

    # ===== 过滤无效预测 & 可视化 =====
    filtered = [(t, p) for t, p in predictions if p is not None]
    if not filtered:
        print("无法生成有效预测")
        return

    times, preds = zip(*filtered)

    times = np.array(times)
    preds = np.array(preds)


    # === 非线性变换函数 ===
    def nonlinear_transform(y):
        return (y - 10) ** 4  # 例如：非线性提升（根号放缓高段变化）

    # 变换后的绘图数据
    transformed_preds = nonlinear_transform(preds)

    # 初始化图像
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(times, transformed_preds, color="blue")
    ax.set_title("谱面分段预测难度随时间变化（非线性）")
    ax.set_xlabel("谱面时间（秒）")
    ax.set_xlim(times[0], times[-1])

    # 设置Y轴刻度：选择几个关键原始值，显示它们在非线性轴的位置
    raw_ticks = [10, 15] + [x for x in range(16, int(preds[-1]) + 2)] if preds[-1] > 14 else []
    trans_ticks = nonlinear_transform(np.array(raw_ticks + [preds[-1] + 1]))
    ax.set_yticks(trans_ticks)
    ax.set_yticklabels([f"{v:.0f}" for v in raw_ticks] + [" "])
    ax.set_ylabel("预测难度（非线性显示）")
    ax.grid(True)

    # 动态元素
    vline = ax.axvline(x=times[0], color='red', linestyle='--')
    time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, verticalalignment='top', fontsize=24)

    # 控制变量
    start_time = None
    play_started = [False]

    # 空格开始播放
    def on_key(event):
        if event.key == ' ' and not play_started[0]:
            play_started[0] = True
            global start_time
            start_time = time.time()
            print("开始播放")

    fig.canvas.mpl_connect('key_press_event', on_key)

    # 实时动画更新
    def update(frame):
        global start_time
        if not play_started[0] or start_time is None:
            return vline, time_text

        elapsed = time.time() - start_time

        # 播放超出时间范围则停止更新
        if elapsed > times[-1]:
            elapsed = times[-1]

        # 插值计算预测值（线性）
        current_pred = np.interp(elapsed, times, preds)

        # 更新动态元素
        vline.set_xdata([elapsed, elapsed])
        time_text.set_text(f"时间: {elapsed:.2f} s\n预测值: {current_pred:.2f}")
        return vline, time_text

    # 创建高帧率动画（约60FPS）
    ani = animation.FuncAnimation(
        fig,
        update,
        interval=16,              # ~60帧每秒
        blit=True,
        cache_frame_data=False    # 避免 warning 和缓存过多帧
    )

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main("187-4_29.json")
