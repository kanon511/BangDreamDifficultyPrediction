import numpy as np
import json
import os

WINDOW_SIZE_BEAT = 4
STRIDE_BEAT = 2

def beat_to_time(beat, bpm_events):
    time = 0
    last_beat = 0
    current_bpm = bpm_events[0]['bpm']
    for bpm in bpm_events[1:]:
        if beat < bpm['beat']:
            break
        duration = (bpm['beat'] - last_beat) * 60 / current_bpm
        time += duration
        last_beat = bpm['beat']
        current_bpm = bpm['bpm']
    time += (beat - last_beat) * 60 / current_bpm
    return time

def assign_hand(lane):
    return 'L' if lane <= 3 else 'R'

def beat_to_time(beat, bpm_events):
    # Placeholder: Replace with your actual conversion logic
    return beat  # assume 1 beat = 1 second for now

def compute_slide_span(connections):
    diffs = [abs(c2['lane'] - c1['lane']) for c1, c2 in zip(connections[:-1], connections[1:])]
    if all(d <= 1 for d in diffs):
        return 0
    return sum(diffs)

def extract_hand_features(notes, bpm_events):
    for note in notes:
        if note['type'] == 'Slide':
            for c in note['connections']:
                c['time'] = beat_to_time(c['beat'], bpm_events)
        else:
            note['time'] = beat_to_time(note['beat'], bpm_events)

    max_beat = max(
        [c['beat'] if n['type'] == 'Slide' else n['beat'] for n in notes for c in (n.get('connections', [n]))]
    )
    features = []
    beat = 0
    while beat < max_beat:
        end = beat + WINDOW_SIZE_BEAT
        segment = [n for n in notes if any(beat <= (c['beat'] if n['type'] == 'Slide' else n['beat']) < end for c in n.get('connections', [n]))]

        hand_data = {
            'L': {
                'key_lanes': [], 'key_times': [],  # 普通按键 + Slide 起始点
                'flicks': 0,
                'slide_spans': [], 'slide_max_span': 0
            },
            'R': {
                'key_lanes': [], 'key_times': [],
                'flicks': 0,
                'slide_spans': [], 'slide_max_span': 0
            }
        }

        for n in segment:
            if n['type'] == 'Slide':
                conns = n['connections']
                slide_span = compute_slide_span(conns)
                if conns:
                    start = conns[0]
                    end_node = conns[-1]

                    # 起点特征加入
                    h_start = assign_hand(start['lane'])
                    hand_data[h_start]['key_lanes'].append(start['lane'])
                    hand_data[h_start]['key_times'].append(start['time'])

                    # 统计 flick（仅起点/终点参与 flick 统计）
                    if start.get('flick', False):
                        hand_data[h_start]['flicks'] += 1
                    if end_node.get('flick', False):
                        h_end = assign_hand(end_node['lane'])
                        hand_data[h_end]['flicks'] += 1

                    # Slide 跨度特征（不参与 key span 统计）
                    hand_data[h_start]['slide_spans'].append(slide_span)
                    hand_data[h_start]['slide_max_span'] = max(
                        hand_data[h_start]['slide_max_span'], slide_span
                    )
            else:
                h = assign_hand(n['lane'])
                hand_data[h]['key_lanes'].append(n['lane'])
                hand_data[h]['key_times'].append(n['time'])
                if n.get('flick', False):
                    hand_data[h]['flicks'] += 1

        feat_vector = []
        for hand in ['L', 'R']:
            lanes = hand_data[hand]['key_lanes']
            times = hand_data[hand]['key_times']

            # 跨度计算：相邻 key_lanes 的 lane 跨度
            spans = [abs(l2 - l1) for l1, l2 in zip(lanes[:-1], lanes[1:])]
            avg_span = np.mean(spans) if spans else 0
            max_span = max(spans) if spans else 0

            density = len(times) / WINDOW_SIZE_BEAT
            flick_count = hand_data[hand]['flicks']
            slide_avg_span = np.mean(hand_data[hand]['slide_spans']) if hand_data[hand]['slide_spans'] else 0
            slide_max_span = hand_data[hand]['slide_max_span']

            feat_vector.extend([avg_span, max_span, density, flick_count, slide_avg_span, slide_max_span])

        features.append(feat_vector)
        beat += STRIDE_BEAT

    return np.array(features, dtype=np.float32)

def parse_chart_json(chart):
    bpm_events = sorted([e for e in chart if e['type'] == 'BPM'], key=lambda x: x['beat'])
    notes = sorted([n for n in chart if n['type'] != 'BPM' and n['type'] != 'Directional'], key=lambda x: x.get('beat', 0))
    return extract_hand_features(notes, bpm_events)

def load_all_from_folder(folder):
    X_data = []
    y_data = []
    for fname in os.listdir(folder):
        if not fname.endswith(".json"):
            continue
        try:
            label = int(fname.split(".")[0])
        except ValueError:
            continue
        fpath = os.path.join(folder, fname)
        try:
            with open(fpath, 'r') as f:
                chart = json.load(f)
            feat = parse_chart_json(chart)
            if len(feat) > 0:
                X_data.append(feat)
                y_data.append(label)
        except:
            continue
    return X_data, y_data

def annotate_with_time(chart_data):
    """
    给所有音符添加 time 字段，Slide 保持原结构，在 connections 中嵌入 time。
    依赖外部提供的 beat_to_time 工具。
    """
    # 提取 BPM 信息
    bpm_list = [e for e in chart_data if e["type"] == "BPM"]
    bpm_list.sort(key=lambda x: x["beat"])

    def beat_to_time(beat, bpm_events):
        time = 0
        last_beat = 0
        current_bpm = bpm_events[0]['bpm']
        for bpm in bpm_events[1:]:
            if beat < bpm['beat']:
                break
            duration = (bpm['beat'] - last_beat) * 60 / current_bpm
            time += duration
            last_beat = bpm['beat']
            current_bpm = bpm['bpm']
        time += (beat - last_beat) * 60 / current_bpm
        return time

    annotated = []

    for event in chart_data:
        if event["type"] == "BPM":
            annotated.append(event)
        elif event["type"] == "Slide":
            new_event = dict(event)
            new_connections = []
            for conn in event["connections"]:
                conn = dict(conn)
                conn["time"] = beat_to_time(conn["beat"], bpm_list)
                new_connections.append(conn)
            new_event["connections"] = new_connections

            # 添加 Slide 本体时间：使用第一个 connection 的时间
            if new_connections:
                new_event["time"] = new_connections[0]["time"]
            annotated.append(new_event)
        else:
            new_event = dict(event)
            new_event["time"] = beat_to_time(event["beat"], bpm_list)
            annotated.append(new_event)

    return annotated

def mirror_chart(chart_data):
    """
    镜像翻转谱面，同时镜像所有音符的位置。
    """
    # 镜像所有音符的位置
    for event in chart_data:
        if event["type"] == "Slide":
            for conn in event["connections"]:
                conn["lane"] = 6 - conn["lane"]
        else:
            if "lane" in event.keys():
                event["lane"] = 6 - event["lane"]

    return chart_data