# -*- coding: utf-8 -*-
"""
对 D:\\大创资料\\dcshuju 中前 50 个 GNT 文件中的汉字打分，每个 GNT 单独一个评分标签文件。
标签文件：1_label.txt ~ 50_label.txt，内容为一行 "汉字,分数"（启发式规则打分）。
"""
import os
import sys
import csv
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DCSHUJU = r"D:\大创资料\dcshuju"
NUM_LABELED = 50


def get_char_from_gnt(path):
    try:
        with open(path, "rb") as f:
            data = f.read(6)
        if len(data) < 6:
            return ""
        return bytes([data[4], data[5]]).decode("gb2312", errors="replace")
    except Exception:
        return ""


def read_online_csv(path):
    """返回 (N, 5) [x, y, f, speed, dt]。"""
    t_list, x_list, y_list, f_list = [], [], [], []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t_list.append(float(row.get("timestamp", row.get("t", 0))))
            x_list.append(float(row.get("x", 0)))
            y_list.append(float(row.get("y", 0)))
            f_list.append(float(row.get("f", 0)))
    t = np.array(t_list, dtype=np.float32)
    x = np.array(x_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    f = np.array(f_list, dtype=np.float32)
    dt = np.zeros_like(t)
    dt[1:] = t[1:] - t[:-1]
    dx = np.zeros_like(x)
    dy = np.zeros_like(y)
    dx[1:] = x[1:] - x[:-1]
    dy[1:] = y[1:] - y[:-1]
    speed = np.zeros_like(x)
    valid = dt > 1e-6
    speed[valid] = np.sqrt(dx[valid] ** 2 + dy[valid] ** 2) / dt[valid]
    return np.stack([x, y, f, speed, dt], axis=1)


def heuristic_score_from_traj(traj):
    """
    无模型时的启发式评分：基于轨迹平滑度、力度稳定性等，归一化到 0~100。
    """
    if traj is None or len(traj) < 2:
        return 50.0
    x, y, f, speed, dt = traj[:, 0], traj[:, 1], traj[:, 2], traj[:, 3], traj[:, 4]
    # 平滑度：相邻点加速度小更好
    dx = np.diff(x)
    dy = np.diff(y)
    vx = dx / (np.clip(dt[1:], 1e-6, None))
    vy = dy / (np.clip(dt[1:], 1e-6, None))
    if len(vx) < 2:
        return 50.0
    ax = np.diff(vx)
    ay = np.diff(vy)
    acc = np.sqrt(ax**2 + ay**2)
    smooth = 1.0 / (1.0 + np.median(acc))
    # 力度稳定性：f 的变异系数小更好
    f_valid = f[f > 0]
    f_stable = 1.0 / (1.0 + np.std(f_valid)) if len(f_valid) > 0 else 0.5
    score = (smooth * 50 + f_stable * 50)
    return float(np.clip(score, 0, 100))


def main():
    print("使用启发式规则打分（轨迹平滑度 + 力度稳定性）")
    print("仅对前 %d 个 GNT 生成评分标签，每个一个文件：1_label.txt ~ %d_label.txt" % (NUM_LABELED, NUM_LABELED))

    for i in range(1, NUM_LABELED + 1):
        gnt_name = "%d.gnt" % i
        gnt_path = os.path.join(DCSHUJU, gnt_name)
        online_path = os.path.join(DCSHUJU, "%d_online.csv" % i)
        if not os.path.isfile(gnt_path):
            continue
        char = get_char_from_gnt(gnt_path)
        traj = None
        if os.path.isfile(online_path):
            traj = read_online_csv(online_path)
        score = round(heuristic_score_from_traj(traj), 2)
        label_path = os.path.join(DCSHUJU, "%d_label.txt" % i)
        with open(label_path, "w", encoding="utf-8") as f:
            f.write("%s,%.2f\n" % (char, score))
    print("已保存 1_label.txt ~ %d_label.txt 到目录: %s" % (NUM_LABELED, DCSHUJU))


if __name__ == "__main__":
    main()
