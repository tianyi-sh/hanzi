# -*- coding: utf-8 -*-
"""解析在线轨迹数据：时间戳 t、坐标 (x,y)、压力 f，并生成 (x, y, f, speed, dt)。"""
import csv
import numpy as np


def read_online_csv(path):
    """
    读取在线轨迹 CSV（表头 timestamp,x,y,f）。
    Returns:
        traj: np.ndarray (N, 5), 列顺序 [x, y, f, speed, dt]
        N 为点数；dt 为与上一帧时间差（首帧 dt=0），speed 为瞬时速度（首帧 0）。
    """
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
    traj = np.stack([x, y, f, speed, dt], axis=1)
    return traj
