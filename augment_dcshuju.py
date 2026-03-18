# -*- coding: utf-8 -*-
"""
在 D:\\大创资料\\dcshuju 中扩充数据集：
- 每个原始 GNT（1.gnt~50.gnt）生成 5 个加噪 GNT，编号 51~300
- 每个新 GNT 对应生成一份加噪的在线 CSV（51_online.csv~300_online.csv）
- 不生成 51~300 的标签，仅保留 50 个原始样本的标签
结果：300 个 GNT + 300 个在线 CSV + 50 条标签
"""
import os
import csv
import random
import struct

DCSHUJU = r"D:\大创资料\dcshuju"
GNT_HEADER_SIZE = 10
COPIES_PER_FILE = 5
NUM_ORIGINAL = 50
# GNT 像素噪声
GNT_NOISE_PIXELS = 80
GNT_NOISE_DELTA = 4
# 在线轨迹噪声 (x,y,f)
ONLINE_XY_NOISE = 0.08
ONLINE_F_NOISE = 0.15


def add_noise_to_gnt_bytes(data: bytearray, rng: random.Random) -> bytearray:
    if len(data) <= GNT_HEADER_SIZE:
        return data
    out = bytearray(data)
    pixel_start = GNT_HEADER_SIZE
    pixel_len = len(out) - pixel_start
    n = min(GNT_NOISE_PIXELS, pixel_len)
    indices = rng.sample(range(pixel_len), n)
    for i in indices:
        idx = pixel_start + i
        old = out[idx]
        delta = rng.randint(-GNT_NOISE_DELTA, GNT_NOISE_DELTA)
        out[idx] = max(0, min(255, old + delta))
    return out


def read_online_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "timestamp": float(row.get("timestamp", row.get("t", 0))),
                "x": float(row.get("x", 0)),
                "y": float(row.get("y", 0)),
                "f": float(row.get("f", 0)),
            })
    return rows


def add_noise_to_online(rows, rng):
    out = []
    for r in rows:
        t = r["timestamp"]
        x = r["x"] + rng.gauss(0, ONLINE_XY_NOISE)
        y = r["y"] + rng.gauss(0, ONLINE_XY_NOISE)
        f = r["f"] + rng.gauss(0, ONLINE_F_NOISE)
        f = max(0.0, min(5.0, f))
        x = max(0.0, min(10.0, x))
        y = max(0.0, min(10.0, y))
        out.append({"timestamp": t, "x": x, "y": y, "f": f})
    return out


def write_online_csv(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["timestamp", "x", "y", "f"])
        w.writeheader()
        w.writerows(rows)


def main():
    if not os.path.isdir(DCSHUJU):
        print("目录不存在:", DCSHUJU)
        return
    # 只处理 1.gnt ~ 50.gnt
    gnt_count = 0
    next_id = NUM_ORIGINAL + 1
    for i in range(1, NUM_ORIGINAL + 1):
        gnt_name = f"{i}.gnt"
        online_name = f"{i}_online.csv"
        gnt_path = os.path.join(DCSHUJU, gnt_name)
        online_path = os.path.join(DCSHUJU, online_name)
        if not os.path.isfile(gnt_path) or not os.path.isfile(online_path):
            continue
        with open(gnt_path, "rb") as f:
            gnt_data = bytearray(f.read())
        online_rows = read_online_csv(online_path)
        for k in range(COPIES_PER_FILE):
            rng = random.Random(i * 1000 + k)
            new_gnt_data = add_noise_to_gnt_bytes(gnt_data, rng)
            new_id = next_id + k
            new_gnt_path = os.path.join(DCSHUJU, f"{new_id}.gnt")
            with open(new_gnt_path, "wb") as f:
                f.write(new_gnt_data)
            new_online = add_noise_to_online(online_rows, rng)
            new_online_path = os.path.join(DCSHUJU, f"{new_id}_online.csv")
            write_online_csv(new_online_path, new_online)
        next_id += COPIES_PER_FILE
        gnt_count += 1
    total_gnt = NUM_ORIGINAL + gnt_count * COPIES_PER_FILE
    print("扩充完成: 原始 50 个 GNT + 每个 5 份加噪 = %d 个 GNT，%d 个在线 CSV" % (total_gnt, total_gnt))
    print("编号: 1~50 为原始，51~%d 为加噪副本。标签仅保留 1~50（见 gnt_scores.csv）。" % total_gnt)


if __name__ == "__main__":
    main()
