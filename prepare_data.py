# -*- coding: utf-8 -*-
"""从 dcshuju 随机选 10 对 gnt+online csv 到 data/raw，生成 pairs.csv。"""
import os
import random
import shutil
import csv

DCSHUJU = r"D:\大创资料\dcshuju"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_GNT = os.path.join(ROOT, "data", "raw", "gnt")
RAW_ONLINE = os.path.join(ROOT, "data", "raw", "online")
PAIRS_CSV = os.path.join(ROOT, "data", "raw", "pairs.csv")


def get_char_from_gnt(path):
    try:
        with open(path, "rb") as f:
            d = f.read(6)
        if len(d) < 6:
            return ""
        return bytes([d[4], d[5]]).decode("gb2312", errors="ignore")
    except Exception:
        return ""


def main():
    os.makedirs(RAW_GNT, exist_ok=True)
    os.makedirs(RAW_ONLINE, exist_ok=True)
    candidates = []
    for i in range(1, 51):
        gnt = os.path.join(DCSHUJU, f"{i}.gnt")
        csv_path = os.path.join(DCSHUJU, f"{i}_online.csv")
        if os.path.isfile(gnt) and os.path.isfile(csv_path):
            candidates.append((i, gnt, csv_path))
    chosen = random.sample(candidates, min(10, len(candidates)))
    rows = []
    for idx, (num, gnt_src, csv_src) in enumerate(chosen):
        sample_id = f"sample_{idx:02d}"
        gnt_dst = os.path.join(RAW_GNT, f"{sample_id}.gnt")
        online_dst = os.path.join(RAW_ONLINE, f"{sample_id}_online.csv")
        shutil.copy2(gnt_src, gnt_dst)
        shutil.copy2(csv_src, online_dst)
        char = get_char_from_gnt(gnt_dst)
        rows.append({"sample_id": sample_id, "gnt_path": gnt_dst, "online_path": online_dst, "writer_id": 0, "char": char})
    with open(PAIRS_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sample_id", "gnt_path", "online_path", "writer_id", "char"])
        w.writeheader()
        w.writerows(rows)
    print("已复制 %d 对样本到 data/raw，pairs.csv 已生成。" % len(rows))


if __name__ == "__main__":
    random.seed(42)
    main()
