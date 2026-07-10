# -*- coding: utf-8 -*-
"""从外部数据目录随机选 10 对 GNT + 在线 CSV 到 data/raw。"""
import argparse
import os
import random
import shutil
import csv

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


def default_source_dir():
    """优先使用环境变量，其次使用仓库同级的“部分实验数据”。"""
    from_env = os.environ.get("HANZI_DATA_DIR")
    if from_env:
        return os.path.abspath(from_env)
    sibling = os.path.join(os.path.dirname(ROOT), "部分实验数据")
    if os.path.isdir(sibling):
        return sibling
    return os.path.join(ROOT, "data", "source")


def main(source_dir=None):
    source_dir = os.path.abspath(source_dir or default_source_dir())
    if not os.path.isdir(source_dir):
        raise FileNotFoundError(
            f"数据目录不存在: {source_dir}\n"
            "请使用 --source-dir 或 HANZI_DATA_DIR 指定包含 *.gnt 和 *_online.csv 的目录。"
        )
    os.makedirs(RAW_GNT, exist_ok=True)
    os.makedirs(RAW_ONLINE, exist_ok=True)
    candidates = []
    for i in range(1, 51):
        gnt = os.path.join(source_dir, f"{i}.gnt")
        csv_path = os.path.join(source_dir, f"{i}_online.csv")
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
        rows.append({
            "sample_id": sample_id,
            "gnt_path": os.path.relpath(gnt_dst, os.path.dirname(PAIRS_CSV)),
            "online_path": os.path.relpath(online_dst, os.path.dirname(PAIRS_CSV)),
            "writer_id": 0,
            "char": char,
        })
    with open(PAIRS_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sample_id", "gnt_path", "online_path", "writer_id", "char"])
        w.writeheader()
        w.writerows(rows)
    print("已复制 %d 对样本到 data/raw，pairs.csv 已生成。" % len(rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", help="包含 GNT 和在线 CSV 的外部数据目录")
    args = parser.parse_args()
    random.seed(42)
    main(args.source_dir)
