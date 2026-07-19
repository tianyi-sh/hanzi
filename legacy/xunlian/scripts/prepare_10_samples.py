# -*- coding: utf-8 -*-
"""为旧版 xunlian 工程准备配对的 GNT 与在线轨迹样本。"""

import argparse
import csv
import os
import random
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT_DIR = os.path.join(ROOT, "data", "raw")


def get_char_from_gnt(path):
    try:
        with open(path, "rb") as handle:
            data = handle.read(6)
        if len(data) < 6:
            return ""
        return bytes([data[4], data[5]]).decode("gb2312", errors="ignore")
    except OSError:
        return ""


def default_source_dir():
    return os.path.abspath(
        os.environ.get("HANZI_LEGACY_DATA_DIR")
        or os.environ.get("HANZI_DATA_DIR")
        or os.path.join(ROOT, "data", "source")
    )


def main(source_dir=None, sample_count=10, seed=42):
    source_dir = os.path.abspath(source_dir or default_source_dir())
    if not os.path.isdir(source_dir):
        raise FileNotFoundError(
            f"数据目录不存在: {source_dir}\n"
            "请使用 --source-dir、HANZI_LEGACY_DATA_DIR 或 HANZI_DATA_DIR 指定数据目录。"
        )
    candidates = []
    for index in range(1, 51):
        gnt_path = os.path.join(source_dir, f"{index}.gnt")
        online_path = os.path.join(source_dir, f"{index}_online.csv")
        if os.path.isfile(gnt_path) and os.path.isfile(online_path):
            candidates.append((gnt_path, online_path))
    if len(candidates) < sample_count:
        raise ValueError(
            f"数据目录只有 {len(candidates)} 对有效样本，少于请求的 {sample_count} 对"
        )

    raw_gnt = os.path.join(DEFAULT_OUTPUT_DIR, "gnt")
    raw_online = os.path.join(DEFAULT_OUTPUT_DIR, "online")
    pairs_csv = os.path.join(DEFAULT_OUTPUT_DIR, "pairs.csv")
    os.makedirs(raw_gnt, exist_ok=True)
    os.makedirs(raw_online, exist_ok=True)
    chosen = random.Random(seed).sample(candidates, sample_count)
    rows = []
    for index, (gnt_source, online_source) in enumerate(chosen):
        sample_id = f"sample_{index:02d}"
        gnt_target = os.path.join(raw_gnt, f"{sample_id}.gnt")
        online_target = os.path.join(raw_online, f"{sample_id}_online.csv")
        shutil.copy2(gnt_source, gnt_target)
        shutil.copy2(online_source, online_target)
        rows.append({
            "sample_id": sample_id,
            "gnt_path": os.path.relpath(gnt_target, DEFAULT_OUTPUT_DIR),
            "online_path": os.path.relpath(online_target, DEFAULT_OUTPUT_DIR),
            "writer_id": 0,
            "char": get_char_from_gnt(gnt_target),
        })
    with open(pairs_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "gnt_path", "online_path", "writer_id", "char"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"已复制 {len(rows)} 对样本到 {DEFAULT_OUTPUT_DIR}，pairs.csv 已生成。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", help="包含配对 GNT 与在线 CSV 的目录")
    parser.add_argument("--sample-count", type=int, default=10, help="抽取样本数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()
    main(args.source_dir, args.sample_count, args.seed)
