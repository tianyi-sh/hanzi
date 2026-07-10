# -*- coding: utf-8 -*-
"""将 raw 中的 gnt+online 配对构建为 .pt 样本，存入 data/processed/samples。"""
import os
import csv
import torch
from .gnt_reader import read_gnt
from .online_reader import read_online_csv


def build_processed_dataset(pairs_csv, out_dir):
    """
    pairs_csv: data/raw/pairs.csv，列 sample_id, gnt_path, online_path, writer_id, char
    每条样本保存为 out_dir/{sample_id}.pt，包含:
      - image: (1, H, W) float32
      - traj: (N, 5) float32 [x, y, f, speed, dt]
      - char: str
      - sample_id: str
      - writer_id: int
    """
    os.makedirs(out_dir, exist_ok=True)
    with open(pairs_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        sample_id = row["sample_id"]
        gnt_path = row["gnt_path"]
        online_path = row["online_path"]
        writer_id = int(row.get("writer_id", 0))
        char = row.get("char", "")
        if not os.path.isfile(gnt_path) or not os.path.isfile(online_path):
            continue
        image, _ = read_gnt(gnt_path)
        traj = read_online_csv(online_path)
        image_t = torch.from_numpy(image).unsqueeze(0)
        traj_t = torch.from_numpy(traj)
        torch.save({
            "image": image_t,
            "traj": traj_t,
            "char": char,
            "sample_id": sample_id,
            "writer_id": writer_id,
        }, os.path.join(out_dir, f"{sample_id}.pt"))
    return len(rows)
