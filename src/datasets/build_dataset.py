# -*- coding: utf-8 -*-
"""根据 pairs.csv 构建 processed/samples 与 struct_graphs。"""
import os
import csv
import torch
import numpy as np
from .gnt_reader import read_gnt
from .online_reader import read_online_csv
from .struct_builder import build_and_save_struct
from .align_utils import trajectory_to_struct_space, soft_coverage_prior

IMG_SIZE = (224, 224)


def build_processed_dataset(pairs_csv, processed_dir, struct_graphs_dir):
    """
    读取 pairs_csv，对每条：解析 GNT -> [1,224,224]，解析 online -> traj [T,5]，
    构建结构图并保存到 struct_graphs_dir，样本保存到 processed_dir。
    每条样本 .pt 含: image, traj, struct_nodes, struct_edges, pi_prior, char, sample_id.
    """
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(struct_graphs_dir, exist_ok=True)
    with open(pairs_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    pairs_dir = os.path.dirname(os.path.abspath(pairs_csv))
    for row in rows:
        sample_id = row["sample_id"]
        gnt_path = row["gnt_path"]
        online_path = row["online_path"]
        if not os.path.isabs(gnt_path):
            gnt_path = os.path.normpath(os.path.join(pairs_dir, gnt_path))
        if not os.path.isabs(online_path):
            online_path = os.path.normpath(os.path.join(pairs_dir, online_path))
        char = row.get("char", "")
        if not os.path.isfile(gnt_path) or not os.path.isfile(online_path):
            continue
        img, _ = read_gnt(gnt_path, target_size=IMG_SIZE)
        traj = read_online_csv(online_path)
        traj_xy = traj[:, :2]
        struct_path = os.path.join(struct_graphs_dir, f"{sample_id}_struct.npz")
        gs = build_and_save_struct(img, struct_path)
        nodes = gs["nodes"]
        edges = gs["edges"]
        traj_in_struct = trajectory_to_struct_space(
            traj_xy, nodes, edges, IMG_SIZE, padding=0.1
        )
        traj_rc = np.column_stack([traj_in_struct[:, 1], traj_in_struct[:, 0]])
        pi_prior = soft_coverage_prior(traj_rc, nodes, edges, sigma=0.5)
        torch.save({
            "image": img,
            "traj": torch.from_numpy(traj).float(),
            "struct_nodes": torch.from_numpy(nodes).float(),
            "struct_edges": edges,
            "pi_prior": torch.from_numpy(pi_prior).float(),
            "char": char,
            "sample_id": sample_id,
        }, os.path.join(processed_dir, f"{sample_id}.pt"))
    return len(rows)


if __name__ == "__main__":
    import sys
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    pairs = os.path.join(root, "data", "raw", "pairs.csv")
    out = os.path.join(root, "data", "processed", "samples")
    struct = os.path.join(root, "data", "processed", "struct_graphs")
    build_processed_dataset(pairs, out, struct)
