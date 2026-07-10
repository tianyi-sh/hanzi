# -*- coding: utf-8 -*-
"""PyTorch Dataset：加载 .pt 样本（image [1,224,224], traj [T,5], struct, pi_prior）。"""
import os
import torch
from torch.utils.data import Dataset


class StructConstraintDataset(Dataset):
    """加载 data/processed/samples/*.pt。"""

    def __init__(self, samples_dir, struct_graphs_dir=None, sample_ids=None):
        self.samples_dir = samples_dir
        self.struct_graphs_dir = struct_graphs_dir
        if sample_ids is None:
            self.sample_ids = [
                os.path.splitext(f)[0] for f in os.listdir(samples_dir)
                if f.endswith(".pt")
            ]
            self.sample_ids.sort()
        else:
            self.sample_ids = list(sample_ids)

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sid = self.sample_ids[idx]
        path = os.path.join(self.samples_dir, f"{sid}.pt")
        data = torch.load(path, map_location="cpu", weights_only=False)
        return {
            "image": data["image"],
            "traj": data["traj"],
            "struct_nodes": data["struct_nodes"],
            "struct_edges": data["struct_edges"],
            "pi_prior": data["pi_prior"],
            "char": data.get("char", ""),
            "sample_id": data.get("sample_id", sid),
        }
