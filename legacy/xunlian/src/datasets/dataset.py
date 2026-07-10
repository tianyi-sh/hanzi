# -*- coding: utf-8 -*-
"""PyTorch Dataset 加载 .pt 样本。"""
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

DEFAULT_IMG_SIZE = (64, 64)


class GntOnlineDataset(Dataset):
    """加载 data/processed/samples/*.pt，返回 image (1,H,W) 统一缩放到 img_size。"""

    def __init__(self, samples_dir, sample_ids=None, img_size=None):
        self.samples_dir = samples_dir
        self.img_size = img_size or DEFAULT_IMG_SIZE
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
        data = torch.load(path, map_location="cpu", weights_only=True)
        img = data["image"]
        if img.dim() == 2:
            img = img.unsqueeze(0)
        if img.shape[1] != self.img_size[0] or img.shape[2] != self.img_size[1]:
            img = F.interpolate(
                img.unsqueeze(0), size=self.img_size, mode="bilinear", align_corners=False
            ).squeeze(0)
        return {
            "image": img,
            "traj": data["traj"],
            "char": data.get("char", ""),
            "sample_id": data.get("sample_id", sid),
            "writer_id": data.get("writer_id", 0),
        }
