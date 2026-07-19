# -*- coding: utf-8 -*-
"""融合头部：对齐、拼接等。"""
import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    """单模态投影到共同空间。"""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        return self.fc(x)


class ConcatFusion(nn.Module):
    """拼接图像与轨迹特征后 MLP。"""

    def __init__(self, img_dim, traj_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(img_dim + traj_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, img_feat, traj_feat):
        return self.mlp(torch.cat([img_feat, traj_feat], dim=1))
