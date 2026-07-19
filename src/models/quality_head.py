# -*- coding: utf-8 -*-
"""质量得分头：轨迹表示 -> 标量质量分。"""
import torch
import torch.nn as nn


class QualityHead(nn.Module):
    def __init__(self, embed_dim, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.mlp(x).squeeze(-1)


def build_quality_head(embed_dim, hidden=64, **kwargs):
    return QualityHead(embed_dim=embed_dim, hidden=hidden, **kwargs)
