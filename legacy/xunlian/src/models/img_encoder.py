# -*- coding: utf-8 -*-
"""图像编码器：将离线字形图编码为向量。"""
import torch
import torch.nn as nn


class CNNImageEncoder(nn.Module):
    """简单 CNN：1xHxW -> embed_dim。"""

    def __init__(self, in_ch=1, hidden=128, embed_dim=128, max_hw=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(hidden, embed_dim)

    def forward(self, x):
        # x: (B, 1, H, W)
        h = self.conv(x)
        h = h.flatten(1)
        return self.fc(h)


def build_img_encoder(embed_dim=128, **kwargs):
    return CNNImageEncoder(embed_dim=embed_dim, **kwargs)
