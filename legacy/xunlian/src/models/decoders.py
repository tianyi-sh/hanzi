# -*- coding: utf-8 -*-
"""解码器：轨迹重建、质量评分等。"""
import torch
import torch.nn as nn


class TrajReconstructDecoder(nn.Module):
    """从轨迹编码（或遮蔽后的序列）解码重建 (x, y, f)。"""

    def __init__(self, hidden_size, num_layers, input_dim=5, pred_dim=3):
        super().__init__()
        self.pred_dim = pred_dim
        self.lstm = nn.LSTM(
            input_dim, hidden_size, num_layers,
            batch_first=True, dropout=0.1 if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_size, pred_dim)

    def forward(self, x):
        # x: (B, N, 5)
        out, _ = self.lstm(x)
        return self.fc(out)


class ScoreHead(nn.Module):
    """轨迹编码 -> 质量标量。"""

    def __init__(self, embed_dim, score_dim=1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, score_dim),
        )

    def forward(self, x):
        return self.fc(x).squeeze(-1)
