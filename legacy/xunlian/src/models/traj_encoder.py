# -*- coding: utf-8 -*-
"""轨迹编码器：将 (x, y, f, speed, dt) 序列编码为向量。"""
import torch
import torch.nn as nn


class LSTMTrajEncoder(nn.Module):
    """LSTM 编码轨迹序列，取最后隐状态或池化作为序列表示。"""

    def __init__(self, input_dim=5, hidden_size=128, num_layers=2, dropout=0.1, embed_dim=None):
        super().__init__()
        embed_dim = embed_dim or hidden_size
        self.lstm = nn.LSTM(
            input_dim, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
        )
        self.proj = nn.Linear(hidden_size, embed_dim)

    def forward(self, x):
        # x: (B, N, 5)
        out, (h_n, _) = self.lstm(x)
        h = h_n[-1]
        return self.proj(h)


def build_traj_encoder(input_dim=5, hidden_size=128, num_layers=2, embed_dim=128, **kwargs):
    return LSTMTrajEncoder(
        input_dim=input_dim, hidden_size=hidden_size, num_layers=num_layers,
        embed_dim=embed_dim, **kwargs
    )
