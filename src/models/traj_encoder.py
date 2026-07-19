# -*- coding: utf-8 -*-
"""轨迹编码器：traj [B,T,5] -> 序列表示或聚合表示。"""
import torch
import torch.nn as nn


class LSTMTrajEncoder(nn.Module):
    """LSTM 编码轨迹，可返回最后隐状态或逐时刻隐状态。"""

    def __init__(self, input_dim=5, hidden_size=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
        )
        self.hidden_size = hidden_size

    def forward(self, x, return_sequence=False):
        # x: (B, T, 5)
        out, (h_n, _) = self.lstm(x)
        if return_sequence:
            return out
        return h_n[-1]


def build_traj_encoder(input_dim=5, hidden_size=128, num_layers=2, **kwargs):
    return LSTMTrajEncoder(input_dim=input_dim, hidden_size=hidden_size, num_layers=num_layers, **kwargs)
