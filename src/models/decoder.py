# -*- coding: utf-8 -*-
"""解码器：轨迹序列 -> 重建 (x, y, f)。"""
import torch
import torch.nn as nn


class TrajDecoder(nn.Module):
    """LSTM 解码 (x,y,f)，pred_dim=3。"""

    def __init__(self, input_dim=5, hidden_size=128, num_layers=2, pred_dim=3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_size, num_layers,
            batch_first=True, dropout=0.1 if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_size, pred_dim)
        self.pred_dim = pred_dim

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out)


def build_decoder(hidden_size=128, num_layers=2, input_dim=5, pred_dim=3, **kwargs):
    return TrajDecoder(input_dim=input_dim, hidden_size=hidden_size, num_layers=num_layers, pred_dim=pred_dim, **kwargs)
