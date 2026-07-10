# -*- coding: utf-8 -*-
"""遮蔽重建损失：MAE。"""
import torch
import torch.nn as nn


def reconstruction_mae(pred, target, mask=None):
    """
    pred, target: (B, N, 3) 预测与真值 (x,y,f)
    mask: (B, N) 可选，True 表示参与损失
    """
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
    return nn.functional.l1_loss(pred, target)
