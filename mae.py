# -*- coding: utf-8 -*-
"""重建 MAE 损失。"""
import torch
import torch.nn as nn


def reconstruction_mae(pred, target, mask=None):
    """
    pred, target: (B, T, 3) 预测与真值 (x,y,f)
    mask: (B, T) 可选
    """
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
    return nn.functional.l1_loss(pred, target)
