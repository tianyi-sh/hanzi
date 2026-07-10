# -*- coding: utf-8 -*-
"""对比损失：InfoNCE（图像-轨迹对齐）。"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def infonce_loss(img_embed, traj_embed, temperature=0.07):
    """
    img_embed, traj_embed: (B, D), L2 归一化
    对角线为正样本，其余为负样本。
    """
    img_embed = F.normalize(img_embed, dim=1)
    traj_embed = F.normalize(traj_embed, dim=1)
    logits = (img_embed @ traj_embed.T) / temperature
    labels = torch.arange(img_embed.size(0), device=img_embed.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    return (loss_i2t + loss_t2i) / 2
