# -*- coding: utf-8 -*-
"""排序损失：Hinge margin。"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def ranking_hinge_loss(score_good, score_bad, margin=0.5):
    """
    score_good: (B,) 优质样本得分
    score_bad: (B,) 劣质样本得分
    hinge: max(0, margin - (score_good - score_bad))
    """
    return F.relu(margin - (score_good - score_bad)).mean()
