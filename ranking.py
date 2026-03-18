# -*- coding: utf-8 -*-
"""L_rank = max(0, m - (s⁺ - s⁻))。"""
import torch
import torch.nn.functional as F


def ranking_hinge_loss(score_good, score_bad, margin=0.5):
    """
    score_good: (B,) 优质样本得分
    score_bad: (B,) 劣质样本得分
    """
    return F.relu(margin - (score_good - score_bad)).mean()
