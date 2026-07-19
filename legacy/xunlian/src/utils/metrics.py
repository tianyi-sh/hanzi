# -*- coding: utf-8 -*-
import torch
import numpy as np


def reconstruction_mae_metric(pred, target, mask=None):
    """pred/target (B,N,3); return scalar."""
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
    return torch.nn.functional.l1_loss(pred, target).item()


def recall_at_k(logits, k=1):
    """logits (B,B), 对角线为正样本。"""
    B = logits.size(0)
    k = min(k, logits.size(1))
    if k < 1:
        return 0.0
    _, idx = logits.topk(k, dim=1)
    correct = (idx == torch.arange(B, device=logits.device).unsqueeze(1)).any(1)
    return correct.float().mean().item()


def pairwise_rank_accuracy(score_good, score_bad):
    """优质得分应大于劣质。"""
    return (score_good > score_bad).float().mean().item()


def margin_mean(score_good, score_bad):
    return (score_good - score_bad).mean().item()
