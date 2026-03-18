# -*- coding: utf-8 -*-
"""L_align = KL(π || a)。"""
import torch
import torch.nn.functional as F


def align_kl_loss(pi_prior, a_pred, eps=1e-8):
    """
    pi_prior: (B, K) 或 (K,) 软覆盖先验
    a_pred: (B, T, K) 或 (B, K) 语义对齐分布
    Returns: scalar
    """
    if a_pred.dim() == 3:
        a_pred = a_pred.mean(dim=1)
    if pi_prior.dim() == 1:
        pi_prior = pi_prior.unsqueeze(0).expand(a_pred.size(0), -1)
    pi_prior = pi_prior.clamp(eps, 1)
    a_pred = a_pred.clamp(eps, 1)
    return F.kl_div(torch.log(a_pred), pi_prior, reduction="batchmean")
