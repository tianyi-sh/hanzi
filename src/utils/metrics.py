# -*- coding: utf-8 -*-
import torch
import numpy as np


def reconstruction_mae_metric(pred, target, mask=None):
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
    return torch.nn.functional.l1_loss(pred, target).item()


def align_kl_metric(pi_prior, a_pred, eps=1e-8):
    if a_pred.dim() == 3:
        a_pred = a_pred.mean(dim=1)
    pi_prior = pi_prior.clamp(eps, 1)
    a_pred = a_pred.clamp(eps, 1)
    return (pi_prior * (torch.log(pi_prior) - torch.log(a_pred))).sum().item()


def align_entropy_metric(a_pred, eps=1e-8):
    a_pred = a_pred.clamp(eps, 1)
    return -(a_pred * torch.log(a_pred)).sum(dim=-1).mean().item()


def pairwise_rank_accuracy(score_good, score_bad):
    return (score_good > score_bad).float().mean().item()


def margin_mean(score_good, score_bad):
    return (score_good - score_bad).mean().item()
