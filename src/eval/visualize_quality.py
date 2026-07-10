# -*- coding: utf-8 -*-
"""可视化：质量排序、轨迹异常。"""
import os
import sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)


def plot_ranking_scores(scores_good, scores_bad, save_path=None):
    """优质/劣质得分分布。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if torch.is_tensor(scores_good):
        scores_good = scores_good.detach().cpu().numpy()
    if torch.is_tensor(scores_bad):
        scores_bad = scores_bad.detach().cpu().numpy()
    plt.figure(figsize=(5, 3))
    plt.hist(scores_good, bins=10, alpha=0.6, label="Good")
    plt.hist(scores_bad, bins=10, alpha=0.6, label="Bad")
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_traj_anomaly(traj_orig, traj_degraded, save_path=None):
    """原始轨迹 vs 劣化轨迹。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if torch.is_tensor(traj_orig):
        traj_orig = traj_orig.detach().cpu().numpy()
    if torch.is_tensor(traj_degraded):
        traj_degraded = traj_degraded.detach().cpu().numpy()
    plt.figure(figsize=(6, 6))
    plt.plot(traj_orig[:, 0], traj_orig[:, 1], "b-", label="Original")
    plt.plot(traj_degraded[:, 0], traj_degraded[:, 1], "r--", alpha=0.7, label="Degraded")
    plt.legend()
    plt.axis("equal")
    if save_path:
        plt.savefig(save_path)
    plt.close()
