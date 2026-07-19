# -*- coding: utf-8 -*-
"""可视化：轨迹、排序得分等。"""
import os
import sys
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)


def plot_traj_and_reconstruct(traj, pred, save_path=None):
    """绘制轨迹与重建 (x,y)。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if torch.is_tensor(traj):
        traj = traj.detach().cpu().numpy()
    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()
    plt.figure(figsize=(6, 6))
    plt.plot(traj[:, 0], traj[:, 1], "b-", label="GT")
    plt.plot(pred[:, 0], pred[:, 1], "r--", alpha=0.7, label="Pred")
    plt.legend()
    plt.axis("equal")
    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_ranking_scores(scores_good, scores_bad, save_path=None):
    """绘制优质/劣质得分分布。"""
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
