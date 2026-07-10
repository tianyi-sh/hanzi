# -*- coding: utf-8 -*-
"""可视化：结构热力图、轨迹-结构对齐。"""
import os
import sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)


def plot_struct_heatmap(struct_nodes, struct_edges, img_shape, save_path=None):
    """在图像尺寸上绘制结构图热力（节点+边）。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    h, w = img_shape[0], img_shape[1]
    canvas = np.zeros((h, w))
    for (i, j) in struct_edges:
        if i < len(struct_nodes) and j < len(struct_nodes):
            r0, c0 = int(struct_nodes[i, 0]), int(struct_nodes[i, 1])
            r1, c1 = int(struct_nodes[j, 0]), int(struct_nodes[j, 1])
            rr = np.linspace(r0, r1, max(2, int(np.hypot(r1 - r0, c1 - c0))))
            cc = np.linspace(c0, c1, len(rr))
            for r, c in zip(rr, cc):
                if 0 <= r < h and 0 <= c < w:
                    canvas[int(r), int(c)] = 1
    for n in struct_nodes:
        r, c = int(n[0]), int(n[1])
        if 0 <= r < h and 0 <= c < w:
            canvas[r, c] = 2
    plt.figure(figsize=(5, 5))
    plt.imshow(canvas, cmap="hot")
    plt.axis("off")
    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_traj_on_struct(traj_xy, struct_nodes, struct_edges, img_shape, save_path=None):
    """轨迹叠加在结构图上。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    for (i, j) in struct_edges:
        if i < len(struct_nodes) and j < len(struct_nodes):
            ax.plot([struct_nodes[i, 1], struct_nodes[j, 1]], [struct_nodes[i, 0], struct_nodes[j, 0]], "b-", alpha=0.7)
    ax.scatter(struct_nodes[:, 1], struct_nodes[:, 0], c="blue", s=20)
    ax.plot(traj_xy[:, 0], traj_xy[:, 1], "r-", alpha=0.8, label="traj")
    ax.legend()
    ax.axis("equal")
    if save_path:
        plt.savefig(save_path)
    plt.close()
