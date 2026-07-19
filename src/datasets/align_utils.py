# -*- coding: utf-8 -*-
"""对齐工具：空间对齐、软覆盖先验 π(i,k)、语义对齐分布 a(i,k)。"""
import numpy as np
import torch
import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from src.utils.geometry import bbox_from_points, scale_traj_to_bbox, point_to_segment_distance, trajectory_to_edge_distances


def trajectory_to_struct_space(traj_xy, struct_nodes, struct_edges, img_shape, padding=0.1):
    """
    将轨迹 bbox 缩放到与结构图相同坐标空间（结构图在 img 坐标系，即 H x W）。
    traj_xy: (T, 2) 原始 x,y（如 0~10）-> 映射到 [0,W-1],[0,H-1] 与 struct 一致。
    """
    h, w = img_shape[0], img_shape[1]
    target_bbox = (0, 0, w - 1, h - 1)
    source_bbox = bbox_from_points(traj_xy, padding)
    traj_scaled = scale_traj_to_bbox(traj_xy, target_bbox, source_bbox)
    return traj_scaled


def soft_coverage_prior(traj_xy, struct_nodes, struct_edges, sigma=0.5):
    """
    软覆盖先验 π(i,k) = softmax(-d/σ)。
    i: 轨迹段/点索引, k: 结构边索引。
    这里简化为：对每条边 k，计算轨迹点到该边的平均距离 d_k，则 π_k = softmax(-d_k/σ)。
    traj_xy: (T, 2), struct_nodes: (N, 2) row,col, struct_edges: list of (i,j)
    Returns: (num_edges,) 分布 π
    """
    if len(struct_edges) == 0:
        return np.array([1.0])
    d_list = []
    for (i, j) in struct_edges:
        a = struct_nodes[i]
        b = struct_nodes[j]
        d = trajectory_to_edge_distances(traj_xy, np.array([a, b]))
        d_list.append(d)
    d_arr = np.array(d_list, dtype=np.float64)
    logits = -d_arr / max(sigma, 1e-6)
    pi = np.exp(logits - logits.max())
    pi = pi / (pi.sum() + 1e-10)
    return pi


def semantic_alignment_distribution(z_traj, z_struct, sigma_sim=0.1):
    """
    a(i,k) = softmax(sim(z_S,i, z_G,k) / sigma)。
    z_traj: (T, D) 轨迹各时刻表示, z_struct: (K, D) 各结构边/节点表示。
    Returns: (T, K) 每个轨迹步对 K 条边的对齐分布 a
    """
    sim = torch.mm(z_traj, z_struct.t()) / max(sigma_sim, 1e-6)
    a = torch.softmax(sim, dim=1)
    return a


def compute_align_kl(pi, a, eps=1e-8):
    """L_align = KL(π || a)。pi: (K,) 先验, a: (T,K) 或 (K,) 预测分布。"""
    if isinstance(pi, np.ndarray):
        pi = torch.from_numpy(pi).float()
    if isinstance(a, np.ndarray):
        a = torch.from_numpy(a).float()
    if a.dim() == 2:
        a = a.mean(dim=0)
    pi = pi.clamp(eps, 1)
    a = a.clamp(eps, 1)
    return (pi * (torch.log(pi) - torch.log(a))).sum().item()


def align_entropy(a, eps=1e-8):
    """对齐分布熵。a: (T,K)。"""
    if isinstance(a, np.ndarray):
        a = torch.from_numpy(a).float()
    a = a.clamp(eps, 1)
    return -(a * torch.log(a)).sum(dim=1).mean().item()
