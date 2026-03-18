# -*- coding: utf-8 -*-
"""L_cons = ||z_S - Σ_k a(i,k) z_G,k||²。"""
import torch


def consistency_loss(z_traj, z_struct, a):
    """
    z_traj: (B, T, D) 轨迹各时刻表示
    z_struct: (B, K, D) 各结构边表示
    a: (B, T, K) 对齐分布
    L_cons = mean over B,T of ||z_traj - Σ_k a(b,t,k)*z_struct(b,k)||^2
    """
    # (B, T, K) @ (B, K, D) -> (B, T, D)
    z_recon = torch.bmm(a, z_struct)
    return torch.nn.functional.mse_loss(z_traj, z_recon)
