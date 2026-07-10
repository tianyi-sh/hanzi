# -*- coding: utf-8 -*-
"""对齐模块：轨迹表示与结构边表示 -> 语义对齐分布 a(i,k)。"""
import torch
import torch.nn as nn


class AlignModule(nn.Module):
    """
    轨迹序列表示 z_traj (B,T,D) 与结构边表示 z_struct (B,K,D) -> a (B,T,K)。
    """
    def __init__(self, embed_dim=128, sigma_sim=0.1):
        super().__init__()
        self.sigma_sim = sigma_sim
        self.embed_dim = embed_dim

    def forward(self, z_traj, z_struct, sigma_sim=None):
        # z_traj: (B, T, D), z_struct: (B, K, D)
        sigma = sigma_sim or self.sigma_sim
        sim = torch.bmm(z_traj, z_struct.transpose(1, 2)) / max(sigma, 1e-6)
        a = torch.softmax(sim, dim=-1)
        return a


def build_align_module(embed_dim=128, sigma_sim=0.1, **kwargs):
    return AlignModule(embed_dim=embed_dim, sigma_sim=sigma_sim, **kwargs)
