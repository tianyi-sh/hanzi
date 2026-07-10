# -*- coding: utf-8 -*-
"""结构图编码器：节点+边 -> 每条边的表示 z_G,k。"""
import torch
import torch.nn as nn
import numpy as np


class StructEncoder(nn.Module):
    """
    输入：struct_nodes (N, 2), struct_edges list of (i,j)
    输出：边表示 (K, D)，K 为边数。
    """
    def __init__(self, node_dim=2, hidden_size=128, out_dim=128):
        super().__init__()
        self.node_embed = nn.Sequential(
            nn.Linear(node_dim, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, out_dim),
        )
        self.edge_aggregate = nn.Sequential(
            nn.Linear(out_dim * 2, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, nodes, edges):
        """
        nodes: (N, 2), edges: list of (i,j) or (E, 2) tensor
        """
        if isinstance(edges, (list, tuple)):
            edges = torch.tensor(edges, dtype=torch.long, device=nodes.device)
        if edges.dim() == 1:
            edges = edges.unsqueeze(0)
        N = nodes.size(0)
        h = self.node_embed(nodes)
        i, j = edges[:, 0], edges[:, 1]
        hi = h[i]
        hj = h[j]
        h_edge = self.edge_aggregate(torch.cat([hi, hj], dim=1))
        return h_edge


def build_struct_encoder(hidden_size=128, out_dim=128, **kwargs):
    return StructEncoder(hidden_size=hidden_size, out_dim=out_dim, **kwargs)
