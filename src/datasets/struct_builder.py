# -*- coding: utf-8 -*-
"""从字形图像构建结构图 G_S = { nodes: [N,2], edges: list_of_edges }。"""
import os
import numpy as np
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from src.utils.skeleton import skeletonize
from src.utils.graph_ops import build_struct_graph


def build_struct_from_binary_img(binary_img, method="medial_axis"):
    """
    binary_img: (H, W) 0/1，1 为笔画。
    Returns: G_S = { "nodes": (N, 2) row,col, "edges": [(i,j), ...] }
    """
    skel = skeletonize(binary_img, method=method)
    return build_struct_graph(skel)


def build_and_save_struct(img_tensor, save_path):
    """
    img_tensor: [1, H, W] 或 [H, W]，二值。
    将 G_S 保存为 npz: nodes, edges（edges 存为 Nx2 数组）。
    """
    img = img_tensor
    if hasattr(img, "numpy"):
        img = img.numpy()
    while img.ndim > 2:
        img = img.squeeze(0)
    img = (img > 0.5).astype(np.float32)
    gs = build_struct_from_binary_img(img)
    nodes = gs["nodes"]
    edges = gs["edges"]
    edge_arr = np.array(edges, dtype=np.int64) if edges else np.zeros((0, 2), dtype=np.int64)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    np.savez(save_path, nodes=nodes, edges=edge_arr)
    return gs
