# -*- coding: utf-8 -*-
"""结构图操作：从骨架提取端点/分叉点、构建边。"""
import numpy as np


def is_endpoint(skel, i, j):
    """骨架点 (i,j) 是否为端点（邻域内仅 1 个前景）。"""
    h, w = skel.shape
    count = 0
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue
            ni, nj = i + di, j + dj
            if 0 <= ni < h and 0 <= nj < w and skel[ni, nj] > 0.5:
                count += 1
    return count == 1


def is_branch(skel, i, j):
    """骨架点 (i,j) 是否为分叉点（邻域内 >= 3 个前景）。"""
    h, w = skel.shape
    count = 0
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue
            ni, nj = i + di, j + dj
            if 0 <= ni < h and 0 <= nj < w and skel[ni, nj] > 0.5:
                count += 1
    return count >= 3


def extract_nodes(skel):
    """
    从骨架图提取节点：端点与分叉点。
    Returns: nodes (N, 2) 每行 [y, x] 或 [row, col]，node_types list of "endpoint"/"branch"
    """
    h, w = skel.shape
    nodes = []
    node_types = []
    for i in range(h):
        for j in range(w):
            if skel[i, j] < 0.5:
                continue
            if is_endpoint(skel, i, j):
                nodes.append([i, j])
                node_types.append("endpoint")
            elif is_branch(skel, i, j):
                nodes.append([i, j])
                node_types.append("branch")
    return np.array(nodes, dtype=np.float32) if nodes else np.zeros((0, 2), dtype=np.float32), node_types


def trace_edges_from_skeleton(skel, nodes_rc):
    """
    从骨架与节点得到边：相邻节点对（简化：基于距离连接最近节点）。
    nodes_rc: (N, 2) row,col
    Returns: list of (i, j) 边索引对
    """
    n = len(nodes_rc)
    if n <= 1:
        return []
    from scipy.spatial.distance import cdist
    D = cdist(nodes_rc, nodes_rc)
    np.fill_diagonal(D, np.inf)
    edges = set()
    for i in range(n):
        j = np.argmin(D[i])
        if D[i, j] < max(skel.shape) * 0.5:
            edges.add((min(i, j), max(i, j)))
    return list(edges)


def build_struct_graph(skel):
    """
    构建 G_S = { nodes: [N,2], edges: list_of_edges }。
    nodes 使用 (row, col) 即 (y, x) 与图像坐标一致。
    """
    nodes, _ = extract_nodes(skel)
    if len(nodes) == 0:
        return {"nodes": nodes, "edges": []}
    edges = trace_edges_from_skeleton(skel, nodes)
    return {"nodes": nodes, "edges": edges}
