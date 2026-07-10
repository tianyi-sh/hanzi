# -*- coding: utf-8 -*-
"""几何工具：点到线段距离、边界框等。"""
import numpy as np


def point_to_segment_distance(p, a, b):
    """
    点 p 到线段 ab 的距离。
    p: (2,) or (N,2), a,b: (2,)
    Returns: scalar or (N,)
    """
    p = np.asarray(p)
    a, b = np.asarray(a), np.asarray(b)
    single = p.ndim == 1
    if single:
        p = p.reshape(1, -1)
    ap = p - a
    ab = b - a
    ab_len_sq = np.sum(ab ** 2) + 1e-10
    t = np.clip(np.dot(ap, ab) / ab_len_sq, 0, 1)
    if t.ndim == 0:
        proj = a + t * ab
    else:
        proj = a + t[:, np.newaxis] * ab
    d = np.linalg.norm(p - proj, axis=1)
    return d[0] if single else d


def trajectory_to_edge_distances(traj_xy, edge_nodes):
    """
    轨迹点集到某条结构边的平均距离。
    traj_xy: (T, 2), edge_nodes: (2, 2) 即 [a, b]
    """
    a, b = edge_nodes[0], edge_nodes[1]
    d = point_to_segment_distance(traj_xy, a, b)
    return float(np.mean(d))


def bbox_from_points(pts, padding=0.0):
    """pts: (N,2)。返回 (xmin, ymin, xmax, ymax)，可加 padding 比例。"""
    pts = np.asarray(pts)
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    if padding > 0:
        w, h = xmax - xmin, ymax - ymin
        xmin -= w * padding
        ymin -= h * padding
        xmax += w * padding
        ymax += h * padding
    return xmin, ymin, xmax, ymax


def scale_traj_to_bbox(traj_xy, target_bbox, source_bbox=None):
    """
    将轨迹从 source_bbox 线性缩放到 target_bbox 范围内。
    traj_xy: (T, 2). target_bbox: (xmin, ymin, xmax, ymax).
    source_bbox: 若为 None 则用 traj 自身 bbox。
    """
    traj_xy = np.asarray(traj_xy, dtype=np.float64)
    if source_bbox is None:
        source_bbox = bbox_from_points(traj_xy, 0)
    sxmin, symin, sxmax, symax = source_bbox
    txmin, tymin, txmax, tymax = target_bbox
    sw, sh = sxmax - sxmin, symax - symin
    tw, th = txmax - txmin, tymax - tymin
    if sw < 1e-6:
        sw = 1e-6
    if sh < 1e-6:
        sh = 1e-6
    x = (traj_xy[:, 0] - sxmin) / sw * tw + txmin
    y = (traj_xy[:, 1] - symin) / sh * th + tymin
    return np.stack([x, y], axis=1)
