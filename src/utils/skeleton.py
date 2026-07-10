# -*- coding: utf-8 -*-
"""骨架化：二值图像 -> 骨架（1-pixel 宽中心线）。"""
import numpy as np


def _thinning_approx(binary_img):
    """形态学腐蚀近似骨架，保证连通性。"""
    from scipy import ndimage
    binary_img = np.asarray(binary_img)
    while binary_img.ndim > 2:
        binary_img = binary_img.squeeze(0)
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    eroded = (binary_img > 0.5).astype(np.uint8).copy()
    n_iter = min(max(binary_img.shape) // 4, 30)
    for _ in range(n_iter):
        e = ndimage.binary_erosion(eroded, structure=structure)
        if np.all(e == 0):
            break
        eroded = e
    return eroded.astype(np.float32)


def medial_axis_skeleton(binary_img):
    """骨架化。优先尝试 skimage，失败则用腐蚀近似。"""
    try:
        from skimage.morphology import skeletonize as skel
        out = skel(binary_img > 0.5).astype(np.float32)
        return out
    except Exception:
        return _thinning_approx(binary_img)


def skeletonize(binary_img, method="medial_axis"):
    """统一接口。binary_img: (H,W) 0/1。返回 (H,W) 骨架。"""
    return medial_axis_skeleton(binary_img)
