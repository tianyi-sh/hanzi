# -*- coding: utf-8 -*-
"""解析 GNT 格式，输出 img tensor [1, 224, 224]，灰度化 + 二值化 + 归一化。"""
import struct
import numpy as np
import torch

GNT_HEADER_SIZE = 10


def read_gnt(path, target_size=(224, 224)):
    """
    读取 GNT 文件。
    Returns:
        img: tensor [1, 224, 224] float32，归一化到 [0,1]，二值化（笔画为 1）
        meta: dict (width, height, char, ...)
    """
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < GNT_HEADER_SIZE:
        raise ValueError(f"GNT file too short: {path}")
    width = struct.unpack("<H", data[6:8])[0]
    height = struct.unpack("<H", data[8:10])[0]
    try:
        char = bytes([data[4], data[5]]).decode("gb2312", errors="replace")
    except Exception:
        char = ""
    pixel_len = width * height
    if len(data) < GNT_HEADER_SIZE + pixel_len:
        raise ValueError(f"GNT pixel data short: {path}")
    pixels = np.frombuffer(data[10 : 10 + pixel_len], dtype=np.uint8)
    img = pixels.reshape((height, width)).astype(np.float32) / 255.0
    # 二值化：Otsu 或简单阈值
    thresh = 0.5
    if img.size > 0:
        try:
            from skimage.filters import threshold_otsu
            thresh = threshold_otsu(img)
        except Exception:
            thresh = 0.5
    img = (img > thresh).astype(np.float32)
    # 缩放到 target_size
    from torch.nn.functional import interpolate
    t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
    t = interpolate(t, size=target_size, mode="bilinear", align_corners=False)
    t = (t.squeeze(0) > 0.5).float()
    meta = {"width": width, "height": height, "char": char}
    return t, meta
