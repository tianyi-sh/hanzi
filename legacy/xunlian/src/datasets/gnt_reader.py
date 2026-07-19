# -*- coding: utf-8 -*-
"""解析 GNT 格式离线字形图像。"""
import struct
import numpy as np

GNT_HEADER_SIZE = 10


def read_gnt(path):
    """
    读取 GNT 文件，返回图像与元信息。
    Returns:
        image: np.ndarray (H, W), dtype float32, 值域 [0, 1]
        meta: dict with keys: width, height, char (GB2312 解码), sample_size
    """
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < GNT_HEADER_SIZE:
        raise ValueError(f"GNT file too short: {path}")
    sample_size = struct.unpack("<I", data[0:4])[0]
    tag = struct.unpack("<H", data[4:6])[0]
    width = struct.unpack("<H", data[6:8])[0]
    height = struct.unpack("<H", data[8:10])[0]
    try:
        char = bytes([data[4], data[5]]).decode("gb2312", errors="replace")
    except Exception:
        char = ""
    pixel_len = width * height
    if len(data) < GNT_HEADER_SIZE + pixel_len:
        raise ValueError(f"GNT pixel data short: need {pixel_len}, got {len(data) - GNT_HEADER_SIZE}")
    pixels = np.frombuffer(data[10 : 10 + pixel_len], dtype=np.uint8)
    image = pixels.reshape((height, width)).astype(np.float32) / 255.0
    meta = {"width": width, "height": height, "char": char, "sample_size": sample_size}
    return image, meta
