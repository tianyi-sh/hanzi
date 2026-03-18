# -*- coding: utf-8 -*-
"""生成结构热力图、轨迹-结构叠加、轨迹异常、排序得分图，保存到指定文件夹。"""
import os
import sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(os.path.realpath(__file__))))
sys.path.insert(0, ROOT)

# 输出目录：优先使用当前脚本所在项目下的 outputs/figures（绝对路径）
OUTPUT_DIR = os.path.join(ROOT, "outputs", "figures")
OUTPUT_DIR = os.path.abspath(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAMPLES_DIR = os.path.join(ROOT, "data", "processed", "samples")
IMG_SHAPE = (224, 224)


def _save_fig(path, draw_fn):
    path = os.path.abspath(path)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        draw_fn(plt)
        plt.savefig(path, bbox_inches="tight", dpi=100)
        plt.close("all")
        if os.path.isfile(path):
            print("已保存:", path)
            return True
    except Exception as e:
        print("保存失败", path, ":", e)
    return False


def scale_traj_to_img(traj_xy, out_h=224, out_w=224):
    x = np.clip(traj_xy[:, 0], 0, 10) / 10.0 * (out_w - 1)
    y = np.clip(traj_xy[:, 1], 0, 10) / 10.0 * (out_h - 1)
    return np.column_stack([x, y])


def main():
    print("输出目录（绝对路径）:", OUTPUT_DIR)
    sample_id = "sample_00"
    path = os.path.join(SAMPLES_DIR, f"{sample_id}.pt")
    if not os.path.isfile(path):
        print("未找到 processed 样本，请先运行 run_pipeline.py")
        print("期望路径:", os.path.abspath(path))
        return
    data = torch.load(path, map_location="cpu", weights_only=False)
    traj = data["traj"]
    struct_nodes = data["struct_nodes"].numpy()
    struct_edges = data["struct_edges"]
    if isinstance(struct_edges, torch.Tensor):
        struct_edges = struct_edges.tolist()
    elif isinstance(struct_edges, np.ndarray):
        struct_edges = struct_edges.tolist()
    if isinstance(struct_edges, list) and len(struct_edges) > 0 and isinstance(struct_edges[0], (list, np.ndarray)):
        struct_edges = [tuple(e) for e in struct_edges]

    traj_xy = traj.numpy()[:, :2]
    traj_in_img = scale_traj_to_img(traj_xy, IMG_SHAPE[0], IMG_SHAPE[1])
    h, w = IMG_SHAPE[0], IMG_SHAPE[1]

    # 1. 结构热力图
    def _draw_heatmap(plt):
        canvas = np.zeros((h, w))
        for (i, j) in struct_edges:
            if i < len(struct_nodes) and j < len(struct_nodes):
                r0, c0 = int(struct_nodes[i, 0]), int(struct_nodes[i, 1])
                r1, c1 = int(struct_nodes[j, 0]), int(struct_nodes[j, 1])
                rr = np.linspace(r0, r1, max(2, int(np.hypot(r1 - r0, c1 - c0))))
                cc = np.linspace(c0, c1, len(rr))
                for r, c in zip(rr, cc):
                    if 0 <= r < h and 0 <= c < w:
                        canvas[int(r), int(c)] = 1
        for n in struct_nodes:
            r, c = int(n[0]), int(n[1])
            if 0 <= r < h and 0 <= c < w:
                canvas[r, c] = 2
        plt.figure(figsize=(5, 5))
        plt.imshow(canvas, cmap="hot")
        plt.axis("off")
    _save_fig(os.path.join(OUTPUT_DIR, "struct_heatmap.png"), _draw_heatmap)

    # 2. 轨迹叠加在结构图上
    def _draw_traj_on_struct(plt):
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        for (i, j) in struct_edges:
            if i < len(struct_nodes) and j < len(struct_nodes):
                ax.plot([struct_nodes[i, 1], struct_nodes[j, 1]], [struct_nodes[i, 0], struct_nodes[j, 0]], "b-", alpha=0.7)
        ax.scatter(struct_nodes[:, 1], struct_nodes[:, 0], c="blue", s=20)
        ax.plot(traj_in_img[:, 0], traj_in_img[:, 1], "r-", alpha=0.8, label="traj")
        ax.legend()
        ax.axis("equal")
    _save_fig(os.path.join(OUTPUT_DIR, "traj_on_struct.png"), _draw_traj_on_struct)

    # 3. 轨迹异常：原始 vs 加噪
    np.random.seed(42)
    traj_degraded = traj_xy + np.random.randn(*traj_xy.shape).astype(np.float32) * 0.5
    def _draw_anomaly(plt):
        plt.figure(figsize=(6, 6))
        plt.plot(traj_xy[:, 0], traj_xy[:, 1], "b-", label="Original")
        plt.plot(traj_degraded[:, 0], traj_degraded[:, 1], "r--", alpha=0.7, label="Degraded")
        plt.legend()
        plt.axis("equal")
    _save_fig(os.path.join(OUTPUT_DIR, "traj_anomaly.png"), _draw_anomaly)

    # 4. 排序得分分布
    ckpt_path = os.path.join(ROOT, "outputs", "runs")
    if os.path.isdir(ckpt_path):
        runs = sorted([d for d in os.listdir(ckpt_path) if d.startswith("run_")], reverse=True)
        for r in runs:
            stage3_ckpt = os.path.join(ckpt_path, r, "stage3", "checkpoints", "best.pt")
            if os.path.isfile(stage3_ckpt):
                from src.datasets.dataset import StructConstraintDataset
                from src.models.traj_encoder import build_traj_encoder
                from src.models.quality_head import build_quality_head
                from torch.utils.data import DataLoader
                def _collate(b):
                    traj = torch.nn.utils.rnn.pad_sequence([x["traj"] for x in b], batch_first=True, padding_value=0)
                    return {"traj": traj}
                ds = StructConstraintDataset(SAMPLES_DIR)
                loader = DataLoader(ds, batch_size=min(10, len(ds)), collate_fn=_collate)
                traj_enc = build_traj_encoder(hidden_size=128, num_layers=2)
                proj = torch.nn.Linear(128, 128)
                quality_head = build_quality_head(embed_dim=128)
                ckpt = torch.load(stage3_ckpt, map_location="cpu", weights_only=True)
                traj_enc.load_state_dict(ckpt["traj_encoder"])
                proj.load_state_dict(ckpt["proj_traj"])
                quality_head.load_state_dict(ckpt["quality_head"])
                traj_enc.eval()
                proj.eval()
                quality_head.eval()
                with torch.no_grad():
                    for batch in loader:
                        t = batch["traj"]
                        t_bad = t + torch.randn_like(t) * 0.3
                        s_g = quality_head(proj(traj_enc(t)))
                        s_b = quality_head(proj(traj_enc(t_bad)))
                        break
                def _draw_ranking(plt):
                    plt.figure(figsize=(5, 3))
                    plt.hist(s_g.detach().cpu().numpy(), bins=10, alpha=0.6, label="Good")
                    plt.hist(s_b.detach().cpu().numpy(), bins=10, alpha=0.6, label="Bad")
                    plt.legend()
                _save_fig(os.path.join(OUTPUT_DIR, "ranking_scores.png"), _draw_ranking)
                break
    else:
        print("未找到 stage3 checkpoint，跳过 ranking_scores.png")

    # 列出最终目录内容
    if os.path.isdir(OUTPUT_DIR):
        files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".png")]
        print("当前目录下 PNG 文件:", files if files else "(无)")

    # 同时复制到工作区（你当前打开的项目目录），方便在资源管理器中查看
    try:
        ws_dir = os.path.abspath(r"d:\QTxiangmu\页面设计\xunlian2_figures")
        if os.path.isdir(OUTPUT_DIR) and os.path.isdir(os.path.dirname(ws_dir)):
            os.makedirs(ws_dir, exist_ok=True)
            import shutil
            for f in os.listdir(OUTPUT_DIR):
                if f.endswith(".png"):
                    shutil.copy2(os.path.join(OUTPUT_DIR, f), os.path.join(ws_dir, f))
            print("已复制到工作区:", ws_dir)
    except Exception as e:
        print("复制到工作区失败:", e)


if __name__ == "__main__":
    main()
