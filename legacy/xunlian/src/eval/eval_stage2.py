# -*- coding: utf-8 -*-
"""Stage 2 评估：Recall@1/5、对齐损失。"""
import os
import sys
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.datasets.dataset import GntOnlineDataset
from src.models.img_encoder import build_img_encoder
from src.models.traj_encoder import build_traj_encoder
from src.models.fusion_heads import ProjectionHead
from src.losses.contrastive import infonce_loss
from src.utils.metrics import recall_at_k


def collate(batch):
    img = torch.stack([b["image"] for b in batch])
    traj = torch.nn.utils.rnn.pad_sequence([b["traj"] for b in batch], batch_first=True, padding_value=0)
    return {"image": img, "traj": traj}


def eval_stage2(processed_dir, checkpoint_path, embed_dim=128):
    dataset = GntOnlineDataset(processed_dir)
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate)
    img_enc = build_img_encoder(hidden=128, embed_dim=embed_dim)
    traj_enc = build_traj_encoder(hidden_size=128, num_layers=2, embed_dim=embed_dim)
    img_proj = ProjectionHead(embed_dim, embed_dim)
    traj_proj = ProjectionHead(embed_dim, embed_dim)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    img_enc.load_state_dict(ckpt["img_encoder"])
    traj_enc.load_state_dict(ckpt["traj_encoder"])
    img_proj.load_state_dict(ckpt["img_proj"])
    traj_proj.load_state_dict(ckpt["traj_proj"])
    img_enc.eval()
    traj_enc.eval()
    img_proj.eval()
    traj_proj.eval()
    r1_list, r5_list, loss_list = [], [], []
    with torch.no_grad():
        for batch in loader:
            img_feat = img_enc(batch["image"])
            traj_feat = traj_enc(batch["traj"])
            img_embed = img_proj(img_feat)
            traj_embed = traj_proj(traj_feat)
            loss_list.append(infonce_loss(img_embed, traj_embed).item())
            logits = (img_embed @ traj_embed.T) / 0.07
            r1_list.append(recall_at_k(logits, 1))
            r5_list.append(recall_at_k(logits, 5))
    return {
        "recall_at_1": sum(r1_list) / len(r1_list) if r1_list else 0,
        "recall_at_5": sum(r5_list) / len(r5_list) if r5_list else 0,
        "align_loss": sum(loss_list) / len(loss_list) if loss_list else 0,
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", default="data/processed/samples")
    ap.add_argument("--checkpoint", default="outputs/runs/stage2_run/checkpoints/best.pt")
    args = ap.parse_args()
    processed_dir = os.path.join(ROOT, args.processed_dir)
    metrics = eval_stage2(processed_dir, os.path.join(ROOT, args.checkpoint))
    print(metrics)
