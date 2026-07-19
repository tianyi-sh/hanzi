# -*- coding: utf-8 -*-
"""Stage 2: 跨模态对齐 (InfoNCE)。"""
import os
import sys
import yaml
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.datasets.dataset import GntOnlineDataset
from src.models.img_encoder import build_img_encoder
from src.models.traj_encoder import build_traj_encoder
from src.models.fusion_heads import ProjectionHead
from src.losses.contrastive import infonce_loss
from src.utils.seed import set_seed
from src.utils.logging import log_metrics, save_metrics_json
from src.utils.metrics import recall_at_k


def collate(batch):
    img = torch.stack([b["image"] for b in batch])
    traj = torch.nn.utils.rnn.pad_sequence([b["traj"] for b in batch], batch_first=True, padding_value=0)
    return {"image": img, "traj": traj}


def train_stage2(cfg_path, run_dir):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    set_seed(42)
    data_dir = os.path.join(ROOT, cfg["data"]["processed_dir"])
    embed_dim = cfg["model"]["embed_dim"]
    temperature = cfg["model"].get("temperature", 0.07)
    batch_size = int(cfg["train"]["batch_size"])
    epochs = int(cfg["train"]["epochs"])
    lr = float(cfg["train"]["lr"])

    dataset = GntOnlineDataset(data_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)

    img_enc = build_img_encoder(hidden=cfg["model"]["img_encoder"]["hidden"], embed_dim=embed_dim)
    traj_enc = build_traj_encoder(
        hidden_size=cfg["model"]["traj_encoder"]["hidden_size"],
        num_layers=cfg["model"]["traj_encoder"]["num_layers"],
        embed_dim=embed_dim,
    )
    img_proj = ProjectionHead(embed_dim, embed_dim)
    traj_proj = ProjectionHead(embed_dim, embed_dim)
    opt = torch.optim.Adam(
        list(img_enc.parameters()) + list(traj_enc.parameters())
        + list(img_proj.parameters()) + list(traj_proj.parameters()),
        lr=lr,
    )

    log_path = os.path.join(run_dir, "logs.jsonl")
    best_r1 = 0.0
    for ep in range(epochs):
        img_enc.train()
        traj_enc.train()
        img_proj.train()
        traj_proj.train()
        total_loss = 0.0
        n_b = 0
        for batch in loader:
            img_feat = img_enc(batch["image"])
            traj_feat = traj_enc(batch["traj"])
            img_embed = img_proj(img_feat)
            traj_embed = traj_proj(traj_feat)
            loss = infonce_loss(img_embed, traj_embed, temperature)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_b += 1
        avg_loss = total_loss / max(n_b, 1)
        with torch.no_grad():
            logits = (img_embed @ traj_embed.T) / temperature
            r1 = recall_at_k(logits, 1)
            r5 = recall_at_k(logits, 5)
        log_metrics(log_path, {"epoch": ep, "loss": avg_loss, "recall@1": r1, "recall@5": r5}, step=ep)
        if r1 > best_r1:
            best_r1 = r1
            torch.save({
                "img_encoder": img_enc.state_dict(),
                "traj_encoder": traj_enc.state_dict(),
                "img_proj": img_proj.state_dict(),
                "traj_proj": traj_proj.state_dict(),
            }, os.path.join(run_dir, "checkpoints", "best.pt"))
    save_metrics_json(os.path.join(run_dir, "metrics.json"), {"recall_at_1": best_r1, "align_loss": avg_loss})
    return best_r1


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stage2.yaml")
    ap.add_argument("--run_dir", default=None)
    args = ap.parse_args()
    run_dir = args.run_dir or os.path.join(ROOT, "outputs", "runs", "stage2_run")
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    cfg_path = os.path.join(ROOT, args.config)
    train_stage2(cfg_path, run_dir)
