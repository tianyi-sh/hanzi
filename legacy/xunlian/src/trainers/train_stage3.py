# -*- coding: utf-8 -*-
"""Stage 3: 质量敏感排序 (Ranking Hinge)。"""
import os
import sys
import yaml
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.datasets.dataset import GntOnlineDataset
from src.models.traj_encoder import build_traj_encoder
from src.models.decoders import ScoreHead
from src.losses.ranking import ranking_hinge_loss
from src.utils.seed import set_seed
from src.utils.logging import log_metrics, save_metrics_json
from src.utils.metrics import pairwise_rank_accuracy, margin_mean


def collate(batch):
    img = torch.stack([b["image"] for b in batch])
    traj = torch.nn.utils.rnn.pad_sequence([b["traj"] for b in batch], batch_first=True, padding_value=0)
    return {"image": img, "traj": traj}


def train_stage3(cfg_path, run_dir):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    set_seed(42)
    data_dir = os.path.join(ROOT, cfg["data"]["processed_dir"])
    noise_scale = cfg["data"].get("degrade_noise_scale", 0.3)
    hidden = cfg["model"]["traj_encoder"]["hidden_size"]
    num_layers = cfg["model"]["traj_encoder"]["num_layers"]
    embed_dim = hidden
    margin = float(cfg["train"].get("margin", 0.5))
    batch_size = int(cfg["train"]["batch_size"])
    epochs = int(cfg["train"]["epochs"])
    lr = float(cfg["train"]["lr"])

    dataset = GntOnlineDataset(data_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)

    traj_enc = build_traj_encoder(hidden_size=hidden, num_layers=num_layers, embed_dim=embed_dim)
    score_head = ScoreHead(embed_dim, score_dim=1)
    opt = torch.optim.Adam(list(traj_enc.parameters()) + list(score_head.parameters()), lr=lr)

    log_path = os.path.join(run_dir, "logs.jsonl")
    best_acc = 0.0
    for ep in range(epochs):
        traj_enc.train()
        score_head.train()
        total_loss = 0.0
        n_b = 0
        for batch in loader:
            traj = batch["traj"]
            traj_bad = traj + torch.randn_like(traj) * noise_scale
            h_good = traj_enc(traj)
            h_bad = traj_enc(traj_bad)
            score_good = score_head(h_good)
            score_bad = score_head(h_bad)
            loss = ranking_hinge_loss(score_good, score_bad, margin)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_b += 1
        avg_loss = total_loss / max(n_b, 1)
        with torch.no_grad():
            acc = pairwise_rank_accuracy(score_good, score_bad)
            marg = margin_mean(score_good, score_bad)
        log_metrics(log_path, {"epoch": ep, "loss": avg_loss, "pairwise_rank_acc": acc, "margin_mean": marg}, step=ep)
        if acc > best_acc:
            best_acc = acc
            torch.save({
                "traj_encoder": traj_enc.state_dict(),
                "score_head": score_head.state_dict(),
            }, os.path.join(run_dir, "checkpoints", "best.pt"))
    save_metrics_json(os.path.join(run_dir, "metrics.json"),
                      {"pairwise_rank_accuracy": best_acc, "margin_mean": marg})
    return best_acc


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stage3.yaml")
    ap.add_argument("--run_dir", default=None)
    args = ap.parse_args()
    run_dir = args.run_dir or os.path.join(ROOT, "outputs", "runs", "stage3_run")
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    cfg_path = os.path.join(ROOT, args.config)
    train_stage3(cfg_path, run_dir)
