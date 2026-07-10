# -*- coding: utf-8 -*-
"""Stage 1: 轨迹遮蔽重建 (MAE)。"""
import os
import sys
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.datasets.dataset import GntOnlineDataset
from src.models.decoders import TrajReconstructDecoder
from src.losses.mae import reconstruction_mae
from src.utils.seed import set_seed
from src.utils.logging import log_metrics, save_metrics_json


def collate(batch):
    img = torch.stack([b["image"] for b in batch])
    traj = torch.nn.utils.rnn.pad_sequence(
        [b["traj"] for b in batch], batch_first=True, padding_value=0
    )
    return {"image": img, "traj": traj, "sample_id": [b["sample_id"] for b in batch]}


def train_stage1(cfg_path, run_dir):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    set_seed(42)
    data_dir = os.path.join(ROOT, cfg["data"]["processed_dir"])
    traj_dim = int(cfg["data"].get("traj_dim", 5))
    mask_ratio = float(cfg["data"].get("mask_ratio", 0.25))
    hidden = cfg["model"]["hidden_size"]
    num_layers = cfg["model"]["num_layers"]
    pred_dim = cfg["model"].get("pred_dim", 3)
    batch_size = int(cfg["train"]["batch_size"])
    epochs = int(cfg["train"]["epochs"])
    lr = float(cfg["train"]["lr"])

    dataset = GntOnlineDataset(data_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)

    decoder = TrajReconstructDecoder(hidden, num_layers, input_dim=traj_dim, pred_dim=pred_dim)
    opt = torch.optim.Adam(decoder.parameters(), lr=lr)

    log_path = os.path.join(run_dir, "logs.jsonl")
    best_mae = 1e9
    for ep in range(epochs):
        decoder.train()
        total_loss = 0.0
        n_b = 0
        for batch in loader:
            traj = batch["traj"]
            B, N, _ = traj.shape
            mask = torch.rand(B, N, device=traj.device) < mask_ratio
            traj_masked = traj.clone()
            traj_masked[mask] = 0
            pred = decoder(traj_masked)
            target = traj[:, :, :pred_dim]
            loss = reconstruction_mae(pred, target, mask)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_b += 1
        avg = total_loss / max(n_b, 1)
        mae = avg
        log_metrics(log_path, {"epoch": ep, "loss": avg, "reconstruction_mae": mae}, step=ep)
        if mae < best_mae:
            best_mae = mae
            torch.save({"decoder": decoder.state_dict()}, os.path.join(run_dir, "checkpoints", "best.pt"))
    save_metrics_json(os.path.join(run_dir, "metrics.json"), {"reconstruction_mae": best_mae})
    return best_mae


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stage1.yaml")
    ap.add_argument("--run_dir", default=None)
    args = ap.parse_args()
    run_dir = args.run_dir or os.path.join(ROOT, "outputs", "runs", "stage1_run")
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    cfg_path = os.path.join(ROOT, args.config)
    train_stage1(cfg_path, run_dir)
