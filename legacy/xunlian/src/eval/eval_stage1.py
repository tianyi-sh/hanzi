# -*- coding: utf-8 -*-
"""Stage 1 评估：Reconstruction MAE。"""
import os
import sys
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.datasets.dataset import GntOnlineDataset
from src.models.decoders import TrajReconstructDecoder
from src.utils.metrics import reconstruction_mae_metric


def collate(batch):
    traj = torch.nn.utils.rnn.pad_sequence([b["traj"] for b in batch], batch_first=True, padding_value=0)
    return {"traj": traj}


def eval_stage1(processed_dir, checkpoint_path, traj_dim=5, hidden_size=128, num_layers=2, pred_dim=3, mask_ratio=0.25):
    dataset = GntOnlineDataset(processed_dir)
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate)
    decoder = TrajReconstructDecoder(hidden_size, num_layers, input_dim=traj_dim, pred_dim=pred_dim)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    decoder.load_state_dict(ckpt["decoder"])
    decoder.eval()
    total_mae = 0.0
    n_b = 0
    with torch.no_grad():
        for batch in loader:
            traj = batch["traj"]
            B, N, _ = traj.shape
            mask = torch.rand(B, N) < mask_ratio
            traj_masked = traj.clone()
            traj_masked[mask] = 0
            pred = decoder(traj_masked)
            target = traj[:, :, :pred_dim]
            total_mae += reconstruction_mae_metric(pred, target, mask)
            n_b += 1
    return total_mae / max(n_b, 1)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", default="data/processed/samples")
    ap.add_argument("--checkpoint", default="outputs/runs/stage1_run/checkpoints/best.pt")
    args = ap.parse_args()
    processed_dir = os.path.join(ROOT, args.processed_dir)
    mae = eval_stage1(processed_dir, os.path.join(ROOT, args.checkpoint))
    print("Reconstruction MAE:", mae)
