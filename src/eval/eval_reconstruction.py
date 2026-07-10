# -*- coding: utf-8 -*-
"""评估重建：Recon MAE。"""
import os
import sys
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.datasets.dataset import StructConstraintDataset
from src.models.traj_encoder import build_traj_encoder
from src.models.decoder import build_decoder
from src.utils.metrics import reconstruction_mae_metric


def collate(batch):
    traj = torch.nn.utils.rnn.pad_sequence([b["traj"] for b in batch], batch_first=True, padding_value=0)
    return {"traj": traj}


def eval_reconstruction(processed_dir, checkpoint_path, traj_dim=5, hidden_size=128, num_layers=2, pred_dim=3):
    dataset = StructConstraintDataset(processed_dir)
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate)
    traj_enc = build_traj_encoder(input_dim=traj_dim, hidden_size=hidden_size, num_layers=num_layers)
    decoder = build_decoder(hidden_size=hidden_size, num_layers=num_layers, pred_dim=pred_dim)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    traj_enc.load_state_dict(ckpt["traj_encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    traj_enc.eval()
    decoder.eval()
    total_mae = 0.0
    n_b = 0
    with torch.no_grad():
        for batch in loader:
            traj = batch["traj"]
            pred = decoder(traj)
            target = traj[:, :, :pred_dim]
            total_mae += reconstruction_mae_metric(pred, target)
            n_b += 1
    return total_mae / max(n_b, 1)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", default="data/processed/samples")
    ap.add_argument("--checkpoint", default="outputs/runs/stage1_run/checkpoints/best.pt")
    args = ap.parse_args()
    mae = eval_reconstruction(os.path.join(ROOT, args.processed_dir), os.path.join(ROOT, args.checkpoint))
    print("Recon MAE:", mae)
