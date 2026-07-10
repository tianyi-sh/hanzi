# -*- coding: utf-8 -*-
"""评估排序：Ranking accuracy、Margin mean。"""
import os
import sys
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.datasets.dataset import StructConstraintDataset
from src.models.traj_encoder import build_traj_encoder
from src.models.quality_head import build_quality_head
from src.utils.metrics import pairwise_rank_accuracy, margin_mean


def collate(batch):
    traj = torch.nn.utils.rnn.pad_sequence([b["traj"] for b in batch], batch_first=True, padding_value=0)
    return {"traj": traj}


def eval_ranking(processed_dir, checkpoint_path, embed_dim=128, hidden_size=128, noise_scale=0.3):
    dataset = StructConstraintDataset(processed_dir)
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate)
    traj_enc = build_traj_encoder(hidden_size=hidden_size, num_layers=2)
    proj_traj = torch.nn.Linear(hidden_size, embed_dim)
    quality_head = build_quality_head(embed_dim=embed_dim)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    traj_enc.load_state_dict(ckpt["traj_encoder"])
    proj_traj.load_state_dict(ckpt["proj_traj"])
    quality_head.load_state_dict(ckpt["quality_head"])
    traj_enc.eval()
    proj_traj.eval()
    quality_head.eval()
    acc_list, marg_list = [], []
    with torch.no_grad():
        for batch in loader:
            traj = batch["traj"]
            traj_bad = traj + torch.randn_like(traj) * noise_scale
            h_good = traj_enc(traj)
            h_bad = traj_enc(traj_bad)
            s_good = quality_head(proj_traj(h_good))
            s_bad = quality_head(proj_traj(h_bad))
            acc_list.append(pairwise_rank_accuracy(s_good, s_bad))
            marg_list.append(margin_mean(s_good, s_bad))
    return {"ranking_accuracy": sum(acc_list) / len(acc_list) if acc_list else 0, "margin_mean": sum(marg_list) / len(marg_list) if marg_list else 0}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", default="data/processed/samples")
    ap.add_argument("--checkpoint", default="outputs/runs/stage3_run/checkpoints/best.pt")
    args = ap.parse_args()
    m = eval_ranking(os.path.join(ROOT, args.processed_dir), os.path.join(ROOT, args.checkpoint))
    print(m)
