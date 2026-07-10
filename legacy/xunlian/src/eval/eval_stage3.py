# -*- coding: utf-8 -*-
"""Stage 3 评估：Pairwise Rank Accuracy、Margin Mean。"""
import os
import sys
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.datasets.dataset import GntOnlineDataset
from src.models.traj_encoder import build_traj_encoder
from src.models.decoders import ScoreHead
from src.utils.metrics import pairwise_rank_accuracy, margin_mean


def collate(batch):
    traj = torch.nn.utils.rnn.pad_sequence([b["traj"] for b in batch], batch_first=True, padding_value=0)
    return {"traj": traj}


def eval_stage3(processed_dir, checkpoint_path, embed_dim=128, noise_scale=0.3):
    dataset = GntOnlineDataset(processed_dir)
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate)
    traj_enc = build_traj_encoder(hidden_size=128, num_layers=2, embed_dim=embed_dim)
    score_head = ScoreHead(embed_dim, 1)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    traj_enc.load_state_dict(ckpt["traj_encoder"])
    score_head.load_state_dict(ckpt["score_head"])
    traj_enc.eval()
    score_head.eval()
    acc_list, marg_list = [], []
    with torch.no_grad():
        for batch in loader:
            traj = batch["traj"]
            traj_bad = traj + torch.randn_like(traj) * noise_scale
            s_good = score_head(traj_enc(traj))
            s_bad = score_head(traj_enc(traj_bad))
            acc_list.append(pairwise_rank_accuracy(s_good, s_bad))
            marg_list.append(margin_mean(s_good, s_bad))
    return {
        "pairwise_rank_accuracy": sum(acc_list) / len(acc_list) if acc_list else 0,
        "margin_mean": sum(marg_list) / len(marg_list) if marg_list else 0,
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", default="data/processed/samples")
    ap.add_argument("--checkpoint", default="outputs/runs/stage3_run/checkpoints/best.pt")
    args = ap.parse_args()
    processed_dir = os.path.join(ROOT, args.processed_dir)
    metrics = eval_stage3(processed_dir, os.path.join(ROOT, args.checkpoint))
    print(metrics)
