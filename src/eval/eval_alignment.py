# -*- coding: utf-8 -*-
"""评估对齐：Align KL、Align entropy。"""
import os
import sys
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.datasets.dataset import StructConstraintDataset
from src.models.traj_encoder import build_traj_encoder
from src.models.struct_encoder import build_struct_encoder
from src.models.align_module import build_align_module
from src.utils.metrics import align_kl_metric, align_entropy_metric


def collate(batch):
    traj = torch.nn.utils.rnn.pad_sequence([b["traj"] for b in batch], batch_first=True, padding_value=0)
    pi_list = [b["pi_prior"] for b in batch]
    max_k = max(p.size(0) for p in pi_list)
    pi_padded = torch.zeros(len(batch), max_k)
    for i, p in enumerate(pi_list):
        pi_padded[i, : p.size(0)] = p
    return {"traj": traj, "struct_nodes": [b["struct_nodes"] for b in batch], "struct_edges": [b["struct_edges"] for b in batch], "pi_prior": pi_padded}


def eval_alignment(processed_dir, checkpoint_path, embed_dim=128, hidden_size=128):
    dataset = StructConstraintDataset(processed_dir)
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate)
    traj_enc = build_traj_encoder(hidden_size=hidden_size, num_layers=2)
    struct_enc = build_struct_encoder(hidden_size=hidden_size, out_dim=embed_dim)
    align_mod = build_align_module(embed_dim=embed_dim)
    proj_traj = torch.nn.Linear(hidden_size, embed_dim)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    traj_enc.load_state_dict(ckpt["traj_encoder"])
    struct_enc.load_state_dict(ckpt["struct_encoder"])
    align_mod.load_state_dict(ckpt["align_module"])
    proj_traj.load_state_dict(ckpt["proj_traj"])
    traj_enc.eval()
    struct_enc.eval()
    align_mod.eval()
    proj_traj.eval()
    total_kl = total_ent = 0.0
    n_b = 0
    with torch.no_grad():
        for batch in loader:
            traj = batch["traj"]
            B, T, _ = traj.shape
            z_seq = traj_enc(traj, return_sequence=True)
            z_traj = proj_traj(z_seq)
            z_struct_list = []
            for b in range(B):
                nodes = batch["struct_nodes"][b]
                edges = batch["struct_edges"][b]
                if isinstance(edges, list):
                    edges = torch.tensor(edges, dtype=torch.long)
                if edges.numel() == 0:
                    z_struct_list.append(torch.zeros(1, embed_dim))
                else:
                    if edges.dim() == 1:
                        edges = edges.unsqueeze(0)
                    z_struct_list.append(struct_enc(nodes, edges))
            max_k = max(z.size(0) for z in z_struct_list)
            z_struct = torch.zeros(B, max_k, embed_dim)
            for b in range(B):
                k = z_struct_list[b].size(0)
                z_struct[b, :k] = z_struct_list[b]
            a = align_mod(z_traj, z_struct)
            pi = batch["pi_prior"][:, :max_k] / (batch["pi_prior"][:, :max_k].sum(1, keepdim=True) + 1e-8)
            total_kl += align_kl_metric(pi, a)
            total_ent += align_entropy_metric(a)
            n_b += 1
    return {"align_kl": total_kl / max(n_b, 1), "align_entropy": total_ent / max(n_b, 1)}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", default="data/processed/samples")
    ap.add_argument("--checkpoint", default="outputs/runs/stage1_run/checkpoints/best.pt")
    args = ap.parse_args()
    m = eval_alignment(os.path.join(ROOT, args.processed_dir), os.path.join(ROOT, args.checkpoint))
    print(m)
