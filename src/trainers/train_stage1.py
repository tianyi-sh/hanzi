# -*- coding: utf-8 -*-
"""Stage 1: L = L_mae + λ1 L_align"""
import os
import sys
import yaml
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.datasets.dataset import StructConstraintDataset
from src.models.traj_encoder import build_traj_encoder
from src.models.struct_encoder import build_struct_encoder
from src.models.align_module import build_align_module
from src.models.decoder import build_decoder
from src.losses.mae import reconstruction_mae
from src.losses.align_kl import align_kl_loss
from src.datasets.align_utils import compute_align_kl, align_entropy
from src.utils.seed import set_seed
from src.utils.logging import log_metrics, save_metrics_json


def collate(batch):
    images = torch.stack([b["image"] for b in batch])
    traj = torch.nn.utils.rnn.pad_sequence([b["traj"] for b in batch], batch_first=True, padding_value=0)
    pi_list = [b["pi_prior"] for b in batch]
    max_k = max(p.size(0) for p in pi_list)
    pi_padded = torch.zeros(len(batch), max_k)
    for i, p in enumerate(pi_list):
        pi_padded[i, : p.size(0)] = p
    return {
        "image": images,
        "traj": traj,
        "struct_nodes": [b["struct_nodes"] for b in batch],
        "struct_edges": [b["struct_edges"] for b in batch],
        "pi_prior": pi_padded,
        "sample_id": [b["sample_id"] for b in batch],
    }


def train_stage1(cfg_path, run_dir):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    set_seed(42)
    data_dir = os.path.join(ROOT, cfg["data"]["processed_dir"])
    traj_dim = int(cfg["data"].get("traj_dim", 5))
    hidden = int(cfg["model"]["traj_encoder"]["hidden_size"])
    num_layers = int(cfg["model"]["traj_encoder"]["num_layers"])
    embed_dim = int(cfg["model"]["align_module"]["embed_dim"])
    pred_dim = int(cfg["model"]["decoder"]["pred_dim"])
    lambda_mae = float(cfg["train"]["lambda_mae"])
    lambda_align = float(cfg["train"]["lambda_align"])
    batch_size = int(cfg["train"]["batch_size"])
    epochs = int(cfg["train"]["epochs"])
    lr = float(cfg["train"]["lr"])

    dataset = StructConstraintDataset(data_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)

    traj_enc = build_traj_encoder(input_dim=traj_dim, hidden_size=hidden, num_layers=num_layers)
    struct_enc = build_struct_encoder(hidden_size=hidden, out_dim=embed_dim)
    align_mod = build_align_module(embed_dim=embed_dim)
    decoder = build_decoder(hidden_size=hidden, num_layers=num_layers, pred_dim=pred_dim)
    proj_traj = torch.nn.Linear(hidden, embed_dim)
    opt = torch.optim.Adam(
        list(traj_enc.parameters()) + list(struct_enc.parameters())
        + list(align_mod.parameters()) + list(decoder.parameters()) + list(proj_traj.parameters()),
        lr=lr,
    )

    log_path = os.path.join(run_dir, "logs.jsonl")
    best_mae = 1e9
    for ep in range(epochs):
        traj_enc.train()
        struct_enc.train()
        align_mod.train()
        decoder.train()
        total_loss = total_mae = total_kl = 0.0
        n_b = 0
        for batch in loader:
            traj = batch["traj"]
            B, T, _ = traj.shape
            z_seq = traj_enc(traj, return_sequence=True)
            z_traj = proj_traj(z_seq)
            z_struct_list = []
            pi_prior = batch["pi_prior"]
            K = pi_prior.size(1)
            for b in range(B):
                nodes = batch["struct_nodes"][b]
                edges = batch["struct_edges"][b]
                if isinstance(edges, list):
                    edges = torch.tensor(edges, dtype=torch.long, device=traj.device)
                if edges.numel() == 0:
                    z_struct_list.append(torch.zeros(1, embed_dim, device=traj.device))
                else:
                    nodes = nodes.to(traj.device)
                    if edges.dim() == 1:
                        edges = edges.unsqueeze(0)
                    edges = edges.to(traj.device)
                    z_struct_list.append(struct_enc(nodes, edges))
            max_k = max(z.size(0) for z in z_struct_list)
            z_struct = torch.zeros(B, max_k, embed_dim, device=traj.device)
            for b in range(B):
                k = z_struct_list[b].size(0)
                z_struct[b, :k] = z_struct_list[b]
            if max_k > K:
                pi_prior = torch.nn.functional.pad(pi_prior, (0, max_k - K))
            elif max_k < K:
                z_struct = torch.nn.functional.pad(z_struct, (0, 0, 0, K - max_k))
                max_k = K
            a = align_mod(z_traj, z_struct)
            pi_prior = pi_prior[:, :max_k].to(traj.device)
            pi_prior = pi_prior / (pi_prior.sum(1, keepdim=True) + 1e-8)
            l_align = align_kl_loss(pi_prior, a)
            pred = decoder(traj)
            target = traj[:, :, :pred_dim]
            l_mae = reconstruction_mae(pred, target)
            loss = lambda_mae * l_mae + lambda_align * l_align
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            total_mae += l_mae.item()
            total_kl += l_align.item()
            n_b += 1
        avg_loss = total_loss / max(n_b, 1)
        avg_mae = total_mae / max(n_b, 1)
        avg_kl = total_kl / max(n_b, 1)
        log_metrics(log_path, {"epoch": ep, "loss": avg_loss, "recon_mae": avg_mae, "align_kl": avg_kl}, step=ep)
        if avg_mae < best_mae:
            best_mae = avg_mae
            torch.save({
                "traj_encoder": traj_enc.state_dict(),
                "struct_encoder": struct_enc.state_dict(),
                "align_module": align_mod.state_dict(),
                "decoder": decoder.state_dict(),
                "proj_traj": proj_traj.state_dict(),
            }, os.path.join(run_dir, "checkpoints", "best.pt"))
    save_metrics_json(os.path.join(run_dir, "metrics.json"), {"recon_mae": best_mae, "align_kl": avg_kl})
    return best_mae


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stage1.yaml")
    ap.add_argument("--run_dir", default=None)
    args = ap.parse_args()
    run_dir = args.run_dir or os.path.join(ROOT, "outputs", "runs", "stage1_run")
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    train_stage1(os.path.join(ROOT, args.config), run_dir)
