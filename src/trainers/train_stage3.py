# -*- coding: utf-8 -*-
"""Stage 3: L = L_mae + λ1 L_align + λ2 L_cons + λ3 L_rank"""
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
from src.models.quality_head import build_quality_head
from src.losses.mae import reconstruction_mae
from src.losses.align_kl import align_kl_loss
from src.losses.consistency import consistency_loss
from src.losses.ranking import ranking_hinge_loss
from src.utils.seed import set_seed
from src.utils.logging import log_metrics, save_metrics_json
from src.utils.metrics import pairwise_rank_accuracy, margin_mean


def collate(batch):
    images = torch.stack([b["image"] for b in batch])
    traj = torch.nn.utils.rnn.pad_sequence([b["traj"] for b in batch], batch_first=True, padding_value=0)
    pi_list = [b["pi_prior"] for b in batch]
    max_k = max(p.size(0) for p in pi_list)
    pi_padded = torch.zeros(len(batch), max_k)
    for i, p in enumerate(pi_list):
        pi_padded[i, : p.size(0)] = p
    return {
        "image": images, "traj": traj,
        "struct_nodes": [b["struct_nodes"] for b in batch],
        "struct_edges": [b["struct_edges"] for b in batch],
        "pi_prior": pi_padded,
        "sample_id": [b["sample_id"] for b in batch],
    }


def train_stage3(cfg_path, run_dir):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    set_seed(42)
    data_dir = os.path.join(ROOT, cfg["data"]["processed_dir"])
    traj_dim = int(cfg["data"].get("traj_dim", 5))
    noise_scale = float(cfg["data"].get("degrade_noise_scale", 0.3))
    hidden = int(cfg["model"]["traj_encoder"]["hidden_size"])
    num_layers = int(cfg["model"]["traj_encoder"]["num_layers"])
    embed_dim = int(cfg["model"]["align_module"]["embed_dim"])
    pred_dim = int(cfg["model"]["decoder"]["pred_dim"])
    lambda_mae = float(cfg["train"]["lambda_mae"])
    lambda_align = float(cfg["train"]["lambda_align"])
    lambda_cons = float(cfg["train"]["lambda_cons"])
    lambda_rank = float(cfg["train"]["lambda_rank"])
    margin = float(cfg["train"].get("rank_margin", 0.5))
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
    quality_head = build_quality_head(embed_dim=embed_dim, hidden=int(cfg["model"]["quality_head"]["hidden"]))
    opt = torch.optim.Adam(
        list(traj_enc.parameters()) + list(struct_enc.parameters()) + list(align_mod.parameters())
        + list(decoder.parameters()) + list(proj_traj.parameters()) + list(quality_head.parameters()),
        lr=lr,
    )

    log_path = os.path.join(run_dir, "logs.jsonl")
    best_acc = -1.0
    for ep in range(epochs):
        traj_enc.train()
        struct_enc.train()
        align_mod.train()
        decoder.train()
        quality_head.train()
        total_loss = total_mae = total_kl = total_cons = total_rank = 0.0
        n_b = 0
        for batch in loader:
            traj = batch["traj"]
            B, T, _ = traj.shape
            traj_bad = traj + torch.randn_like(traj, device=traj.device) * noise_scale
            z_seq = traj_enc(traj, return_sequence=True)
            z_traj = proj_traj(z_seq)
            z_struct_list = []
            pi_prior = batch["pi_prior"]
            K = pi_prior.size(1)
            for b in range(B):
                nodes = batch["struct_nodes"][b].to(traj.device)
                edges = batch["struct_edges"][b]
                if isinstance(edges, list):
                    edges = torch.tensor(edges, dtype=torch.long, device=traj.device)
                if edges.numel() == 0:
                    z_struct_list.append(torch.zeros(1, embed_dim, device=traj.device))
                else:
                    if edges.dim() == 1:
                        edges = edges.unsqueeze(0)
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
            pi_prior = pi_prior[:, :max_k].to(traj.device) / (pi_prior[:, :max_k].sum(1, keepdim=True).to(traj.device) + 1e-8)
            a = align_mod(z_traj, z_struct)
            l_align = align_kl_loss(pi_prior, a)
            l_cons = consistency_loss(z_traj, z_struct, a)
            pred = decoder(traj)
            target = traj[:, :, :pred_dim]
            l_mae = reconstruction_mae(pred, target)
            h_good = traj_enc(traj)
            h_bad = traj_enc(traj_bad)
            s_good = quality_head(proj_traj(h_good))
            s_bad = quality_head(proj_traj(h_bad))
            l_rank = ranking_hinge_loss(s_good, s_bad, margin)
            loss = lambda_mae * l_mae + lambda_align * l_align + lambda_cons * l_cons + lambda_rank * l_rank
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            total_mae += l_mae.item()
            total_kl += l_align.item()
            total_cons += l_cons.item()
            total_rank += l_rank.item()
            n_b += 1
        avg_mae = total_mae / max(n_b, 1)
        with torch.no_grad():
            acc = pairwise_rank_accuracy(s_good, s_bad)
            marg = margin_mean(s_good, s_bad)
        log_metrics(log_path, {"epoch": ep, "loss": total_loss / n_b, "recon_mae": avg_mae, "align_kl": total_kl / n_b, "cons_loss": total_cons / n_b, "ranking_accuracy": acc, "margin_mean": marg}, step=ep)
        if acc > best_acc:
            best_acc = acc
            torch.save({
                "traj_encoder": traj_enc.state_dict(), "struct_encoder": struct_enc.state_dict(),
                "align_module": align_mod.state_dict(), "decoder": decoder.state_dict(),
                "proj_traj": proj_traj.state_dict(), "quality_head": quality_head.state_dict(),
            }, os.path.join(run_dir, "checkpoints", "best.pt"))
    save_metrics_json(os.path.join(run_dir, "metrics.json"), {"recon_mae": avg_mae, "ranking_accuracy": best_acc, "margin_mean": marg})
    return best_acc


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stage3.yaml")
    ap.add_argument("--run_dir", default=None)
    args = ap.parse_args()
    run_dir = args.run_dir or os.path.join(ROOT, "outputs", "runs", "stage3_run")
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    train_stage3(os.path.join(ROOT, args.config), run_dir)
