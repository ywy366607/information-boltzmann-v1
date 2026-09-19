"""Fast, accurate mathematical and empirical stability audit of GDN vs GDN-2."""
from __future__ import annotations

import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(script_dir))
if sys.path and os.path.abspath(sys.path[0]) == script_dir:
    sys.path.pop(0)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import json
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from scripts.ib_local.gated_deltanet import GatedDeltaNetLM
from scripts.ib_local.gated_deltanet_2 import GatedDeltaNet2LM


@torch.inference_mode()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_data = np.load("data/ib_owt_gpt2/validation.npy", mmap_mode="r")

    # 1. Load trained models
    gdn = GatedDeltaNetLM().to(device).eval()
    gdn.load_state_dict(torch.load("results/gated_deltanet_d128_3000/BBest.pt")["model"])

    gdn2 = GatedDeltaNet2LM().to(device).eval()
    gdn2.load_state_dict(torch.load("results/gated_deltanet_2_d128_3000/BBest.pt")["model"])

    # -------------------------------------------------------------
    # 2. Operator Analysis on 2048 Real Tokens
    # -------------------------------------------------------------
    N_TOK = 2048
    tokens_sub = torch.as_tensor(val_data[:N_TOK].copy(), device=device, dtype=torch.long)[None]

    # GDN-1 layer 0
    h1 = gdn.source.embedding(tokens_sub)
    n1 = gdn.layers[0].norm(h1)
    k1 = F.normalize(gdn.layers[0].k_proj(n1).view(1, N_TOK, 4, 32), dim=-1)
    a1 = torch.sigmoid(gdn.layers[0].alpha_proj(n1)) # [1, N_TOK, 4]
    b1 = torch.sigmoid(gdn.layers[0].beta_proj(n1))  # [1, N_TOK, 4]

    # GDN-2 layer 0
    h2 = gdn2.source.embedding(tokens_sub)
    n2 = gdn2.layers[0].norm(h2)
    k2 = F.normalize(gdn2.layers[0].k_proj(n2).view(1, N_TOK, 4, 32), dim=-1)
    D2 = torch.sigmoid(gdn2.layers[0].decay_proj(n2)).view(1, N_TOK, 4, 32)
    b2 = torch.sigmoid(gdn2.layers[0].erase_proj(n2)).view(1, N_TOK, 4, 32)
    w2 = torch.sigmoid(gdn2.layers[0].write_proj(n2)).view(1, N_TOK, 4, 32)

    # Batched matrix construction [N_TOK * 4, 32, 32]
    # GDN-1: A1 = a * (I - b * k k^T)
    k1_flat = k1.reshape(-1, 32) # [8192, 32]
    a1_flat = a1.reshape(-1, 1, 1) # [8192, 1, 1]
    b1_flat = b1.reshape(-1, 1, 1) # [8192, 1, 1]
    I32 = torch.eye(32, device=device).unsqueeze(0).expand(8192, -1, -1)
    A1_batch = a1_flat * (I32 - b1_flat * torch.bmm(k1_flat.unsqueeze(-1), k1_flat.unsqueeze(1)))

    # GDN-2: A2 = (I - k (b*k)^T) @ Diag(D)
    k2_flat = k2.reshape(-1, 32) # [8192, 32]
    b2_flat = b2.reshape(-1, 32) # [8192, 32]
    D2_flat = D2.reshape(-1, 32) # [8192, 32]
    bk2 = b2_flat * k2_flat
    P_erase = I32 - torch.bmm(k2_flat.unsqueeze(-1), bk2.unsqueeze(1))
    A2_batch = P_erase * D2_flat.unsqueeze(1) # column-wise scaling by D

    # SVD & Eigenvalues
    s1 = torch.linalg.svdvals(A1_batch)
    e1 = torch.linalg.eigvals(A1_batch).abs()
    s2 = torch.linalg.svdvals(A2_batch)
    e2 = torch.linalg.eigvals(A2_batch).abs()

    s1_max = s1[:, 0].cpu().numpy()
    e1_max = e1.amax(-1).cpu().numpy()
    s2_max = s2[:, 0].cpu().numpy()
    e2_max = e2.amax(-1).cpu().numpy()

    # -------------------------------------------------------------
    # 3. Transition Products (multi-step amplification)
    # -------------------------------------------------------------
    # Sample 16 blocks of 64 steps
    prod_lengths = [1, 2, 4, 8, 16, 32, 64]
    prod_norms1 = {L: [] for L in prod_lengths}
    prod_norms2 = {L: [] for L in prod_lengths}

    for start in range(0, 8192 - 64, 512):
        P1 = torch.eye(32, device=device)
        P2 = torch.eye(32, device=device)
        for step in range(64):
            idx = start + step
            P1 = A1_batch[idx] @ P1
            P2 = A2_batch[idx] @ P2
            L = step + 1
            if L in prod_lengths:
                prod_norms1[L].append(float(torch.linalg.svdvals(P1).max()))
                prod_norms2[L].append(float(torch.linalg.svdvals(P2).max()))

    # -------------------------------------------------------------
    # 4. Long Sequence State Frobenius Norm (2048, 8192, 32768)
    # -------------------------------------------------------------
    lengths = [2048, 8192, 32768]
    rollouts = {}
    CHUNK = 128

    for L in lengths:
        sub = torch.as_tensor(val_data[:L].copy(), device=device, dtype=torch.long)

        # GDN un-reset
        s_gdn = gdn.initial_state(1, device=device)
        norms_gdn = []
        for i in range(0, L, CHUNK):
            c = sub[i:i+CHUNK][None]
            if c.shape[1] < CHUNK: break
            _, s_gdn, _ = gdn(c, state=s_gdn)
            norms_gdn.append(float(s_gdn.square().sum().sqrt()))

        # GDN-2 un-reset
        s_gdn2 = gdn2.initial_state(1, device=device)
        norms_gdn2 = []
        for i in range(0, L, CHUNK):
            c = sub[i:i+CHUNK][None]
            if c.shape[1] < CHUNK: break
            _, s_gdn2, _ = gdn2(c, state=s_gdn2)
            norms_gdn2.append(float(s_gdn2.square().sum().sqrt()))

        # GDN-2 reset every 2048
        s_gdn2_res = gdn2.initial_state(1, device=device)
        norms_gdn2_res = []
        for i in range(0, L, CHUNK):
            if i > 0 and i % 2048 == 0:
                s_gdn2_res = gdn2.initial_state(1, device=device)
            c = sub[i:i+CHUNK][None]
            if c.shape[1] < CHUNK: break
            _, s_gdn2_res, _ = gdn2(c, state=s_gdn2_res)
            norms_gdn2_res.append(float(s_gdn2_res.square().sum().sqrt()))

        rollouts[L] = {
            "gdn_final": norms_gdn[-1],
            "gdn_max": max(norms_gdn),
            "gdn2_final": norms_gdn2[-1],
            "gdn2_max": max(norms_gdn2),
            "gdn2_reset2048_max": max(norms_gdn2_res),
        }

    results = {
        "operator": {
            "gdn1": {
                "max_sigma_max": float(np.max(s1_max)),
                "mean_sigma_max": float(np.mean(s1_max)),
                "frac_sigma_gt_1": float(np.mean(s1_max > 1.0)),
                "max_spectral_radius": float(np.max(e1_max)),
            },
            "gdn2": {
                "max_sigma_max": float(np.max(s2_max)),
                "mean_sigma_max": float(np.mean(s2_max)),
                "frac_sigma_gt_1": float(np.mean(s2_max > 1.0)),
                "max_spectral_radius": float(np.max(e2_max)),
            }
        },
        "transition_products": {
            "gdn1": {L: float(np.mean(prod_norms1[L])) for L in prod_lengths},
            "gdn2": {L: float(np.mean(prod_norms2[L])) for L in prod_lengths},
        },
        "rollouts": rollouts,
        "gates": {
            "gdn1_alpha_mean": float(a1.mean()),
            "gdn1_beta_mean": float(b1.mean()),
            "gdn2_decay_mean": float(D2.mean()),
            "gdn2_erase_mean": float(b2.mean()),
            "gdn2_write_mean": float(w2.mean()),
        }
    }

    with open("results/gdn_stability_audit_final.json", "w") as f:
        json.dump(results, f, indent=2)
    print("SUCCESS_WRITTEN")


if __name__ == "__main__":
    main()
