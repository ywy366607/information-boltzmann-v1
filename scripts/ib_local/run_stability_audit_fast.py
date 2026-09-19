"""Rigorous stability audit comparing GDN vs NVIDIA GDN-2.

Measures:
1. Operator norm sigma_max(A_t) vs spectral radius rho(A_t) (non-normal amplification).
2. Multi-step transition product norm || prod_{i=0}^{L-1} A_{t+i} ||_2.
3. State Frobenius norm accumulation over un-reset long sequences (2k, 8k, 32k, 100k tokens).
4. Causal attribution: Is growth driven by non-normal transition amplification or additive write accumulation?
5. Comparison: GDN vs GDN-2 (un-reset) vs GDN-2 (reset every 2048).
"""
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
def run_audit():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_data = np.load("data/ib_owt_gpt2/validation.npy", mmap_mode="r")

    # 1. Load checkpoints
    gdn = GatedDeltaNetLM().to(device).eval()
    gdn.load_state_dict(torch.load("results/gated_deltanet_d128_3000/BBest.pt")["model"])

    gdn2 = GatedDeltaNet2LM().to(device).eval()
    gdn2.load_state_dict(torch.load("results/gated_deltanet_2_d128_3000/BBest.pt")["model"])

    # -------------------------------------------------------------
    # 2. Mathematical Audit: Operator norm vs Spectral Radius
    # -------------------------------------------------------------
    N_TOK = 2048
    tokens_sub = torch.as_tensor(val_data[:N_TOK].copy(), device=device, dtype=torch.long)[None]

    # GDN-1 layer 0
    h1 = gdn.source.embedding(tokens_sub)
    n1 = gdn.layers[0].norm(h1)
    k1 = F.normalize(gdn.layers[0].k_proj(n1).view(1, N_TOK, 4, 32), dim=-1)
    a1 = torch.sigmoid(gdn.layers[0].alpha_proj(n1))
    b1 = torch.sigmoid(gdn.layers[0].beta_proj(n1))

    # GDN-2 layer 0
    h2 = gdn2.source.embedding(tokens_sub)
    n2 = gdn2.layers[0].norm(h2)
    k2 = F.normalize(gdn2.layers[0].k_proj(n2).view(1, N_TOK, 4, 32), dim=-1)
    D2 = torch.sigmoid(gdn2.layers[0].decay_proj(n2)).view(1, N_TOK, 4, 32)
    b2 = torch.sigmoid(gdn2.layers[0].erase_proj(n2)).view(1, N_TOK, 4, 32)
    w2 = torch.sigmoid(gdn2.layers[0].write_proj(n2)).view(1, N_TOK, 4, 32)

    gdn1_smax, gdn1_rho = [], []
    gdn2_smax, gdn2_rho = [], []

    for t in range(N_TOK):
        for h in range(4):
            # GDN-1
            kt1 = k1[0, t, h]
            A1 = a1[0, t, h] * (torch.eye(32, device=device) - b1[0, t, h] * torch.outer(kt1, kt1))
            gdn1_smax.append(float(torch.linalg.svdvals(A1).max()))
            gdn1_rho.append(float(torch.linalg.eigvals(A1).abs().max()))

            # GDN-2
            kt2 = k2[0, t, h]
            bt2 = b2[0, t, h]
            Dt2 = torch.diag(D2[0, t, h])
            A2 = (torch.eye(32, device=device) - torch.outer(kt2, bt2 * kt2)) @ Dt2
            gdn2_smax.append(float(torch.linalg.svdvals(A2).max()))
            gdn2_rho.append(float(torch.linalg.eigvals(A2).abs().max()))

    # -------------------------------------------------------------
    # 3. Transition Product Norm: || prod_{i=0}^{L-1} A_{t+i} ||
    # -------------------------------------------------------------
    product_lengths = [1, 2, 4, 8, 16, 32, 64]
    prod_norms_gdn1 = {L: [] for L in product_lengths}
    prod_norms_gdn2 = {L: [] for L in product_lengths}

    # Sample 16 starts to keep computation fast (<1s)
    starts = list(range(0, N_TOK - 64, 128))
    for start in starts:
        for h in range(4):
            P1 = torch.eye(32, device=device)
            P2 = torch.eye(32, device=device)
            for step in range(64):
                t = start + step
                kt1 = k1[0, t, h]
                A1 = a1[0, t, h] * (torch.eye(32, device=device) - b1[0, t, h] * torch.outer(kt1, kt1))
                P1 = A1 @ P1

                kt2 = k2[0, t, h]
                bt2 = b2[0, t, h]
                Dt2 = torch.diag(D2[0, t, h])
                A2 = (torch.eye(32, device=device) - torch.outer(kt2, bt2 * kt2)) @ Dt2
                P2 = A2 @ P2

                L = step + 1
                if L in product_lengths:
                    prod_norms_gdn1[L].append(float(torch.linalg.svdvals(P1).max()))
                    prod_norms_gdn2[L].append(float(torch.linalg.svdvals(P2).max()))

    # -------------------------------------------------------------
    # 4. Long Sequence State Frobenius Norm: 2k, 8k, 32k, 100k
    # -------------------------------------------------------------
    lengths = [2048, 8192, 32768, 100000]
    rollout_results = {}

    for total_len in lengths:
        sub = torch.as_tensor(val_data[:total_len].copy(), device=device, dtype=torch.long)
        CHUNK = 128

        # A: GDN un-reset
        s_gdn = gdn.initial_state(1, device=device)
        norms_gdn = []
        for i in range(0, total_len, CHUNK):
            c = sub[i:i+CHUNK][None]
            if c.shape[1] < CHUNK: break
            _, s_gdn, _ = gdn(c, state=s_gdn)
            norms_gdn.append(float(s_gdn.square().sum().sqrt()))

        # B: GDN-2 un-reset
        s_gdn2 = gdn2.initial_state(1, device=device)
        norms_gdn2 = []
        for i in range(0, total_len, CHUNK):
            c = sub[i:i+CHUNK][None]
            if c.shape[1] < CHUNK: break
            _, s_gdn2, _ = gdn2(c, state=s_gdn2)
            norms_gdn2.append(float(s_gdn2.square().sum().sqrt()))

        # C: GDN-2 reset every 2048
        s_gdn2_reset = gdn2.initial_state(1, device=device)
        norms_gdn2_reset = []
        for i in range(0, total_len, CHUNK):
            if i > 0 and i % 2048 == 0:
                s_gdn2_reset = gdn2.initial_state(1, device=device)
            c = sub[i:i+CHUNK][None]
            if c.shape[1] < CHUNK: break
            _, s_gdn2_reset, _ = gdn2(c, state=s_gdn2_reset)
            norms_gdn2_reset.append(float(s_gdn2_reset.square().sum().sqrt()))

        # D: Zero-write rollout (homogenous recurrence S_t = A_t S_{t-1} without new writes)
        # to test if A_t alone explodes an arbitrary initial state S_0
        s_gdn2_nowrite = torch.randn_like(s_gdn2) # start with unit random state
        s_gdn2_nowrite = s_gdn2_nowrite / s_gdn2_nowrite.norm()
        norms_gdn2_nowrite = []
        for i in range(0, min(total_len, 8192), CHUNK):
            c = sub[i:i+CHUNK][None]
            if c.shape[1] < CHUNK: break
            # Compute A_t only
            h = gdn2.source.embedding(c)
            n = gdn2.layers[0].norm(h)
            k = F.normalize(gdn2.layers[0].k_proj(n).view(1, CHUNK, 4, 32), dim=-1)
            D = torch.sigmoid(gdn2.layers[0].decay_proj(n)).view(1, CHUNK, 4, 32)
            b = torch.sigmoid(gdn2.layers[0].erase_proj(n)).view(1, CHUNK, 4, 32)
            S_curr = s_gdn2_nowrite[0, 0] # [4, 32, 32]
            for t in range(CHUNK):
                kt = k[0, t] # [4, 32]
                bt = b[0, t] # [4, 32]
                Dt = D[0, t] # [4, 32]
                S_decayed = Dt.unsqueeze(-1) * S_curr
                retrieved = torch.einsum("hk,hkd->hd", bt * kt, S_decayed)
                erased = torch.einsum("hk,hd->hkd", kt, retrieved)
                S_curr = S_decayed - erased
            s_gdn2_nowrite[0, 0] = S_curr
            norms_gdn2_nowrite.append(float(s_gdn2_nowrite.norm()))

        rollout_results[total_len] = {
            "gdn_final_norm": norms_gdn[-1] if norms_gdn else 0,
            "gdn_max_norm": max(norms_gdn) if norms_gdn else 0,
            "gdn2_final_norm": norms_gdn2[-1] if norms_gdn2 else 0,
            "gdn2_max_norm": max(norms_gdn2) if norms_gdn2 else 0,
            "gdn2_reset_max_norm": max(norms_gdn2_reset) if norms_gdn2_reset else 0,
            "nowrite_decay_ratio": norms_gdn2_nowrite[-1] / (norms_gdn2_nowrite[0] + 1e-8) if norms_gdn2_nowrite else 0,
        }

    report = {
        "operator_properties": {
            "gdn1": {
                "max_sigma_max": float(np.max(gdn1_smax)),
                "mean_sigma_max": float(np.mean(gdn1_smax)),
                "frac_sigma_gt_1": float(np.mean([s > 1.0 for s in gdn1_smax])),
                "max_spectral_radius": float(np.max(gdn1_rho)),
            },
            "gdn2": {
                "max_sigma_max": float(np.max(gdn2_smax)),
                "mean_sigma_max": float(np.mean(gdn2_smax)),
                "frac_sigma_gt_1": float(np.mean([s > 1.0 for s in gdn2_smax])),
                "max_spectral_radius": float(np.max(gdn2_rho)),
            }
        },
        "transition_product_norms": {
            "gdn1": {L: float(np.mean(prod_norms_gdn1[L])) for L in product_lengths},
            "gdn2": {L: float(np.mean(prod_norms_gdn2[L])) for L in product_lengths},
        },
        "rollout_results": rollout_results,
        "gates_summary": {
            "gdn2_decay_mean": float(D2.mean()),
            "gdn2_erase_mean": float(b2.mean()),
            "gdn2_write_mean": float(w2.mean()),
            "gdn1_decay_mean": float(a1.mean()),
            "gdn1_beta_mean": float(b1.mean()),
        }
    }

    with open("results/gdn_stability_audit_final.json", "w") as f:
        json.dump(report, f, indent=2)
    print("AUDIT_COMPLETE")


if __name__ == "__main__":
    run_audit()
