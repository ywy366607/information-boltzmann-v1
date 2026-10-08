"""Rigorous Self-Supervised Audit: LeJEPA / LeWM Brain World Model.

Protocol:
1. Self-Supervised Learning: Encoder and Predictor trained strictly with
   L_WM = MSE(ẑ_{t+1}, z_{t+1}) + lambda * SIGReg(Z).
   NO token supervision or cross-entropy during world model training.
2. Anti-Collapse Verification: Exact SIGReg isotropic Gaussianity check
   (mean, std, rank, condition number).
3. Linear Probe Protocol: Linear readout trained on Z.detach() on train split,
   evaluated prequentially on held-out validation split.
4. Multi-Horizon Rollout Audit: Measure latent cosine sim and validation NLL
   across physical delay horizons k = 1..16.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ["PYTORCH_ALLOC_CONF"] = "max_split_size_mb:64"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from information_boltzmann.core.fly_bptt_learning import (
    FlyPhysicalState,
    advance_fly_input_event,
)
from information_boltzmann.core.sigreg import SIGReg
from information_boltzmann.core.lejepa_world_model import LeJEPAWorldModel
from scripts.ib.run_full_diagnosis import load_checkpoint, clone_state


def extract_stream_trajectories(
    model,
    state: FlyPhysicalState,
    data: np.ndarray,
    start_cursor: int,
    num_tokens: int,
    prev_tok: int,
    options: dict,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, FlyPhysicalState]:
    """Runs the biological brain through a continuous token stream and records states."""
    tokens = torch.as_tensor(data[start_cursor:start_cursor + num_tokens], device=device, dtype=torch.long)
    input_tokens = torch.cat([tokens.new_tensor([prev_tok]), tokens[:-1]])

    motor_list = []
    emb_list = []
    curr_state = clone_state(state)

    with torch.no_grad():
        for t in range(num_tokens):
            tok = input_tokens[t:t+1]
            emb = model.embedding(tok)
            emb_list.append(emb)
            curr_state = advance_fly_input_event(
                model, curr_state, tok, settle_ticks=0, writer_baseline_clock="input", **options
            )
            m = (curr_state.h - curr_state.h_mean)[:, model.read_indices] if model.read_centering else curr_state.h[:, model.read_indices]
            motor_list.append(m)

    H_motor = torch.cat(motor_list, dim=0) # [T, 2333]
    E_tokens = torch.cat(emb_list, dim=0)   # [T, 128]
    return H_motor, E_tokens, tokens, curr_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("artifacts/active_training/q8_fly_bptt32_gamma_2333_wide115_100k/last.pt"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--train-tokens", type=int, default=512)
    parser.add_argument("--val-tokens", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--probe-epochs", type=int, default=30)
    parser.add_argument("--lambda-sigreg", type=float, default=1.0)
    args = parser.parse_args()

    print(f"Loading checkpoint from {args.checkpoint} on {args.device}...", flush=True)
    saved, learner = load_checkpoint(args.checkpoint, device=args.device)
    model = learner.model
    s_init = clone_state(learner.state)

    data = np.load(ROOT / saved["config"]["data"] / "train.npy", mmap_mode="r")
    cursor = int(saved["train_cursor"])
    prev_tok = int(saved["learner"]["previous_token"])

    rates, thresholds = model.get_decay_rates(), model.get_thresholds()
    gains, alif, stp = model.get_conductance_gains(), model.get_alif_params(), model.get_stp_params()
    options = dict(base_rates=rates, thresholds=thresholds, conductance_gains=gains, alif_params=alif, stp_params=stp)

    # 1. Continuous stream extraction: Train split followed immediately by Validation split
    print(f"\nExtracting train trajectory ({args.train_tokens} tokens)...", flush=True)
    H_train, E_train, Y_train, s_val = extract_stream_trajectories(
        model, s_init, data, cursor, args.train_tokens, prev_tok, options, args.device
    )

    print(f"Extracting held-out validation trajectory ({args.val_tokens} tokens)...", flush=True)
    H_val, E_val, Y_val, _ = extract_stream_trajectories(
        model, s_val, data, cursor + args.train_tokens, args.val_tokens, int(Y_train[-1].item()), options, args.device
    )

    # Static Unigram Baselines
    bias_train = model.decoder.bias.unsqueeze(0).expand(args.train_tokens, -1)
    bias_val = model.decoder.bias.unsqueeze(0).expand(args.val_tokens, -1)
    unigram_train_nll = F.cross_entropy(bias_train, Y_train).item()
    unigram_val_nll = F.cross_entropy(bias_val, Y_val).item()
    print(f"\nUnigram Baselines: Train NLL = {unigram_train_nll:.4f} nats | Val NLL = {unigram_val_nll:.4f} nats.")

    # 2. Instantiate Professional LeJEPA World Model
    d_latent = 128
    d_emb = model.embedding.embedding_dim
    n_motor = len(model.read_indices)

    wm = LeJEPAWorldModel(
        n_motor=n_motor,
        d_latent=d_latent,
        d_emb=d_emb,
        vocab_size=model.decoder.out_features,
        lambda_sigreg=args.lambda_sigreg,
        num_slices=128,
    ).to(args.device)

    # Phase 1: Pure Self-Supervised World Model Training
    # Only optimizes Encoder E_theta and Predictor P_phi via MSE + SIGReg. Probe is NOT involved.
    wm_params = list(wm.encoder.parameters()) + list(wm.predictor.parameters())
    opt_wm = torch.optim.AdamW(wm_params, lr=1e-3, weight_decay=1e-4)

    print("\n" + "=" * 80)
    print("PHASE 1: PURE SELF-SUPERVISED TRAINING (MSE + SIGReg, Zero Token Supervision)")
    print("=" * 80)
    print(f"{'Epoch':<6} | {'Total WM Loss':<14} | {'Latent MSE':<12} | {'SIGReg Loss':<13} | {'Latent Mean':<12} | {'Latent Std':<10}")
    print("-" * 80)

    t0 = time.perf_counter()
    for ep in range(1, args.epochs + 1):
        opt_wm.zero_grad()
        Z = wm.encode(H_train)
        z_curr = Z[:-1]
        z_target = Z[1:]
        e_curr = E_train[:-1]

        z_pred = wm.predict_next(z_curr, e_curr)
        loss_wm, metrics = wm.compute_jepa_loss(z_pred, z_target, Z)
        loss_wm.backward()
        opt_wm.step()

        if ep in (1, 2, 5, 10, 20, 30, 40, 50):
            with torch.no_grad():
                l_mean = Z.mean().item()
                l_std = Z.std(dim=0).mean().item()
            print(f"Ep {ep:<3d} | {metrics['loss_total']:<14.4f} | {metrics['loss_pred_mse']:<12.6f} | {metrics['loss_sigreg']:<13.5f} | {l_mean:<12.4f} | {l_std:<10.4f}")

    t_wm = time.perf_counter() - t0
    print(f"\nSelf-supervised training finished in {t_wm:.2f}s ({t_wm / args.epochs * 1000:.1f} ms/epoch).")

    # Anti-collapse and Representation Geometry Audit on Train & Val
    with torch.no_grad():
        Z_train = wm.encode(H_train)
        Z_val = wm.encode(H_val)

        # Dimension-wise variance
        train_stds = Z_train.std(dim=0)
        val_stds = Z_val.std(dim=0)
        dead_dims_train = (train_stds < 0.1).sum().item()
        dead_dims_val = (val_stds < 0.1).sum().item()

        # Effective rank via singular values
        s_vals = torch.linalg.svdvals(Z_val - Z_val.mean(dim=0))
        p_vals = s_vals / s_vals.sum()
        entropy = -(p_vals * torch.log(p_vals + 1e-12)).sum()
        effective_rank = torch.exp(entropy).item()

    print("\n" + "=" * 80)
    print("REPRESENTATION GEOMETRY AUDIT (Validation Split)")
    print("=" * 80)
    print(f"Latent Dimensionality:       {d_latent}")
    print(f"Mean Latent Std:             {val_stds.mean().item():.4f} (target: ~1.0 for isotropic normal)")
    print(f"Dead Dimensions (std < 0.1): {dead_dims_val} / {d_latent}")
    print(f"Effective Rank:              {effective_rank:.2f} / {d_latent}")

    # Phase 2: Downstream Linear Probe Evaluation (Standard SSL Protocol)
    # Train linear probe on Z_train.detach() -> Y_train, test on Z_val.detach() -> Y_val
    print("\n" + "=" * 80)
    print("PHASE 2: DOWNSTREAM LINEAR PROBING (Trained on Z.detach(), Evaluated on Fresh Stream)")
    print("=" * 80)
    print(f"{'Epoch':<6} | {'Train Probe NLL':<16} | {'Val Probe NLL':<14} | {'Val vs Unigram':<15}")
    print("-" * 80)

    probe = nn.Linear(d_latent, model.decoder.out_features).to(args.device)
    # Initialize probe bias to unigram distribution to start on equal footing
    probe.bias.data.copy_(model.decoder.bias.data)
    probe.weight.data.zero_()

    opt_probe = torch.optim.AdamW(probe.parameters(), lr=1e-2, weight_decay=1e-4)
    Z_tr_det = Z_train.detach()
    Z_val_det = Z_val.detach()

    for ep in range(1, args.probe_epochs + 1):
        opt_probe.zero_grad()
        logits_tr = probe(Z_tr_det)
        loss_tr = F.cross_entropy(logits_tr, Y_train)
        loss_tr.backward()
        opt_probe.step()

        if ep in (1, 2, 5, 10, 15, 20, 25, 30):
            with torch.no_grad():
                logits_val = probe(Z_val_det)
                loss_val = F.cross_entropy(logits_val, Y_val).item()
                gap_val = loss_val - unigram_val_nll
            print(f"Ep {ep:<3d} | {loss_tr.item():<16.4f} | {loss_val:<14.4f} | {gap_val:<+15.4f}")

    # Phase 3: Multi-Step Latent Trajectory Rollout Audit (Horizons k = 1..16)
    print("\n" + "=" * 80)
    print("PHASE 3: MULTI-HORIZON LATENT TRAJECTORY ROLLOUT ON HELD-OUT VALIDATION STREAM")
    print("=" * 80)
    print(f"{'Horizon k':<10} | {'Latent Cos Sim':<16} | {'Latent MSE':<12} | {'Val Rollout NLL':<16} | {'vs Unigram':<12}")
    print("-" * 80)

    horizons = [1, 2, 3, 4, 6, 8, 10, 12, 14, 16]
    with torch.no_grad():
        for k in horizons:
            if k >= args.val_tokens:
                break
            # Roll out from z_val[t] for k steps using actual token actions
            z_roll = Z_val[:-k].clone()
            for step in range(k):
                z_roll = wm.predict_next(z_roll, E_val[step:args.val_tokens - k + step])

            z_target_k = Z_val[k:]
            cos_sim = F.cosine_similarity(z_roll, z_target_k, dim=-1).mean().item()
            mse_val = F.mse_loss(z_roll, z_target_k).item()

            # Predict tokens from rolled-out latent state
            rollout_logits = probe(z_roll)
            target_tokens_k = Y_val[k:]
            rollout_nll = F.cross_entropy(rollout_logits, target_tokens_k).item()
            bias_k = model.decoder.bias.unsqueeze(0).expand(args.val_tokens - k, -1)
            unigram_k = F.cross_entropy(bias_k, target_tokens_k).item()
            gap_k = rollout_nll - unigram_k

            print(f"k = {k:<6d} | {cos_sim:<16.4f} | {mse_val:<12.6f} | {rollout_nll:<16.4f} | {gap_k:<+12.4f}")

    # Save curated results
    out_dict = {
        "config": {
            "train_tokens": args.train_tokens,
            "val_tokens": args.val_tokens,
            "epochs": args.epochs,
            "lambda_sigreg": args.lambda_sigreg,
            "d_latent": d_latent,
        },
        "baselines": {
            "unigram_train_nll": unigram_train_nll,
            "unigram_val_nll": unigram_val_nll,
        },
        "representation": {
            "mean_val_std": float(val_stds.mean().item()),
            "dead_dims": dead_dims_val,
            "effective_rank": float(effective_rank),
        },
        "downstream_probe": {
            "final_train_nll": float(loss_tr.item()),
            "final_val_nll": float(loss_val),
            "val_gap_vs_unigram": float(gap_val),
        },
    }
    out_path = ROOT / "results" / "published" / "fly_lejepa_audit_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_dict, f, indent=2)
    print(f"\nAudit results saved to {out_path}.")


if __name__ == "__main__":
    main()
