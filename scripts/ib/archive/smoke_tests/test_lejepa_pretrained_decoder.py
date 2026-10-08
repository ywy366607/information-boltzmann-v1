"""Test LeJEPA / LeWM Latent Predictor with Pretrained Mature Decoder.

Objective:
Directly evaluate whether LeJEPA latent predictive coding can bridge the 8-13 tick
conduction delay in the fruit fly brain when paired with the mature, pretrained
768-dim language decoder (avoiding cold-probe overfitting).
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
from scripts.ib.run_full_diagnosis import load_checkpoint, clone_state


class LatentWorldModel(nn.Module):
    """Action-conditioned Latent Predictor with SIGReg on 768-dim brain features."""
    def __init__(self, d_model: int = 768, d_emb: int = 128, lambda_sigreg: float = 0.5):
        super().__init__()
        self.d_model = d_model
        self.d_emb = d_emb
        self.lambda_sigreg = lambda_sigreg

        # Predictor: maps (z_t, e(x_t)) -> delta_z
        self.predictor = nn.Sequential(
            nn.Linear(d_model + d_emb, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.sigreg = SIGReg(dim=d_model, num_slices=128)

    def predict_next(self, z_curr: torch.Tensor, token_emb: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([z_curr, token_emb], dim=-1)
        return z_curr + self.predictor(inp)

    def loss(self, z_pred: torch.Tensor, z_target: torch.Tensor, z_all: torch.Tensor):
        l_mse = F.mse_loss(z_pred, z_target)
        l_sig = 0.5 * (self.sigreg(z_all) + self.sigreg(z_pred))
        total = l_mse + self.lambda_sigreg * l_sig
        return total, l_mse.item(), l_sig.item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("artifacts/active_training/q8_fly_bptt32_gamma_2333_wide115_100k/last.pt"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--train-tokens", type=int, default=512)
    parser.add_argument("--val-tokens", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
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

    total_tokens = args.train_tokens + args.val_tokens
    tokens = torch.as_tensor(data[cursor:cursor + total_tokens], device=args.device, dtype=torch.long)
    input_tokens = torch.cat([tokens.new_tensor([prev_tok]), tokens[:-1]])

    print(f"\nStepping brain connectome across {total_tokens} continuous OWT tokens...", flush=True)
    feats_list = []
    emb_list = []
    curr_state = clone_state(s_init)

    with torch.no_grad():
        for t in range(total_tokens):
            tok = input_tokens[t:t+1]
            emb = model.embedding(tok)
            emb_list.append(emb)
            curr_state = advance_fly_input_event(
                model, curr_state, tok, settle_ticks=0, writer_baseline_clock="input", **options
            )
            read_in = (curr_state.h - curr_state.h_mean
                       if getattr(model, 'read_centering', False)
                       and curr_state.h_mean.numel() == curr_state.h.numel() else curr_state.h)
            motor = read_in[:, model.read_indices]
            feat = model.read_norm(model.output_read(motor))
            feats_list.append(feat)

    Feats = torch.cat(feats_list, dim=0) # [T, 768]
    Embs = torch.cat(emb_list, dim=0)    # [T, 128]

    # Split into train and held-out validation
    Feats_tr, Feats_val = Feats[:args.train_tokens], Feats[args.train_tokens:]
    Embs_tr, Embs_val = Embs[:args.train_tokens], Embs[args.train_tokens:]
    Y_tr, Y_val = tokens[:args.train_tokens], tokens[args.train_tokens:]

    # 1. Baseline Evaluations with mature pretrained decoder
    with torch.no_grad():
        # Unigram baseline (bias only)
        bias_logits = model.decoder.bias.unsqueeze(0)
        unigram_tr = F.cross_entropy(bias_logits.expand(args.train_tokens, -1), Y_tr).item()
        unigram_val = F.cross_entropy(bias_logits.expand(args.val_tokens, -1), Y_val).item()

        # Direct Delayed Physical Motor Reading into Decoder (no predictor)
        delayed_logits_tr = model.decoder(Feats_tr)
        delayed_logits_val = model.decoder(Feats_val)
        delayed_nll_tr = F.cross_entropy(delayed_logits_tr, Y_tr).item()
        delayed_nll_val = F.cross_entropy(delayed_logits_val, Y_val).item()

    print("\n" + "=" * 80)
    print("BASELINE COMPARISON (Using 1.5M Pretrained Decoder)")
    print("=" * 80)
    print(f"Train Split: Unigram NLL = {unigram_tr:.4f} | Delayed Brain Read = {delayed_nll_tr:.4f} ({delayed_nll_tr - unigram_tr:+.4f})")
    print(f"Val Split:   Unigram NLL = {unigram_val:.4f} | Delayed Brain Read = {delayed_nll_val:.4f} ({delayed_nll_val - unigram_val:+.4f})")
    print("Note: Delayed Brain Read is worse than unigram because brain states lag by 8-13 ticks!")

    # 2. Train LeJEPA Latent Predictor with SIGReg on Feats_tr
    wm = LatentWorldModel(
        d_model=model.embedding.embedding_dim if hasattr(model, 'd_model') else 768,
        d_emb=model.embedding.embedding_dim,
        lambda_sigreg=0.2,
    ).to(args.device)

    opt = torch.optim.AdamW(wm.parameters(), lr=5e-4, weight_decay=1e-3)

    print("\n" + "=" * 80)
    print("TRAINING LEJEPA LATENT WORLD MODEL ON 768-DIM LANGUAGE FEATURES")
    print("=" * 80)
    print(f"{'Epoch':<6} | {'Total Loss':<12} | {'Latent MSE':<12} | {'SIGReg Loss':<13} | {'Val Pred NLL':<14} | {'Val vs Delayed':<15}")
    print("-" * 80)

    for ep in range(1, args.epochs + 1):
        opt.zero_grad()
        z_curr = Feats_tr[:-1]
        z_target = Feats_tr[1:]
        e_curr = Embs_tr[:-1]

        z_pred = wm.predict_next(z_curr, e_curr)
        loss, mse_val, sig_val = wm.loss(z_pred, z_target, Feats_tr)
        loss.backward()
        opt.step()

        if ep in (1, 2, 5, 10, 20, 30, 40, 50):
            with torch.no_grad():
                z_val_pred = wm.predict_next(Feats_val[:-1], Embs_val[:-1])
                pred_logits_val = model.decoder(z_val_pred)
                pred_nll_val = F.cross_entropy(pred_logits_val, Y_val[1:]).item()
                gap_delayed = pred_nll_val - delayed_nll_val
            print(f"Ep {ep:<3d} | {loss.item():<12.4f} | {mse_val:<12.6f} | {sig_val:<13.5f} | {pred_nll_val:<14.4f} | {gap_delayed:<+15.4f}")

    # 3. Multi-Step Latent Trajectory Rollout Audit (Horizons k = 1..16)
    print("\n" + "=" * 80)
    print("MULTI-HORIZON LATENT ROLLOUT AUDIT ON HELD-OUT STREAM (Using Pretrained Decoder)")
    print("=" * 80)
    print(f"{'Horizon k':<10} | {'Latent Cos Sim':<16} | {'Latent MSE':<12} | {'Val Rollout NLL':<16} | {'vs Delayed':<12} | {'vs Unigram':<12}")
    print("-" * 80)

    horizons = [1, 2, 3, 4, 6, 8, 10, 12, 14, 16]
    with torch.no_grad():
        for k in horizons:
            if k >= args.val_tokens:
                break
            z_roll = Feats_val[:-k].clone()
            for step in range(k):
                z_roll = wm.predict_next(z_roll, Embs_val[step:args.val_tokens - k + step])

            z_actual = Feats_val[k:]
            cos_sim = F.cosine_similarity(z_roll, z_actual, dim=-1).mean().item()
            mse_k = F.mse_loss(z_roll, z_actual).item()

            roll_logits = model.decoder(z_roll)
            target_tokens = Y_val[k:]
            roll_nll = F.cross_entropy(roll_logits, target_tokens).item()

            # Compare against delayed baseline on the same slice
            slice_delayed_logits = model.decoder(Feats_val[k:])
            slice_delayed_nll = F.cross_entropy(slice_delayed_logits, target_tokens).item()

            slice_bias = model.decoder.bias.unsqueeze(0).expand(args.val_tokens - k, -1)
            slice_unigram_nll = F.cross_entropy(slice_bias, target_tokens).item()

            gap_delayed = roll_nll - slice_delayed_nll
            gap_unigram = roll_nll - slice_unigram_nll

            print(f"k = {k:<6d} | {cos_sim:<16.4f} | {mse_k:<12.6f} | {roll_nll:<16.4f} | {gap_delayed:<+12.4f} | {gap_unigram:<+12.4f}")

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)


if __name__ == "__main__":
    main()
