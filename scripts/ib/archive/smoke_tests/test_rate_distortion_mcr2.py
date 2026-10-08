"""Empirical GPU test of Rate-Distortion Theory and MCR2 on the MaleCNS fruit fly connectome:
1. Rate-Distortion Curve: R(D) where Rate R = 1/tau (throughput) and Distortion D is representation/NLL error.
2. MCR^2 (Maximal Coding Rate Reduction): Does coding rate volume detect subspace aliasing vs orthogonal packing?
"""

import gc
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from information_boltzmann.core.fly_reservoir import FlyReservoirLM
from information_boltzmann.core.fly_bptt_learning import (
    FlyPhysicalState,
    extract_fly_motor_latent,
    step_fly_physical_tick,
)


def compute_coding_rate(Z, eps=0.5):
    """Computes Gaussian coding rate R(Z) = 0.5 * log det(I + (d / (m * eps^2)) * Z Z^T).
    Z is of shape [d, m] where d is feature dim, m is number of samples.
    """
    d, m = Z.shape
    alpha = d / (m * (eps ** 2))
    # ZZ^T is [d, d]. If m < d, use Sylvester's identity: det(I + alpha Z Z^T) = det(I + alpha Z^T Z)
    if m < d:
        gram = torch.eye(m, device=Z.device) + alpha * (Z.t() @ Z)
    else:
        gram = torch.eye(d, device=Z.device) + alpha * (Z @ Z.t())
    slogdet = torch.slogdet(gram)
    return 0.5 * slogdet.logabsdet.item()


def load_model_and_checkpoint(device):
    graph_path = Path("data/malecns_v1/fly_reservoir_coba.npz")
    ckpt_path = Path("E:/ib_checkpoints/q8_fly_ctm_settle14_100k/last.pt")

    model = FlyReservoirLM(
        str(graph_path),
        vocab_size=50257,
        d_model=768,
        injection='topographic',
        read_surface='output',
        synapse_model='coba',
        use_alif=True,
        use_stp=True,
    ).to(device)

    saved = torch.load(ckpt_path, map_location='cpu', weights_only=False, mmap=True)
    weights = {k: v for k, v in saved['model'].items() if k not in ('edge_weight_e', 'edge_weight_i')}
    with torch.no_grad():
        for name in ('edge_weight_e', 'edge_weight_i'):
            getattr(model, name).copy_(saved['model'][name])
        model.load_state_dict(weights, strict=False)
    model.eval()

    phys = saved['learner']['physical']
    state_dict = {
        k: v.to(device) if isinstance(v, torch.Tensor)
        else tuple(t.to(device) for t in v) if isinstance(v, tuple)
        else v
        for k, v in phys.items()
    }
    initial_state = FlyPhysicalState(**state_dict)

    del saved, weights, phys
    gc.collect()
    torch.cuda.empty_cache()

    rates = model.get_decay_rates()
    thresholds = model.get_thresholds()
    gains = model.get_conductance_gains()
    alif = model.get_alif_params()
    stp = model.get_stp_params()
    options = dict(
        base_rates=rates,
        thresholds=thresholds,
        conductance_gains=gains,
        alif_params=alif,
        stp_params=stp,
    )

    return model, initial_state, options


def evaluate_rate_distortion_curve(model, initial_state, options, device, num_tokens=20):
    val_data = np.load("data/ib_owt_gpt2/validation.npy", mmap_mode='r')
    tokens = val_data[:num_tokens + 1].astype(np.int64)
    quiet_source = torch.zeros_like(initial_state.h)

    # Test settling intervals: tau in [1, 2, 3, 4, 6, 8, 10, 14]
    tau_list = [1, 2, 3, 4, 6, 8, 10, 14]
    rd_results = []

    # First compute reference (clean isolated tau=14) latents
    print("Computing clean reference latents (tau=14)...")
    clean_latents = []
    state = initial_state.detached()
    with torch.no_grad():
        for i in range(num_tokens):
            inp = torch.tensor([tokens[i]], dtype=torch.long, device=device)
            drv, bsl = model.topographic_writer.forward_with_state(
                model.embedding(inp), state.h, state.baseline)
            for t in range(14):
                src = drv if t == 0 else quiet_source
                base = bsl if t == 0 else state.baseline
                state = step_fly_physical_tick(model, state, inp if t==0 else None, src, base, options)
            lat = extract_fly_motor_latent(model, state)
            clean_latents.append(lat.squeeze(0))
    clean_Z = torch.stack(clean_latents, dim=1) # [d, num_tokens]
    clean_rate = compute_coding_rate(clean_Z)
    print(f"Clean Z shape: {clean_Z.shape}, Coding Rate R(Z_clean) = {clean_rate:.3f} nats")

    for tau in tau_list:
        state = initial_state.detached()
        tau_latents = []
        nlls = []
        accs = []
        with torch.no_grad():
            for i in range(num_tokens):
                inp = torch.tensor([tokens[i]], dtype=torch.long, device=device)
                target = torch.tensor([tokens[i+1]], dtype=torch.long, device=device)

                drv, bsl = model.topographic_writer.forward_with_state(
                    model.embedding(inp), state.h, state.baseline)
                for t in range(tau):
                    src = drv if t == 0 else quiet_source
                    base = bsl if t == 0 else state.baseline
                    state = step_fly_physical_tick(model, state, inp if t==0 else None, src, base, options)

                lat = extract_fly_motor_latent(model, state)
                logits = model.decoder(lat)
                loss = torch.nn.functional.cross_entropy(logits, target).item()
                pred = logits.argmax(dim=-1).item()

                tau_latents.append(lat.squeeze(0))
                nlls.append(loss)
                accs.append(int(pred == target.item()))

        Z_tau = torch.stack(tau_latents, dim=1) # [d, num_tokens]
        coding_rate_R = compute_coding_rate(Z_tau)

        # Compute distortion D
        # 1. MSE Distortion in representation space: ||Z_tau - Z_clean||^2 / ||Z_clean||^2
        dist_rep = ((Z_tau - clean_Z) ** 2).sum() / (clean_Z ** 2).sum()
        dist_rep = dist_rep.item()

        # 2. Rate: bits/tokens per tick = 1 / tau
        channel_rate = 1.0 / tau
        mean_nll = float(np.mean(nlls))
        acc = float(np.mean(accs))

        # 3. Subspace dimensionality / volume ratio
        vol_ratio = coding_rate_R / clean_rate

        res = {
            "tau": tau,
            "rate_tokens_per_tick": channel_rate,
            "mean_nll": mean_nll,
            "acc": acc,
            "dist_rep_mse": dist_rep,
            "coding_rate_R": coding_rate_R,
            "coding_rate_ratio": vol_ratio,
        }
        rd_results.append(res)
        print(f"tau={tau:2d} | Rate={channel_rate:5.3f} tok/tick | NLL={mean_nll:5.2f} | Acc={acc*100:4.1f}% | Dist_MSE={dist_rep:5.3f} | CodingRate={coding_rate_R:6.2f} (Ratio={vol_ratio*100:5.1f}%)")

    # Plot Rate-Distortion & MCR2 figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    plt.style.use('dark_background')

    # Panel 1: Classical Rate-Distortion Curve R(D)
    ax1 = axes[0]
    rates = [r['rate_tokens_per_tick'] for r in rd_results]
    dists = [r['dist_rep_mse'] for r in rd_results]
    ax1.plot(dists, rates, 'o-', color='#38bdf8', linewidth=2.5, markersize=8)
    for r in rd_results:
        ax1.annotate(f"tau={r['tau']}", (r['dist_rep_mse'], r['rate_tokens_per_tick']),
                     textcoords="offset points", xytext=(8, -4), color='#f1f5f9', fontsize=9)
    ax1.set_title("1. Rate-Distortion Curve R(D)\n(Transmission Rate vs Representation Distortion)", fontsize=11, fontweight='bold', color='#f1f5f9')
    ax1.set_xlabel("Representation Distortion D (MSE vs S=14)", fontsize=10, color='#94a3b8')
    ax1.set_ylabel("Rate R (Tokens per Physical Tick)", fontsize=10, color='#94a3b8')
    ax1.grid(True, alpha=0.15)

    # Panel 2: Coding Rate Volume vs Settling Interval (MCR2 expansion)
    ax2 = axes[1]
    taus = [r['tau'] for r in rd_results]
    c_rates = [r['coding_rate_R'] for r in rd_results]
    ax2.plot(taus, c_rates, 's-', color='#10b981', linewidth=2.5, markersize=8)
    ax2.axhline(clean_rate, color='#ef4444', linestyle='--', label=f"Uncompressed Volume ({clean_rate:.2f})")
    ax2.set_title("2. Representation Space Volume R(Z)\n(MCR2 Subspace Capacity Expansion)", fontsize=11, fontweight='bold', color='#f1f5f9')
    ax2.set_xlabel("Settling Interval tau (ticks)", fontsize=10, color='#94a3b8')
    ax2.set_ylabel("Gaussian Coding Rate R(Z) [nats]", fontsize=10, color='#94a3b8')
    ax2.legend(framealpha=0.3, fontsize=9)
    ax2.grid(True, alpha=0.15)

    # Panel 3: Distortion vs NLL Loss
    ax3 = axes[2]
    nlls = [r['mean_nll'] for r in rd_results]
    ax3.plot(taus, nlls, '^-', color='#f59e0b', linewidth=2.5, markersize=8)
    ax3.set_title("3. Predictive Loss vs Settling Interval tau", fontsize=11, fontweight='bold', color='#f1f5f9')
    ax3.set_xlabel("Settling Interval tau (ticks)", fontsize=10, color='#94a3b8')
    ax3.set_ylabel("Validation NLL Loss", fontsize=10, color='#94a3b8')
    ax3.grid(True, alpha=0.15)

    plt.tight_layout()
    out_png = Path("present/rate_distortion_mcr2_verification.png")
    plt.savefig(out_png, dpi=200)
    plt.close()

    out_json = Path("present/rate_distortion_mcr2_report.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"rd_results": rd_results, "clean_rate": clean_rate}, f, indent=2)

    print(f"\nSaved R(D) report to {out_json}")
    print(f"Saved R(D) figure to {out_png}")


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, initial_state, options = load_model_and_checkpoint(device)
    evaluate_rate_distortion_curve(model, initial_state, options, device, num_tokens=20)
