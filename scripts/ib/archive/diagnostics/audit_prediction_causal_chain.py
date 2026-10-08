"""Causal Chain Diagnostic Audit for Fly Reservoir Language Prediction.

Inspects the 3-step causal chain:
1. WRITER: Does topographic writer inject distinguishable content differences?
2. CONNECTOME: Do differences propagate to the 2,333 motor neurons and preserve history?
3. READOUT & DECODER: Does Gamma + Decoder extract these differences into conditional predictions?
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from information_boltzmann.core.fly_reservoir import FlyReservoirLM
from information_boltzmann.core.fly_bptt_learning import (
    FlyBPTTLearner,
    FlyPhysicalState,
    advance_fly_input_event,
)


def load_model_and_state(checkpoint_path: Path, device: str = "cuda"):
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg, old = saved["config"], saved["learner"]
    model = FlyReservoirLM(
        ROOT / cfg["graph"],
        vocab_size=50257,
        d_model=cfg["d_model"],
        injection="topographic",
        read_surface="output",
        synapse_model="coba",
        use_alif=True,
        use_stp=True,
        decoder_bias=cfg["decoder_bias"],
        read_centering=old.get("read_centering", cfg.get("read_centering", False)),
        use_read_gamma_trace=old.get("use_read_gamma_trace", cfg.get("use_read_gamma_trace", True)),
    ).to(device)

    with torch.no_grad():
        for name in ("edge_weight_e", "edge_weight_i"):
            if name in saved["model"]:
                getattr(model, name).copy_(saved["model"][name].to(device))
        model.load_state_dict(
            {k: v.to(device) for k, v in saved["model"].items() if k not in ("edge_weight_e", "edge_weight_i")},
            strict=False,
        )
    model.eval()

    physical = FlyPhysicalState(**{
        k: tuple(t.to(device) for t in v) if k == "ring" else v.to(device)
        for k, v in old["physical"].items()
    })
    return model, physical, cfg, old, saved


@torch.no_grad()
def audit_writer(model: FlyReservoirLM, sample_tokens: list[int]):
    """1. WRITER AUDIT: Does topographic writer create distinguishable drive?"""
    print("\n" + "=" * 60)
    print("STEP 1: TOPOGRAPHIC WRITER INPUT DIFFERENTIATION AUDIT")
    print("=" * 60)

    device = next(model.parameters()).device
    embs = model.embedding(torch.tensor(sample_tokens, device=device))
    n_tokens = len(sample_tokens)

    # 1.1 Embedding similarity
    embs_norm = F.normalize(embs, dim=-1)
    cos_sim = (embs_norm @ embs_norm.T).detach().cpu().numpy()
    off_diag = cos_sim[~np.eye(n_tokens, dtype=bool)]
    print(f"Embedding Cosine Similarity: mean = {off_diag.mean():.4f}, std = {off_diag.std():.4f}, min = {off_diag.min():.4f}, max = {off_diag.max():.4f}")

    # 1.2 Writer gates
    writer = model.topographic_writer
    gates = F.softmax(writer.gate_linear(embs), dim=-1).detach().cpu().numpy()
    print(f"Modality Routing Gates [Vis, Chemo, Mech]:")
    print(f"  Visual mean = {gates[:, 0].mean():.3f} (std = {gates[:, 0].std():.3f})")
    print(f"  Chemo  mean = {gates[:, 1].mean():.3f} (std = {gates[:, 1].std():.3f})")
    print(f"  Mech   mean = {gates[:, 2].mean():.3f} (std = {gates[:, 2].std():.3f})")

    # 1.3 Drive vector differences
    h_dummy = torch.zeros(1, model.n_neurons, device=device)
    drives = []
    with torch.no_grad():
        for i in range(n_tokens):
            t_emb = embs[i:i+1]
            drv, _ = writer.forward_with_state(t_emb, h_dummy, writer.a_adapt)
            drives.append(drv)
    drives = torch.cat(drives, dim=0)  # [n_tokens, N]

    drive_norms = drives.norm(dim=-1).detach().cpu().numpy()
    diffs = []
    rel_diffs = []
    for i in range(n_tokens):
        for j in range(i + 1, n_tokens):
            diff = (drives[i] - drives[j]).norm().item()
            diffs.append(diff)
            rel_diffs.append(diff / max(drives[i].norm().item(), 1e-6))
    diffs = np.array(diffs)
    rel_diffs = np.array(rel_diffs)
    print(f"Sensory Drive Norm: mean = {drive_norms.mean():.4f}, std = {drive_norms.std():.4f}")
    print(f"Pairwise ||Drive(i) - Drive(j)||: mean = {diffs.mean():.4f}")
    print(f"Relative Difference ||Delta Drive|| / ||Drive||: mean = {rel_diffs.mean():.4f} (range [{rel_diffs.min():.4f}, {rel_diffs.max():.4f}])")

    # Check non-zero drive sites
    active_sites = (drives.abs() > 1e-4).float().sum(dim=-1).detach().cpu().numpy()
    print(f"Active Sensory Neurons hit per token: mean = {active_sites.mean():.1f} / {writer.n_total}")
    return {
        "emb_cos_sim_mean": float(off_diag.mean()),
        "drive_norm_mean": float(drive_norms.mean()),
        "rel_drive_diff_mean": float(rel_diffs.mean()),
    }


@torch.no_grad()
def audit_connectome(model: FlyReservoirLM, base_state: FlyPhysicalState, sample_tokens: list[int]):
    """2. CONNECTOME AUDIT: Do differences reach motor neurons across time?"""
    print("\n" + "=" * 60)
    print("STEP 2: CONNECTOME PROPAGATION & MOTOR READOUT REACHABILITY AUDIT")
    print("=" * 60)

    # 2.1 Measure graph topological distance (shortest paths) from sensory to motor
    # Graph structure
    graph_path = ROOT / "data/malecns_v1/fly_reservoir_coba.npz"
    packed = np.load(graph_path, allow_pickle=False)
    n_neurons = int(packed["neuron_body_ids"].shape[0])
    pre = packed["edge_pre"].astype(np.int64)
    post = packed["edge_post"].astype(np.int64)
    delays = packed["edge_delay"].astype(np.int32) if "edge_delay" in packed else np.ones_like(pre, dtype=np.int32)

    sens_idx = set(model.topographic_writer.injection_index.cpu().numpy().tolist())
    motor_idx = set(model.read_indices.cpu().numpy().tolist())

    # Check 1-hop connections from sensory to motor
    is_sens_pre = np.isin(pre, list(sens_idx))
    is_motor_post = np.isin(post, list(motor_idx))
    direct_edges = np.sum(is_sens_pre & is_motor_post)
    print(f"Direct 1-hop Sensory -> Motor Synapses: {direct_edges} / {len(pre)} ({direct_edges/len(pre):.4%})")

    device = next(model.parameters()).device
    # 2.2 Dynamic impulse response: Token w_A vs w_B
    # Run 16 ticks from the same physical state with different inputs at t=0
    token_a = torch.tensor([sample_tokens[0]], device=device)
    token_b = torch.tensor([sample_tokens[1]], device=device)

    with torch.no_grad():
        # Advance 1 step with token_a vs token_b
        state_a = advance_fly_input_event(model, base_state, token_a, settle_ticks=0, writer_baseline_clock="input")
        state_b = advance_fly_input_event(model, base_state, token_b, settle_ticks=0, writer_baseline_clock="input")

        read_a0 = state_a.h[0, model.read_indices]
        read_b0 = state_b.h[0, model.read_indices]
        diff_0 = (read_a0 - read_b0).norm().item()
        norm_0 = read_a0.norm().item()
        print(f"Tick 0 (Immediate after 1 input tick):")
        print(f"  Motor Membrane Norm ||h_motor|| = {norm_0:.4f}")
        print(f"  Motor Difference ||h_motor(A) - h_motor(B)|| = {diff_0:.6f} (Relative = {diff_0 / max(norm_0, 1e-6):.6%})")

        # Now let subsequent tokens be identical (or quiet) to observe propagation wave
        diff_curve = [diff_0]
        st_a, st_b = state_a, state_b
        quiet_token = torch.tensor([sample_tokens[2]], device=device)
        for t in range(1, 16):
            st_a = advance_fly_input_event(model, st_a, quiet_token, settle_ticks=0, writer_baseline_clock="input")
            st_b = advance_fly_input_event(model, st_b, quiet_token, settle_ticks=0, writer_baseline_clock="input")
            ra = st_a.h[0, model.read_indices]
            rb = st_b.h[0, model.read_indices]
            d = (ra - rb).norm().item()
            diff_curve.append(d)

    print(f"Motor Difference Impulse Response Curve across 16 ticks:")
    for t, d in enumerate(diff_curve):
        bar = "#" * int(min(d / max(diff_curve[0], 1e-6) * 20, 50))
        print(f"  t={t:2d}: ||Delta h_motor|| = {d:.6f} {bar}")

    # Peak response tick
    peak_t = int(np.argmax(diff_curve))
    print(f"Peak Motor Differentiation Occurs at Tick t = {peak_t} (Max Delta = {diff_curve[peak_t]:.6f})")

    return {
        "direct_edges": int(direct_edges),
        "immediate_motor_diff": float(diff_0),
        "peak_motor_tick": peak_t,
        "peak_motor_diff": float(diff_curve[peak_t]),
        "impulse_curve": [float(x) for x in diff_curve],
    }


@torch.no_grad()
def audit_gamma_and_decoder(model: FlyReservoirLM, base_state: FlyPhysicalState, validation_data_path: Path):
    """3. GAMMA & DECODER AUDIT: How are motor states translated into predictions?"""
    print("\n" + "=" * 60)
    print("STEP 3: GAMMA FILTER & DECODER LOGITS DECOMPOSITION AUDIT")
    print("=" * 60)

    device = next(model.parameters()).device
    # 3.1 Gamma time constants distribution
    gamma_decay = model.get_read_gamma_decay().detach().squeeze().cpu().numpy()
    tau_arr = gamma_decay / np.maximum(1.0 - gamma_decay, 1e-6)
    group_delays = 2.0 * tau_arr
    frac_t1 = 1.0 - (1.0 + 1.0 / tau_arr) * np.exp(-1.0 / tau_arr)
    frac_t4 = 1.0 - (1.0 + 4.0 / tau_arr) * np.exp(-4.0 / tau_arr)
    frac_t32 = 1.0 - (1.0 + 32.0 / tau_arr) * np.exp(-32.0 / tau_arr)
    print(f"Learned Readout Gamma Parameters (N=2,333):")
    print(f"  Decay Factor gamma: mean = {gamma_decay.mean():.4f}, std = {gamma_decay.std():.4f}, min = {gamma_decay.min():.4f}, max = {gamma_decay.max():.4f}")
    print(f"  Time Constant tau : mean = {tau_arr.mean():.2f}, median = {np.median(tau_arr):.2f}, min = {tau_arr.min():.2f}, max = {tau_arr.max():.2f}")
    print(f"\nTwo-Stage Gamma Filter Delay Dynamics:")
    print(f"  Group Delay (2 * tau): mean = {group_delays.mean():.1f} ticks, median = {np.median(group_delays):.1f} ticks, max = {group_delays.max():.1f} ticks")
    print(f"  Cumulative Release at t=1  tick (Immediate next-token): mean = {frac_t1.mean()*100:.2f}%, min = {frac_t1.min()*100:.4f}%")
    print(f"  Cumulative Release at t=4  ticks (Short phrase)      : mean = {frac_t4.mean()*100:.2f}%, min = {frac_t4.min()*100:.4f}%")
    print(f"  Cumulative Release at t=32 ticks (BPTT Window Horizon): mean = {frac_t32.mean()*100:.2f}%, min = {frac_t32.min()*100:.2f}%")

    # 3.2 Evaluate on real OWT tokens: Decompose logits into b_dec and Delta_logits
    val_data = np.load(validation_data_path, mmap_mode="r")
    eval_tokens = torch.from_numpy(val_data[:512].astype(np.int64)).to(device)
    inputs = eval_tokens[:-1]
    targets = eval_tokens[1:]

    with torch.no_grad():
        rates, thresholds = model.get_decay_rates(), model.get_thresholds()
        gains, alif, stp = model.get_conductance_gains(), model.get_alif_params(), model.get_stp_params()
        latents = []
        st = base_state.detached()
        for token in inputs:
            st = advance_fly_input_event(
                model, st, token.view(1),
                settle_ticks=0, writer_baseline_clock="input",
                base_rates=rates, thresholds=thresholds,
                conductance_gains=gains, alif_params=alif, stp_params=stp,
            )
            latents.append(model.output_read(st.gamma_z2))
        features = torch.cat(latents, dim=0)

        # Deconstruct decoder:
        # logits = features_normed @ W_dec.T + b_dec
        feat_norm = model.read_norm(features)  # [T, d_model]
        w_dec = model.decoder.weight            # [vocab, d_model]
        b_dec = model.decoder.bias              # [vocab]

        delta_logits = feat_norm @ w_dec.T      # [T, vocab]
        full_logits = delta_logits + b_dec      # [T, vocab]

    # Variance and norm comparison
    b_dec_var = float(b_dec.var().item())
    b_dec_norm = float(b_dec.norm().item())
    delta_var = float(delta_logits.var().item())
    delta_norm = float(delta_logits.norm(dim=-1).mean().item())

    print(f"\nLogits Energy Decomposition:")
    print(f"  Static Unigram Bias b_dec   : Norm = {b_dec_norm:.4f}, Variance = {b_dec_var:.4f}")
    print(f"  Conditioned Delta_logits    : Norm = {delta_norm:.4f}, Variance = {delta_var:.4f}")
    print(f"  Ratio ||Delta_logits|| / ||b_dec|| = {delta_norm / max(b_dec_norm, 1e-6):.4f}")

    # 3.3 The Crucial Test: Interpolating b_dec + alpha * Delta_logits
    print("\n" + "-" * 50)
    print("THE GOLDEN ABLATION: b_dec + alpha * Delta_logits")
    print("-" * 50)
    alphas = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    res_alphas = {}
    for a in alphas:
        interpolated = b_dec + a * delta_logits
        nll = F.cross_entropy(interpolated, targets, reduction="mean").item()
        res_alphas[a] = nll
        marker = " <-- [CURRENT MODEL]" if a == 1.0 else (" <-- [PURE STATIC UNIGRAM]" if a == 0.0 else "")
        print(f"  alpha = {a:4.2f} : Cross-Entropy NLL = {nll:.4f} {marker}")

    best_a = min(res_alphas, key=res_alphas.get)
    best_nll = res_alphas[best_a]
    print(f"\nOptimal Coupling Alpha: {best_a:.2f} (Achieves NLL = {best_nll:.4f})")
    if best_a < 1.0:
        print(f"DIAGNOSIS: Delta_logits is over-scaled or noisy! Shrinking its amplitude by {best_a:.2f} improves NLL by {res_alphas[1.0] - best_nll:.4f}!")
    else:
        print("DIAGNOSIS: Delta_logits scale is well-calibrated or under-scaled.")

    # 3.4 Target Token Alignment: Does Delta_logits actually rank the true target higher?
    target_deltas = delta_logits.gather(1, targets[:, None]).squeeze(-1)  # [T]
    mean_target_delta = target_deltas.mean().item()
    mean_other_delta = (delta_logits.sum(dim=-1) - target_deltas).mean().item() / (delta_logits.shape[-1] - 1)
    print(f"\nTrue Target Logit Boost:")
    print(f"  Delta on True Target Token : {mean_target_delta:.4f}")
    print(f"  Delta on Average Other Token: {mean_other_delta:.4f}")
    print(f"  Target Advantage (True - Other): {mean_target_delta - mean_other_delta:.4f}")

    return {
        "b_dec_norm": b_dec_norm,
        "delta_logits_norm": delta_norm,
        "nll_alpha_0_unigram": res_alphas[0.0],
        "nll_alpha_1_model": res_alphas[1.0],
        "best_alpha": best_a,
        "best_nll": best_nll,
        "target_advantage": mean_target_delta - mean_other_delta,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("artifacts/active_training/q8_fly_bptt32_gamma_2333_wide115_100k/last.pt"))
    parser.add_argument("--validation", type=Path,
                        default=Path("data/ib_owt_gpt2_31m/validation.npy"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/published/fly_prediction_causal_chain_audit.json"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Loading checkpoint from {args.checkpoint} on {args.device}...")
    model, physical, cfg, old, saved = load_model_and_state(args.checkpoint, device=args.device)
    print(f"Model loaded: {model.n_neurons} neurons, trained tokens = {saved['bptt_train_tokens']}")

    sample_tokens = [18371, 1165, 3382, 262, 1628, 284, 6758, 290, 481, 21923]

    res_writer = audit_writer(model, sample_tokens)
    res_connectome = audit_connectome(model, physical, sample_tokens)
    res_decoder = audit_gamma_and_decoder(model, physical, args.validation)

    summary = {
        "checkpoint": str(args.checkpoint),
        "bptt_train_tokens": saved["bptt_train_tokens"],
        "writer_audit": res_writer,
        "connectome_audit": res_connectome,
        "decoder_audit": res_decoder,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nAudit complete! Full results saved to {args.output}")


if __name__ == "__main__":
    main()
