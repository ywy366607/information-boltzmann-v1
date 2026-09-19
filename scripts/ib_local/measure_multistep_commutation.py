"""Multi-Step Dynamical Commutation and Semigroup Flow Consistency Test.

Theoretical Formulation:
Tests whether iterated numerical realizations G_h^n converge to a continuous-time
semigroup flow Phi_tau: X -> X on function space L^2(T^3), rather than diverging exponentially.

Two Regimes:
1. Autonomous Kinetic Flow: Phi_tau = (D o C o T)^n for n in [1, 2, 4, 8, 16, 32, 40]
   Measures pure unforced continuous flow preservation over long internal time tau = n * delta_tau.
2. Driven Autoregressive Sequence: Iterated full steps with real token stream over N = 32 steps.
   Measures multi-step continuous stream consistency: field error, JS divergence, and Top-1 match.

Grid Pair: (8, 8, 4) vs (16, 16, 8)
"""
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(script_dir))
if sys.path and os.path.abspath(sys.path[0]) == script_dir:
    sys.path.pop(0)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from pathlib import Path
import json
import math

import numpy as np
import torch
import torch.nn.functional as F

from scripts.ib_local.cbim_torus3d import CBIMTorus3D


def spectral_resample(field: torch.Tensor, target_shape: tuple[int, int, int]) -> torch.Tensor:
    """Spectral resampling preserving continuous function values u(x) on T^3."""
    B, X, Y, Z, D = field.shape
    tx, ty, tz = target_shape
    if (X, Y, Z) == (tx, ty, tz):
        return field

    freq = torch.fft.fftn(field, dim=(1, 2, 3), norm="backward")
    freq_target = torch.zeros(B, tx, ty, tz, D, dtype=freq.dtype, device=field.device)

    def mode_slices(N_src, N_tgt):
        c = min(N_src, N_tgt)
        pos = (c + 1) // 2
        neg = c // 2
        src_pos = slice(0, pos)
        tgt_pos = slice(0, pos)
        src_neg = slice(N_src - neg, N_src) if neg > 0 else slice(0, 0)
        tgt_neg = slice(N_tgt - neg, N_tgt) if neg > 0 else slice(0, 0)
        return (src_pos, tgt_pos), (src_neg, tgt_neg)

    (sx_p, tx_p), (sx_n, tx_n) = mode_slices(X, tx)
    (sy_p, ty_p), (sy_n, ty_n) = mode_slices(Y, ty)
    (sz_p, tz_p), (sz_n, tz_n) = mode_slices(Z, tz)

    scale = (tx * ty * tz) / (X * Y * Z)
    for sx, tx_s in [(sx_p, tx_p), (sx_n, tx_n)]:
        for sy, ty_s in [(sy_p, ty_p), (sy_n, ty_n)]:
            for sz, tz_s in [(sz_p, tz_p), (sz_n, tz_n)]:
                freq_target[:, tx_s, ty_s, tz_s] = freq[:, sx, sy, sz] * scale

    return torch.fft.ifftn(freq_target, dim=(1, 2, 3), norm="backward").real


def relative_l2_error(u1: torch.Tensor, u2: torch.Tensor, eps: float = 1e-8) -> float:
    diff_norm = (u1 - u2).square().sum().sqrt().item()
    ref_norm = u2.square().sum().sqrt().item()
    return diff_norm / max(ref_norm, eps)


def jensen_shannon_div(p: torch.Tensor, q: torch.Tensor) -> float:
    m = 0.5 * (p + q)
    kl_p_m = F.kl_div(m.log(), p, reduction="batchmean")
    kl_q_m = F.kl_div(m.log(), q, reduction="batchmean")
    return float(0.5 * (kl_p_m + kl_q_m).item())


def main():
    ckpt_path = Path("results/cbim_torus3d_w2_8x8x4_arm_c_cont_q8_unified_dissipation_3000/BBest.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading reference model from {ckpt_path}...")
    saved = torch.load(ckpt_path, map_location="cuda")

    m8 = CBIMTorus3D(
        shape=(8, 8, 4), velocities=8, content_dim=16,
        v2_coordinate_components=True,
        readout_type="kernel_r1", write_type="w2_impedance",
        micro_steps=3, adaptive_clock=True, continuous_velocities=True,
        dissipation_type="unified", dissipation_rank=4
    ).cuda()
    m8.load_state_dict(saved["model"], strict=True)
    m8.eval()

    m16 = CBIMTorus3D(
        shape=(16, 16, 8), velocities=8, content_dim=16,
        v2_coordinate_components=True,
        readout_type="kernel_r1", write_type="w2_impedance",
        micro_steps=3, adaptive_clock=True, continuous_velocities=True,
        dissipation_type="unified", dissipation_rank=4
    ).cuda()
    m16.load_state_dict(saved["model"], strict=True)
    m16.eval()

    common_shape = (8, 8, 4)

    # 1. Harvest a mature continuous state
    val_data = np.load("data/ib_owt_gpt2/validation.npy", mmap_mode="r")
    with torch.no_grad():
        state = m8.initial_state(1, "cuda", warm_start=True)
        for i in range(8):
            chunk = torch.as_tensor(np.array(val_data[i*128:(i+1)*128]), dtype=torch.long, device="cuda")[None]
            targets = torch.as_tensor(np.array(val_data[i*128+1:(i+1)*128+1]), dtype=torch.long, device="cuda")[None]
            _, state, _ = m8(chunk, targets, state)
        u_mature_8 = state.clone()
        u_mature_16 = spectral_resample(u_mature_8, (16, 16, 8))

    print("\n" + "=" * 80)
    print("      PART 1: AUTONOMOUS CONTINUOUS KINETIC FLOW (Phi_tau = (D o C o T)^n)")
    print("=" * 80)
    print(f"{'Step n':<8} | {'eps_field (8->16)':<20} | {'E_8(n) / E_0':<16} | {'E_16(n) / E_0':<16}")
    print("-" * 80)

    horizons = [1, 2, 4, 8, 16, 24, 32, 40]
    flow_results = {}

    with torch.no_grad():
        u8_cur = u_mature_8.clone()
        u16_cur = u_mature_16.clone()
        e8_0 = 0.5 * u8_cur.square().sum(-1).mean().item()
        e16_0 = 0.5 * u16_cur.square().sum(-1).mean().item()

        tok_mock = m8.source.embedding(torch.tensor([100], device="cuda"))

        current_n = 0
        for target_n in horizons:
            while current_n < target_n:
                # Single microstep T -> C -> D
                t8 = m8.transport(u8_cur)[0]
                c8 = m8.collision(t8)[0]
                u8_cur = m8.bath(c8, delta_tau=1.0, tok_embed=tok_mock)[0]

                t16 = m16.transport(u16_cur)[0]
                c16 = m16.collision(t16)[0]
                u16_cur = m16.bath(c16, delta_tau=1.0, tok_embed=tok_mock)[0]

                current_n += 1

            u16_proj = spectral_resample(u16_cur, common_shape)
            err_field = relative_l2_error(u8_cur, u16_proj)
            e8_n = 0.5 * u8_cur.square().sum(-1).mean().item() / e8_0
            e16_n = 0.5 * u16_cur.square().sum(-1).mean().item() / e16_0

            flow_results[str(target_n)] = {
                "err_field": err_field,
                "e8_ratio": e8_n,
                "e16_ratio": e16_n,
            }
            print(f"{target_n:<8d} | {err_field:<20.4%} | {e8_n:<16.4f} | {e16_n:<16.4f}")

    print("=" * 80)

    print("\n" + "=" * 88)
    print("      PART 2: DRIVEN AUTOREGRESSIVE STREAM FLOW (N = 32 SEQUENTIAL TOKENS)")
    print("=" * 88)
    print(f"{'Token Step':<10} | {'Field L2 Error':<16} | {'JS Divergence':<16} | {'Top-1 Match':<14} | {'Top-5 Overlap'}")
    print("-" * 88)

    ar_results = []
    with torch.no_grad():
        s8 = u_mature_8.clone()
        s16 = u_mature_16.clone()

        js_list = []
        top1_list = []
        top5_list = []
        err_list = []

        for step_idx in range(1, 33):
            tok = torch.as_tensor([val_data[1024 + step_idx]], dtype=torch.long, device="cuda")

            l8, s8, _ = m8.step(s8, tok)
            l16, s16, _ = m16.step(s16, tok)

            s16_p = spectral_resample(s16, common_shape)
            err_f = relative_l2_error(s8, s16_p)
            err_list.append(err_f)

            p8 = F.softmax(l8, dim=-1)
            p16 = F.softmax(l16, dim=-1)
            js = jensen_shannon_div(p8, p16)
            js_list.append(js)

            top1 = bool((l8.argmax() == l16.argmax()).item())
            top1_list.append(top1)

            t5_8 = set(l8.topk(5).indices[0].cpu().tolist())
            t5_16 = set(l16.topk(5).indices[0].cpu().tolist())
            top5 = len(t5_8.intersection(t5_16)) / 5.0
            top5_list.append(top5)

            if step_idx in (1, 2, 4, 8, 16, 24, 32):
                print(f"{step_idx:<10d} | {err_f:<16.4%} | {js:<16.6f} | {str(top1):<14s} | {top5:<13.1%}")

            ar_results.append({
                "step": step_idx,
                "field_error": err_f,
                "js_divergence": js,
                "top1_match": top1,
                "top5_overlap": top5,
            })

    print("=" * 88)
    print("\n" + "=" * 60)
    print("           DRIVEN AUTOREGRESSIVE STREAM SUMMARY (32 STEPS)")
    print("=" * 60)
    print(f"Mean Field Commutation Error:   {np.mean(err_list):.2%}")
    print(f"Step 32 Final Field Error:      {err_list[-1]:.2%}")
    print(f"Mean JS Divergence:             {np.mean(js_list):.6f} nats")
    print(f"Step 32 Final JS Divergence:    {js_list[-1]:.6f} nats")
    print(f"Overall Top-1 Match Rate:       {np.mean(top1_list):.1%} ({sum(top1_list)}/32 tokens)")
    print(f"Overall Top-5 Overlap Rate:     {np.mean(top5_list):.1%}")
    print("=" * 60)

    out_file = Path("results/multistep_commutation_flow.json")
    out_file.write_text(json.dumps({
        "autonomous_flow": flow_results,
        "autoregressive_stream": ar_results,
        "summary": {
            "mean_field_error": float(np.mean(err_list)),
            "step32_field_error": float(err_list[-1]),
            "mean_js": float(np.mean(js_list)),
            "step32_js": float(js_list[-1]),
            "top1_match_rate": float(np.mean(top1_list)),
            "top5_overlap_rate": float(np.mean(top5_list)),
        }
    }, indent=2), encoding="utf-8")
    print(f"\nReport written to {out_file}")


if __name__ == "__main__":
    main()
