"""Multi-State Operator Commutation and Discretization Convergence Test.

Theoretical Formulation (Neural Operator Consistency Test, Berner et al., Nature Machine Intelligence 2025/2026):
Continuous operator G: X -> Y on periodic 3-torus T^3.
Strictly self-similar isotropic grid hierarchy:
  h1 = (4, 4, 2)   [32 nodes,   h_x=0.250, h_y=0.250, h_z=0.500]
  h2 = (8, 8, 4)   [256 nodes,  h_x=0.125, h_y=0.125, h_z=0.250, native training resolution]
  h3 = (16, 16, 8) [2048 nodes, h_x=0.0625, h_y=0.0625, h_z=0.125, fine reference resolution]
Grid refinement ratio is strictly 2.0 across all axes.

Discretization error with respect to fine reference resolution:
  epsilon_h = || P_common(G_fine(u_fine)) - P_common(G_h(u_h)) ||_L2 / || P_common u ||_L2

Convergence rate p on log-log grid:
  p = log2(epsilon_h1 / epsilon_h2)

Evaluates across 16 mature NESS states along the validation text stream.
Also measures probability distribution divergence (Jensen-Shannon) and Top-k agreement.
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


def theoretical_spectral_tail(sigma: float = 0.15, fine_shape=(16, 16, 8), target_shape=(8, 8, 4)) -> float:
    """Calculate exact theoretical energy ratio outside coarse grid Nyquist band."""
    wave_axes = [torch.fft.fftfreq(n, d=1.0/n) * 2.0 * math.pi for n in fine_shape]
    wave = torch.stack(torch.meshgrid(*wave_axes, indexing="ij"), -1)
    k_sq = wave.square().sum(-1)
    fourier_coeff = torch.exp(-0.5 * (sigma ** 2) * k_sq)
    total_energy = fourier_coeff.square().sum().item()

    # Mask modes inside target_shape
    c_x, c_y, c_z = target_shape
    def mode_slice(N_fine, N_coarse):
        pos, neg = (N_coarse + 1) // 2, N_coarse // 2
        return [slice(0, pos), slice(N_fine - neg, N_fine)]

    mask = torch.zeros(fine_shape, dtype=torch.bool)
    for sx in mode_slice(fine_shape[0], c_x):
        for sy in mode_slice(fine_shape[1], c_y):
            for sz in mode_slice(fine_shape[2], c_z):
                mask[sx, sy, sz] = True

    resolved_energy = fourier_coeff[mask].square().sum().item()
    tail_energy = max(total_energy - resolved_energy, 0.0)
    return math.sqrt(tail_energy / total_energy)


def main():
    ckpt_path = Path("results/cbim_torus3d_w2_8x8x4_arm_c_cont_q8_unified_dissipation_3000/BBest.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading reference model from {ckpt_path}...")
    saved = torch.load(ckpt_path, map_location="cuda")

    # Strictly isotropic self-similar grid family
    configs = {
        "4x4x2": (4, 4, 2),
        "8x8x4": (8, 8, 4),
        "16x16x8": (16, 16, 8),
    }
    models = {}
    for name, shape in configs.items():
        m = CBIMTorus3D(
            shape=shape, velocities=8, content_dim=16,
            v2_coordinate_components=True,
            readout_type="kernel_r1", write_type="w2_impedance",
            micro_steps=3, adaptive_clock=True, continuous_velocities=True,
            dissipation_type="unified", dissipation_rank=4
        ).cuda()
        m.load_state_dict(saved["model"], strict=True)
        m.eval()
        models[name] = m

    m4, m8, m16 = models["4x4x2"], models["8x8x4"], models["16x16x8"]
    common_shape = (4, 4, 2)

    # Calculate theoretical spectral tail floor for Gaussian sigma = 0.15
    floor_8 = theoretical_spectral_tail(0.15, fine_shape=(16, 16, 8), target_shape=(8, 8, 4))
    floor_4 = theoretical_spectral_tail(0.15, fine_shape=(16, 16, 8), target_shape=(4, 4, 2))
    print(f"Theoretical Spectral Tail Error Floor: outside (8x8x4) = {floor_8:.2%}, outside (4x4x2) = {floor_4:.2%}")

    # Harvest 16 diverse mature NESS states
    val_data = np.load("data/ib_owt_gpt2/validation.npy", mmap_mode="r")
    num_states = 16
    chunk_tokens = 128
    print(f"Harvesting {num_states} mature NESS states along validation stream ({num_states * chunk_tokens} tokens)...")

    mature_states = []
    tokens_list = []
    with torch.no_grad():
        state = m8.initial_state(1, "cuda", warm_start=True)
        for i in range(num_states):
            offset = i * chunk_tokens
            chunk = torch.as_tensor(np.array(val_data[offset:offset + chunk_tokens]), dtype=torch.long, device="cuda")[None]
            targets = torch.as_tensor(np.array(val_data[offset + 1:offset + chunk_tokens + 1]), dtype=torch.long, device="cuda")[None]
            _, state, _ = m8(chunk, targets, state)
            mature_states.append(state.clone())
            tokens_list.append(torch.as_tensor([val_data[offset + chunk_tokens]], dtype=torch.long, device="cuda"))

    modules = [
        "Transport (T)",
        "Collision (C)",
        "Dissipation (D)",
        "Write (W)",
        "Readout Probe (R)",
        "Kinetic Cycle (T->C->D)",
        "Full Step Field (F_next)",
        "Output Logits (vocab)",
    ]
    state_errors = {mod: {"err_4_8": [], "err_8_16": [], "err_4_16": []} for mod in modules}
    js_divs = []
    top1_matches = []
    top5_overlaps = []
    abs_l2_list = []

    print(f"Executing commutation tests across {num_states} distinct physical states on isotropic grid hierarchy...")
    with torch.no_grad():
        for idx in range(num_states):
            u_ref = mature_states[idx]
            tok_id = tokens_list[idx]
            tok_emb = m8.source.embedding(tok_id)

            u4 = spectral_resample(u_ref, (4, 4, 2))
            u8 = u_ref
            u16 = spectral_resample(u_ref, (16, 16, 8))

            # 1. Transport
            t4 = m4.transport(u4)[0]
            t8 = m8.transport(u8)[0]
            t16 = m16.transport(u16)[0]
            t8_p = spectral_resample(t8, common_shape)
            t16_p = spectral_resample(t16, common_shape)
            state_errors["Transport (T)"]["err_4_8"].append(relative_l2_error(t4, t8_p))
            state_errors["Transport (T)"]["err_8_16"].append(relative_l2_error(t8_p, t16_p))
            state_errors["Transport (T)"]["err_4_16"].append(relative_l2_error(t4, t16_p))

            # 2. Collision
            c4 = m4.collision(u4)[0]
            c8 = m8.collision(u8)[0]
            c16 = m16.collision(u16)[0]
            c8_p = spectral_resample(c8, common_shape)
            c16_p = spectral_resample(c16, common_shape)
            state_errors["Collision (C)"]["err_4_8"].append(relative_l2_error(c4, c8_p))
            state_errors["Collision (C)"]["err_8_16"].append(relative_l2_error(c8_p, c16_p))
            state_errors["Collision (C)"]["err_4_16"].append(relative_l2_error(c4, c16_p))

            # 3. Dissipation
            d4 = m4.bath(u4, delta_tau=1.0, tok_embed=tok_emb)[0]
            d8 = m8.bath(u8, delta_tau=1.0, tok_embed=tok_emb)[0]
            d16 = m16.bath(u16, delta_tau=1.0, tok_embed=tok_emb)[0]
            d8_p = spectral_resample(d8, common_shape)
            d16_p = spectral_resample(d16, common_shape)
            state_errors["Dissipation (D)"]["err_4_8"].append(relative_l2_error(d4, d8_p))
            state_errors["Dissipation (D)"]["err_8_16"].append(relative_l2_error(d8_p, d16_p))
            state_errors["Dissipation (D)"]["err_4_16"].append(relative_l2_error(d4, d16_p))

            # 4. Write
            w4 = m4.source(u4, tok_id)[0]
            w8 = m8.source(u8, tok_id)[0]
            w16 = m16.source(u16, tok_id)[0]
            w8_p = spectral_resample(w8, common_shape)
            w16_p = spectral_resample(w16, common_shape)
            state_errors["Write (W)"]["err_4_8"].append(relative_l2_error(w4, w8_p))
            state_errors["Write (W)"]["err_8_16"].append(relative_l2_error(w8_p, w16_p))
            state_errors["Write (W)"]["err_4_16"].append(relative_l2_error(w4, w16_p))

            # 5. Readout Probe
            r4 = m4.readout(u4, tok_emb)[0]
            r8 = m8.readout(u8, tok_emb)[0]
            r16 = m16.readout(u16, tok_emb)[0]
            state_errors["Readout Probe (R)"]["err_4_8"].append(relative_l2_error(r4, r8))
            state_errors["Readout Probe (R)"]["err_8_16"].append(relative_l2_error(r8, r16))
            state_errors["Readout Probe (R)"]["err_4_16"].append(relative_l2_error(r4, r16))

            # 6. Kinetic Cycle
            def microstep(m, u):
                t = m.transport(u)[0]
                c = m.collision(t)[0]
                d = m.bath(c, delta_tau=1.0, tok_embed=tok_emb)[0]
                return d

            k4 = microstep(m4, u4)
            k8 = microstep(m8, u8)
            k16 = microstep(m16, u16)
            k8_p = spectral_resample(k8, common_shape)
            k16_p = spectral_resample(k16, common_shape)
            state_errors["Kinetic Cycle (T->C->D)"]["err_4_8"].append(relative_l2_error(k4, k8_p))
            state_errors["Kinetic Cycle (T->C->D)"]["err_8_16"].append(relative_l2_error(k8_p, k16_p))
            state_errors["Kinetic Cycle (T->C->D)"]["err_4_16"].append(relative_l2_error(k4, k16_p))

            # 7. Full Step Field and Logits
            l4, fs4, _ = m4.step(u4, tok_id)
            l8, fs8, _ = m8.step(u8, tok_id)
            l16, fs16, _ = m16.step(u16, tok_id)
            fs8_p = spectral_resample(fs8, common_shape)
            fs16_p = spectral_resample(fs16, common_shape)
            state_errors["Full Step Field (F_next)"]["err_4_8"].append(relative_l2_error(fs4, fs8_p))
            state_errors["Full Step Field (F_next)"]["err_8_16"].append(relative_l2_error(fs8_p, fs16_p))
            state_errors["Full Step Field (F_next)"]["err_4_16"].append(relative_l2_error(fs4, fs16_p))

            state_errors["Output Logits (vocab)"]["err_4_8"].append(relative_l2_error(l4, l8))
            state_errors["Output Logits (vocab)"]["err_8_16"].append(relative_l2_error(l8, l16))
            state_errors["Output Logits (vocab)"]["err_4_16"].append(relative_l2_error(l4, l16))

            # Distribution metrics (8 vs 16)
            p8 = F.softmax(l8, dim=-1)
            p16 = F.softmax(l16, dim=-1)
            js = jensen_shannon_div(p8, p16)
            js_divs.append(js)
            top1_matches.append(bool((l8.argmax() == l16.argmax()).item()))
            top5_8 = set(l8.topk(5).indices[0].cpu().tolist())
            top5_16 = set(l16.topk(5).indices[0].cpu().tolist())
            top5_overlaps.append(len(top5_8.intersection(top5_16)) / 5.0)
            abs_l2_list.append((l8 - l16).norm().item())

    # Summary table with Percentiles
    summary = {}
    print("\n" + "=" * 98)
    print("  ISOTROPIC SELF-SIMILAR OPERATOR COMMUTATION & CONVERGENCE: (4x4x2) -> (8x8x4) -> (16x16x8)")
    print("=" * 98)
    print(f"{'Module / Operator':<26} | {'eps(4x2, 8x4)':<17} | {'eps(8x4, 16x8)':<17} | {'P50 (8->16)':<11} | {'P90 (8->16)':<11} | {'Order p'}")
    print("-" * 98)

    for mod in modules:
        e4_8 = np.array(state_errors[mod]["err_4_8"])
        e8_16 = np.array(state_errors[mod]["err_8_16"])
        e4_16 = np.array(state_errors[mod]["err_4_16"])

        m_4_8 = float(e4_8.mean())
        m_8_16 = float(e8_16.mean())
        m_4_16 = float(e4_16.mean())
        p50 = float(np.percentile(e8_16, 50))
        p90 = float(np.percentile(e8_16, 90))
        pmax = float(e8_16.max())

        # Exact isotropic convergence order p: h_1 / h_2 = 2.0
        p = math.log2(max(m_4_16, 1e-8) / max(m_8_16, 1e-8)) if m_8_16 > 0 else 0.0

        summary[mod] = {
            "err_4_8_mean": m_4_8, "err_8_16_mean": m_8_16, "err_4_16_mean": m_4_16,
            "err_8_16_p50": p50, "err_8_16_p90": p90, "err_8_16_max": pmax,
            "convergence_order_p": p, "convergent": p > 0,
        }

        p_str = f"{p:+.2f} ({'PASS' if p > 0 else 'FAIL'})"
        print(f"{mod:<26} | {m_4_8:.4f}           | {m_8_16:.4f}           | {p50:.4f}      | {p90:.4f}      | {p_str}")

    print("=" * 98)

    # Distribution divergence report
    js_arr = np.array(js_divs)
    top1_arr = np.array(top1_matches)
    top5_arr = np.array(top5_overlaps)
    abs_arr = np.array(abs_l2_list)

    print("\n" + "=" * 70)
    print("       OUTPUT PREDICTION DISTRIBUTION METRICS (8x8x4 vs 16x16x8)")
    print("=" * 70)
    print(f"Top-1 Token Prediction Agreement:   {top1_arr.mean():.1%} ({top1_arr.sum()}/{len(top1_arr)} states match)")
    print(f"Top-5 Token Set Overlap:            {top5_arr.mean():.1%}")
    print(f"Jensen-Shannon Divergence Mean:     {js_arr.mean():.6f} nats")
    print(f"Jensen-Shannon Divergence Median:   {np.percentile(js_arr, 50):.6f} nats")
    print(f"Jensen-Shannon Divergence P90:      {np.percentile(js_arr, 90):.6f} nats")
    print(f"Jensen-Shannon Divergence Max:      {js_arr.max():.6f} nats")
    print(f"Absolute Logits L2 Error Mean:      {abs_arr.mean():.2f}")
    print(f"Theoretical Spectral Tail Floor:    {floor_8:.2%} (outside 8x8x4)")
    print("=" * 70)

    summary["distribution_metrics"] = {
        "top1_agreement": float(top1_arr.mean()),
        "top5_overlap": float(top5_arr.mean()),
        "js_mean": float(js_arr.mean()),
        "js_median": float(np.percentile(js_arr, 50)),
        "js_p90": float(np.percentile(js_arr, 90)),
        "js_max": float(js_arr.max()),
        "theoretical_spectral_tail_floor_8": float(floor_8),
    }

    out_file = Path("results/operator_commutation_isotropic_16states.json")
    out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nReport written to {out_file}")


if __name__ == "__main__":
    main()
