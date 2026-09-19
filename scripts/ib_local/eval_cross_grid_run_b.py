"""Zero-shot cross-grid resolution evaluation of CBIM Three-Clock v1 (Run B).

Evaluates:
- (4, 4, 2): 32 nodes (8x coarser)
- (4, 4, 4): 64 nodes (4x coarser)
- (8, 8, 4): 256 nodes (native training resolution)
- (8, 8, 8): 512 nodes (2x denser)
- (16, 16, 8): 2048 nodes (8x denser zero-shot super-resolution)

STRICT RULE:
- NEVER reset the field state to 0 or cold vacuum.
- Directly inherit the mature continuous physical state saved from the training stream (saved["state"]).
- For non-native grid resolutions, apply continuous Fourier-Galerkin spectral resampling.
"""
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
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from scripts.ib_local.cbim_torus3d import CBIMTorus3D


def resample_field_spectral(field: torch.Tensor, target_shape: tuple[int, int, int]) -> torch.Tensor:
    """Exact continuous Fourier-Galerkin spectral resampling of a 3D field across resolutions."""
    B, Xs, Ys, Zs, D = field.shape
    Xt, Yt, Zt = target_shape
    if (Xs, Ys, Zs) == (Xt, Yt, Zt):
        return field.clone()

    freq_src = torch.fft.fftn(field, dim=(1, 2, 3), norm="ortho")
    freq_tgt = torch.zeros(B, Xt, Yt, Zt, D, dtype=freq_src.dtype, device=field.device)

    def mode_slices(N_src, N_tgt):
        c = min(N_src, N_tgt)
        pos, neg = (c + 1) // 2, c // 2
        return (slice(0, pos), slice(0, pos)), (slice(N_src - neg, N_src), slice(N_tgt - neg, N_tgt))

    (sx_p, tx_p), (sx_n, tx_n) = mode_slices(Xs, Xt)
    (sy_p, ty_p), (sy_n, ty_n) = mode_slices(Ys, Yt)
    (sz_p, tz_p), (sz_n, tz_n) = mode_slices(Zs, Zt)

    for sx, tx_s in [(sx_p, tx_p), (sx_n, tx_n)]:
        for sy, ty_s in [(sy_p, ty_p), (sy_n, ty_n)]:
            for sz, tz_s in [(sz_p, tz_p), (sz_n, tz_n)]:
                freq_tgt[:, tx_s, ty_s, tz_s] = freq_src[:, sx, sy, sz]

    scale = math.sqrt((Xt * Yt * Zt) / (Xs * Ys * Zs))
    freq_tgt = freq_tgt * scale
    return torch.fft.ifftn(freq_tgt, dim=(1, 2, 3), norm="ortho").real


def main():
    ckpt_path = Path("results/cbim_three_clock_w2_8x8x4_k3_3000/BBest.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading Run B weights and mature persistent state from {ckpt_path}...", flush=True)
    saved = torch.load(ckpt_path, map_location="cuda")
    state_dict = saved["model"]
    mature_state_native = saved["state"].detach().cuda()  # [1, 8, 8, 4, 128]
    val_data = np.load("data/ib_owt_gpt2/validation.npy", mmap_mode="r")

    num_eval_tokens = 1024
    shapes = [
        (4, 4, 2),
        (4, 4, 4),
        (8, 8, 4),
        (8, 8, 8),
        (16, 16, 8),
    ]

    results = {}

    print("\n" + "=" * 95, flush=True)
    print("   CBIM THREE-CLOCK V1 (RUN B) ZERO-SHOT CROSS-GRID EVALUATION (NO RESET, 1024 TOKENS)")
    print("=" * 95, flush=True)
    print(f"{'Grid Shape':<14} | {'Nodes':<7} | {'Val NLL':<12} | {'Delta vs Native':<18} | {'Speed (tok/s)':<14} | {'Mean Energy'}", flush=True)
    print("-" * 95, flush=True)

    # First evaluate native resolution (8, 8, 4) to establish the true baseline
    native_model = CBIMTorus3D(
        shape=(8, 8, 4), velocities=8, content_dim=16,
        v2_coordinate_components=True, readout_type="kernel_r1",
        write_type="w2_impedance", micro_steps=3, adaptive_clock=True,
        continuous_velocities=True, dissipation_type="unified", dissipation_rank=4,
        three_clock=True, tau_mem=3.0, nu_s_init=0.020
    ).cuda()
    native_model.load_state_dict(state_dict, strict=False)
    native_model.eval()

    with torch.no_grad():
        state = mature_state_native.clone()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        total_loss = 0.0
        energies = []
        for t in range(num_eval_tokens):
            inp = torch.as_tensor([val_data[t]], dtype=torch.long, device="cuda")
            tgt = torch.as_tensor([val_data[t + 1]], dtype=torch.long, device="cuda")
            logits, state, diag = native_model.step(state, inp, micro_steps=3)
            loss = F.cross_entropy(logits, tgt).item()
            total_loss += loss
            energies.append(float(diag["energy"]))
        torch.cuda.synchronize()
        native_nll = total_loss / num_eval_tokens

    for sh in shapes:
        torch.cuda.empty_cache()
        nodes = sh[0] * sh[1] * sh[2]
        model = CBIMTorus3D(
            shape=sh, velocities=8, content_dim=16,
            v2_coordinate_components=True, readout_type="kernel_r1",
            write_type="w2_impedance", micro_steps=3, adaptive_clock=True,
            continuous_velocities=True, dissipation_type="unified", dissipation_rank=4,
            three_clock=True, tau_mem=3.0, nu_s_init=0.020
        ).cuda()
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        with torch.no_grad():
            # STRICT INHERITANCE: resample the exact mature physical state to the new grid
            state = resample_field_spectral(mature_state_native, sh)

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            total_loss = 0.0
            energies = []

            for t in range(num_eval_tokens):
                inp = torch.as_tensor([val_data[t]], dtype=torch.long, device="cuda")
                tgt = torch.as_tensor([val_data[t + 1]], dtype=torch.long, device="cuda")
                logits, state, diag = model.step(state, inp, micro_steps=3)
                loss = F.cross_entropy(logits, tgt).item()
                total_loss += loss
                energies.append(float(diag["energy"]))

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            speed = num_eval_tokens / elapsed
            mean_nll = total_loss / num_eval_tokens
            mean_e = float(np.mean(energies))
            delta_nll = mean_nll - native_nll

            results[str(sh)] = {
                "nodes": nodes,
                "val_nll": mean_nll,
                "delta_vs_native": delta_nll,
                "speed_tok_sec": speed,
                "mean_energy": mean_e,
            }

            print(f"{str(sh):<14} | {nodes:<7d} | {mean_nll:<12.4f} | {delta_nll:<+18.4f} | {speed:<14.1f} | {mean_e:.6f}", flush=True)

    print("=" * 95, flush=True)
    out_file = Path("results/run_b_cross_grid_evaluation.json")
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nReport written to {out_file}", flush=True)


if __name__ == "__main__":
    main()
