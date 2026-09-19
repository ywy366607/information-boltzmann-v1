"""Evaluate Run B model across K in [1, 2, 3, 4, 8, 16, 32, 64] on 512 validation tokens."""
import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

from scripts.ib_local.cbim_torus3d import CBIMTorus3D


def main():
    ckpt_path = Path("results/cbim_three_clock_w2_8x8x4_k3_3000/BBest.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading Run B model from {ckpt_path}...", flush=True)
    saved = torch.load(ckpt_path, map_location="cuda")
    val_data = np.load("data/ib_owt_gpt2/validation.npy", mmap_mode="r")

    model = CBIMTorus3D(
        shape=(8, 8, 4), velocities=8, content_dim=16,
        v2_coordinate_components=True, readout_type="kernel_r1",
        write_type="w2_impedance", micro_steps=3, adaptive_clock=True,
        continuous_velocities=True, dissipation_type="unified", dissipation_rank=4,
        three_clock=True, tau_mem=3.0, nu_s_init=0.020
    ).cuda()
    model.load_state_dict(saved["model"])
    model.eval()

    num_eval_tokens = 512
    horizons = [1, 2, 3, 4, 8, 16, 32, 64]
    results = {}

    print("\n" + "=" * 60, flush=True)
    print("   RUN B ZERO-SHOT PONDERING EXTRAPOLATION (512 TOKENS)", flush=True)
    print("=" * 60, flush=True)
    print(f"{'Horizon K':<12} | {'Validation NLL':<18} | {'Delta vs K=3'}", flush=True)
    print("-" * 60, flush=True)

    with torch.no_grad():
        for k in horizons:
            state = model.initial_state(1, "cuda", warm_start=True)
            # Warmup 64 tokens
            for t in range(64):
                inp = torch.as_tensor([val_data[t]], dtype=torch.long, device="cuda")
                tgt = torch.as_tensor([val_data[t + 1]], dtype=torch.long, device="cuda")
                _, state, _ = model.step(state, inp, micro_steps=k)

            total_loss = 0.0
            for t in range(64, 64 + num_eval_tokens):
                inp = torch.as_tensor([val_data[t]], dtype=torch.long, device="cuda")
                tgt = torch.as_tensor([val_data[t + 1]], dtype=torch.long, device="cuda")
                logits, state, _ = model.step(state, inp, micro_steps=k)
                loss = F.cross_entropy(logits, tgt).item()
                total_loss += loss
            mean_nll = total_loss / num_eval_tokens
            results[k] = mean_nll
            k3_loss = results.get(3, mean_nll)
            delta = mean_nll - k3_loss
            print(f"K = {k:<9d} | {mean_nll:<18.4f} | {delta:<+12.4f} nats", flush=True)

    print("=" * 60, flush=True)
    out_path = Path("results/run_b_zero_shot_pondering_k512.json")
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Results saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
