"""Comprehensive Zero-Field and Context Ablation Controls.

Answers the fundamental question:
"What is the true baseline language loss when the field is completely dead / zeroed out?"

Evaluates on 1024 tokens:
1. Uniform Distribution Ceiling: ln(50257) = 10.8249 nats.
2. Absolute Vacuum: F = 0, tok_embed = 0 -> readout -> decoder.
3. Pure Zero-Field with Token Query: F = 0, tok_embed = embed(x_t) -> readout -> decoder.
4. Memoryless Instantaneous Write: F_prior = 0, F = Write(0, x_t) -> readout -> decoder (0 steps).
5. Memoryless Standard Step: F_prior = 0, step(0, x_t, micro_steps=3) -> logits.
6. Full Continuous Physical Stream (Reference): state carried over from previous tokens.
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
import numpy as np
import torch
import torch.nn.functional as F

from scripts.ib_local.cbim_torus3d import CBIMTorus3D


def main():
    ckpt_path = Path("results/cbim_torus3d_w2_8x8x4_arm_c_cont_q8_unified_dissipation_3000/BBest.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading reference model from {ckpt_path}...")
    saved = torch.load(ckpt_path, map_location="cuda")
    val_data = np.load("data/ib_owt_gpt2/validation.npy", mmap_mode="r")

    num_tokens = 1024
    warmup_tokens = 256

    model = CBIMTorus3D(
        shape=(8, 8, 4), velocities=8, content_dim=16,
        v2_coordinate_components=True,
        readout_type="kernel_r1", write_type="w2_impedance",
        micro_steps=3, adaptive_clock=True, continuous_velocities=True,
        dissipation_type="unified", dissipation_rank=4
    ).cuda()
    model.load_state_dict(saved["model"], strict=True)
    model.eval()

    uniform_nll = float(np.log(50257))

    losses = {
        "1. Uniform Random Guessing": [],
        "2. Absolute Vacuum (F=0, tok_embed=0)": [],
        "3. Pure Zero-Field (F=0, tok_embed=embed(x_t))": [],
        "4. Memoryless Write 0-step (F_prior=0, write x_t, K=0)": [],
        "5. Memoryless Standard (F_prior=0, write x_t, K=3)": [],
        "6. Full Continuous Stream (Normal Physical Memory)": []
    }

    with torch.no_grad():
        state = model.initial_state(1, "cuda", warm_start=True)
        # Warmup continuous memory
        for offset in range(0, warmup_tokens, 128):
            ids = torch.as_tensor(np.array(val_data[offset:offset + 128]), dtype=torch.long, device="cuda")[None]
            targets = torch.as_tensor(np.array(val_data[offset + 1:offset + 129]), dtype=torch.long, device="cuda")[None]
            _, state, _ = model(ids, targets, state)

        for t in range(warmup_tokens, warmup_tokens + num_tokens):
            inp_id = torch.as_tensor([val_data[t]], dtype=torch.long, device="cuda")
            tgt_id = torch.as_tensor([val_data[t + 1]], dtype=torch.long, device="cuda")
            tok_embed = model.source.embedding(inp_id)
            zero_field = torch.zeros(1, *model.state_shape, device="cuda")

            # 1. Uniform
            losses["1. Uniform Random Guessing"].append(uniform_nll)

            # 2. Absolute Vacuum: F = 0, tok_embed = 0
            zero_tok = torch.zeros_like(tok_embed)
            feat_vac, _ = model.readout(zero_field, zero_tok, return_diag=False)
            l_vac = F.cross_entropy(model.decoder(feat_vac), tgt_id).item()
            losses["2. Absolute Vacuum (F=0, tok_embed=0)"].append(l_vac)

            # 3. Pure Zero Field: F = 0, tok_embed = embed(x_t)
            feat_zero, _ = model.readout(zero_field, tok_embed, return_diag=False)
            l_zero = F.cross_entropy(model.decoder(feat_zero), tgt_id).item()
            losses["3. Pure Zero-Field (F=0, tok_embed=embed(x_t))"].append(l_zero)

            # 4. Memoryless Write 0-step: F_prior = 0, Write(0, x_t) -> Readout
            f_memless_w, _, _ = model.source(zero_field, inp_id)
            feat_w0, _ = model.readout(f_memless_w, tok_embed, return_diag=False)
            l_w0 = F.cross_entropy(model.decoder(feat_w0), tgt_id).item()
            losses["4. Memoryless Write 0-step (F_prior=0, write x_t, K=0)"].append(l_w0)

            # 5. Memoryless Standard: F_prior = 0, step(0, x_t, micro_steps=3)
            logits_memless_k3, _, _ = model.step(zero_field, inp_id, micro_steps=3)
            l_mem_k3 = F.cross_entropy(logits_memless_k3, tgt_id).item()
            losses["5. Memoryless Standard (F_prior=0, write x_t, K=3)"].append(l_mem_k3)

            # 6. Full Continuous Stream: normal step from ongoing physical state
            logits_phys, state, _ = model.step(state, inp_id, micro_steps=3)
            l_phys = F.cross_entropy(logits_phys, tgt_id).item()
            losses["6. Full Continuous Stream (Normal Physical Memory)"].append(l_phys)

            if (t - warmup_tokens + 1) % 256 == 0:
                print(f"  Processed {t - warmup_tokens + 1} / {num_tokens} tokens...")

    print("\n" + "=" * 90)
    print("      COMPREHENSIVE ZERO-FIELD AND MEMORYLESS CONTROLS (N = 1024 TOKENS)")
    print("=" * 90)
    print(f"{'Condition':<55} | {'Validation NLL':<16} | {'Delta vs Normal':<16}")
    print("-" * 90)

    means = {cond: float(np.mean(vals)) for cond, vals in losses.items()}
    ref_nll = means["6. Full Continuous Stream (Normal Physical Memory)"]

    for cond, m_val in means.items():
        delta = m_val - ref_nll
        delta_str = f"{delta:+.4f} nats" if cond != "6. Full Continuous Stream (Normal Physical Memory)" else "Reference (0.0)"
        print(f"{cond:<55} | {m_val:<16.4f} | {delta_str:<16}")

    print("=" * 90)

    out_file = Path("results/comprehensive_zero_field_controls.json")
    out_file.write_text(json.dumps(means, indent=2), encoding="utf-8")
    print(f"\nSaved report to {out_file}")


if __name__ == "__main__":
    main()
