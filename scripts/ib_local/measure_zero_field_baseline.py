"""Measure zero-field control baseline L_zero-field.

Directly tests what NLL the model achieves when:
1. F = 0 (completely dead / erased field).
2. F = Gaussian white noise with matched norm.
3. F = Normal physical state (reference).
4. Uniform vocabulary baseline ln(50257) = 10.8249 nats.
5. Unigram frequency baseline.
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

    num_tokens = 512
    warmup_tokens = 128

    model = CBIMTorus3D(
        shape=(8, 8, 4), velocities=8, content_dim=16,
        v2_coordinate_components=True,
        readout_type="kernel_r1", write_type="w2_impedance",
        micro_steps=3, adaptive_clock=True, continuous_velocities=True,
        dissipation_type="unified", dissipation_rank=4
    ).cuda()
    model.load_state_dict(saved["model"], strict=True)
    model.eval()

    loss_zero = []
    loss_white_noise = []
    loss_physical = []
    loss_uniform = []

    uniform_nll = float(np.log(50257))

    with torch.no_grad():
        state = model.initial_state(1, "cuda", warm_start=True)
        # Warm up
        for offset in range(0, warmup_tokens, 128):
            ids = torch.as_tensor(np.array(val_data[offset:offset + 128]), dtype=torch.long, device="cuda")[None]
            targets = torch.as_tensor(np.array(val_data[offset + 1:offset + 129]), dtype=torch.long, device="cuda")[None]
            _, state, _ = model(ids, targets, state)

        for t in range(warmup_tokens, warmup_tokens + num_tokens):
            inp_id = torch.as_tensor([val_data[t]], dtype=torch.long, device="cuda")
            tgt_id = torch.as_tensor([val_data[t + 1]], dtype=torch.long, device="cuda")
            tok_embed = model.source.embedding(inp_id)

            # 1. Normal physical step
            logits_phys, state, diag = model.step(state, inp_id, micro_steps=3)
            loss_phys = F.cross_entropy(logits_phys, tgt_id).item()
            loss_physical.append(loss_phys)

            # 2. Zero-field control: F = 0
            zero_field = torch.zeros_like(state)
            feat_zero, _ = model.readout(zero_field, tok_embed, return_diag=False)
            logits_zero = model.decoder(feat_zero)
            loss_z = F.cross_entropy(logits_zero, tgt_id).item()
            loss_zero.append(loss_z)

            # 3. White noise field control: random Gaussian with matched norm
            field_norm = state.norm()
            noise_field = torch.randn_like(state)
            noise_field = noise_field * (field_norm / (noise_field.norm() + 1e-8))
            feat_noise, _ = model.readout(noise_field, tok_embed, return_diag=False)
            logits_noise = model.decoder(feat_noise)
            loss_noise = F.cross_entropy(logits_noise, tgt_id).item()
            loss_white_noise.append(loss_noise)

            loss_uniform.append(uniform_nll)

    mean_phys = float(np.mean(loss_physical))
    mean_zero = float(np.mean(loss_zero))
    mean_noise = float(np.mean(loss_white_noise))

    # Also inspect zero-field output entropy and top-1 probability
    with torch.no_grad():
        probs_zero = F.softmax(logits_zero, dim=-1)
        entropy_zero = -(probs_zero * (probs_zero + 1e-12).log()).sum(-1).item()
        top1_p_zero = probs_zero.max(-1).values.item()

    print("\n" + "=" * 70)
    print("           ZERO-FIELD AND BASELINE LANGUAGE CONTROLS")
    print("=" * 70)
    print(f"Uniform Vocabulary Floor (ln 50257):     {uniform_nll:.4f} nats")
    print(f"Zero-Field Readout L_zero-field (F = 0):  {mean_zero:.4f} nats")
    print(f"White Noise Field L_noise (|F| matched):  {mean_noise:.4f} nats")
    print(f"Normal Physical Field L_phys (K=3 step):  {mean_phys:.4f} nats")
    print("-" * 70)
    print(f"Zero-Field Information Gap (L_zero - L_phys):    {mean_zero - mean_phys:+.4f} nats")
    print(f"Zero-Field vs Pure Uniform (Uniform - L_zero):  {uniform_nll - mean_zero:+.4f} nats")
    print(f"Zero-Field Output Entropy H(p_0):              {entropy_zero:.4f} nats (max {uniform_nll:.4f})")
    print(f"Zero-Field Top-1 Probability:                  {top1_p_zero * 100:.2f}%")
    print("=" * 70)

    out = {
        "uniform_floor": uniform_nll,
        "l_zero_field": mean_zero,
        "l_white_noise": mean_noise,
        "l_physical": mean_phys,
        "information_gap": mean_zero - mean_phys,
        "tok_embed_prior_gain": uniform_nll - mean_zero,
        "zero_field_entropy": entropy_zero,
        "zero_field_top1_p": top1_p_zero,
    }
    out_path = Path("results/zero_field_baseline_control.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
