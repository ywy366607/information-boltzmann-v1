"""Step 1: Backward-only Collision Tangent Diagnosis across H=32..1500.

Evaluates:
- Forward pass: 100% BIT-EXACT UNCHANGED.
- Backward pass: angle_input = self.norm(flat.detach()) in Collision operator.
- Measures ||grad_theta(H)|| across H in {32, 64, 128, 256, 512, 768, 1024, 1500}.
"""
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(script_dir))
if sys.path and os.path.abspath(sys.path[0]) == script_dir:
    sys.path.pop(0)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import argparse
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from scripts.ib_local.cbim_torus3d import CBIMTorus3D


def patch_collision_tangent_decoupling(model):
    """Detach field from angle_input inside collision: Q(theta(F)) F becomes strictly orthogonal in tangent space."""
    orig_collision_forward = model.collision.forward

    def decoupled_collision_forward(field, delta_tau=1.0):
        batch = field.shape[0]
        flat = field.reshape(batch, -1, model.collision.d)
        nullspace = model.collision.nullspace.to(dtype=flat.dtype)
        coefficient = torch.einsum("dk,bnd->bnk", nullspace, flat)
        conserved = flat - torch.einsum("dk,bnk->bnd", nullspace, coefficient)

        # DETACH flat from angle_input to make tangent map strictly orthogonal (sigma = 1.0000)
        angle_input = model.collision.norm(flat.detach())
        if model.collision.position_conditioned:
            position = model.collision.position_features.to(flat)[None].expand(batch, -1, -1)
            angle_input = torch.cat((angle_input, position), -1)
        angles = model.collision.angle(angle_input).reshape(
            batch, flat.shape[1], model.collision.layers, model.collision.nullity // 2)

        if isinstance(delta_tau, torch.Tensor):
            dt = delta_tau.view(batch, 1, 1, 1)
        else:
            dt = float(delta_tau)
        scaled_angles = angles * dt
        value = coefficient
        for layer in range(model.collision.layers):
            pair = model.collision.schedules[layer]
            left, right = value[..., pair[:, 0]], value[..., pair[:, 1]]
            theta = scaled_angles[:, :, layer]
            cosine, sine = theta.cos(), theta.sin()
            updated = value.clone()
            updated[..., pair[:, 0]] = cosine * left - sine * right
            updated[..., pair[:, 1]] = sine * left + cosine * right
            value = updated
        output = conserved + torch.einsum("dk,bnk->bnd", nullspace, value)
        coll_in_power = coefficient.square().sum(-1).mean()
        cons_in_power = conserved.square().sum(-1).mean()
        coll_out_power = value.square().sum(-1).mean()
        return output.reshape_as(field), {
            "collision_angle_abs_mean": scaled_angles.detach().abs().mean(),
            "collision_angle_abs_max": scaled_angles.detach().abs().amax(),
            "collision_input_snr": (coll_in_power / cons_in_power.clamp_min(1e-8)).detach(),
            "collision_output_snr": (coll_out_power / cons_in_power.clamp_min(1e-8)).detach(),
        }

    model.collision.forward = decoupled_collision_forward


def flatten_grads(parameters):
    grads = []
    for p in parameters:
        if p.grad is not None:
            grads.append(p.grad.detach().flatten())
        else:
            grads.append(torch.zeros_like(p).flatten())
    return torch.cat(grads)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("results/cbim_three_clock_bptt128_8x8x4_k3_3000/BBest.pt"))
    parser.add_argument("--data", type=Path, default=Path("data/ib_owt_gpt2/validation.npy"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/published/cbim_collision_tangent_diagnosis_h1500.json"))
    parser.add_argument("--warmup", type=int, default=256)
    parser.add_argument("--t-ref", type=int, default=1500)
    parser.add_argument("--chunk-size", type=int, default=16)
    args = parser.parse_args()

    print(f"Loading champion model from {args.checkpoint}...", flush=True)
    saved = torch.load(args.checkpoint, map_location="cuda")
    cfg = saved["config"]

    model = CBIMTorus3D(
        shape=tuple(cfg["shape"]),
        velocities=cfg["velocities"],
        content_dim=cfg["content_dim"],
        v2_coordinate_components=True,
        readout_type=cfg.get("readout_type", "kernel_r1"),
        write_type=cfg.get("write_type", "w2_impedance"),
        micro_steps=3,
        adaptive_clock=True,
        continuous_velocities=True,
        dissipation_type="unified",
        dissipation_rank=4,
        three_clock=True,
        tau_mem=3.0,
        nu_s_init=0.020,
        decouple_source_feedback=True
    ).cuda()
    model.load_state_dict(saved["model"])
    model.eval()

    # Apply collision tangent decoupling
    patch_collision_tangent_decoupling(model)

    val_data = np.load(args.data, mmap_mode="r")
    mature_state_base = saved["state"].detach().cuda()

    T_ref = args.t_ref
    start_offset = 8192
    chunk_size = args.chunk_size

    print(f"Running {args.warmup}-token warmup...", flush=True)
    state = mature_state_base.clone()
    with torch.no_grad():
        for t in range(args.warmup):
            inp = torch.as_tensor([val_data[start_offset + t]], dtype=torch.long, device="cuda")
            _, state, _ = model.step(state, inp, micro_steps=3)

    seq_start = start_offset + args.warmup
    seq = torch.as_tensor(val_data[seq_start:seq_start + T_ref + 2].copy(), dtype=torch.long, device="cuda")

    print(f"Collecting forward trajectory for {T_ref} tokens...", flush=True)
    chunk_states = {0: state.clone()}
    curr = state.clone()
    with torch.no_grad():
        for step_i in range(T_ref):
            inp = seq[step_i:step_i+1]
            _, curr, _ = model.step(curr, inp, micro_steps=3)
            if (step_i + 1) % chunk_size == 0 or (step_i + 1) == T_ref:
                chunk_states[step_i + 1] = curr.clone()

    def run_chunk(s, chunk_toks):
        for tok in chunk_toks:
            _, s, _ = model.step(s, tok.unsqueeze(0), micro_steps=3)
        return s

    def compute_grad(H):
        start_step = T_ref - H
        cp_step = (start_step // chunk_size) * chunk_size
        s_init = chunk_states[cp_step].clone()

        if start_step > cp_step:
            with torch.no_grad():
                for i in range(cp_step, start_step):
                    _, s_init, _ = model.step(s_init, seq[i:i+1], micro_steps=3)

        model.zero_grad(set_to_none=True)
        curr = s_init.clone()

        for st in range(start_step, T_ref, chunk_size):
            end = min(st + chunk_size, T_ref)
            curr = cp.checkpoint(run_chunk, curr, seq[st:end], use_reentrant=False)

        last_tok = seq[T_ref - 1:T_ref]
        tok_embed = model.source.embedding(last_tok)
        feat, _ = model.readout(curr, tok_embed)
        logits = model.decoder(feat)
        loss = F.cross_entropy(logits, seq[T_ref:T_ref+1])
        loss.backward()

        return flatten_grads(model.parameters()), float(loss.item())

    # Reference at H=1500
    print(f"\nComputing reference gradient at H = {T_ref} with decoupled collision tangent...", flush=True)
    t0 = time.perf_counter()
    grad_ref, loss_ref = compute_grad(T_ref)
    norm_ref = float(grad_ref.norm().item())
    print(f"Reference H={T_ref} computed in {time.perf_counter() - t0:.2f}s: Norm = {norm_ref:.4e}, Loss = {loss_ref:.4f}", flush=True)

    horizons = [32, 64, 128, 256, 512, 768, 1024, 1500]
    results = {}

    print("\n" + "=" * 105)
    print("   COLLISION TANGENT DECOUPLING DIAGNOSIS: GRADIENT CONVERGENCE vs H=1500")
    print("=" * 105)
    print(f"{'Horizon H':<12} | {'Grad Norm':<14} | {'Norm Ratio (H/1500)':<22} | {'Cosine (c_H)':<16} | {'Rel Error (delta_H)':<22} | {'Alignment'}")
    print("-" * 105)

    for H in horizons:
        if H == T_ref:
            grad_h = grad_ref
            norm_h = norm_ref
        else:
            grad_h, _ = compute_grad(H)
            norm_h = float(grad_h.norm().item())

        dot = float(torch.dot(grad_h, grad_ref).item())
        cos_sim = dot / (norm_h * norm_ref + 1e-12)
        diff_norm = float((grad_h - grad_ref).norm().item())
        rel_err = diff_norm / (norm_ref + 1e-12)
        norm_ratio = norm_h / (norm_ref + 1e-12)

        alignment = ">= 99% Exact" if cos_sim >= 0.99 else (">= 95% Strong" if cos_sim >= 0.95 else (">= 80% Moderate" if cos_sim >= 0.80 else "Weak / Distorted"))
        print(f"{H:<12d} | {norm_h:<14.4e} | {norm_ratio * 100:<20.2f}% | {cos_sim:<16.4f} | {rel_err * 100:<20.2f}% | {alignment}", flush=True)

        results[str(H)] = {
            "H": int(H),
            "grad_norm": norm_h,
            "norm_ratio": norm_ratio,
            "cosine_similarity": cos_sim,
            "rel_error": rel_err,
        }

    print("=" * 105)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved diagnosis report to {args.output}")


if __name__ == "__main__":
    main()
