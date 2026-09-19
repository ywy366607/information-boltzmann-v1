"""Comprehensive diagnostic: Velocity head steering jitter & cold bath reverberation."""
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
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from scripts.ib_local.cbim_torus3d import CBIMTorus3D


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("results/cbim_torus3d_w2_16ch_k3_adaptive_continuous_q8_3000/BBest.pt"))
    parser.add_argument("--data", type=Path, default=Path("data/ib_owt_gpt2"))
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=256)
    args = parser.parse_args()

    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = saved["config"]
    model = CBIMTorus3D(
        shape=tuple(cfg["shape"]), velocities=cfg["velocities"],
        content_dim=cfg["content_dim"],
        collision_layers=cfg.get("collision_layers", 2),
        relative_address=cfg.get("relative_address", False),
        v2_coordinate_components=cfg.get("v2_coordinate_components", True),
        readout_type=cfg.get("readout_type", "baseline"),
        readout_probes=cfg.get("readout_probes", 8),
        readout_rounds=cfg.get("readout_rounds", 1),
        write_type=cfg.get("write_type", "w0_baseline"),
        micro_steps=cfg.get("micro_steps", 1),
        adaptive_clock=cfg.get("adaptive_clock", False),
        continuous_velocities=cfg.get("continuous_velocities", False),
        alpha_causal=cfg.get("alpha_causal", 0.90),
        alpha_max=cfg.get("alpha_max", 2.50)
    ).cuda().eval()
    model.load_state_dict(saved["model"])

    valid = np.load(args.data / "validation.npy", mmap_mode="r")
    start = 8192
    eval_len = args.warmup + args.tokens

    # =========================================================================
    # PART 1: MEASURE STEERING ANGULAR JITTER (Microsteps vs Tokens)
    # =========================================================================
    print("=" * 70)
    print("PART 1: MEASURING VELOCITY HEAD STEERING JITTER")
    print("=" * 70, flush=True)

    state = model.initial_state(1, "cuda")
    micro_disps = []    # delta theta between microstep k and k+1
    token_disps = []    # delta theta between token t (microstep 3) and t+1 (microstep 1)
    base_disps = []     # angle from D3Q8 baseline
    all_dirs_saved = [] # [T, K, H, 3]

    last_micro3_dir = None

    for t in range(eval_len):
        tok_id = torch.as_tensor([valid[start + t]], dtype=torch.long, device="cuda")
        tok_embed = model.source.embedding(tok_id)
        field, _, _ = model.source(state, tok_id)

        dirs_this_tok = []
        for k in range(model.micro_steps):
            dir_k = model.direction_controller(field, tok_embed)  # [1, 8, 3]
            dirs_this_tok.append(dir_k)

            # Evolve field through micro-step
            alpha_k = model.clock(field, tok_embed) if model.adaptive_clock else 1.0
            dt_k = alpha_k * model.tau_0_tensor
            mult, _ = model.transport.multiplier(dt_k, direction=dir_k)
            field = model.transport.apply_multiplier(field, mult)
            field, _ = model.collision(field, dt_k)
            field, _ = model.bath(field, dt_k)

        state = field
        if t >= args.warmup:
            # Measure micro-step angular shifts: k=0->1, k=1->2
            for k in range(len(dirs_this_tok) - 1):
                cos_k = (dirs_this_tok[k] * dirs_this_tok[k + 1]).sum(dim=-1).clamp(-1.0, 1.0)
                deg_k = torch.rad2deg(torch.acos(cos_k)).mean().item()
                micro_disps.append(deg_k)

            # Measure token-to-token shift: t-1 (micro 3) -> t (micro 1)
            if last_micro3_dir is not None:
                cos_tok = (last_micro3_dir * dirs_this_tok[0]).sum(dim=-1).clamp(-1.0, 1.0)
                deg_tok = torch.rad2deg(torch.acos(cos_tok)).mean().item()
                token_disps.append(deg_tok)

            # Measure displacement from D3Q8 base
            base = model.direction_controller.base_dirs[None]
            cos_base = (dirs_this_tok[-1] * base).sum(dim=-1).clamp(-1.0, 1.0)
            deg_base = torch.rad2deg(torch.acos(cos_base)).mean().item()
            base_disps.append(deg_base)

            dirs_stack = torch.stack(dirs_this_tok, dim=1).cpu()  # [1, K, H, 3]
            all_dirs_saved.append(dirs_stack)

        last_micro3_dir = dirs_this_tok[-1]

    mean_micro_jitter = float(np.mean(micro_disps))
    max_micro_jitter = float(np.max(micro_disps))
    std_micro_jitter = float(np.std(micro_disps))
    mean_tok_shift = float(np.mean(token_disps))
    max_tok_shift = float(np.max(token_disps))
    std_tok_shift = float(np.std(token_disps))
    mean_base_disp = float(np.mean(base_disps))

    print(f"Displacement from D3Q8 baseline:   {mean_base_disp:.2f}°")
    print(f"Intra-token micro-step shift:      mean={mean_micro_jitter:.2f}°, std={std_micro_jitter:.2f}°, max={max_micro_jitter:.2f}°")
    print(f"Token-to-token steering shift:     mean={mean_tok_shift:.2f}°, std={std_tok_shift:.2f}°, max={max_tok_shift:.2f}°")

    # =========================================================================
    # PART 2: INTERVENTION ABLATIONS ON STEERING NOISE & MOMENTUM
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 2: INTERVENTIONS ON STEERING JITTER (NLL EVALUATION)")
    print("=" * 70, flush=True)

    # Helper function to evaluate model with custom direction policy
    def evaluate_direction_policy(policy_fn, desc):
        state = model.initial_state(1, "cuda")
        # Warmup
        for t in range(args.warmup):
            x = torch.as_tensor(np.array(valid[start + t:start + t + 1]), dtype=torch.long, device="cuda")[None]
            y = torch.as_tensor(np.array(valid[start + t + 1:start + t + 2]), dtype=torch.long, device="cuda")[None]
            loss, state, _ = model(x, y, state)

        # Evaluation with policy
        total_loss = 0.0
        policy_state = {}
        for t in range(args.tokens):
            tok_id = torch.as_tensor([valid[start + args.warmup + t]], dtype=torch.long, device="cuda")
            target_id = torch.as_tensor([valid[start + args.warmup + t + 1]], dtype=torch.long, device="cuda")
            tok_embed = model.source.embedding(tok_id)
            field, _, _ = model.source(state, tok_id)

            for k in range(model.micro_steps):
                raw_dir = model.direction_controller(field, tok_embed)
                dir_k = policy_fn(raw_dir, t, k, policy_state)
                alpha_k = model.clock(field, tok_embed) if model.adaptive_clock else 1.0
                dt_k = alpha_k * model.tau_0_tensor
                mult, _ = model.transport.multiplier(dt_k, direction=dir_k)
                field = model.transport.apply_multiplier(field, mult)
                field, _ = model.collision(field, dt_k)
                field, _ = model.bath(field, dt_k)

            if model.readout_type == "baseline":
                feat = model.readout(field)
            else:
                feat, _ = model.readout(field, tok_embed, return_diag=False)
            logits = model.decoder(feat)
            loss = F.cross_entropy(logits, target_id)
            total_loss += float(loss)
            state = field

        nll = total_loss / args.tokens
        print(f"  {desc:<45}: NLL = {nll:.4f}")
        return nll

    # Policies:
    # 0. Base: raw dynamic directions
    nll_base = evaluate_direction_policy(lambda raw, t, k, s: raw, "Policy 0: Base Dynamic (per-microstep)")

    # 1. Token-locked: microsteps 2 & 3 reuse microstep 1 direction
    def policy_token_locked(raw, t, k, s):
        if k == 0:
            s["tok_dir"] = raw
        return s["tok_dir"]
    nll_tok_locked = evaluate_direction_policy(policy_token_locked, "Policy 1: Token-Locked (freeze across microsteps)")

    # 2. Momentum EMA smoothing across tokens (beta=0.8, 0.9, 0.95)
    for beta in [0.5, 0.8, 0.9, 0.95]:
        def make_momentum_policy(b):
            def policy_momentum(raw, t, k, s):
                base = model.direction_controller.base_dirs[None]
                delta = raw - base
                if "momentum" not in s:
                    s["momentum"] = delta
                else:
                    s["momentum"] = b * s["momentum"] + (1.0 - b) * delta
                smoothed_u = base + s["momentum"]
                return F.normalize(smoothed_u, p=2, dim=-1, eps=1e-6)
            return policy_momentum
        evaluate_direction_policy(make_momentum_policy(beta), f"Policy 2: Momentum EMA (beta={beta})")

    # 3. Rigid D3Q8 lock: zero continuous delta
    def policy_rigid_d3q8(raw, t, k, s):
        return model.direction_controller.base_dirs[None]
    nll_rigid = evaluate_direction_policy(policy_rigid_d3q8, "Policy 3: Rigid D3Q8 Lock (delta = 0)")

    # =========================================================================
    # PART 3: COLD BATH DISSIPATION & MULTI-TURN ECHO REVERBERATION
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 3: COLD BATH DISSIPATION & WAVE REVERBERATION")
    print("=" * 70, flush=True)

    # 1. Steady-state energy balance over validation stream
    state = model.initial_state(1, "cuda")
    e_incident, e_accepted, e_bath_dissipated, e_field = [], [], [], []
    for t in range(args.warmup + 256):
        x = torch.as_tensor(np.array(valid[start + t:start + t + 1]), dtype=torch.long, device="cuda")[None]
        y = torch.as_tensor(np.array(valid[start + t + 1:start + t + 2]), dtype=torch.long, device="cuda")[None]
        loss, state, diag = model(x, y, state)
        if t >= args.warmup:
            e_incident.append(float(diag["incident_energy"]))
            e_accepted.append(float(diag["accepted_energy"]))
            e_bath_dissipated.append(float(diag["bath_out_energy"]))
            e_field.append(float(diag["energy"]))

    mean_incident = float(np.mean(e_incident))
    mean_accepted = float(np.mean(e_accepted))
    mean_dissipated = float(np.mean(e_bath_dissipated))
    mean_field_energy = float(np.mean(e_field))
    dissipation_balance = mean_dissipated / (mean_accepted + 1e-8)

    print(f"Mean Field Stored Energy E_field:   {mean_field_energy:.4f}")
    print(f"Mean Incident Energy E_incident:    {mean_incident:.4f}")
    print(f"Mean Accepted Write E_accepted:     {mean_accepted:.4f} ({mean_accepted / (mean_incident + 1e-8) * 100:.1f}%)")
    print(f"Mean Dissipated Energy E_bath:      {mean_dissipated:.4f}")
    print(f"Bath Dissipation Balance Ratio:     {dissipation_balance:.3f} (1.0 = exact steady-state equilibrium)")

    # 2. Impulse response & wave reverberation decay
    # Inject an impulse on the mature steady-state field, then feed silence (zero write) for 40 steps
    # and observe how fast the perturbation energy decays and if acoustic reflections appear.
    print("\nMeasuring Impulse Perturbation Echo Decay over 40 silent steps...")
    mature_state = state.clone()

    # Create an impulse perturbation
    impulse_token = torch.tensor([100], dtype=torch.long, device="cuda")
    field_perturbed, _, _ = model.source(mature_state.clone(), impulse_token)
    perturbation_initial = field_perturbed - mature_state
    init_pert_energy = perturbation_initial.square().sum().item()

    # Silent evolution (only TCB, no token write)
    f_silent = field_perturbed.clone()
    decay_curve = []
    neutral_tok_embed = model.source.embedding(torch.tensor([0], device="cuda"))

    for step_silent in range(40):
        # TCB evolution
        for k in range(model.micro_steps):
            alpha_k = model.clock(f_silent, neutral_tok_embed) if model.adaptive_clock else 1.0
            dt_k = alpha_k * model.tau_0_tensor
            dir_k = model.direction_controller(f_silent, neutral_tok_embed) if model.continuous_velocities else None
            mult, _ = model.transport.multiplier(dt_k, direction=dir_k)
            f_silent = model.transport.apply_multiplier(f_silent, mult)
            f_silent, _ = model.collision(f_silent, dt_k)
            f_silent, _ = model.bath(f_silent, dt_k)

        pert_current = f_silent - mature_state
        rel_energy = pert_current.square().sum().item() / (init_pert_energy + 1e-12)
        decay_curve.append(rel_energy)

    print("Impulse Perturbation Residual Energy E(t)/E(0):")
    for s in [1, 2, 3, 5, 8, 12, 16, 20, 30, 40]:
        print(f"  Step {s:2d}: {decay_curve[s-1] * 100:.2f}%")

    # Check for half-life
    half_life = None
    for idx, e in enumerate(decay_curve):
        if e <= 0.50:
            half_life = idx + 1
            break
    print(f"\nPerturbation Half-life t_1/2: {half_life} steps")

    # Output JSON summary
    report = {
        "steering_jitter": {
            "displacement_from_d3q8_deg": mean_base_disp,
            "microstep_shift_deg": {"mean": mean_micro_jitter, "std": std_micro_jitter, "max": max_micro_jitter},
            "token_shift_deg": {"mean": mean_tok_shift, "std": std_tok_shift, "max": max_tok_shift},
        },
        "jitter_intervention_nll": {
            "policy_0_base_dynamic": nll_base,
            "policy_1_token_locked": nll_tok_locked,
            "policy_3_rigid_d3q8": nll_rigid,
        },
        "bath_dissipation": {
            "mean_field_energy": mean_field_energy,
            "mean_incident_energy": mean_incident,
            "mean_accepted_energy": mean_accepted,
            "mean_dissipated_energy": mean_dissipated,
            "dissipation_balance_ratio": dissipation_balance,
            "half_life_steps": half_life,
            "residual_energy_curve": decay_curve,
        }
    }
    out_file = Path("results/published/cbim_direction_jitter_and_bath_diagnostic.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport saved to {out_file}")


if __name__ == "__main__":
    main()
