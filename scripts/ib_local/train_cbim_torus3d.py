"""Train the pure T^3 CBIM with the current d128 mechanisms."""
from __future__ import annotations

import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(script_dir))
if sys.path and os.path.abspath(sys.path[0]) == script_dir:
    sys.path.pop(0)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import shutil

import numpy as np
import torch

from scripts.ib_local.cbim_torus3d import CBIMTorus3D
from scripts.ib_local.train_cbim_malecns_internal_time import (
    TruncatedInternalTimeGraphTrainer,
)


def atomic_json(path: Path, data, required=False):
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    payload = json.dumps(data, allow_nan=False)
    for attempt in range(60):
        try:
            temporary.write_text(payload, encoding="utf-8")
            os.replace(temporary, path)
            return True
        except PermissionError:
            time.sleep(0.05 * min(attempt + 1, 4))
    if required:
        raise PermissionError(f"Unable to update {path}")
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/ib_owt_gpt2"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/cbim_torus3d_d128_3000"))
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--shape", type=int, nargs=3, default=(4, 4, 4))
    parser.add_argument("--velocities", type=int, default=8, choices=[8, 27],
                        help="Number of discrete velocities (8 for D3Q8, 27 for D3Q27)")
    parser.add_argument("--content-dim", type=int, default=16)
    parser.add_argument("--collision-layers", type=int, default=2)
    parser.add_argument("--v2-coordinate-components", action="store_true",
                        help="Change only write addressing; retain v2 collision, bath, and readout coordinates")
    parser.add_argument("--relative-address", action="store_true", default=False,
                        help="Use relative coordinate displacement for write")
    parser.add_argument("--readout-type", type=str, default="baseline",
                        choices=["baseline", "dynamic_linear", "kernel_r1", "kernel_r2"],
                        help="Readout probe mechanism (Arm A baseline, Arm B dynamic linear, Arm C kernel R=1, Arm D kernel R=2)")
    parser.add_argument("--readout-probes", type=int, default=8,
                        help="Number of probes for CharacteristicKernelReadout")
    parser.add_argument("--readout-rounds", type=int, default=1,
                        help="Number of recurrent controller rounds for CharacteristicKernelReadout")
    parser.add_argument("--write-type", type=str, default="w0_baseline",
                        choices=["w0_baseline", "w1_unbounded", "w2_impedance", "w3_additive"],
                        help="Boundary write operator mechanism")
    parser.add_argument("--micro-steps", type=int, default=1,
                        help="Number of internal kinetic micro-steps per external token")
    parser.add_argument("--adaptive-clock", action="store_true", default=False,
                        help="Adaptive internal clock rate per micro-step bounded by causal horizon")
    parser.add_argument("--continuous-velocities", action="store_true", default=False,
                        help="Continuous adaptive velocity directions on S^2 (Run 3)")
    parser.add_argument("--dissipation-type", type=str, default="unified",
                        choices=["unified", "quadratic"],
                        help="Dissipation mechanism (unified 3-layer operator or legacy quadratic bath)")
    parser.add_argument("--dissipation-rank", type=int, default=4,
                        help="Subspace rank for content-selective forgetting (R=4 or 8)")
    parser.add_argument("--gamma0-init", type=float, default=0.010,
                        help="Initial scalar base leakage rate gamma_0")
    parser.add_argument("--nu-init", type=float, default=0.020,
                        help="Initial spectral viscosity coefficient nu")
    parser.add_argument("--three-clock", action="store_true", default=False,
                        help="CBIM Three-Clock v1 architecture (Continuous Write -> (TC)^K -> Readout -> D_mem)")
    parser.add_argument("--tau-mem", type=float, default=3.0,
                        help="Token memory retention time scale tau_mem for D_mem")
    parser.add_argument("--nu-s-init", type=float, default=0.020,
                        help="Initial source spectral viscosity nu_s merged into packet spectrum")
    parser.add_argument("--alpha-causal", type=float, default=0.90,
                        help="Lower bound on clock rate to ensure causal horizon coverage")
    parser.add_argument("--alpha-max", type=float, default=2.50,
                        help="Upper bound on clock rate")
    parser.add_argument("--chunk-tokens", type=int, default=None,
                        help="BPTT chunk size for CUDAGraph unrolling (defaults to 32 if micro_steps > 1 else tokens)")
    parser.add_argument("--validate-every", type=int, default=250)
    parser.add_argument("--validation-tokens", type=int, default=4096)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    args.shape = tuple(args.shape)

    torch.set_num_threads(2)
    torch.manual_seed(11)
    args.output.mkdir(parents=True, exist_ok=True)
    dashboard = Path("present/cbim_torus3d_live.html")
    if dashboard.exists():
        shutil.copy2(dashboard, args.output / "index.html")
    if (args.output / "last.pt").exists() and args.resume is None:
        raise ValueError("Existing run requires --resume or a new output directory")
    train = np.load(args.data / "train.npy", mmap_mode="r")
    valid = np.load(args.data / "validation.npy", mmap_mode="r")
    if args.steps * args.tokens + 1 > len(train):
        raise ValueError("Training stream is shorter than requested budget")

    model = CBIMTorus3D(
        shape=args.shape, velocities=args.velocities, content_dim=args.content_dim,
        collision_layers=args.collision_layers,
        relative_address=args.relative_address,
        v2_coordinate_components=args.v2_coordinate_components,
        readout_type=args.readout_type,
        readout_probes=args.readout_probes,
        readout_rounds=args.readout_rounds,
        write_type=args.write_type,
        micro_steps=args.micro_steps,
        adaptive_clock=args.adaptive_clock,
        continuous_velocities=args.continuous_velocities,
        alpha_causal=args.alpha_causal,
        alpha_max=args.alpha_max,
        dissipation_type=args.dissipation_type,
        dissipation_rank=args.dissipation_rank,
        gamma0_init=args.gamma0_init,
        nu_init=args.nu_init,
        three_clock=args.three_clock,
        tau_mem=args.tau_mem,
        nu_s_init=args.nu_s_init).cuda()
    chunk_tokens = args.chunk_tokens or args.tokens
    runner = TruncatedInternalTimeGraphTrainer(
        model, tokens=args.tokens, chunk_tokens=chunk_tokens)
    channels = model.d
    sources = [Path(__file__), Path("scripts/ib_local/cbim_torus3d.py"), Path("scripts/ib_local/readout_probes.py")]
    config = {
        "architecture": model.architecture,
        "three_clock": args.three_clock,
        "tau_mem": args.tau_mem,
        "nu_s_init": args.nu_s_init,
        "data": str(args.data), "output": str(args.output),
        "steps": args.steps, "tokens": args.tokens, "shape": args.shape,
        "velocities": args.velocities, "content_dim": args.content_dim,
        "channels": channels, "collision_layers": args.collision_layers,
        "grid_points": math.prod(args.shape), "bptt_chunk_tokens": chunk_tokens,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "precision": "FP32", "seed": 11, "lr": 3e-4,
        "validate_every": args.validate_every,
        "validation_tokens": args.validation_tokens,
        "boundary": "periodic T^3", "anatomy": None,
        "relative_address": args.relative_address,
        "v2_coordinate_components": args.v2_coordinate_components,
        "readout_type": args.readout_type,
        "readout_probes": args.readout_probes,
        "readout_rounds": args.readout_rounds,
        "write_type": args.write_type,
        "micro_steps": args.micro_steps,
        "adaptive_clock": args.adaptive_clock,
        "continuous_velocities": args.continuous_velocities,
        "alpha_causal": args.alpha_causal,
        "alpha_max": args.alpha_max,
        "dissipation_type": args.dissipation_type,
        "dissipation_rank": args.dissipation_rank,
        "gamma0_init": args.gamma0_init,
        "nu_init": args.nu_init,
        "stopping": "fixed update budget; convergence not established",
        "manifest": json.loads((args.data / "manifest.json").read_text()),
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sources},
    }
    atomic_json(args.output / "config.json", config, required=True)
    atomic_json(args.output / "progress.json", {
        "status": "capturing", "step": 0, "target_steps": args.steps})
    step, best = 0, float("inf")
    if args.resume:
        saved = torch.load(args.resume, map_location="cuda", weights_only=False)
        for key in ("architecture", "shape", "channels", "tokens"):
            if saved["config"][key] != config[key]:
                raise ValueError(f"Resume mismatch: {key}")
        model.load_state_dict(saved["model"])
        runner.optimizer.load_state_dict(saved["optimizer"])
        runner.state.copy_(saved["state"])
        step, best = saved["step"], saved["best_validation_nll"]

    def batch(data, offset):
        return tuple(torch.as_tensor(np.array(
            data[offset + shift:offset + shift + args.tokens]),
            dtype=torch.long, device="cuda")[None] for shift in (0, 1))

    def save(name):
        model.set_ness_prior(runner.state)
        temporary = args.output / f"{name}.{os.getpid()}.tmp"
        torch.save({"model": model.state_dict(),
                    "optimizer": runner.optimizer.state_dict(),
                    "state": runner.state.detach(), "step": step,
                    "events": step * args.tokens,
                    "best_validation_nll": best, "config": config}, temporary)
        os.replace(temporary, args.output / name)

    def log(row):
        with (args.output / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row), flush=True)

    @torch.no_grad()
    def validate():
        # Thermodynamic NESS Random-Phase Warm Start:
        # Matches exact steady-state energy E_NESS (no cold vacuum shock) while phase-randomizing to eliminate train-leak.
        if step > 0:
            model.set_ness_prior(runner.state)
            state = model.initial_state(1, "cuda", warm_start=True)
            total = 0.0
            for offset in range(0, args.validation_tokens, args.tokens):
                ids, targets = batch(valid, offset)
                loss, state, _ = model(ids, targets, state)
                total += float(loss) * args.tokens
            return total / args.validation_tokens
        else:
            state, total = model.initial_state(1, "cuda", warm_start=False), 0.0
            for offset in range(0, args.validation_tokens + args.tokens, args.tokens):
                ids, targets = batch(valid, offset)
                loss, state, _ = model(ids, targets, state)
                if offset:
                    total += float(loss) * args.tokens
            return total / args.validation_tokens

    try:
        if args.resume is None:
            best = validate()
            log({"kind": "validation", "step": 0, "validation_nll": best})
            save("BBest.pt"); save("last.pt")
        while step < args.steps:
            ids, targets = batch(train, step * args.tokens)
            torch.cuda.synchronize(); started = time.perf_counter()
            loss, state, diagnostics = runner.step(ids, targets)
            torch.cuda.synchronize(); elapsed = time.perf_counter() - started
            step += 1
            if not math.isfinite(float(loss)):
                raise FloatingPointError("Non-finite training loss")
            if elapsed > 15.0 and step > 2:
                raise RuntimeError(f"Training step exceeded 15.0 seconds: {elapsed:.3f}")
            if step == 1 or step % 10 == 0 or step == args.steps:
                field = state.detach()
                if hasattr(model.readout, "attention_weights"):
                    attention = model.readout.attention_weights(field)
                    attn_entropy = float(
                        -(attention * attention.clamp_min(1e-12).log()).sum(-1).mean())
                    attn_heads_json = attention[0].cpu().tolist()
                else:
                    attn_entropy = float(diagnostics.get("read_attention_entropy", 0.0))
                    attn_heads_json = []
                row = {
                    "kind": "train", "step": step, "events": step * args.tokens,
                    "nll": float(loss), "seconds": elapsed,
                    "energy": float(diagnostics["energy"]),
                    "incident_energy": float(diagnostics["incident_energy"]),
                    "reflected_energy": float(diagnostics["reflected_energy"]),
                    "accepted_energy": float(diagnostics["accepted_energy"]),
                    "accepted_fraction": float(diagnostics["accepted_fraction"]),
                    "t_packet": float(diagnostics.get("t_packet", diagnostics["accepted_fraction"])),
                    "delta_e_field": float(diagnostics.get("delta_e_field", diagnostics["accepted_energy"])),
                    "cross_interference": float(diagnostics.get("cross_interference", 0.0)),
                    "write_to_f_ratio": float(diagnostics.get("write_to_f_ratio", 0.0)),
                    "collision_input_snr": float(diagnostics.get("collision_input_snr", 0.0)),
                    "collision_output_snr": float(diagnostics.get("collision_output_snr", 0.0)),
                    "bath_out_energy": float(diagnostics["bath_out_energy"]),
                    "write_angle_abs_mean": float(diagnostics["write_angle_abs_mean"]),
                    "write_angle_peak_mean": float(diagnostics["write_angle_peak_mean"]),
                    "write_spatial_support": float(diagnostics["write_spatial_support"]),
                    "collision_angle_abs_mean": float(diagnostics["collision_angle_abs_mean"]),
                    "transport_norm_residual": float(diagnostics["transport_norm_residual"]),
                    "read_attention_entropy": attn_entropy,
                    "alpha_1": float(diagnostics["alpha_1"]) if "alpha_1" in diagnostics else None,
                    "alpha_2": float(diagnostics["alpha_2"]) if "alpha_2" in diagnostics else None,
                    "alpha_3": float(diagnostics["alpha_3"]) if "alpha_3" in diagnostics else None,
                    "delta_tau_total": float(diagnostics["delta_tau_total"]) if "delta_tau_total" in diagnostics else None,
                    "collision_exposure": float(diagnostics["collision_exposure"]) if "collision_exposure" in diagnostics else None,
                    "dir_disp_deg": float(diagnostics["dir_disp_deg"]) if "dir_disp_deg" in diagnostics else None,
                    "dir_pairwise_sep_deg": float(diagnostics["dir_pairwise_sep_deg"]) if "dir_pairwise_sep_deg" in diagnostics else None,
                    "dir_change_micro_deg": float(diagnostics["dir_change_micro_deg"]) if "dir_change_micro_deg" in diagnostics else None,
                    "grad_norm": float(runner.grad_norm),
                    "grad_norm_lexical": float(runner.group_grad_norms[0]),
                    "grad_norm_dynamics": float(runner.group_grad_norms[1]),
                    "variance_x": float(field.var(1, unbiased=False).mean()),
                    "variance_y": float(field.var(2, unbiased=False).mean()),
                    "variance_z": float(field.var(3, unbiased=False).mean()),
                    "allocated_mib": torch.cuda.memory_allocated() / 2**20,
                    "reserved_mib": torch.cuda.memory_reserved() / 2**20,
                }
                log(row)
                atomic_json(args.output / "progress.json", {
                    "status": "running", "target_steps": args.steps, **row})
                atomic_json(args.output / "live_state.json", {
                    **row, "run_id": args.output.name,
                    "volume": field[0].square().mean(-1).sqrt().cpu().tolist(),
                    "write_center": diagnostics["source_center"][0].cpu().tolist(),
                    "attention_heads": attn_heads_json,
                })
            if step % args.validate_every == 0 or step == args.steps:
                score = validate(); improved = score < best
                if improved: best = score
                log({"kind": "validation", "step": step,
                     "validation_nll": score, "best_validation_nll": best})
                if improved: save("BBest.pt")
                # BBest and last are the durable checkpoints.  Periodic age
                # snapshots duplicate them and can exhaust small experiment
                # volumes during long runs.
                save("last.pt")
        atomic_json(args.output / "progress.json", {
            "status": "complete", "step": step,
            "target_steps": args.steps, "best_validation_nll": best})
    except BaseException as error:
        atomic_json(args.output / "progress.json", {
            "status": "failed", "step": step, "error": str(error)})
        raise


if __name__ == "__main__":
    main()
