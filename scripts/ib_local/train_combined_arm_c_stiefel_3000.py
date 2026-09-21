"""CBIM-Sudoku Combined Model: Arm C Corrective Flow + Newton-Schulz Stiefel Polar Decomposition.
Synthesizes:
1. Steering: Arm C Learned Isoenergetic Corrective Flow on Riemannian Tangent Space steers the state towards puzzle constraints.
2. Stability: Newton-Schulz 5th-order Stiefel Polar Decomposition orthonormalizes the 8 velocity channels in each cell (St(8, 32)), preventing channel collapse and unconstrained MLP distortion.
3. Transport: Exact Complex Exponential Spectral Advection exp(-i * dt * omega) on 9x9 Torus.
4. Scale: Time-Step Continuum Scaling (dt = 1.0 / K).
5. Macrostep H: 2-Port State-and-Problem Aware Scattering theta(F, P) + Quadratic Passive Bath.

Trained on ONLY K in [1, 2] for 3,000 steps (Cosine LR, Never Reset).
Evaluates zero-shot extrapolation across K in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512].
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
import math
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.ib_local.cbim_sudoku_unified_dissipative import (
    ContextualBoundaryWrite2Port,
    QuadraticPassiveRadiationBath
)
from scripts.ib_local.train_stage1_dt_scaling_3000 import ContinuousConditionedLieCollision
from scripts.ib_local.train_stage2_exact_exponential_3000 import ExactExponentialTorusTransport
from scripts.ib_local.cbim_sudoku_stiefel_unified import newton_schulz
from scripts.ib_local.train_never_reset_unified_3000 import evaluate_never_reset_stream
from external.TinyRecursiveModels.adam_atan2 import AdamATan2


class CombinedArmCStiefelCBIMSudokuModel(nn.Module):
    """CBIM Sudoku Model Combining Arm C Corrective Flow AND Newton-Schulz Stiefel Projection."""

    def __init__(self, vocab_size: int = 11, d_channels: int = 256, alpha_max: float = 0.25):
        super().__init__()
        self.vocab_size = vocab_size
        self.d = d_channels
        self.n_v = 8
        self.d_c = d_channels // self.n_v
        self.alpha_max = float(alpha_max)

        self.embed_tokens = nn.Embedding(vocab_size, d_channels)
        self.clue_proj = nn.Sequential(
            nn.Linear(d_channels, d_channels),
            nn.SiLU(),
            nn.Linear(d_channels, d_channels)
        )

        # 1. H-Boundary 2-Port State-Aware Scattering theta(F, P)
        self.boundary_write = ContextualBoundaryWrite2Port(d=d_channels)

        # 2. Exact Complex Exponential Transport on 9x9 Torus
        self.transport = ExactExponentialTorusTransport(n_v=self.n_v, d_c=self.d_c)

        # 3. Continuous Lie Collision (Synchronized with dt!)
        self.collision = ContinuousConditionedLieCollision(d=d_channels)

        # 4. Arm C Corrective Flow (Active Steering)
        self.corrective_net = nn.Sequential(
            nn.Linear(2 * d_channels, d_channels),
            nn.SiLU(),
            nn.Linear(d_channels, d_channels)
        )
        self.angle_gate = nn.Sequential(
            nn.Linear(2 * d_channels, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )
        nn.init.zeros_(self.corrective_net[-1].weight)
        nn.init.zeros_(self.corrective_net[-1].bias)
        nn.init.zeros_(self.angle_gate[-1].weight)
        nn.init.constant_(self.angle_gate[-1].bias, -2.0)

        # 5. H-Boundary Quadratic Passive Bath
        self.bath = QuadraticPassiveRadiationBath(d=d_channels)

        # 6. Readout
        self.readout_mlp = nn.Sequential(
            nn.Linear(d_channels, 256),
            nn.SiLU(),
            nn.Linear(256, vocab_size)
        )

    def forward_stream_step(self, persistent_state: torch.Tensor, inp: torch.Tensor, k_step: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        b = inp.shape[0]
        clue_field = self.clue_proj(self.embed_tokens(inp).view(b, 9, 9, self.d))

        # 1. Macro-Step H Boundary: 2-Port Unitary Scattering theta(F, P)
        f_absorbed, reflected, write_diag = self.boundary_write(persistent_state, clue_field)
        curr = f_absorbed
        orig_norm = torch.linalg.vector_norm(curr.float(), dim=(1, 2, 3), keepdim=True).to(curr.dtype)

        dt = 1.0 / max(1, k_step)

        # 2. Microstep Loop K: Transport -> Collision -> Arm C Steering -> Stiefel Newton-Schulz Projection!
        for _ in range(k_step):
            # Step A: Exact Exponential Transport
            f_5d = curr.view(b, 9, 9, self.n_v, self.d_c)
            f_tr = self.transport(f_5d, dt=dt).view(b, 9, 9, self.d)

            # Step B: Problem-Conditioned Lie Collision
            f_star = self.collision(f_tr, cond=clue_field, dt=dt)

            # Step C: Arm C Corrective Flow (Steers towards constraint satisfaction)
            cat_in = torch.cat([f_star, clue_field], dim=-1)
            v_k = self.corrective_net(cat_in)
            dot_p = torch.sum(f_star.float() * v_k.float(), dim=(1, 2, 3), keepdim=True)
            norm_sq = torch.sum(f_star.float() ** 2, dim=(1, 2, 3), keepdim=True) + 1e-8
            u_k = v_k.float() - (dot_p / norm_sq) * f_star.float()
            u_hat = u_k / (torch.linalg.vector_norm(u_k, dim=(1, 2, 3), keepdim=True) + 1e-8)

            gate_val = torch.sigmoid(torch.mean(self.angle_gate(cat_in), dim=(1, 2, 3), keepdim=True))
            alpha_k = (dt * self.alpha_max) * gate_val.to(curr.dtype)

            f_norm_step = torch.linalg.vector_norm(f_star.float(), dim=(1, 2, 3), keepdim=True)
            f_steered = torch.cos(alpha_k) * f_star + f_norm_step * torch.sin(alpha_k) * u_hat.to(curr.dtype)

            # Step D: Newton-Schulz Stiefel Polar Projection on [B, 9, 9, 8, 32] (Guarantees velocity orthogonality & compact stability!)
            f_5d_steered = f_steered.view(b, 9, 9, self.n_v, self.d_c)
            f_stiefel = newton_schulz(f_5d_steered, steps=5).view(b, 9, 9, self.d)

            # Scale to exact norm
            stiefel_norm = torch.linalg.vector_norm(f_stiefel.float(), dim=(1, 2, 3), keepdim=True) + 1e-8
            curr = f_stiefel * (orig_norm / stiefel_norm.to(curr.dtype))

        flat = curr.view(b, 81, self.d)
        logits = self.readout_mlp(flat)

        # 3. Macro-Step H Boundary Exit: Quadratic Radiation Bath
        state_next, bath_out, bath_diag = self.bath(curr)

        return state_next, logits, {**write_diag, **bath_diag}


def main():
    parser = argparse.ArgumentParser(description="Combined Arm C + Stiefel Polar Projection (3000 steps)")
    parser.add_argument("--steps", type=int, default=3000, help="Total training steps")
    parser.add_argument("--d-channels", type=int, default=256, help="Phase space feature channels")
    parser.add_argument("--micro-batch-size", type=int, default=16, help="Micro-batch size")
    parser.add_argument("--grad-accum-steps", type=int, default=2, help="Gradient accumulation steps (effective batch 32)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate")
    parser.add_argument("--min-lr", type=float, default=1e-5, help="Minimum learning rate")
    parser.add_argument("--eval-interval", type=int, default=500, help="Validation interval")
    parser.add_argument("--output-dir", type=str, default="results/combined_arm_c_stiefel_3000", help="Output directory")
    args = parser.parse_args()

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    metrics_log = out_path / "metrics.jsonl"

    print("=" * 110)
    print("   COMBINED EXPERIMENT: ARM C CORRECTIVE FLOW + NEWTON-SCHULZ STIEFEL PROJECTION")
    print(f"   Architecture: Arm C Active Steering + Stiefel Orthonormal Velocity Frame St(8, 32)")
    print(f"   Transport: Exact Complex Exponential exp(-i * dt * omega) | Time-Step Scaling: dt = 1.0 / K")
    print(f"   Training Horizons: STRICTLY K in [1, 2] | Steps: {args.steps} | Eff Batch: {args.micro_batch_size * args.grad_accum_steps}")
    print("=" * 110)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_path = Path("data/sudoku-extreme-1k-aug-100")
    train_in = np.load(data_path / "train" / "all__inputs.npy", mmap_mode="r")
    train_lbl = np.load(data_path / "train" / "all__labels.npy", mmap_mode="r")
    test_in = np.load(data_path / "test" / "all__inputs.npy")
    test_lbl = np.load(data_path / "test" / "all__labels.npy")
    num_train = len(train_in)

    model = CombinedArmCStiefelCBIMSudokuModel(vocab_size=11, d_channels=args.d_channels).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {num_params:,} ({num_params / 1e6:.3f}M)\n")

    optimizer = AdamATan2(model.parameters(), lr=args.lr, weight_decay=1.0)
    scaler = torch.amp.GradScaler("cuda")

    init_sample = torch.as_tensor(train_in[:args.micro_batch_size], dtype=torch.long, device=device)
    with torch.no_grad():
        persistent_state = model.clue_proj(model.embed_tokens(init_sample).view(args.micro_batch_size, 9, 9, args.d_channels))

    training_horizons = [1, 2]
    probs = [0.5, 0.5]
    rng = np.random.default_rng(42)

    best_loss = float("inf")
    best_cell_acc = 0.0
    start_time = time.perf_counter()

    for step in range(1, args.steps + 1):
        t0 = time.perf_counter()
        model.train()

        progress = step / args.steps
        current_lr = args.min_lr + 0.5 * (args.lr - args.min_lr) * (1.0 + math.cos(math.pi * progress))
        for g in optimizer.param_groups:
            g["lr"] = current_lr

        k_step = int(rng.choice(training_horizons, p=probs))
        accum_loss = 0.0
        diag_last = {}

        for _ in range(args.grad_accum_steps):
            idx = rng.integers(0, num_train, size=args.micro_batch_size)
            inp = torch.as_tensor(train_in[idx], dtype=torch.long, device=device)
            lbl = torch.as_tensor(train_lbl[idx], dtype=torch.long, device=device)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                state_next, logits, diag_last = model.forward_stream_step(persistent_state, inp, k_step=k_step)
                loss = F.cross_entropy(logits.view(-1, 11), lbl.view(-1))
                loss = loss / args.grad_accum_steps

            scaler.scale(loss).backward()
            accum_loss += loss.item() * args.grad_accum_steps

            with torch.no_grad():
                persistent_state.copy_(state_next.detach())

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        torch.cuda.synchronize()
        step_time_ms = (time.perf_counter() - t0) * 1000.0
        vram_mb = torch.cuda.max_memory_reserved() / (1024 ** 2)

        if step % 100 == 0 or step == 1:
            eff_batch = args.micro_batch_size * args.grad_accum_steps
            throughput = eff_batch / (step_time_ms / 1000.0)
            theta_m = diag_last.get("theta_mean", 0.0)
            cool_p = diag_last.get("cooling_power", 0.0)
            print(f"Step {step:4d}/{args.steps} | K: {k_step:2d} | Loss: {accum_loss:6.3f} | LR: {current_lr:.1e} | theta: {theta_m:.3f} | cool: {cool_p:.2e} | Time: {step_time_ms:5.1f}ms | Throughput: {throughput:5.1f} puz/s", flush=True)

            with open(metrics_log, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "kind": "train", "step": step, "k": k_step, "loss": accum_loss, "lr": current_lr,
                    "theta_mean": float(theta_m), "cooling_power": float(cool_p),
                    "step_time_ms": step_time_ms, "throughput": throughput, "vram_mb": vram_mb
                }) + "\n")

        # Periodic Zero-Shot Extrapolation Check
        if step % args.eval_interval == 0 or step == args.steps:
            val_results = evaluate_never_reset_stream(
                model, test_in, test_lbl,
                mature_state=persistent_state,
                horizons=[1, 2, 4, 8, 16, 32, 64],
                batch_size=32
            )
            print(f"\n[INTERIM COMBINED EXTRAPOLATION Step {step}] Trained ONLY on K in [1, 2]:", flush=True)
            for k in [1, 2, 4, 8, 16, 32, 64]:
                m = val_results[k]
                tag = " (In-Domain)" if k in [1, 2] else " (Zero-Shot Extrapolated!)"
                print(f"   K = {k:3d} | Cell Acc: {m['cell_accuracy']*100:5.2f}% | Loss: {m['mean_loss']:6.3f}{tag}", flush=True)

            acc_k16 = val_results[16]["cell_accuracy"]
            if acc_k16 > best_cell_acc:
                best_cell_acc = acc_k16
                torch.save({
                    "model": model.state_dict(),
                    "persistent_state": persistent_state.detach().cpu(),
                    "step": step,
                    "val_results": val_results
                }, out_path / "Best_Combined_ArmC_Stiefel_d256.pt")
                print(f"Saved new best model to {out_path / 'Best_Combined_ArmC_Stiefel_d256.pt'}\n", flush=True)

    total_time_min = (time.perf_counter() - start_time) / 60.0
    print(f"\nTraining completed in {total_time_min:.2f} minutes!")

    # Post-training full test-time zero-shot pondering sweep out to K=512
    print("\n" + "=" * 105)
    print("   POST-TRAINING FULL ZERO-SHOT PONDERING EXTRAPOLATION SWEEP (K in [1..512])")
    print("   (Model trained ONLY on K=1, 2! With COMBINED ARM C STEERING + STIEFEL ORTHOGONALITY!)")
    print("=" * 105)
    final_sweep = evaluate_never_reset_stream(
        model, test_in, test_lbl,
        mature_state=persistent_state,
        horizons=[1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512],
        batch_size=32
    )
    for k, m in final_sweep.items():
        tag = " (In-Domain)" if k in [1, 2] else " (Zero-Shot Extrapolation!)"
        print(f"  Ponder Depth K = {k:3d} | Exact Acc: {m['exact_accuracy']*100:5.2f}% | Cell Acc: {m['cell_accuracy']*100:5.2f}% | Loss: {m['mean_loss']:6.3f}{tag}")

    with open(out_path / "final_report.json", "w", encoding="utf-8") as f:
        json.dump({
            "mechanism": "Combined Arm C Corrective Flow + Newton-Schulz Stiefel Polar Projection",
            "trained_horizons": [1, 2], "steps": args.steps, "d_channels": args.d_channels,
            "total_time_min": total_time_min, "final_sweep": final_sweep
        }, f, indent=2)


if __name__ == "__main__":
    main()
