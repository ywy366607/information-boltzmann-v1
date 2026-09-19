"""Train and evaluate readout probes on frozen CBIM Torus3D physical field.

Ablation arms:
- arm_a_baseline: Original static linear readout on frozen physical field (Reference baseline).
- arm_b_dynamic_linear: Dynamic token query Q(x_t) + linear multi-head attention + token residual.
- arm_c_kernel_r1: Dynamic Characteristic Gaussian Kernel Readout (R=1 single round).
- arm_d_kernel_r2: Dynamic Characteristic Gaussian Kernel Readout with Recurrent Controller (R=2 rounds).
"""
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
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from scripts.ib_local.cbim_torus3d import CBIMTorus3D
from scripts.ib_local.readout_probes import (
    DynamicLinearReadout,
    CharacteristicKernelReadout,
)


def evaluate_baseline(
    base_model: CBIMTorus3D,
    valid_data: np.ndarray,
    val_tokens: int = 4096,
    tokens: int = 128,
) -> float:
    """Evaluates the original frozen checkpoint readout."""
    val_state = base_model.initial_state(1, "cuda")
    total_loss = 0.0
    n_chunks = val_tokens // tokens
    multiplier = base_model.transport.multiplier()[0]

    with torch.no_grad():
        for chunk_idx in range(n_chunks):
            offset = chunk_idx * tokens
            x = torch.as_tensor(
                np.array(valid_data[offset : offset + tokens]),
                dtype=torch.long,
                device="cuda",
            )[None]
            y = torch.as_tensor(
                np.array(valid_data[offset + 1 : offset + 1 + tokens]),
                dtype=torch.long,
                device="cuda",
            )[None]

            features = []
            for t in range(tokens):
                val_state, _, _ = base_model.source(val_state, x[:, t])
                val_state = base_model.transport.apply_multiplier(val_state, multiplier)
                val_state, _ = base_model.collision(val_state)
                val_state, _ = base_model.bath(val_state)
                features.append(base_model.readout(val_state))

            logits = base_model.decoder(torch.stack(features, dim=1))
            loss = F.cross_entropy(
                logits.reshape(-1, base_model.vocab_size), y.reshape(-1)
            )
            total_loss += float(loss) * tokens

    return total_loss / (n_chunks * tokens)


def run_probe_experiment(
    arm: str,
    checkpoint_path: Path,
    data_dir: Path,
    output_dir: Path,
    steps: int = 500,
    tokens: int = 128,
    lr: float = 3e-4,
    validate_every: int = 100,
    val_tokens: int = 4096,
):
    torch.manual_seed(11)
    torch.cuda.manual_seed(11)

    print(f"\n========================================================")
    print(f"Starting Frozen-Field Probe Experiment: Arm = {arm}")
    print(f"Base Checkpoint: {checkpoint_path}")
    print(f"Steps: {steps}, Tokens/Step: {tokens}, LR: {lr}")
    print(f"========================================================")

    # 1. Load base frozen model
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = saved["config"]
    base_model = CBIMTorus3D(
        shape=tuple(cfg["shape"]),
        velocities=cfg["velocities"],
        content_dim=cfg["content_dim"],
        collision_layers=cfg.get("collision_layers", 2),
        relative_address=cfg.get("relative_address", False),
        v2_coordinate_components=cfg.get(
            "v2_coordinate_components",
            cfg.get("architecture") == "CBIM-Torus3D-fullrank-d3q8-v2",
        ),
    ).cuda()
    base_model.load_state_dict(saved["model"])

    for param in base_model.parameters():
        param.requires_grad = False
    base_model.eval()

    train_data = np.load(data_dir / "train.npy", mmap_mode="r")
    valid_data = np.load(data_dir / "validation.npy", mmap_mode="r")
    multiplier = base_model.transport.multiplier()[0]

    if arm == "arm_a_baseline":
        val_nll = evaluate_baseline(base_model, valid_data, val_tokens, tokens)
        print(f"Arm A (Baseline) Validation NLL: {val_nll:.5f}")
        return {
            "arm": "arm_a_baseline",
            "trainable_parameters": 0,
            "steps": 0,
            "initial_val_nll": val_nll,
            "best_val_nll": val_nll,
            "final_val_nll": val_nll,
            "best_step": 0,
            "total_time_seconds": 0.0,
        }

    # 2. Instantiate Probe
    if arm == "arm_b_dynamic_linear":
        probe = DynamicLinearReadout(
            shape=base_model.shape, d=base_model.d, heads=8
        ).cuda()
    elif arm == "arm_c_kernel_r1":
        probe = CharacteristicKernelReadout(
            shape=base_model.shape, d=base_model.d, num_probes=8, rounds=1
        ).cuda()
    elif arm == "arm_d_kernel_r2":
        probe = CharacteristicKernelReadout(
            shape=base_model.shape, d=base_model.d, num_probes=8, rounds=2
        ).cuda()
    else:
        raise ValueError(f"Unknown arm: {arm}")

    param_count = sum(p.numel() for p in probe.parameters() if p.requires_grad)
    print(f"Probe architecture: {probe.__class__.__name__}, trainable parameters: {param_count:,}")

    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-2)

    def forward_step(field, token_ids, return_diag=False):
        with torch.no_grad():
            field, _, _ = base_model.source(field, token_ids)
            field = base_model.transport.apply_multiplier(field, multiplier)
            field, _ = base_model.collision(field)
            field, _ = base_model.bath(field)
            token_embed = base_model.source.embedding(token_ids)
        feature, diag = probe(field, token_embed, return_diag=return_diag)
        return field, feature, diag

    @torch.no_grad()
    def evaluate(val_tokens_budget: int = 4096) -> Tuple[float, Dict[str, float]]:
        probe.eval()
        val_state = base_model.initial_state(1, "cuda")
        total_loss = 0.0
        n_chunks = val_tokens_budget // tokens
        all_diags = []

        for chunk_idx in range(n_chunks):
            offset = chunk_idx * tokens
            x = torch.as_tensor(
                np.array(valid_data[offset : offset + tokens]),
                dtype=torch.long,
                device="cuda",
            )[None]
            y = torch.as_tensor(
                np.array(valid_data[offset + 1 : offset + 1 + tokens]),
                dtype=torch.long,
                device="cuda",
            )[None]

            features = []
            for t in range(tokens):
                val_state, feat, diag = forward_step(
                    val_state, x[:, t], return_diag=(t == tokens - 1)
                )
                features.append(feat)
                if diag is not None:
                    all_diags.append(diag)

            logits = base_model.decoder(torch.stack(features, dim=1))
            loss = F.cross_entropy(
                logits.reshape(-1, base_model.vocab_size), y.reshape(-1)
            )
            total_loss += float(loss) * tokens

        probe.train()
        mean_diag = {}
        if all_diags:
            for k in all_diags[0].keys():
                mean_diag[k] = float(np.mean([d[k] for d in all_diags]))
        return total_loss / (n_chunks * tokens), mean_diag

    print("Evaluating initial zero-shot / step 0 validation NLL...")
    val_nll_init, val_diag_init = evaluate(val_tokens)
    print(f"Initial Validation NLL: {val_nll_init:.5f}")

    output_arm_dir = output_dir / arm
    output_arm_dir.mkdir(parents=True, exist_ok=True)
    metrics_log = []

    def log_metric(record):
        metrics_log.append(record)
        with (output_arm_dir / "metrics.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    log_metric({"step": 0, "kind": "validation", "val_nll": val_nll_init, **val_diag_init})

    state = base_model.initial_state(1, "cuda")
    best_val_nll = val_nll_init
    best_step = 0
    t0_global = time.perf_counter()

    for step in range(1, steps + 1):
        probe.train()
        optimizer.zero_grad()

        offset = (step - 1) * tokens
        x = torch.as_tensor(
            np.array(train_data[offset : offset + tokens]),
            dtype=torch.long,
            device="cuda",
        )[None]
        y = torch.as_tensor(
            np.array(train_data[offset + 1 : offset + 1 + tokens]),
            dtype=torch.long,
            device="cuda",
        )[None]

        t_step0 = time.perf_counter()
        features = []
        last_step_diag = None
        for t in range(tokens):
            state, feat, diag = forward_step(
                state, x[:, t], return_diag=(t == tokens - 1)
            )
            features.append(feat)
            if diag is not None:
                last_step_diag = diag

        logits = base_model.decoder(torch.stack(features, dim=1))
        loss = F.cross_entropy(
            logits.reshape(-1, base_model.vocab_size), y.reshape(-1)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
        optimizer.step()
        dt_step = time.perf_counter() - t_step0

        loss_scalar = float(loss.detach())
        field_energy = float(0.5 * state.detach().square().sum(-1).mean())

        if step % 20 == 0 or step == 1:
            diag_str = ""
            if last_step_diag:
                diag_str = (
                    f" | Ent: {last_step_diag.get('probe_attention_entropy', 0):.2f}"
                    f" | logZ: {last_step_diag.get('kernel_response_logZ', 0):.2f}"
                    f" | Ovr: {last_step_diag.get('probe_spatial_overlap', 0):.2f}"
                )
                if last_step_diag.get("round_query_delta", 0) > 0:
                    diag_str += f" | dQ: {last_step_diag['round_query_delta']:.3f}"
            print(
                f"Step {step:4d}/{steps} | Train NLL: {loss_scalar:.5f} | "
                f"E: {field_energy:.3f}{diag_str} | Step Time: {dt_step*1000:.1f}ms"
            )
            log_record = {
                "step": step,
                "kind": "train",
                "train_nll": loss_scalar,
                "field_energy": field_energy,
                "step_ms": dt_step * 1000,
            }
            if last_step_diag:
                log_record.update(last_step_diag)
            log_metric(log_record)

        if step % validate_every == 0 or step == steps:
            val_nll, val_diag = evaluate(val_tokens)
            is_best = val_nll < best_val_nll
            if is_best:
                best_val_nll = val_nll
                best_step = step
                torch.save(
                    {
                        "probe": probe.state_dict(),
                        "step": step,
                        "val_nll": val_nll,
                        "arm": arm,
                    },
                    output_arm_dir / "best_probe.pt",
                )
            print(
                f">>> [VAL @ step {step:4d}] Validation NLL: {val_nll:.5f} "
                f"(Best: {best_val_nll:.5f} at step {best_step})"
            )
            log_record = {
                "step": step,
                "kind": "validation",
                "val_nll": val_nll,
                "best_val_nll": best_val_nll,
            }
            log_record.update(val_diag)
            log_metric(log_record)

    total_time = time.perf_counter() - t0_global
    print(f"\nCompleted arm {arm} in {total_time:.1f}s ({total_time/steps*1000:.1f}ms/step)")
    print(f"Final Best Validation NLL: {best_val_nll:.5f} (Achieved at step {best_step})")

    summary = {
        "arm": arm,
        "trainable_parameters": param_count,
        "steps": steps,
        "initial_val_nll": val_nll_init,
        "final_val_nll": val_nll,
        "best_val_nll": best_val_nll,
        "best_step": best_step,
        "total_time_seconds": total_time,
    }
    with (output_arm_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arm",
        type=str,
        default="all",
        choices=[
            "all",
            "arm_a_baseline",
            "arm_b_dynamic_linear",
            "arm_c_kernel_r1",
            "arm_d_kernel_r2",
        ],
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("results/cbim_torus3d_8x8x4_d128_3000/BBest.pt"),
    )
    parser.add_argument(
        "--data", type=Path, default=Path("data/ib_owt_gpt2")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/probe_ablation_characteristic_kernel")
    )
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--val-tokens", type=int, default=4096)
    args = parser.parse_args()

    arms = (
        [
            "arm_a_baseline",
            "arm_b_dynamic_linear",
            "arm_c_kernel_r1",
            "arm_d_kernel_r2",
        ]
        if args.arm == "all"
        else [args.arm]
    )

    results = {}
    for arm in arms:
        results[arm] = run_probe_experiment(
            arm=arm,
            checkpoint_path=args.checkpoint,
            data_dir=args.data,
            output_dir=args.output,
            steps=args.steps,
            tokens=args.tokens,
            lr=args.lr,
            validate_every=args.validate_every,
            val_tokens=args.val_tokens,
        )

    # Print comprehensive comparison table
    print("\n==========================================================================================")
    print("FROZEN-FIELD CHARACTERISTIC KERNEL READOUT ABLATION SUMMARY")
    print("==========================================================================================")
    print(f"Base Checkpoint: {args.checkpoint}")
    base_nll = results.get("arm_a_baseline", {}).get("best_val_nll", 7.24059)
    print(f"{'Arm':<24} | {'Params':<10} | {'Init NLL':<10} | {'Best NLL':<10} | {'Delta vs Base':<12}")
    print("------------------------------------------------------------------------------------------")
    for arm, res in results.items():
        delta = res["best_val_nll"] - base_nll
        delta_str = f"{delta:+.5f}"
        print(
            f"{arm:<24} | {res['trainable_parameters']:<10,d} | {res['initial_val_nll']:<10.5f} | {res['best_val_nll']:<10.5f} | {delta_str:<12}"
        )
    print("==========================================================================================")

    # Save overall summary
    with (args.output / "overall_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "baseline_original_nll": base_nll,
                "results": results,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
