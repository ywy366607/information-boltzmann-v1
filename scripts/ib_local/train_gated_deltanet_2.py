"""Train the official NVIDIA Gated DeltaNet-2 (GDN-2) on OpenWebText BPE."""
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

import numpy as np
import torch

from scripts.ib_local.gated_deltanet_2 import GatedDeltaNet2LM
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
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/gated_deltanet_2_d128_3000"),
    )
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--d", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--validate-every", type=int, default=250)
    parser.add_argument("--validation-tokens", type=int, default=4096)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()

    torch.set_num_threads(2)
    torch.manual_seed(11)
    torch.cuda.set_per_process_memory_fraction(0.90)
    args.output.mkdir(parents=True, exist_ok=True)

    train = np.load(args.data / "train.npy", mmap_mode="r")
    valid = np.load(args.data / "validation.npy", mmap_mode="r")

    model = GatedDeltaNet2LM(
        vocab_size=50257,
        d=args.d,
        layers=args.layers,
        heads=args.heads,
    ).cuda()

    runner = TruncatedInternalTimeGraphTrainer(
        model, tokens=args.tokens, chunk_tokens=args.tokens
    )

    sources = [Path(__file__), Path("scripts/ib_local/gated_deltanet_2.py")]
    non_emb_params = sum(
        p.numel()
        for n, p in model.named_parameters()
        if "embedding" not in n and "decoder" not in n
    )

    config = {
        "architecture": model.architecture,
        "data": str(args.data),
        "output": str(args.output),
        "steps": args.steps,
        "tokens": args.tokens,
        "d": args.d,
        "layers": args.layers,
        "heads": args.heads,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "non_embedding_parameters": non_emb_params,
        "cbim_reference_non_embedding": 332724,
        "precision": "FP32",
        "seed": 11,
        "lr": 3e-4,
        "validate_every": args.validate_every,
        "validation_tokens": args.validation_tokens,
        "manifest": json.loads((args.data / "manifest.json").read_text()),
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
        },
    }

    atomic_json(args.output / "config.json", config, required=True)
    atomic_json(
        args.output / "progress.json",
        {"status": "capturing", "step": 0, "target_steps": args.steps},
    )

    step, best = 0, float("inf")
    if args.resume:
        saved = torch.load(args.resume, map_location="cuda", weights_only=False)
        model.load_state_dict(saved["model"])
        runner.optimizer.load_state_dict(saved["optimizer"])
        runner.state.copy_(saved["state"])
        step, best = saved["step"], saved["best_validation_nll"]

    def batch(data, offset):
        return tuple(
            torch.as_tensor(
                np.array(data[offset + shift : offset + shift + args.tokens]),
                dtype=torch.long,
                device="cuda",
            )[None]
            for shift in (0, 1)
        )

    def save(name):
        temporary = args.output / f"{name}.{os.getpid()}.tmp"
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": runner.optimizer.state_dict(),
                "state": runner.state.detach(),
                "step": step,
                "events": step * args.tokens,
                "best_validation_nll": best,
                "config": config,
            },
            temporary,
        )
        os.replace(temporary, args.output / name)

    def log(row):
        with (args.output / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row), flush=True)

    @torch.no_grad()
    def validate():
        state, total = model.initial_state(1, "cuda"), 0.0
        for offset in range(0, args.validation_tokens + args.tokens, args.tokens):
            ids, targets = batch(valid, offset)
            loss, state, _ = model(ids, targets, state)
            if offset:
                total += float(loss) * args.tokens
        return total / args.validation_tokens

    try:
        if args.resume is None:
            best = validate()
            log(
                {
                    "kind": "validation",
                    "step": 0,
                    "validation_nll": best,
                    "best_validation_nll": best,
                }
            )
            save("BBest.pt")
            save("last.pt")

        while step < args.steps:
            ids, targets = batch(train, step * args.tokens)
            torch.cuda.synchronize()
            started = time.perf_counter()
            loss, state, diagnostics = runner.step(ids, targets)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            step += 1

            if not math.isfinite(float(loss)):
                raise FloatingPointError("Non-finite training loss")
            if elapsed > 1.5 and step > 2:
                raise RuntimeError(f"Training step exceeded 1.5 seconds: {elapsed:.3f}")

            if step == 1 or step % 10 == 0 or step == args.steps:
                row = {
                    "kind": "train",
                    "step": step,
                    "events": step * args.tokens,
                    "nll": float(loss),
                    "seconds": elapsed,
                    "decay_mean": float(diagnostics["decay_mean"]),
                    "erase_mean": float(diagnostics["erase_mean"]),
                    "write_mean": float(diagnostics["write_mean"]),
                    "state_norm": float(diagnostics["state_norm"]),
                    "state_norm_final": float(diagnostics["state_norm_final"]),
                    "grad_norm": float(runner.grad_norm),
                    "grad_norm_lexical": float(runner.group_grad_norms[0]),
                    "grad_norm_dynamics": float(runner.group_grad_norms[1]),
                    "allocated_mib": torch.cuda.memory_allocated() / 2**20,
                    "reserved_mib": torch.cuda.memory_reserved() / 2**20,
                }
                log(row)
                atomic_json(
                    args.output / "progress.json",
                    {"status": "running", "target_steps": args.steps, **row},
                )

            if step % args.validate_every == 0 or step == args.steps:
                score = validate()
                improved = score < best
                if improved:
                    best = score
                log(
                    {
                        "kind": "validation",
                        "step": step,
                        "validation_nll": score,
                        "best_validation_nll": best,
                    }
                )
                if improved:
                    save("BBest.pt")
                save("last.pt")

        atomic_json(
            args.output / "progress.json",
            {
                "status": "complete",
                "step": step,
                "target_steps": args.steps,
                "best_validation_nll": best,
            },
        )
    except BaseException as error:
        atomic_json(
            args.output / "progress.json",
            {"status": "failed", "step": step, "error": str(error)},
        )
        raise


if __name__ == "__main__":
    main()
