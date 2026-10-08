"""Full-update timing only; synthetic tensors do not establish capability.

This entry point never starts or stops another process. Use a free GPU window
for CUDA measurements. Both scopes retain all fast-state values between updates.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from information_boltzmann.core.plastic_medium import MediumState, PlasticMedium3D
from information_boltzmann.core.plastic_ports import PlasticMediumPorts3D


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("core", "ports"), default="core")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--shape", type=int, nargs=3, default=(8, 8, 4))
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=50257)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--substeps", type=int, default=1)
    parser.add_argument("--event-duration", type=float, required=True,
                        help="Model physical time per event; independent of substeps")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--adaptive-conduction", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plasticity-time-reference", type=float, default=1.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.tokens, args.updates, args.substeps) < 1 or args.warmup < 0:
        parser.error("Positive tokens/updates/substeps and nonnegative warmup required")
    if not math.isfinite(args.event_duration) or args.event_duration <= 0:
        parser.error("event-duration must be finite and positive")
    torch.manual_seed(319)
    device = torch.device(args.device)
    shape = tuple(args.shape)
    if args.scope == "core":
        model = PlasticMedium3D(shape, args.channels,
                               adaptive_conduction=args.adaptive_conduction,
                               plasticity_time_reference=args.plasticity_time_reference).to(device)
        initial = model.initial_state()
        state = MediumState(torch.randn_like(initial.field),
                            tuple(torch.randn_like(x) for x in initial.flux), initial.elapsed,
                            initial.conduction)
        target = torch.randn_like(initial.field)
    else:
        model = PlasticMediumPorts3D(args.vocab_size, shape, args.channels,
                                    adaptive_conduction=args.adaptive_conduction,
                                    plasticity_time_reference=args.plasticity_time_reference).to(device)
        state = model.initial_belief()
        ids = torch.randint(args.vocab_size, (1, args.tokens + 1), device=device)
    runner = torch.compile(model) if args.compile else model
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    def update(current):
        optimizer.zero_grad(set_to_none=True)
        if args.scope == "core":
            losses = []
            for _ in range(args.tokens):
                current, _ = runner(current, args.event_duration, substeps=args.substeps)
                losses.append((current.field - target).square().mean())
            loss = torch.stack(losses).mean()
        else:
            loss, current, _ = runner(ids[:, :-1], ids[:, 1:], current,
                                      event_duration=args.event_duration, substeps=args.substeps)
        loss.backward()
        optimizer.step()
        return current.detach()

    for _ in range(args.warmup):
        state = update(state)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(args.updates):
        state = update(state)
    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = (time.perf_counter() - start) / args.updates
    report = {
        "purpose": "synthetic full-update execution benchmark; no capability claim",
        "scope": args.scope, "device": str(device), "shape": shape,
        "channels": args.channels, "tokens_per_update": args.tokens,
        "event_duration": args.event_duration, "substeps": args.substeps,
        "warmup_updates": args.warmup, "measured_updates": args.updates,
        "compiled": args.compile, "seconds_per_update": seconds,
        "adaptive_conduction": args.adaptive_conduction,
        "plasticity_time_reference": args.plasticity_time_reference,
        "parameters": sum(p.numel() for p in model.parameters()),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved() if device.type == "cuda" else None,
        "pytorch": torch.__version__,
    }
    serialized = json.dumps(report, indent=2, ensure_ascii=False)
    print(serialized)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
