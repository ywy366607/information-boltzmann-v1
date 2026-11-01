"""Audit finite causal score paths of the column body's continuing parameters.

This is an implementation diagnostic, not a learning algorithm.  It uses a
short retained graph only to establish the smallest causal horizon at which a
parameter group can affect a future real-language score.  It never writes a
parameter, and it is explicitly not admissible as BPTT-free credit.

The resulting horizon partition tells the local learning design what must be
reconstructed from H/C/S: one-event boundary terms, two-event H/C terms, and
three-event S-formation terms.  The input is the registered OpenWebText UTF-8
stream rather than a synthetic memory task.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from information_boltzmann.core.cgb_column_body import CGBColumnLanguageBody

BYTE_OFFSET = 4


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lexical_bytes(path: Path, needed: int) -> np.ndarray:
    values = np.load(path, mmap_mode="r")
    output: list[int] = []
    for raw in values:
        value = int(raw)
        if BYTE_OFFSET <= value < BYTE_OFFSET + 256:
            output.append(value - BYTE_OFFSET)
            if len(output) >= needed:
                return np.asarray(output, dtype=np.int64)
    raise ValueError(f"only found {len(output)} lexical bytes in {path}")


def parameter_paths(body: CGBColumnLanguageBody, values: torch.Tensor, *, warmup: int,
                    max_horizon: int) -> dict[str, int | None]:
    """Earliest exact causal score path after a detached real-stream state."""
    with torch.no_grad():
        state = body.initial_state(1, device=values.device)
        for token in values[:warmup]:
            state = body.detach_state(body.step(state, token.view(1), elapsed=1.0).state)
    origin = body.clone_state(state)
    first: dict[str, int | None] = {name: None for name, _ in body.named_parameters()}
    for horizon in range(1, max_horizon + 1):
        body.zero_grad(set_to_none=True)
        state = body.clone_state(origin)
        prepared = None
        for token in values[warmup:warmup + horizon]:
            prepared = body.step(state, token.view(1), elapsed=1.0)
            state = prepared.state
        assert prepared is not None
        target = values[warmup + horizon].view(1)
        F.cross_entropy(prepared.logits, target).backward()
        for name, parameter in body.named_parameters():
            if first[name] is None and parameter.grad is not None and float(parameter.grad.norm()) > 1e-10:
                first[name] = horizon
    return first


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data/ib_owt_10000/train.npy")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/ib_owt_10000/manifest.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/published/cgb_column_parameter_paths_owt_20261101.json")
    parser.add_argument("--warmup", type=int, default=48)
    parser.add_argument("--max-horizon", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--cols", type=int, default=8)
    parser.add_argument("--active-columns", type=int, default=8)
    parser.add_argument("--contact-rank", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.warmup < 1 or args.max_horizon < 1:
        raise ValueError("warmup and max-horizon must be positive")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    raw = lexical_bytes(args.data, args.warmup + args.max_horizon + 1)
    values = torch.as_tensor(raw, device=device, dtype=torch.long)
    body = CGBColumnLanguageBody(args.d_model, (args.rows, args.cols),
                                 active_columns=args.active_columns,
                                 contact_rank=args.contact_rank).to(device)
    first = parameter_paths(body, values, warmup=args.warmup, max_horizon=args.max_horizon)
    groups: dict[str, list[str]] = {}
    for name, horizon in first.items():
        key = "unreached" if horizon is None else f"event_{horizon}"
        groups.setdefault(key, []).append(name)
    report = {
        "status": "causal_parameter_path_audit",
        "claim_scope": "Exact short unroll used only to measure causal reachability from a detached OpenWebText state; no parameters were updated and this is not a BPTT-free learning rule.",
        "not_claimed": ["an online local learning rule", "self-assembled teaching alignment", "long-horizon credit", "language capability"],
        "result": {"continuing_parameter_groups": len(first), "all_reached_within_horizon": all(v is not None for v in first.values()),
                   "earliest_score_horizon": first, "groups": groups},
        "model": {"d_model": args.d_model, "grid_shape": [args.rows, args.cols],
                  "active_columns": args.active_columns, "contact_rank": args.contact_rank,
                  "parameter_numbers": sum(p.numel() for p in body.parameters()),
                  "persistent_state_bytes_fp32": body.persistent_state_bytes(body.initial_state(1, device=device))},
        "protocol": {"warmup_events": args.warmup, "max_horizon": args.max_horizon,
                     "state_history": "warmup state is detached before every measured horizon",
                     "parameter_write": False, "seed": args.seed, "device": str(device)},
        "source": {"path": str(args.data.relative_to(ROOT)), "sha256": sha256(args.data),
                   "manifest_sha256": sha256(args.manifest)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["result"], indent=2))


if __name__ == "__main__":
    main()
