"""Measure the local simultaneous rank-one eligibility load on OpenWebText.

An individual CGB contact event has the exact local response form
``gate_t ⊗ S_t``.  A delayed third factor can only distinguish several past
events if their unresolved factors have not been merged.  This audit does not
teach, update, or add state.  It asks a narrower architectural question on a
real UTF-8 OpenWebText stream: in each geometric column compartment, how many
linearly independent rank-one factors coexist over declared delay windows?

The answer is a workload measurement for a future factorized H/S carrier. It
does not establish that the current H/C/S dynamics already retain those
factors, nor does it define a language-learning rule.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

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
    raw = np.load(path, mmap_mode="r")
    output: list[int] = []
    for value in raw:
        value = int(value)
        if BYTE_OFFSET <= value < BYTE_OFFSET + 256:
            output.append(value - BYTE_OFFSET)
            if len(output) == needed:
                return np.asarray(output, dtype=np.int64)
    raise ValueError(f"not enough lexical bytes in {path}")


def summarize(values: torch.Tensor) -> dict[str, float]:
    values = values.float().flatten().cpu()
    return {
        "mean": float(values.mean()),
        "median": float(values.median()),
        "p95": float(torch.quantile(values, 0.95)),
        "max": float(values.max()),
    }


@torch.no_grad()
def pending_factor_loads(
    body: CGBColumnLanguageBody,
    values: torch.Tensor,
    *,
    warmup: int,
    steps: int,
    windows: tuple[int, ...],
    sample_every: int,
    tolerance: float,
) -> dict[str, dict]:
    """Scan one stream once and sample all requested delay windows together."""
    if values.numel() < warmup + steps:
        raise ValueError("not enough values")
    state = body.initial_state(1, device=values.device)
    for token in values[:warmup]:
        state = body.detach_state(body.step(state, token.view(1), elapsed=1.0).state)

    factors = {window: [] for window in windows}
    factor_ranks = {window: [] for window in windows}
    summed_ranks = {window: [] for window in windows}
    live_counts = {window: [] for window in windows}
    for event_index, token in enumerate(values[warmup:warmup + steps], start=1):
        source = body.clone_state(state)
        prepared = body.step(source, token.view(1), elapsed=1.0)
        # [rows, cols, D, F], one local rank-one factor at each compartment.
        factor = body.contact_response_factor(source, prepared).squeeze(0)
        for window in windows:
            factors[window].append(factor)
            if len(factors[window]) > window:
                factors[window].pop(0)
        # Matrix factorization is a diagnostic, not part of the organism's
        # recurrence.  Sampling it avoids repeatedly spending GPU time on the
        # same adjacent windows while preserving the stream's exact H/C/S run.
        if event_index % sample_every == 0 or event_index == steps:
            for window in windows:
                stack = torch.stack(factors[window], dim=2)  # [rows, cols, time, D, F]
                flattened = stack.flatten(start_dim=-2)  # [rows, cols, time, D*F]
                factor_ranks[window].append(torch.linalg.matrix_rank(flattened, tol=tolerance))
                summed_ranks[window].append(torch.linalg.matrix_rank(stack.sum(dim=2), tol=tolerance))
                live_counts[window].append((flattened.abs().amax(dim=-1) > tolerance).sum(dim=-1))
        state = body.detach_state(prepared.state)
    return {
        str(window): {
            "factor_span_rank": summarize(torch.stack(factor_ranks[window])),
            "summed_outer_product_rank": summarize(torch.stack(summed_ranks[window])),
            "nonzero_unresolved_factor_count": summarize(torch.stack(live_counts[window])),
        } for window in windows
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data/ib_owt_10000/train.npy")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/ib_owt_10000/manifest.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/published/cgb_column_pending_factor_load_owt_20261101.json")
    parser.add_argument("--warmup", type=int, default=128)
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--windows", default="1,2,4,8,16")
    parser.add_argument("--sample-every", type=int, default=16)
    parser.add_argument("--rank-tolerance", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--cols", type=int, default=8)
    parser.add_argument("--active-columns", type=int, default=8)
    parser.add_argument("--contact-rank", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    windows = tuple(sorted({int(value) for value in args.windows.split(",")}))
    if (not windows or windows[0] < 1 or args.warmup < 1 or args.steps < 1 or
            args.sample_every < 1):
        raise ValueError("windows, warmup, steps, and sample interval must be positive")
    if args.rank_tolerance <= 0:
        raise ValueError("rank tolerance must be positive")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    values = torch.as_tensor(lexical_bytes(args.data, args.warmup + args.steps),
                             device=device, dtype=torch.long)
    body = CGBColumnLanguageBody(args.d_model, (args.rows, args.cols),
                                 active_columns=args.active_columns,
                                 contact_rank=args.contact_rank).to(device).eval()
    result = pending_factor_loads(body, values, warmup=args.warmup, steps=args.steps,
                                  windows=windows, sample_every=args.sample_every,
                                  tolerance=args.rank_tolerance)
    report = {
        "status": "local_pending_rank_one_factor_load_audit",
        "claim_scope": "Real-OpenWebText measurement of simultaneous local rank-one response factors over declared delay windows. No parameter or persistent-state update is applied.",
        "not_claimed": ["that current H/C/S preserves this factor set", "a required global rank", "an online learning rule", "language capability", "life closure"],
        "factor": "hard-saturation response gate outer-product pre-arrival typed S participation",
        "interpretation": "factor_span_rank is the number of independent historic factor directions in a compartment before a third factor could distinguish them; summed_outer_product_rank is the rank after their linear superposition. Both are local, not a demand to materialize a global dense eligibility matrix.",
        "result_by_delay_window": result,
        "model": {"d_model": args.d_model, "grid_shape": [args.rows, args.cols],
                  "active_columns": args.active_columns, "contact_rank": args.contact_rank,
                  "parameter_numbers": sum(parameter.numel() for parameter in body.parameters()),
                  "persistent_state_bytes_fp32": body.persistent_state_bytes(body.initial_state(1, device=device))},
        "protocol": {"warmup_events": args.warmup, "steps": args.steps, "windows": windows,
                     "rank_sample_every_events": args.sample_every,
                     "rank_tolerance": args.rank_tolerance, "parameter_write": False,
                     "state_history": "the body state is detached after each event; no retained autograd graph",
                     "seed": args.seed, "device": str(device)},
        "source": {"path": str(args.data.relative_to(ROOT)), "sha256": sha256(args.data),
                   "manifest_sha256": sha256(args.manifest)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
