"""Audit whether the existing H/C/S body can decode a delayed local eligibility.

This is a deliberately strict structural admission audit on registered OpenWebText bytes.  It
does not train the organism, alter any body parameter, or claim a learning
rule.  For each real arrival it records the exact transient local factor
``response_gate * old_S``.  After a declared number of later byte events, a
linear readout is fitted *only as a diagnostic* from the already persistent
local H/C/S state to that factor, then evaluated on the independent validation
split.

The probe is deliberately bounded: both gate and S lie in [0, 1].  Its
coordinatewise residual therefore provides empirical evidence relevant to one
*linear full-factor* ``BoundedEpsilonDecoder`` candidate; it is not a universal
Lean certificate.  A biological factorized carrier may instead retain a small
sum of ``gate ⊗ S`` terms and reconstruct their product only when the current
third factor arrives.  This audit neither establishes nor rejects that weaker,
proper low-rank design.  In particular, observed low rank on one stream never
overrides ``collision_of_excess_keys`` for independently required directions.
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
from information_boltzmann.core.cgb_column_body import CGBColumnLanguageBody, ColumnBodyState

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


@torch.no_grad()
def local_hcs_features(body: CGBColumnLanguageBody, state: ColumnBodyState) -> torch.Tensor:
    """Use only the retained local H/C/S and C's already-present causal code."""
    mean = state.natural / state.precision
    log_precision = state.precision.log()
    participation = state.participation.flatten(start_dim=-3)
    valid = (state.causal_token >= 0).to(dtype=state.activity.dtype).view(-1, 1)
    causal_code = body.causal_address(state.causal_token.clamp_min(0)) * valid
    causal_code = causal_code[:, None, None, :].expand(-1, body.rows, body.cols, -1)
    return torch.cat((state.activity, mean, log_precision, participation, causal_code), dim=-1)


@torch.no_grad()
def collect_probe(
    body: CGBColumnLanguageBody,
    values: torch.Tensor,
    *,
    warmup: int,
    anchors: int,
    delay: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect independent local samples without retaining an autograd graph."""
    required = warmup + anchors * (delay + 1)
    if values.numel() < required:
        raise ValueError("not enough lexical bytes for requested warmup, anchors, and delay")
    state = body.initial_state(1, device=values.device)
    for token in values[:warmup]:
        state = body.detach_state(body.step(state, token.view(1), elapsed=1.0).state)
    cursor = warmup
    features: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for _ in range(anchors):
        source = body.clone_state(state)
        prepared = body.step(source, values[cursor:cursor + 1], elapsed=1.0)
        target = body.contact_response_factor(source, prepared)
        delayed = body.detach_state(prepared.state)
        cursor += 1
        for _ in range(delay):
            delayed = body.detach_state(body.step(delayed, values[cursor:cursor + 1], elapsed=1.0).state)
            cursor += 1
        features.append(local_hcs_features(body, delayed).flatten(end_dim=2).cpu())
        targets.append(target.flatten(end_dim=2).flatten(start_dim=1).cpu())
        state = delayed
    return torch.cat(features), torch.cat(targets)


def fit_and_score(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    validation_x: torch.Tensor,
    validation_y: torch.Tensor,
    ridge: float,
    tolerances: tuple[float, ...],
) -> dict:
    """Fit a bounded-probe diagnostic, reporting coordinatewise held-out error."""
    mean = train_x.mean(dim=0, keepdim=True)
    scale = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)
    train_x = (train_x - mean) / scale
    validation_x = (validation_x - mean) / scale
    train_x = torch.cat((train_x, torch.ones_like(train_x[:, :1])), dim=1)
    validation_x = torch.cat((validation_x, torch.ones_like(validation_x[:, :1])), dim=1)
    gram = train_x.T @ train_x
    gram.diagonal().add_(ridge)
    weights = torch.linalg.solve(gram, train_x.T @ train_y)
    prediction = validation_x @ weights
    error = (prediction - validation_y).abs()
    baseline = validation_y.abs()
    energy = validation_y.square().sum().clamp_min(1e-12)
    decoded_energy = (prediction - validation_y).square().sum()
    return {
        "input_coordinates": int(train_x.shape[1] - 1),
        "target_coordinates": int(train_y.shape[1]),
        "train_local_samples": int(train_x.shape[0]),
        "validation_local_samples": int(validation_x.shape[0]),
        "ridge": ridge,
        "held_out_mean_abs_error": float(error.mean()),
        "held_out_max_abs_error": float(error.max()),
        "zero_decoder_mean_abs_error": float(baseline.mean()),
        "relative_mean_abs_error_to_zero": float(error.mean() / baseline.mean().clamp_min(1e-12)),
        "held_out_explained_target_energy": float(1.0 - decoded_energy / energy),
        "coordinate_fraction_within_epsilon": {
            str(epsilon): float((error <= epsilon).float().mean()) for epsilon in tolerances
        },
        "max_coordinate_error_minus_epsilon": {
            str(epsilon): float((error - epsilon).max()) for epsilon in tolerances
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=ROOT / "data/ib_owt_10000/train.npy")
    parser.add_argument("--validation", type=Path, default=ROOT / "data/ib_owt_10000/validation.npy")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/ib_owt_10000/manifest.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/published/cgb_column_eligibility_decoder_owt_20261101.json")
    parser.add_argument("--warmup", type=int, default=128)
    parser.add_argument("--train-anchors", type=int, default=512)
    parser.add_argument("--validation-anchors", type=int, default=256)
    parser.add_argument("--delays", default="0,1,2,4,8,16,32")
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--cols", type=int, default=8)
    parser.add_argument("--active-columns", type=int, default=8)
    parser.add_argument("--contact-rank", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    delays = tuple(sorted({int(value) for value in args.delays.split(",")}))
    if not delays or delays[0] < 0 or args.warmup < 1 or args.train_anchors < 1 or args.validation_anchors < 1:
        raise ValueError("delays must be nonnegative; warmup and anchors must be positive")
    if args.ridge < 0:
        raise ValueError("ridge must be nonnegative")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    required = args.warmup + max(args.train_anchors, args.validation_anchors) * (max(delays) + 1)
    train_values = torch.as_tensor(lexical_bytes(args.train, required), device=device, dtype=torch.long)
    validation_values = torch.as_tensor(lexical_bytes(args.validation, required), device=device, dtype=torch.long)
    body = CGBColumnLanguageBody(args.d_model, (args.rows, args.cols),
                                 active_columns=args.active_columns,
                                 contact_rank=args.contact_rank).to(device).eval()
    tolerances = (0.05, 0.1, 0.2)
    result: dict[str, dict] = {}
    for delay in delays:
        train_x, train_y = collect_probe(body, train_values, warmup=args.warmup,
                                         anchors=args.train_anchors, delay=delay)
        validation_x, validation_y = collect_probe(body, validation_values, warmup=args.warmup,
                                                    anchors=args.validation_anchors, delay=delay)
        result[str(delay)] = fit_and_score(train_x, train_y, validation_x, validation_y,
                                           args.ridge, tolerances)
    report = {
        "status": "bounded_local_eligibility_decoder_audit",
        "claim_scope": "Held-out OpenWebText diagnostic of whether an explicit linear readout of existing local H/C/S can reconstruct a bounded exact one-event response factor after a stated delay. The organism and its parameters are never trained or updated.",
        "not_claimed": ["a universal BoundedEpsilonDecoder proof", "sufficiency or insufficiency of a factorized low-rank carrier", "an online learning rule", "long-horizon credit", "language capability", "life closure"],
        "target": "per-column transient hard-saturation response gate times pre-arrival typed S participation",
        "state_read": "post-delay local H activity, C natural/precision, S participation, and C causal-token code only",
        "theorem_connection": "GenerationAssociationCapacity: exact_decoder_capacity prohibits claiming universal exact recovery beyond independent key dimension; this finite held-out bounded probe instead reports its declared coordinatewise residual.",
        "result_by_delay_events": result,
        "model": {
            "d_model": args.d_model, "grid_shape": [args.rows, args.cols],
            "active_columns": args.active_columns, "contact_rank": args.contact_rank,
            "parameter_numbers": sum(parameter.numel() for parameter in body.parameters()),
            "persistent_state_bytes_fp32": body.persistent_state_bytes(body.initial_state(1, device=device)),
        },
        "protocol": {"warmup_events": args.warmup, "train_anchors": args.train_anchors,
                     "validation_anchors": args.validation_anchors, "delays": delays,
                     "ridge": args.ridge, "tolerances": tolerances, "parameter_write": False,
                     "state_history": "each event state is detached before the next event; no retained autograd graph",
                     "seed": args.seed, "device": str(device)},
        "source": {"train": str(args.train.relative_to(ROOT)), "validation": str(args.validation.relative_to(ROOT)),
                   "train_sha256": sha256(args.train), "validation_sha256": sha256(args.validation),
                   "manifest_sha256": sha256(args.manifest)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
