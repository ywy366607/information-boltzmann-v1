"""Measure the Lean-admitted GDN byte learner on one continuous OWT stream.

This is deliberately language-only.  Each arm keeps one physical GDN state
through train and validation; prediction is scored before every external byte
is assimilated.  The online parameter updates are the existing G14 affine-head step and, in one
separately reported arm, the Lean-admitted previous-byte value credit. Address,
likelihood, and GDN process parameters remain fixed, so this is an admission
experiment for proved paths, not a claim that all of GDN is trained.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fine_grain.active_gdn2 import ActiveInferenceGDN2, ActiveInferenceState
from information_boltzmann.core.cgb_prequential_bridge import LastByteGDNReadoutPort


BYTE_OFFSET = 4
BYTE_VOCAB = 256


@dataclass
class Arm:
    name: str
    memory: ActiveInferenceGDN2
    encoder: torch.nn.Embedding
    address_encoder: torch.nn.Embedding | None
    likelihood_encoder: torch.nn.Embedding | None
    head: torch.nn.Linear
    port: LastByteGDNReadoutPort


@dataclass
class BiasOnlyG14Arm:
    """The G14 head's feature-free limit: an online byte-frequency control."""

    logits: torch.Tensor
    head_step_size: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_byte_source(data_root: Path, split: str, limit: int) -> tuple[np.ndarray, dict[str, Any]]:
    manifest_path = data_root / "manifest.json"
    tokenizer_path = data_root / "tokenizer.json"
    token_path = data_root / f"{split}.npy"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tokenizer = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    if (manifest.get("source") != "openwebtext"
            or manifest.get("tokenizer") != "byte"
            or manifest.get("vocab_size") != 260
            or tokenizer != {"type": "utf8_bytes", "offset": BYTE_OFFSET,
                             "special_tokens": ["<pad>", "<bos>", "<eos>", "<unk>"]}):
        raise ValueError("expected the registered 260-symbol OWT UTF-8 byte source")
    tokens = np.load(token_path, mmap_mode="r")
    if tokens.ndim != 1:
        raise ValueError(f"expected a one-dimensional stream, got {tokens.shape}")
    if limit < 1:
        raise ValueError("event limit must be positive")
    return tokens[:min(int(limit), len(tokens))], {
        "split": split,
        "path": str(token_path.relative_to(ROOT)),
        "tokens_available": int(len(tokens)),
        "tokens_presented": int(min(int(limit), len(tokens))),
        "source": manifest["source"],
        "revision": manifest["revision"],
        "split_rule": manifest["split_rule"],
        "manifest_sha256": _sha256(manifest_path),
        "tokenizer_sha256": _sha256(tokenizer_path),
        "token_file_sha256": _sha256(token_path),
    }


def _make_arms(
    *, device: torch.device, d_model: int, n_slots: int, seed: int, head_step: float,
    value_step: float,
) -> tuple[Arm, Arm, Arm]:
    torch.manual_seed(seed)
    base_memory = ActiveInferenceGDN2(d_model=d_model, n_slots=n_slots, res=n_slots).to(device)
    base_encoder = torch.nn.Embedding(BYTE_VOCAB, d_model).to(device)
    base_head = torch.nn.Linear(d_model, BYTE_VOCAB).to(device)
    seed_logits = torch.zeros(n_slots, device=device)

    def make(name: str, step: float, value_credit_step: float = 0.0) -> Arm:
        memory = copy.deepcopy(base_memory).to(device)
        encoder = copy.deepcopy(base_encoder).to(device)
        head = copy.deepcopy(base_head).to(device)
        address_encoder = likelihood_encoder = None
        kwargs: dict[str, Any] = {}
        if value_credit_step:
            # These immutable copies make the already-formalized value path
            # literal: its update cannot change address or likelihood.
            address_encoder = copy.deepcopy(base_encoder).to(device)
            likelihood_encoder = copy.deepcopy(base_encoder).to(device)
            for parameter in address_encoder.parameters():
                parameter.requires_grad_(False)
            for parameter in likelihood_encoder.parameters():
                parameter.requires_grad_(False)
            kwargs = {
                "address_encoder": address_encoder,
                "likelihood_encoder": likelihood_encoder,
                "value_step_size": value_credit_step,
            }
        state = memory.initial_state(1, device, torch.float32)
        port = LastByteGDNReadoutPort(
            memory, state, encoder, head, seed_logits, memory.atlas[0].to(device),
            observation_precision=1.0, head_step_size=step, **kwargs,
        )
        return Arm(name, memory, encoder, address_encoder, likelihood_encoder, head, port)

    return (
        make("frozen_head", 0.0),
        make("local_g14_head", head_step),
        make("local_g14_value_credit", head_step, value_step),
    )


def _make_bias_only_g14_arm(
    *, device: torch.device, d_model: int, n_slots: int, seed: int, head_step: float,
) -> BiasOnlyG14Arm:
    """Recreate the common initial affine bias, without a GDN feature readout.

    This is not a competing model.  It is the exact `f = 0` case of the same
    G14 head update, so its only learnable distinction is the marginal byte
    distribution.  A gap to the real GDN arm therefore measures information
    supplied by the GDN's causal readout rather than by the shared bias.
    """
    torch.manual_seed(seed)
    _ = ActiveInferenceGDN2(d_model=d_model, n_slots=n_slots, res=n_slots).to(device)
    _ = torch.nn.Embedding(BYTE_VOCAB, d_model).to(device)
    head = torch.nn.Linear(d_model, BYTE_VOCAB).to(device)
    return BiasOnlyG14Arm(head.bias.detach().clone(), head_step)


def _state_summary(state: ActiveInferenceState) -> dict[str, float]:
    return {
        "natural_l2": float(state.natural.norm().item()),
        "precision_mean": float(state.precision.mean().item()),
        "precision_min": float(state.precision.min().item()),
        "precision_max": float(state.precision.max().item()),
    }


def _persistent_state_numbers(state: ActiveInferenceState) -> int:
    tensors = [state.natural, state.precision, state.velocity_mean, state.velocity_precision]
    tensors.extend(value for value in (
        state.last_tokens, state.last_centers, state.last_observation_precision,
    ) if value is not None)
    return int(sum(value.numel() for value in tensors))


def _phase(
    arm: Arm,
    tokens: Iterable[int],
    *, elapsed: float, report_every: int,
) -> dict[str, Any]:
    """Advance one never-reset stream phase and return first-pass statistics."""
    nll_sum = 0.0
    scored = 0
    special = 0
    progress: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for source_id in tokens:
        source_id = int(source_id)
        # IDs 0..3 are tokenizer controls.  They are not fabricated as bytes:
        # they consume the declared causal tick but cause no lexical write.
        arm.port.advance_time(elapsed)
        if source_id < BYTE_OFFSET:
            special += 1
            continue
        byte = source_id - BYTE_OFFSET
        if not 0 <= byte < BYTE_VOCAB:
            raise ValueError(f"OWT byte id {source_id} lies outside the registered alphabet")
        logits = arm.port.predict()
        nll_sum += float(-torch.log_softmax(logits, dim=0)[byte].item())
        arm.port.observe_pre_scored(byte, external=True)
        scored += 1
        if report_every and scored % report_every == 0:
            progress.append({
                "scored_bytes": scored,
                "mean_nll_nats": nll_sum / scored,
                "bits_per_byte": nll_sum / scored / math.log(2.0),
            })
    if arm.port.state.natural.is_cuda:
        torch.cuda.synchronize(arm.port.state.natural.device)
    seconds = time.perf_counter() - started
    if scored == 0:
        raise ValueError("phase contained no UTF-8 byte tokens")
    return {
        "scored_bytes": scored,
        "special_tokens_elapsed_without_lexical_write": special,
        "mean_nll_nats": nll_sum / scored,
        "bits_per_byte": nll_sum / scored / math.log(2.0),
        "bytes_per_second": scored / seconds,
        "seconds": seconds,
        "progress": progress,
        "state_after": _state_summary(arm.port.state),
    }


def _bias_only_phase(
    arm: BiasOnlyG14Arm,
    tokens: Iterable[int],
    *, report_every: int,
) -> dict[str, Any]:
    """Score then update the G14 feature-free reference on one live stream."""
    nll_sum = 0.0
    scored = 0
    special = 0
    progress: list[dict[str, float | int]] = []
    started = time.perf_counter()
    with torch.no_grad():
        for source_id in tokens:
            source_id = int(source_id)
            if source_id < BYTE_OFFSET:
                special += 1
                continue
            byte = source_id - BYTE_OFFSET
            if not 0 <= byte < BYTE_VOCAB:
                raise ValueError(f"OWT byte id {source_id} lies outside the registered alphabet")
            nll_sum += float(-torch.log_softmax(arm.logits, dim=0)[byte].item())
            innovation = -torch.softmax(arm.logits, dim=0)
            innovation[byte] += 1
            arm.logits.add_(innovation, alpha=arm.head_step_size)
            scored += 1
            if report_every and scored % report_every == 0:
                progress.append({
                    "scored_bytes": scored,
                    "mean_nll_nats": nll_sum / scored,
                    "bits_per_byte": nll_sum / scored / math.log(2.0),
                })
    if arm.logits.is_cuda:
        torch.cuda.synchronize(arm.logits.device)
    seconds = time.perf_counter() - started
    if scored == 0:
        raise ValueError("phase contained no UTF-8 byte tokens")
    return {
        "scored_bytes": scored,
        "special_tokens_elapsed_without_lexical_write": special,
        "mean_nll_nats": nll_sum / scored,
        "bits_per_byte": nll_sum / scored / math.log(2.0),
        "bytes_per_second": scored / seconds,
        "seconds": seconds,
        "progress": progress,
    }


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", suffix=".tmp", delete=False) as tmp:
        json.dump(value, tmp, indent=2, sort_keys=True)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def _checkpoint(
    path: Path,
    arms: tuple[Arm, ...],
    metadata: dict[str, Any],
    *,
    bias_only: BiasOnlyG14Arm,
) -> None:
    payload: dict[str, Any] = {"metadata": metadata, "arms": {}}
    for arm in arms:
        payload["arms"][arm.name] = {
            "memory": arm.memory.state_dict(),
            "encoder": arm.encoder.state_dict(),
            "address_encoder": (None if arm.address_encoder is None
                                else arm.address_encoder.state_dict()),
            "likelihood_encoder": (None if arm.likelihood_encoder is None
                                   else arm.likelihood_encoder.state_dict()),
            "head": arm.head.state_dict(),
            "port_snapshot": arm.port.snapshot(),
            "head_step_size": arm.port.head_step_size,
            "value_step_size": arm.port.value_step_size,
        }
    payload["bias_only_g14"] = {
        "logits": bias_only.logits,
        "head_step_size": bias_only.head_step_size,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.data_root = args.data_root.resolve()
    args.output = args.output.resolve()
    args.checkpoint = args.checkpoint.resolve()
    if args.head_step <= 0 or not math.isfinite(args.head_step):
        raise ValueError("head step must be finite and positive")
    if args.value_step <= 0 or not math.isfinite(args.value_step):
        raise ValueError("value step must be finite and positive")
    if args.elapsed < 0 or not math.isfinite(args.elapsed):
        raise ValueError("elapsed must be finite and nonnegative")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.output.exists() or args.checkpoint.exists():
        raise FileExistsError("refusing to overwrite an existing result or complete-state checkpoint")

    train, train_source = _load_byte_source(args.data_root, "train", args.train_events)
    validation, validation_source = _load_byte_source(args.data_root, "validation", args.validation_events)
    arms = _make_arms(device=device, d_model=args.d_model, n_slots=args.n_slots,
                      seed=args.seed, head_step=args.head_step, value_step=args.value_step)
    results: dict[str, Any] = {}
    for arm in arms:
        train_result = _phase(arm, train, elapsed=args.elapsed, report_every=args.report_every)
        # This is first-pass validation for the same continuing individual.
        # No state, head, clock, or learned parameter is reset at the split.
        validation_result = _phase(arm, validation, elapsed=args.elapsed, report_every=args.report_every)
        base_state_numbers = _persistent_state_numbers(arm.port.state)
        # The value-credit arm keeps exactly the previous byte index and one
        # scalar eligibility beyond the ordinary GDN state; report them rather
        # than hiding their O(1) cost in the interface.
        credit_state_numbers = 2 if arm.port.value_step_size else 0
        results[arm.name] = {
            "head_step_size": arm.port.head_step_size,
            "value_step_size": arm.port.value_step_size,
            "train_first_pass": train_result,
            "validation_first_pass_continuing": validation_result,
            "persistent_state_numbers": base_state_numbers + credit_state_numbers,
            "persistent_state_bytes_fp32": base_state_numbers * 4 + credit_state_numbers * 4,
            "additional_forward_credit_state": {
                "previous_byte_index": 1 if credit_state_numbers else 0,
                "eligibility_scalar": 1 if credit_state_numbers else 0,
            },
            "parameter_numbers": {
                "memory": sum(p.numel() for p in arm.memory.parameters()),
                "value_encoder": sum(p.numel() for p in arm.encoder.parameters()),
                "address_encoder": (0 if arm.address_encoder is None else
                                    sum(p.numel() for p in arm.address_encoder.parameters())),
                "likelihood_encoder": (0 if arm.likelihood_encoder is None else
                                       sum(p.numel() for p in arm.likelihood_encoder.parameters())),
                "head": sum(p.numel() for p in arm.head.parameters()),
            },
        }
    bias_only = _make_bias_only_g14_arm(
        device=device,
        d_model=args.d_model,
        n_slots=args.n_slots,
        seed=args.seed,
        head_step=args.head_step,
    )
    results["bias_only_g14"] = {
        "head_step_size": bias_only.head_step_size,
        "control": "feature-free f = 0 G14 update; learns marginal byte frequency only",
        "train_first_pass": _bias_only_phase(bias_only, train, report_every=args.report_every),
        "validation_first_pass_continuing": _bias_only_phase(
            bias_only, validation, report_every=args.report_every,
        ),
        "persistent_state_numbers": int(bias_only.logits.numel()),
        "persistent_state_bytes_fp32": int(bias_only.logits.numel() * bias_only.logits.element_size()),
        "parameter_numbers": {"head_bias": int(bias_only.logits.numel())},
    }

    frozen = results["frozen_head"]
    learned = results["local_g14_head"]
    value_credit = results["local_g14_value_credit"]
    bias = results["bias_only_g14"]
    report = {
        "status": "language_only_gdn_credit_admission",
        "claim_scope": (
            "Real OWT UTF-8 byte prequential comparison of a fixed GDN state, "
            "its Lean-admitted affine-head update, and its isolated one-arrival "
            "value-credit update."
        ),
        "not_claimed": [
            "full GDN parameter learning",
            "address or likelihood learning",
            "action learning or active language acquisition",
            "credit beyond the next prepared byte score",
            "general language-model capability",
        ],
        "lean_admission": [
            "CGB.GDNLife.finish_saved_nll_improves",
            "CGB.GDNLife.head_update_preserves_memory",
            "CGB.GDNLife.value_future_log_likelihood_deriv",
            "CGB.GDNLife.value_future_score_factor",
            "CGB.GDNLife.carrier_fold",
            "CGB.GDNLife.exists_open_real_gdn_trajectory",
        ],
        "protocol": {
            "external_input": "OpenWebText UTF-8 bytes only",
            "prediction": "prequential: score before the byte is assimilated",
            "individual_continuity": "one state/head per arm across train then validation; no reset",
            "action": "absent",
            "autograd_history": "absent",
            "time": "one declared causal tick per source token; tokenizer controls advance time but are not lexical bytes",
            "fixed_parameters": ["content_address", "likelihood", "process_variance"],
            "online_update": (
                "all learned arms: G14 observed-minus-predicted affine byte-head update; "
                "value-credit arm only: prior-byte value eligibility times next-byte language error"
            ),
            "value_credit_isolation": (
                "value-credit arm uses frozen copies of the initial byte encoder for "
                "address and likelihood; the learned value table cannot alter either path"
            ),
        },
        "model": {
            "runtime": "ActiveInferenceGDN2 + LastByteGDNReadoutPort",
            "d_model": args.d_model,
            "n_slots": args.n_slots,
            "dtype": "float32",
            "device": str(device),
            "seed": args.seed,
        },
        "source": {"train": train_source, "validation": validation_source},
        "arms": results,
        "effect_of_local_head_learning": {
            "train_bits_per_byte_delta_vs_frozen": (
                learned["train_first_pass"]["bits_per_byte"]
                - frozen["train_first_pass"]["bits_per_byte"]
            ),
            "validation_bits_per_byte_delta_vs_frozen": (
                learned["validation_first_pass_continuing"]["bits_per_byte"]
                - frozen["validation_first_pass_continuing"]["bits_per_byte"]
            ),
            "validation_context_delta_vs_same_g14_bias_only": (
                learned["validation_first_pass_continuing"]["bits_per_byte"]
                - bias["validation_first_pass_continuing"]["bits_per_byte"]
            ),
            "value_credit_validation_bits_per_byte_delta_vs_head_only": (
                value_credit["validation_first_pass_continuing"]["bits_per_byte"]
                - learned["validation_first_pass_continuing"]["bits_per_byte"]
            ),
        },
    }
    _atomic_json(args.output, report)
    _checkpoint(args.checkpoint, arms, {
        "result": str(args.output.relative_to(ROOT)),
        "source": report["source"],
        "model": report["model"],
        "protocol": report["protocol"],
    }, bias_only=bias_only)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "ib_owt_10000")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-events", type=int, default=100_000)
    parser.add_argument("--validation-events", type=int, default=100_000)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--n-slots", type=int, default=32)
    parser.add_argument("--head-step", type=float, default=0.01)
    parser.add_argument("--value-step", type=float, default=0.001)
    parser.add_argument("--elapsed", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20261031)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--report-every", type=int, default=10_000)
    args = parser.parse_args()
    if args.d_model < 1 or args.n_slots < 1 or args.train_events < 1 or args.validation_events < 1:
        parser.error("dimensions and event counts must be positive")
    if args.report_every < 0:
        parser.error("report interval must be nonnegative")
    report = run(args)
    effect = report["effect_of_local_head_learning"]
    print(json.dumps({
        "output": str(args.output),
        "checkpoint": str(args.checkpoint),
        "train_bpb_delta": effect["train_bits_per_byte_delta_vs_frozen"],
        "validation_bpb_delta": effect["validation_bits_per_byte_delta_vs_frozen"],
        "validation_context_bpb_delta": effect["validation_context_delta_vs_same_g14_bias_only"],
        "value_credit_validation_bpb_delta": effect[
            "value_credit_validation_bits_per_byte_delta_vs_head_only"
        ],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
