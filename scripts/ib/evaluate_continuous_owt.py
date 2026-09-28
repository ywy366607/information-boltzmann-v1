"""Evaluate language models on fixed warm local OpenWebText contexts.

For kinetic CBIM checkpoints, each held-out site begins from an independent
random-phase reconstruction of the checkpoint's mature NESS: it preserves the
terminal field's DC component and Fourier amplitude spectrum while discarding
the training-document phase.  It then receives its observed local context and
predicts its immediate continuation.  There is no cold start or reset within a
site.  GDN-2 has no field spectrum and therefore retains its explicitly marked
terminal-state fallback.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from information_boltzmann.core.torus3d import CBIMTorus3D, KineticBeliefState
from information_boltzmann.evaluation import WarmSiteSpec


def _gdn2_step(
    self: torch.nn.Module,
    state: torch.Tensor,
    current: torch.Tensor,
    *,
    micro_steps: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Expose the historical GDN-2 recurrence through the common event API."""
    if micro_steps != 1:
        raise ValueError("GDN-2 has no kinetic quadrature; its event update is K=1.")
    h = self.source.embedding(current[:, None])
    next_states = []
    diagnostics: dict[str, torch.Tensor] = {}
    for layer_index, layer in enumerate(self.layers):
        h, next_state, layer_diag = layer(h, state[layer_index])
        next_states.append(next_state)
        for key, value in layer_diag.items():
            diagnostics[f"l{layer_index}_{key}"] = value
    logits = self.decoder(self.final_norm(h))[:, 0]
    for key in ("state_norm", "decay_mean", "erase_mean", "write_mean"):
        diagnostics[key] = torch.stack(
            [diagnostics[f"l{layer_index}_{key}"] for layer_index in range(self.num_layers)]
        ).mean()
    return logits, torch.stack(next_states, dim=0), diagnostics


def _model_from_checkpoint(saved: dict[str, Any]) -> tuple[torch.nn.Module, int, str]:
    config = saved["config"]
    architecture = str(config.get("architecture", ""))
    if architecture.startswith("GatedDeltaNet2"):
        from scripts.ib_local.gated_deltanet_2 import GatedDeltaNet2LM

        model = GatedDeltaNet2LM(
            vocab_size=int(config.get("vocab_size", 50257)),
            d=int(config["d"]),
            layers=int(config["layers"]),
            heads=int(config["heads"]),
        )
        model.step = types.MethodType(_gdn2_step, model)  # type: ignore[attr-defined]
        return model, 1, "gdn2"
    if not architecture.startswith("CBIM"):
        raise ValueError("Expected a GDN-2 reference or a CBIM kinetic checkpoint.")
    return (
        CBIMTorus3D(
            shape=tuple(config["shape"]),
            velocities=int(config["velocities"]),
            content_dim=int(config["content_dim"]),
            collision_layers=int(config["collision_layers"]),
            relative_address=bool(config.get("relative_address", False)),
            v2_coordinate_components=bool(config.get("v2_coordinate_components", False)),
            readout_type=str(config.get("readout_type", "baseline")),
            readout_probes=int(config.get("readout_probes", 8)),
            readout_rounds=int(config.get("readout_rounds", 1)),
            write_type=str(config.get("write_type", "w2_impedance")),
            micro_steps=int(config.get("micro_steps", 1)),
            adaptive_clock=bool(config.get("adaptive_clock", False)),
            continuous_velocities=bool(config.get("continuous_velocities", False)),
            alpha_causal=float(config.get("alpha_causal", 0.90)),
            alpha_max=float(config.get("alpha_max", 2.50)),
            dissipation_type=str(config.get("dissipation_type", "quadratic")),
            dissipation_rank=int(config.get("dissipation_rank", 4)),
            gamma0_init=float(config.get("gamma0_init", 0.010)),
            nu_init=float(config.get("nu_init", 0.020)),
            three_clock=bool(config.get("three_clock", False)),
            tau_mem=float(config.get("tau_mem", 3.0)),
            nu_s_init=float(config.get("nu_s_init", 0.020)),
        ),
        int(config.get("micro_steps", 1)),
        "kinetic",
    )


def _as_float(value: torch.Tensor | float | int | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().mean().cpu())
    return float(value)


@torch.no_grad()
def evaluate_warm_sites(
    model: torch.nn.Module,
    terminal_state: torch.Tensor | KineticBeliefState,
    tokens: np.ndarray,
    spec: WarmSiteSpec,
    micro_steps: int,
    ness_phase_seed: int = 11,
) -> dict[str, Any]:
    """Measure fixed local contexts from a mature kinetic regime without cold starts."""
    belief_mode = isinstance(terminal_state, KineticBeliefState)
    terminal_field = terminal_state.field if belief_mode else terminal_state
    if terminal_field.ndim != 5:
        raise ValueError("Expected a rank-five checkpoint recurrent state")
    batch_axis = 1 if hasattr(model, "num_layers") else 0
    if terminal_field.shape[batch_axis] != 1:
        raise ValueError("Warm-site evaluation accepts exactly one carried stream state")
    if max(spec.site_starts) + spec.required_tokens_per_site > len(tokens):
        raise ValueError("A requested site exceeds the validation stream")

    device = next(model.parameters()).device
    terminal = terminal_field.to(
        device=device, dtype=next(model.parameters()).dtype
    ).detach().clone()
    terminal_precision = (
        terminal_state.precision.to(device=device, dtype=terminal.dtype).detach().clone()
        if belief_mode else None
    )
    model.eval()
    kinetic_ness = (
        hasattr(model, "set_ness_prior")
        and hasattr(model, "initial_state")
        and not hasattr(model, "num_layers")
    )
    if kinetic_ness:
        # Keep the mature field's energy spectrum and DC component but discard
        # its training-text phase before every independent held-out site.
        model.set_ness_prior(terminal)  # type: ignore[attr-defined]

    def initialize_site(site_index: int) -> tuple[torch.Tensor | KineticBeliefState, str, bool]:
        if kinetic_ness:
            torch.manual_seed(ness_phase_seed + site_index)
            torch.cuda.manual_seed_all(ness_phase_seed + site_index)
            field = model.initial_state(  # type: ignore[attr-defined]
                1, device=device, dtype=terminal.dtype, warm_start=True
            )
            if belief_mode:
                if terminal_precision is None:
                    raise RuntimeError("Belief evaluation requires terminal posterior precision")
                return (
                    KineticBeliefState(field=field, precision=terminal_precision.clone()),
                    "mature_ness_spectrum_with_randomized_phase_and_posterior_precision_then_local_warm_in",
                    True,
                )
            return (
                field,
                "mature_ness_spectrum_with_randomized_phase_then_local_warm_in",
                True,
            )
        return (
            terminal.clone(),
            "checkpoint_terminal_state_clone_then_local_warm_in",
            False,
        )

    all_diagnostics: dict[str, list[float]] = {}
    sites: list[dict[str, Any]] = []
    started = time.perf_counter()

    for site_index, start in enumerate(spec.site_starts):
        state, state_policy, random_phase = initialize_site(site_index)
        losses: list[float] = []
        site_diagnostics: dict[str, list[float]] = {}
        warm_diagnostics: dict[str, list[float]] = {}

        def advance(index: int, score: bool) -> None:
            nonlocal state
            current = torch.as_tensor([int(tokens[index])], device=device, dtype=torch.long)
            target = torch.as_tensor([int(tokens[index + 1])], device=device, dtype=torch.long)
            if belief_mode:
                logits, state, diag = model.belief_step(  # type: ignore[attr-defined]
                    state, current, micro_steps=micro_steps)
            else:
                logits, state, diag = model.step(state, current, micro_steps=micro_steps)  # type: ignore[attr-defined]
            if score:
                losses.append(float(F.cross_entropy(logits.float(), target).cpu()))
            destination = site_diagnostics if score else warm_diagnostics
            for key, value in diag.items():
                scalar = _as_float(value)
                if scalar is not None and math.isfinite(scalar):
                    destination.setdefault(key, []).append(scalar)
                    if score:
                        all_diagnostics.setdefault(key, []).append(scalar)

        for index in range(start, start + spec.warm_in_tokens):
            advance(index, score=False)
        score_start = start + spec.warm_in_tokens
        for index in range(score_start, score_start + spec.score_tokens):
            advance(index, score=True)
        sites.append({
            "site": site_index,
            "validation_start": start,
            "warm_in_tokens": spec.warm_in_tokens,
            "score_start": score_start,
            "score_tokens": spec.score_tokens,
            "nll": float(np.mean(losses)),
            "state_policy": state_policy,
            "cold_start": False,
            "random_phase_resampling": random_phase,
            "resets_within_site": 0,
            "warm_in_diagnostics": {
                key: {
                    "first": float(values[0]),
                    "last": float(values[-1]),
                    "mean": float(np.mean(values)),
                }
                for key, values in warm_diagnostics.items()
            },
            "diagnostics": {
                key: float(np.mean(values)) for key, values in site_diagnostics.items()
            },
        })

    elapsed = time.perf_counter() - started
    total_scored = len(sites) * spec.score_tokens
    return {
        "nll": float(np.mean([site["nll"] for site in sites])),
        "sites": sites,
        "sites_count": len(sites),
        "warm_in_tokens": spec.warm_in_tokens,
        "score_tokens_per_site": spec.score_tokens,
        "state_policy": (
            "mature_ness_spectrum_with_randomized_phase_and_posterior_precision_then_fixed_local_warm_in"
            if kinetic_ness and belief_mode else
            "mature_ness_spectrum_with_randomized_phase_then_fixed_local_warm_in"
            if kinetic_ness else
            "one_mature_checkpoint_terminal_state_cloned_for_fixed_independent_local_contexts"
        ),
        "cold_start": False,
        "random_phase_resampling": kinetic_ness,
        "resets_within_site": 0,
        "mature_ness_initializations": len(sites) if kinetic_ness else 0,
        "checkpoint_terminal_state_clones": 0 if kinetic_ness else len(sites),
        "seconds": elapsed,
        "scored_tokens_per_second": total_scored / max(elapsed, 1e-12),
        "diagnostics": {
            key: float(np.mean(values)) for key, values in all_diagnostics.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/ib_owt_gpt2"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warm-in-tokens", type=int, default=256)
    parser.add_argument("--score-tokens", type=int, default=128)
    parser.add_argument("--site-starts", type=str, default="8192,12288,16384,20480")
    parser.add_argument("--micro-steps", type=int,
                        help="Must match saved kinetic K; omitted for its saved value.")
    parser.add_argument("--ness-phase-seed", type=int, default=11,
                        help="Deterministic phase seed for energy-aligned kinetic NESS sites.")
    args = parser.parse_args()
    spec = WarmSiteSpec(
        site_starts=tuple(int(value) for value in args.site_starts.split(",") if value),
        warm_in_tokens=args.warm_in_tokens,
        score_tokens=args.score_tokens,
    )
    if not torch.cuda.is_available():
        parser.error("CUDA is required for this evaluator")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "state" not in saved:
        parser.error("Checkpoint has no mature terminal state; local warm-in must not cold-start.")
    model, trained_k, model_family = _model_from_checkpoint(saved)
    model = model.cuda().eval()
    model.load_state_dict(saved["model"])
    micro_steps = args.micro_steps if args.micro_steps is not None else trained_k
    if micro_steps != trained_k:
        parser.error(f"K mismatch: checkpoint was trained at K={trained_k}, requested K={micro_steps}")
    stream = np.load(args.data / "validation.npy", mmap_mode="r")
    terminal_state: torch.Tensor | KineticBeliefState = saved["state"]
    if getattr(model, "write_agent", None) is not None:
        if "precision" not in saved:
            parser.error("Predictive-port checkpoint has no terminal posterior precision.")
        terminal_state = KineticBeliefState(
            field=saved["state"], precision=saved["precision"])
    report = evaluate_warm_sites(
        model, terminal_state, stream, spec, micro_steps,
        ness_phase_seed=args.ness_phase_seed,
    )
    report.update({
        "protocol": "IB-warm-local-language-v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "checkpoint_step": int(saved.get("step", -1)),
        "architecture": saved["config"].get("architecture"),
        "model_family": model_family,
        "continuous_velocity_channels": int(saved["config"].get("velocities", 0)),
        "K": micro_steps,
        "integration_resolution": "kinetic" if model_family == "kinetic" else "not_applicable",
        "dataset": str(args.data / "validation.npy"),
        "ness_phase_seed": args.ness_phase_seed if model_family == "kinetic" else None,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
