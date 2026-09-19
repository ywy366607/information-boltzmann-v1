"""Measure useful history and alien-history interference on real OWT streams.

This is a frozen-checkpoint diagnostic.  Two independent OWT prefixes are
followed by the same suffix.  The suffix is scored from the correct semantic
field, a zero semantic field, an alien semantic field with matched controller
state, and the complete alien state.  Collision and transport bypasses start
from the correct state and use the same suffix.
"""
from __future__ import annotations

import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if sys.path and os.path.abspath(sys.path[0]) == SCRIPT_DIR:
    sys.path.pop(0)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from scripts.ib_local.cbim_malecns import CBIMMaleCNS
from scripts.ib_local.cbim_malecns_critical import CBIMMaleCNSCritical
from scripts.ib_local.cbim_malecns_critical_port import CBIMMaleCNSCriticalPort
from scripts.ib_local.cbim_malecns_v3 import CBIMMaleCNSV3


DEFAULT_CHECKPOINTS = [
    Path("results/cbim_malecns_256_3000/BBest.pt"),
    Path("results/cbim_malecns_v3_3000/BBest.pt"),
    Path("results/cbim_malecns_critical_3000/BBest.pt"),
    Path("results/cbim_malecns_critical_port_3000/BBest.pt"),
]


def load_model(checkpoint: Path, device: torch.device):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = saved["config"]
    architecture = cfg["architecture"]
    common = dict(
        graph_path=cfg["graph"],
        queries=cfg.get("queries", 4),
        heads=cfg.get("heads", 4),
    )
    if architecture == "CBIM-MaleCNS-graph-v1":
        model = CBIMMaleCNS(d=cfg.get("channels", 64), **common)
    elif architecture == "CBIM-MaleCNS-open-boundary-kinetic-v3":
        model = CBIMMaleCNSV3(
            velocities=cfg.get("velocities", 8),
            content_dim=cfg.get("content_dim", 8),
            **common,
        )
    elif architecture == "CBIM-MaleCNS-critical-Maxwell-v1":
        model = CBIMMaleCNSCritical(
            velocities=cfg.get("velocities", 8),
            content_dim=cfg.get("content_dim", 8),
            probe_epsilon=cfg.get("probe_epsilon", 1e-3),
            controller_ema=cfg.get("controller_ema", .01),
            controller_lr=cfg.get("controller_lr", .01),
            **common,
        )
    elif architecture == "CBIM-MaleCNS-critical-passive-port-v2":
        model = CBIMMaleCNSCriticalPort(
            velocities=cfg.get("velocities", 8),
            content_dim=cfg.get("content_dim", 8),
            checkpoint_tokens=cfg.get("critical_measure_interval", 32),
            probe_epsilon=cfg.get("probe_epsilon", 1e-3),
            controller_ema=cfg.get("controller_ema", .05),
            controller_lr=cfg.get("controller_lr", .02),
            flux_ema=cfg.get("flux_ema", .01),
            flux_lr=cfg.get("flux_lr", .02),
            max_critical_a=cfg.get("max_critical_a", .25),
            **common,
        )
    else:
        raise ValueError(f"unsupported architecture: {architecture}")
    model.load_state_dict(saved["model"], strict=True)
    model.to(device).eval()
    return model, saved, cfg


def semantic_field(model, state: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "d") and state.shape[-1] != model.d:
        return state[..., :model.d]
    return state


def replace_semantic_field(
    model, base: torch.Tensor, replacement: torch.Tensor
) -> torch.Tensor:
    if base.shape[-1] == replacement.shape[-1]:
        return replacement.clone()
    output = base.clone()
    output[..., :model.d] = replacement[..., :model.d]
    return output


@torch.no_grad()
def warm_states(model, prefixes: torch.Tensor, chunk: int = 128) -> torch.Tensor:
    state = None
    for start in range(0, prefixes.shape[1] - 1, chunk):
        stop = min(prefixes.shape[1] - 1, start + chunk)
        _, state, _ = model(
            prefixes[:, start:stop], prefixes[:, start + 1:stop + 1], state
        )
    return state


@torch.no_grad()
def step_model(
    model,
    state: torch.Tensor,
    token: torch.Tensor,
    step_index: int,
    disable_collision: bool = False,
    disable_transport: bool = False,
):
    architecture = model.architecture if hasattr(model, "architecture") else "CBIM-MaleCNS-graph-v1"
    if architecture == "CBIM-MaleCNS-graph-v1":
        field, _ = model.source(state, token)
        if not disable_transport:
            field, _ = model.transport(field)
        if not disable_collision:
            field, _ = model.scattering(field)
        return model.decoder(model.readout(field)), field

    if architecture == "CBIM-MaleCNS-open-boundary-kinetic-v3":
        field, resource, fatigue = model.unpack(state)
        field, resource, envelope, _ = model.source(
            field, resource, fatigue, token
        )
        if not disable_transport:
            field, _ = model.transport(field)
        if not disable_collision:
            field, _ = model.collision(field)
        field, _ = model.outflow(field, resource, fatigue, envelope)
        local_energy = .5 * field.square().sum(-1)
        fatigue = (
            model.fatigue_decay * fatigue
            + (1.0 - model.fatigue_decay) * local_energy
        )
        next_state = torch.cat(
            (field, resource[..., None], fatigue[..., None]), -1
        )
        return model.decoder(model.readout(field)), next_state

    if architecture == "CBIM-MaleCNS-critical-Maxwell-v1":
        next_state, _ = model.evolve(
            state, token, disable_collision, disable_transport
        )
        field = model.unpack(next_state)[0]
        return model.decoder(model.readout(field)), next_state

    if architecture == "CBIM-MaleCNS-critical-passive-port-v2":
        interval = max(1, int(model.checkpoint_tokens))
        next_state, _ = model.evolve(
            state,
            token,
            measure_critical=(step_index % interval == 0),
            disable_collision=disable_collision,
            disable_transport=disable_transport,
        )
        field = model.unpack(next_state)[0]
        return model.decoder(model.readout(field)), next_state

    raise ValueError(architecture)


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint: Path,
    data: np.ndarray,
    device: torch.device,
    prefix_a: int,
    prefix_b: int,
    warmup: int,
    score: int,
    horizons: list[int],
) -> dict[str, Any]:
    model, saved, cfg = load_model(checkpoint, device)
    a = torch.as_tensor(
        np.array(data[prefix_a:prefix_a + warmup + score + 1]),
        device=device,
        dtype=torch.long,
    )
    b = torch.as_tensor(
        np.array(data[prefix_b:prefix_b + warmup + 1]),
        device=device,
        dtype=torch.long,
    )
    prefixes = torch.stack((a[:warmup + 1], b[:warmup + 1]))
    warmed = warm_states(model, prefixes)
    true_state = warmed[:1].clone()
    alien_state = warmed[1:2].clone()
    zero_field = torch.zeros_like(semantic_field(model, true_state))
    zero_state = replace_semantic_field(model, true_state, zero_field)
    alien_field_state = replace_semantic_field(model, true_state, alien_state)

    arm_names = ["true", "zero_field", "alien_field", "full_alien"]
    states = torch.cat(
        (true_state, zero_state, alien_field_state, alien_state), dim=0
    )
    role_states = {
        "no_collision": true_state.clone(),
        "no_transport": true_state.clone(),
    }
    losses = {name: [] for name in arm_names + list(role_states)}
    field_distance = {name: [] for name in arm_names[1:]}
    suffix = a[warmup:warmup + score + 1]

    for index in range(score):
        token = suffix[index].expand(len(arm_names))
        logits, states = step_model(model, states, token, index)
        target = suffix[index + 1].expand(len(arm_names))
        per_arm = F.cross_entropy(logits, target, reduction="none")
        for arm_index, name in enumerate(arm_names):
            losses[name].append(float(per_arm[arm_index]))

        true_field = semantic_field(model, states[:1])
        true_norm = true_field.norm().clamp_min(1e-12)
        for arm_index, name in enumerate(arm_names[1:], start=1):
            distance = (
                semantic_field(model, states[arm_index:arm_index + 1])
                - true_field
            ).norm() / true_norm
            field_distance[name].append(float(distance))

        for name, role_state in list(role_states.items()):
            role_logits, role_states[name] = step_model(
                model,
                role_state,
                suffix[index:index + 1],
                index,
                disable_collision=(name == "no_collision"),
                disable_transport=(name == "no_transport"),
            )
            losses[name].append(
                float(F.cross_entropy(role_logits, suffix[index + 1:index + 2]))
            )

    points = []
    for horizon in horizons:
        row = {"horizon": horizon}
        for name in losses:
            row[f"nll_{name}"] = float(np.mean(losses[name][:horizon]))
        row["useful_history"] = row["nll_zero_field"] - row["nll_true"]
        row["alien_interference"] = (
            row["nll_alien_field"] - row["nll_zero_field"]
        )
        row["history_specificity"] = (
            row["nll_alien_field"] - row["nll_true"]
        )
        row["collision_value"] = row["nll_no_collision"] - row["nll_true"]
        row["transport_value"] = row["nll_no_transport"] - row["nll_true"]
        for name in field_distance:
            row[f"relative_field_distance_{name}"] = float(
                field_distance[name][horizon - 1]
            )
        points.append(row)

    result = {
        "checkpoint": str(checkpoint),
        "architecture": cfg["architecture"],
        "checkpoint_step": saved.get("step"),
        "prefix_a_offset": prefix_a,
        "prefix_b_offset": prefix_b,
        "warmup_tokens": warmup,
        "score_tokens": score,
        "controller_matched_for_field_interventions": True,
        "points": points,
    }
    del model, states, role_states
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", type=Path, default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--data", type=Path, default=Path("data/ib_owt_gpt2/validation.npy"))
    parser.add_argument("--prefix-a", type=int, default=0)
    parser.add_argument("--prefix-b", type=int, default=32768)
    parser.add_argument("--warmup", type=int, default=1024)
    parser.add_argument("--score", type=int, default=512)
    parser.add_argument(
        "--horizons", nargs="+", type=int,
        default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/published/cbim_predictive_interference.json"),
    )
    args = parser.parse_args()
    if max(args.horizons) > args.score:
        raise ValueError("all horizons must be <= score")
    if args.prefix_b + args.warmup + 1 > len(np.load(args.data, mmap_mode="r")):
        raise ValueError("prefix B exceeds validation data")

    torch.manual_seed(20260913)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.load(args.data, mmap_mode="r")
    report = {
        "dataset": str(args.data),
        "device": str(device),
        "definition": {
            "useful_history": "NLL(zero semantic field) - NLL(correct history)",
            "alien_interference": "NLL(alien semantic field) - NLL(zero semantic field)",
            "history_specificity": "NLL(alien semantic field) - NLL(correct history)",
        },
        "models": [],
    }
    for checkpoint in args.checkpoints:
        print(f"measuring {checkpoint}", flush=True)
        result = evaluate_checkpoint(
            checkpoint, data, device, args.prefix_a, args.prefix_b,
            args.warmup, args.score, sorted(set(args.horizons)),
        )
        report["models"].append(result)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(result["points"][-1], indent=2), flush=True)


if __name__ == "__main__":
    main()
