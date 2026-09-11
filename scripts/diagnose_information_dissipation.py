"""Frozen-input controls for information retention versus heat-bath support.

This is a bounded diagnostic. It never trains, writes checkpoints, or makes a
claim about language quality, a NESS, or asymptotic criticality.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fine_grain.information_boltzmann import InformationBoltzmann, PhaseState
from fine_grain.information_boltzmann.behavior import continuation_distance
from fine_grain.information_boltzmann.diagnostics import conditional_response


def clone_state(state: PhaseState) -> PhaseState:
    return PhaseState(state.x.clone(), state.v.clone(), state.time)


def energy(model: InformationBoltzmann, state: PhaseState) -> float:
    return float(((state.v.square() + model.force.kappa * state.x.square()).sum(-1).mean() / 2).detach())


def phase_rms_distance(left: PhaseState, right: PhaseState) -> float:
    """Particle-indexed full phase gap; both arms share their initial sample."""
    delta = torch.cat((left.x - right.x, left.v - right.v), dim=-1)
    return float(delta.square().mean().sqrt().detach())


def mean(values: list[float]) -> float:
    return float(np.mean(values))


def run_arm(model, tokens, prefix_events, arm_seed):
    generator = torch.Generator(device=tokens.device).manual_seed(arm_seed)
    state = model.initialize(torch.tensor([1], device=tokens.device), generator)
    initial = clone_state(state)
    rows = []
    prefix_state = None
    for event, token in enumerate(tokens.tolist(), start=1):
        before = energy(model, state)
        budget = {}
        state, _, report = model._advance_steps(state, int(token), generator, budget=budget)
        predicted = (budget["drive_work"] + budget["trap_work"]
                     - budget["deterministic_damping_loss"]
                     + budget["ou_fluctuation_energy"]
                     + budget["drift_potential_change"]
                     + budget.get("collision_energy_error", 0.0))
        moments = state.moments()
        rows.append({
            "event": event,
            "energy": energy(model, state),
            "energy_change": energy(model, state) - before,
            "accounting_residual": energy(model, state) - before - predicted,
            "drive_work": budget["drive_work"],
            "damping_loss": budget["deterministic_damping_loss"],
            "ou_fluctuation_energy": budget["ou_fluctuation_energy"],
            "position_variance": float(moments["position_variance"].detach()),
            "velocity_variance": float(moments["velocity_variance"].detach()),
            "accepted_collisions": report["accepted"],
        })
        if event == prefix_events:
            prefix_state = clone_state(state)
    response = conditional_response(
        model, initial, tokens.tolist(), torch.Generator(device=tokens.device).manual_seed(arm_seed),
        epsilon=1e-5, burn_in=max(8, len(tokens) // 4),
    )
    return {"prefix_state": prefix_state, "rows": rows,
            "response": {key: value for key, value in response.items() if key != "rates"}}


def summarize(rows):
    midpoint = len(rows) // 2
    early, late = rows[:midpoint], rows[midpoint:]
    return {
        "mean_energy_early": mean([r["energy"] for r in early]),
        "mean_energy_late": mean([r["energy"] for r in late]),
        "mean_position_variance_late": mean([r["position_variance"] for r in late]),
        "mean_velocity_variance_late": mean([r["velocity_variance"] for r in late]),
        "mean_drive_work": mean([r["drive_work"] for r in rows]),
        "mean_damping_loss": mean([r["damping_loss"] for r in rows]),
        "mean_ou_fluctuation_energy": mean([r["ou_fluctuation_energy"] for r in rows]),
        "max_abs_accounting_residual": max(abs(r["accounting_residual"]) for r in rows),
        "accepted_collisions": sum(r["accepted_collisions"] for r in rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--events", type=int, default=96)
    parser.add_argument("--prefix-events", type=int, default=48)
    parser.add_argument("--suffix-events", type=int, default=16)
    parser.add_argument("--gammas", type=float, nargs="+", default=[0.03, 0.1, 0.3])
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.0, 0.1])
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 47])
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.prefix_events + args.suffix_events > args.events:
        raise ValueError("prefix_events + suffix_events must fit inside events")
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    token_values = np.load(args.tokens, mmap_mode="r")[:args.events]
    if len(token_values) != args.events:
        raise ValueError("Token stream is shorter than requested diagnostic")
    base_config = json.loads(args.config.read_text(encoding="utf-8"))
    records = []
    for gamma in args.gammas:
        for temperature in args.temperatures:
            for seed in args.seeds:
                torch.manual_seed(seed)
                configuration = copy.deepcopy(base_config)
                configuration["dynamics"].update(gamma=gamma, temperature=temperature,
                                                   gamma_mode="fixed", adaptive_gamma=False)
                base_model = InformationBoltzmann.from_config(configuration).to(device=device, dtype=torch.float64)
                token_arms = {
                    "ordered": token_values.copy(),
                    "shuffled": np.random.default_rng(100000 + seed).permutation(token_values),
                    "no_drive": token_values.copy(),
                }
                arm_runs = {}
                for name, values in token_arms.items():
                    model = copy.deepcopy(base_model)
                    if name == "no_drive":
                        model.force.amplitude = 0.0
                    sequence = torch.as_tensor(values, device=device)
                    arm_runs[name] = run_arm(model, sequence, args.prefix_events, seed)
                    records.append({
                        "gamma": gamma, "temperature": temperature, "seed": seed, "arm": name,
                        "summary": summarize(arm_runs[name]["rows"]),
                        "response": arm_runs[name]["response"],
                    })
                    print(json.dumps(records[-1]), flush=True)
                continuations = [token_values[args.prefix_events:args.prefix_events + args.suffix_events].tolist()]
                ordered = arm_runs["ordered"]["prefix_state"]
                for name in ("shuffled", "no_drive"):
                    probe = continuation_distance(base_model, ordered, arm_runs[name]["prefix_state"], continuations,
                                                  seed=200000 + seed)
                    matching = next(r for r in records if r["gamma"] == gamma and r["temperature"] == temperature
                                    and r["seed"] == seed and r["arm"] == name)
                    matching["common_suffix_total_variation"] = probe["total_variation_by_suffix"][0]
                    matching["prefix_phase_rms_distance"] = phase_rms_distance(
                        ordered, arm_runs[name]["prefix_state"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "frozen_control_diagnostic_not_NESS_criticality_or_language_quality_evidence",
        "design": {
            "same_random_network_per_seed": True,
            "ordered": "validation stream in order",
            "shuffled": "same token multiset, deterministic permutation",
            "no_drive": "ordered stream, data-drive amplitude set to zero",
            "common_suffix": "ordered and comparison prefix states receive identical future tokens and common random numbers",
            "temperature": "0 removes OU bath noise; 0.1 enables it",
            "training": "none",
        },
        "events": args.events, "prefix_events": args.prefix_events,
        "suffix_events": args.suffix_events, "records": records,
        "interpretation": "Only compare matched arms within a seed/gamma/temperature. Finite common-noise total variation is an empirical behavior probe, not probabilistic bisimulation."
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
