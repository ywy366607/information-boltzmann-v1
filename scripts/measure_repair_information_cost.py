"""Measure useful information and discrete maintenance cost on repaired paths.

This is an exact finite witness measurement.  It does not assign joules to a
discrete unit and it does not turn a zero-information probe into one useful
bit.  The useful-bit value is mutual information between the finite world
branch and the recorded signal under the stated prior.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def entropy(values: list[float]) -> float:
    value = -sum(p * math.log2(p) for p in values if p > 0.0)
    return 0.0 if abs(value) < 1e-15 else value


def mutual_information(joint: dict[tuple[str, str], float]) -> float:
    world_mass: dict[str, float] = {}
    obs_mass: dict[str, float] = {}
    for (world, obs), mass in joint.items():
        world_mass[world] = world_mass.get(world, 0.0) + mass
        obs_mass[obs] = obs_mass.get(obs, 0.0) + mass
    value = 0.0
    for (world, obs), mass in joint.items():
        if mass <= 0.0:
            continue
        value += mass * math.log2(mass / (world_mass[world] * obs_mass[obs]))
    return value


def measurement(name: str, joint: dict[tuple[str, str], float],
                outgoing_units: int, incoming_units: int,
                record_write_units: int = 0,
                non_degenerate_prior: bool = False) -> dict[str, object]:
    world_probs = {}
    obs_probs = {}
    for (world, obs), mass in joint.items():
        world_probs[world] = world_probs.get(world, 0.0) + mass
        obs_probs[obs] = obs_probs.get(obs, 0.0) + mass
    useful_bits = mutual_information(joint)
    return {
        "name": name,
        "joint_world_observation": {
            f"{world}|{obs}": mass for (world, obs), mass in joint.items()
        },
        "world_entropy_bits": entropy(list(world_probs.values())),
        "observation_entropy_bits": entropy(list(obs_probs.values())),
        "useful_bits_mutual_information": useful_bits,
        "outgoing_dissipation_units": outgoing_units,
        "incoming_recharge_units": incoming_units,
        "net_discrete_cost_units": outgoing_units - incoming_units,
        "record_write_units": record_write_units,
        "total_accounted_discrete_units": outgoing_units + record_write_units,
        "non_degenerate_prior": non_degenerate_prior,
        "energy_calibrated": False,
        "energy_per_useful_bit_units": (
            outgoing_units / useful_bits if useful_bits > 0.0 else None
        ),
        "denominator_status": "positive" if useful_bits > 0.0 else "zero_useful_bits",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    # In the registered finite model probeA and probeB both record Signal.probe.
    # Equal prior over the two world labels therefore gives zero predictive gain.
    probe = measurement(
        "protected_probe_equal_prior",
        {("probeA", "probe"): 0.5, ("probeB", "probe"): 0.5},
        outgoing_units=2,
        incoming_units=2,
    )

    # The concrete guarded witness (search, probeA) has the deterministic
    # branch probeA -> answerA -> feedA before the third natural signal.
    guarded = measurement(
        "guarded_search_probeA_third_signal",
        {("feed", "feed"): 1.0},
        outgoing_units=2,
        incoming_units=3,
    )

    # These are registered actual transition rows: a search state receiving
    # answerA or answerB writes the corresponding candidate label.  The
    # balanced prior is a measurement condition, not the fixed initial law.
    answer_branch = measurement(
        "nondegenerate_answer_branch",
        {("probeA", "answerA"): 0.5, ("probeB", "answerB"): 0.5},
        outgoing_units=2,
        incoming_units=1,
        record_write_units=1,
        non_degenerate_prior=True,
    )

    result = {
        "scope": "exact finite path information/cost measurement",
        "source_contract": "RecorderContract remains conditional; this artifact supplies measured inputs",
        "measurements": [probe, guarded, answer_branch],
        "interpretation": {
            "probe": "The current probe consumes and replenishes two discrete units but records zero useful predictive bits because probeA and probeB alias to Signal.probe.",
            "guarded": "The concrete guarded witness has a deterministic third signal under the stated branch, so its mutual information is also zero; positive useful-bit claims require a non-degenerate prior or held-out predictive task.",
            "answer_branch": "The registered answerA/answerB rows carry one useful bit under a balanced branch prior; this is a conditional path measurement, not evidence that the fixed initial law produces that prior.",
            "physical_energy": "Not calibrated: discrete units are not joules.",
            "universal_theorem": False,
        },
    }
    print(json.dumps(result, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
