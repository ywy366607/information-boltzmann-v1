"""Numerical audit: exact collision invariants and FP64 collapse control."""
import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fine_grain.information_boltzmann import InformationBoltzmann, PhaseState
from fine_grain.information_boltzmann.collision import CollisionKernel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(11)
    device = args.device
    kernel = CollisionKernel(2, 8, max_rate=10).to(device=device, dtype=torch.float64)
    v = torch.randn(16, 2, device=device, dtype=torch.float64)
    phase = PhaseState(torch.zeros_like(v), v)
    with torch.no_grad():
        changed, _, report = kernel(phase, 1, torch.Generator(device=device).manual_seed(3))
    invariants = {
        "positions_exact": torch.equal(changed.x, phase.x),
        "momentum_max_error": float((changed.v.sum(0)-v.sum(0)).abs().max()),
        "energy_sum_error": float((changed.v.square().sum()-v.square().sum()).abs()),
        **report,
    }
    config = json.loads(args.config.read_text(encoding="utf-8"))
    torch.manual_seed(config["seed"])
    base = InformationBoltzmann.from_config(config).to(device=device, dtype=torch.float64)
    tokens = np.load(args.tokens)[:64]
    collapse = {}
    for enabled in (True, False):
        model = copy.deepcopy(base)
        if not enabled:
            model.collision.max_rate = 0
        generator = torch.Generator(device=device).manual_seed(11)
        with torch.no_grad():
            state = model.initialize(torch.tensor([1], device=device), generator)
            initial = float(state.belief()[1].trace())
            current = 1
            rows = []
            for i, token in enumerate(tokens):
                state, _, _ = model.advance(state, current, generator)
                current = int(token)
                if (i+1) % 8 == 0:
                    rows.append({"event": i+1, "position_variance": float(state.belief()[1].trace()),
                                 "velocity_variance": float(state.moments()["velocity_variance"])})
        collapse["collision_on" if enabled else "collision_off"] = {
            "initial_position_variance": initial, "trajectory": rows,
            "final_initial_variance_ratio": rows[-1]["position_variance"] / initial,
        }
    ratio = collapse["collision_on"]["final_initial_variance_ratio"]
    if ratio < 1e-4:
        interpretation = "Noncollapse fails even without learning; compare collision-off. This is not an NESS proof."
    else:
        interpretation = "Langevin fluctuation-dissipation sustains finite phase variance across stream; non-collapse verified."
    summary = {"device": device, "dtype": "float64", "collision_invariants": invariants,
               "frozen_parameter_collapse_controls": collapse,
               "interpretation": interpretation}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
