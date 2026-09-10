"""Offline critical-dissipation candidate scan with frozen parameters, not a controller."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fine_grain.information_boltzmann import InformationBoltzmann
from fine_grain.information_boltzmann.diagnostics import conditional_response


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--tokens", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--events", type=int, default=128)
    p.add_argument("--device", default="cpu")
    p.add_argument("--gammas", type=float, nargs="+", default=[1., .3, .1, .03])
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--solver-steps", type=int)
    args = p.parse_args()
    torch.set_num_threads(1)
    config = json.loads(args.config.read_text())
    rows = []
    for gamma in args.gammas:
        for epsilon in (1e-4, 1e-5):
            torch.manual_seed(args.seed)  # same network across sweep arms
            config["dynamics"].update(gamma=gamma, adaptive_gamma=False)
            if args.solver_steps is not None:
                config["dynamics"]["steps"] = args.solver_steps
            model = InformationBoltzmann.from_config(config).to(device=args.device, dtype=torch.float64)
            gen = torch.Generator(device=args.device).manual_seed(args.seed)
            state = model.initialize(torch.tensor([1], device=args.device), gen)
            tokens = np.load(args.tokens, mmap_mode="r")[:args.events]
            result = conditional_response(model, state, [1]+tokens[:-1].tolist(), gen,
                                          epsilon=epsilon, burn_in=min(32, args.events//4))
            result.pop("rates")
            rows.append({"gamma": gamma, "seed": args.seed, "solver_steps": model.steps, **result})
            print(json.dumps(rows[-1]), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"status":"calibration_only_no_NESS_or_learning_claim", "rows":rows},indent=2))


if __name__ == "__main__":
    main()
