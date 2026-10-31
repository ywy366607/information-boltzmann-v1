from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from fine_grain.active_gdn2 import ActiveInferenceGDN2


def run(seed: int = 0, horizon: float = 1.0) -> dict[str, object]:
    torch.manual_seed(seed)
    memory = ActiveInferenceGDN2(d_model=4, n_slots=4, res=4)
    state = memory.initial_state(1, torch.device("cpu"), torch.float32)
    tokens = torch.tensor(
        [[[1.0, -0.5, 0.25, 0.75], [-0.25, 0.5, 1.0, -1.0],
          [0.5, 0.25, -0.75, 0.125], [0.0, 1.0, -0.25, 0.5]]]
    )
    centers = memory.atlas.unsqueeze(0)
    keys = memory.address(tokens, centers)
    empty = memory.assimilate(
        state, tokens, centers, torch.zeros(1), update_motion=False,
    )
    written = memory.assimilate(
        state, tokens, centers, torch.ones(1), update_motion=False,
    )
    erased = memory.dynamic_prior(written, torch.tensor([horizon]))
    zero_write_delta = float((empty.natural - state.natural).abs().max().detach())
    written_error = memory.association_probe_error(written, keys, tokens)
    erased_error = memory.association_probe_error(erased, keys, tokens)
    written_uncertainty = memory.association_probe_uncertainty(written, keys)
    erased_uncertainty = memory.association_probe_uncertainty(erased, keys)
    retention = erased.precision / written.precision.clamp_min(1e-6)
    return {
        "seed": seed,
        "horizon": horizon,
        "zero_write_delta": zero_write_delta,
        "written_probe_error": float(written_error.item()),
        "erased_probe_error": float(erased_error.item()),
        "written_mean_probe_variance": float(written_uncertainty.mean().item()),
        "erased_mean_probe_variance": float(erased_uncertainty.mean().item()),
        "mean_precision_retention": float(retention.mean().item()),
        "max_precision_retention": float(retention.max().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon", type=float, default=1.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.seed, args.horizon)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
