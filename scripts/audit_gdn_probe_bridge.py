from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from fine_grain.active_gdn2 import ActiveInferenceGDN2


def run(seed: int = 0, erase_horizon: float = 64.0, overwrite_precision: float = 8.0, epsilon: float = 0.2) -> dict[str, object]:
    torch.manual_seed(seed)
    memory = ActiveInferenceGDN2(d_model=4, n_slots=4, res=4)
    state = memory.initial_state(1, torch.device("cpu"), torch.float32)
    tokens = torch.tensor(
        [[[1.0, -0.5, 0.25, 0.75], [-0.25, 0.5, 1.0, -1.0],
          [0.5, 0.25, -0.75, 0.125], [0.0, 1.0, -0.25, 0.5]]]
    )
    centers = memory.atlas.unsqueeze(0)
    keys = memory.address(tokens, centers)
    written = memory.assimilate(
        state, tokens, centers, torch.ones(1), update_motion=False,
    )
    erased = memory.dynamic_prior(written, torch.tensor([erase_horizon]))
    overwritten = memory.assimilate(
        written, -tokens, centers, torch.full((1,), overwrite_precision),
        update_motion=False,
    )

    def metrics(st: object) -> dict[str, float]:
        error = memory.association_probe_error(st, keys, tokens)
        variance = memory.association_probe_uncertainty(st, keys)
        return {
            "probe_error": float(error.detach().item()),
            "mean_probe_variance": float(variance.detach().mean().item()),
        }

    pairwise = torch.pdist(keys[0].detach())
    written_metrics = metrics(written)
    erased_metrics = metrics(erased)
    overwritten_metrics = metrics(overwritten)
    return {
        "seed": seed,
        "erase_horizon": erase_horizon,
        "overwrite_precision": overwrite_precision,
        "epsilon": epsilon,
        "address_rank": int(torch.linalg.matrix_rank(keys[0].detach()).item()),
        "address_pairwise_min_distance": float(pairwise.min().item()),
        "written": written_metrics,
        "erased": erased_metrics,
        "overwritten": overwritten_metrics,
        "bridge_conditions": {
            "probe_injective": bool(pairwise.min() > 1e-6),
            "written_probe_recoverable": written_metrics["probe_error"] <= epsilon,
            "overwritten_probe_unrecoverable": overwritten_metrics["probe_error"] > epsilon,
        },
        "zero_observation_write_delta": float(
            (memory.assimilate(state, tokens, centers, torch.zeros(1), update_motion=False).natural
             - state.natural).abs().max().detach().item()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.seed)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
