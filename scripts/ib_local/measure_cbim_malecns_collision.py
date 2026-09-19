"""Frozen scattering intervention for a trained MaleCNS graph-field CBIM."""
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(script_dir))
if sys.path and os.path.abspath(sys.path[0]) == script_dir:
    sys.path.pop(0)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.ib_local.cbim_malecns import CBIMMaleCNS


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/ib_owt_gpt2"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-tokens", type=int, default=4096)
    parser.add_argument("--history", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=128)
    parser.add_argument("--sites", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(2)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = saved["config"]
    model = CBIMMaleCNS(config["graph"], d=config["channels"]).cuda().eval()
    model.load_state_dict(saved["model"])
    window_tokens = config["tokens"]
    data = np.load(args.data / "validation.npy", mmap_mode="r")

    def batch(start):
        ids = torch.as_tensor(np.array(data[start:start + window_tokens]),
                              dtype=torch.long, device="cuda")[None]
        targets = torch.as_tensor(
            np.array(data[start + 1:start + window_tokens + 1]),
            dtype=torch.long, device="cuda")[None]
        return ids, targets

    def score(start, history, horizon, disabled):
        state = torch.zeros(1, *model.state_shape, device="cuda")
        for offset in range(0, history, window_tokens):
            ids, targets = batch(start + offset)
            _, state, _ = model(ids, targets, state,
                                disable_scattering=disabled)
        total = 0.
        for offset in range(history, history + horizon, window_tokens):
            ids, targets = batch(start + offset)
            loss, state, _ = model(ids, targets, state,
                                   disable_scattering=disabled)
            total += float(loss) * window_tokens
        return total / horizon

    full_keep = score(0, window_tokens, args.validation_tokens, False)
    full_off = score(0, window_tokens, args.validation_tokens, True)
    sites = []
    for site in range(args.sites):
        start = 8192 + site * 4096
        keep = score(start, args.history, args.horizon, False)
        off = score(start, args.history, args.horizon, True)
        sites.append({"site": site, "start": start, "full_nll": keep,
                      "no_scattering_nll": off, "delta_nll": off - keep})
    report = {
        "architecture": config["architecture"],
        "checkpoint": str(args.checkpoint),
        "step": saved["step"],
        "protocol": {
            "validation_tokens": args.validation_tokens,
            "history": args.history, "horizon": args.horizon,
            "sites": args.sites,
            "intervention": "bypass only graph conservative scattering",
        },
        "full_validation": {"full_nll": full_keep,
                            "no_scattering_nll": full_off,
                            "delta_nll": full_off - full_keep},
        "sites": sites,
        "site_mean_delta_nll": float(np.mean(
            [site["delta_nll"] for site in sites])),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
