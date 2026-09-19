"""Frozen single-factor causal interventions for anatomical-port CBIM."""
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

from scripts.ib_local.cbim_malecns_port_transport import (
    CBIMMaleCNSPortTransport,
)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/ib_owt_gpt2"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--history", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=128)
    parser.add_argument("--sites", type=int, default=4)
    parser.add_argument("--start", type=int, default=8192)
    args = parser.parse_args()
    torch.set_num_threads(2)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = saved["config"]
    model = CBIMMaleCNSPortTransport(
        config["graph"], velocities=config["velocities"],
        content_dim=config["content_dim"], input_topk=config["input_topk"],
        output_topk=config["output_topk"],
        stream_fraction=config["stream_fraction"]).cuda().eval()
    model.load_state_dict(saved["model"])
    data = np.load(args.data / "validation.npy", mmap_mode="r")
    window = int(config["tokens"])
    if args.history % window or args.horizon % window:
        raise ValueError("history and horizon must be multiples of training tokens")

    def batch(start):
        ids = torch.as_tensor(np.array(data[start:start + window]),
                              dtype=torch.long, device="cuda")[None]
        targets = torch.as_tensor(np.array(data[start + 1:start + window + 1]),
                                  dtype=torch.long, device="cuda")[None]
        return ids, targets

    def advance(state, start, length, disable_collision=False,
                disable_transport=False, score=False):
        total = 0.0
        for offset in range(0, length, window):
            ids, targets = batch(start + offset)
            loss, state, _ = model(
                ids, targets, state, disable_collision=disable_collision,
                disable_transport=disable_transport)
            if score:
                total += float(loss) * window
        return state, total / length if score else None

    records = []
    for site in range(args.sites):
        start = args.start + site * 4096
        initial = model.initial_state(1, device="cuda")
        shared, _ = advance(initial, start, args.history)
        values = {}
        for name, no_collision, no_transport in (
                ("full", False, False),
                ("no_collision", True, False),
                ("no_transport", False, True)):
            _, values[name] = advance(
                shared.clone(), start + args.history, args.horizon,
                disable_collision=no_collision,
                disable_transport=no_transport, score=True)
        records.append({
            "site": site, "start": start, **values,
            "collision_delta_nll": values["no_collision"] - values["full"],
            "transport_delta_nll": values["no_transport"] - values["full"],
        })

    report = {
        "architecture": config["architecture"],
        "checkpoint": str(args.checkpoint), "step": saved["step"],
        "protocol": {
            "dataset": "OpenWebText validation stream",
            "sites": args.sites, "shared_full_history": args.history,
            "intervention_horizon": args.horizon,
            "intervention": "single-factor bypass only during scoring horizon",
        },
        "sites": records,
        "mean_full_nll": float(np.mean([x["full"] for x in records])),
        "mean_collision_delta_nll": float(np.mean(
            [x["collision_delta_nll"] for x in records])),
        "mean_transport_delta_nll": float(np.mean(
            [x["transport_delta_nll"] for x in records])),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
