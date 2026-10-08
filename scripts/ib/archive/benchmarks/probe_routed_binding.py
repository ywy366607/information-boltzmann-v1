"""Test-time wiring perturbation probes on any trained fly reservoir checkpoint.

All probes load the SAME trained I/O weights and differ only in the wiring:
baseline / degree-preserving node relabel / zero transmission / sign flip.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from information_boltzmann.core.fly_reservoir import FlyReservoirLM

GRAPH = ROOT / "data/malecns_v1/fly_reservoir_full.npz"
DATA = np.load(ROOT / "data/ib_owt_gpt2/validation.npy", mmap_mode="r")
SITES = (8192, 12288, 16384, 20480)
WARM, SCORE = 256, 128


def main():
    parser = argparse.ArgumentParser(description="Wiring perturbation probe")
    parser.add_argument("--ckpt", type=str,
                        default="results/q8_fly_reservoir_routed_3000/BBest.pt",
                        help="Path to checkpoint (.pt)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to output json file (defaults to ckpt dir / causality_probe.json)")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt).resolve()
    if not ckpt_path.exists():
        print(f"Error: {ckpt_path} does not exist.")
        sys.exit(1)

    out_file = Path(args.output).resolve() if args.output else ckpt_path.parent / "causality_probe.json"

    print(f"Loading {ckpt_path}...")
    saved = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    cfg = saved["config"]
    print(f"Config: injection={cfg.get('injection', 'broadcast')}, read_surface={cfg.get('read_surface', 'all')}")

    io_state = {k: v for k, v in saved["model"].items()
                if k not in ("edge_index", "edge_pre", "edge_post", "edge_weight")}
    graph_path = Path(cfg.get("graph", GRAPH))
    if not graph_path.is_absolute():
        graph_path = ROOT / graph_path
    packed = dict(np.load(graph_path, allow_pickle=False))
    pre, post = packed["edge_pre"].copy(), packed["edge_post"].copy()
    weight = packed["edge_weight"].copy()
    n = packed["neuron_body_ids"].shape[0]

    tag_prefix = ckpt_path.parent.name
    temp_files = []

    def build(edge_pre, edge_post, edge_weight, tag="probe", perm=None, zero_dan=False):
        out = ROOT / f"data/malecns_v1/_probe_{tag_prefix}_{tag}.npz"
        temp_files.append(out)
        kwargs = {
            "edge_pre": edge_pre,
            "edge_post": edge_post,
            "edge_weight": edge_weight,
            "neuron_body_ids": packed["neuron_body_ids"],
            "superclass_names": packed["superclass_names"],
            "superclass_id": packed["superclass_id"],
        }
        if "delay_splits" in packed:
            kwargs["delay_splits"] = packed["delay_splits"]
        if "lambda_0" in packed:
            kwargs["lambda_0"] = packed["lambda_0"][perm] if perm is not None else packed["lambda_0"]
        if "dan_edge_pre" in packed:
            if perm is not None:
                kwargs["dan_edge_pre"] = perm[packed["dan_edge_pre"]]
                kwargs["dan_edge_post"] = perm[packed["dan_edge_post"]]
            else:
                kwargs["dan_edge_pre"] = packed["dan_edge_pre"]
                kwargs["dan_edge_post"] = packed["dan_edge_post"]
            if zero_dan or tag == "zero":
                kwargs["dan_edge_weight"] = np.zeros_like(packed["dan_edge_weight"])
            else:
                kwargs["dan_edge_weight"] = packed["dan_edge_weight"]
            kwargs["dan_delay_splits"] = packed["dan_delay_splits"]
            kwargs["dan_scale"] = packed["dan_scale"]
        np.savez_compressed(out, **kwargs)
        leak_val = cfg["leak"]
        if isinstance(leak_val, str):
            leak_val = 0.90
        model = FlyReservoirLM(out, vocab_size=cfg.get("vocab_size", 50257),
                               d_model=cfg["d_model"], leak=leak_val,
                               threshold=cfg["threshold"],
                               injection=cfg.get("injection", "broadcast"),
                               read_surface=cfg.get("read_surface", "all")).cuda()
        model.load_state_dict(io_state)
        model.eval()
        return model

    @torch.no_grad()
    def evaluate(model):
        per_site = []
        for start in SITES:
            if getattr(model, "topographic_writer", None) is not None:
                model.topographic_writer.a_adapt.zero_()
            h = torch.zeros(1, model.n_neurons, device="cuda")
            ring = tuple(torch.zeros(1, model.n_neurons, device="cuda") for _ in range(4)) if model.has_delays else None
            i_syn = torch.zeros(1, model.n_neurons, device="cuda")
            total = 0.0
            for offset in range(WARM + SCORE):
                token = torch.tensor([int(DATA[start + offset])], device="cuda")
                if model.has_delays:
                    h, _, ring, i_syn = model.step(h, token, ring, i_syn)
                else:
                    h, _, i_syn = model.step(h, token, i_syn=i_syn)
                if offset >= WARM:
                    target = torch.tensor([int(DATA[start + offset + 1])], device="cuda")
                    logits = model.read(h)
                    total += float(torch.nn.functional.cross_entropy(logits, target))
            per_site.append(total / SCORE)
        return float(sum(per_site) / len(per_site)), [float(s) for s in per_site]

    started = time.perf_counter()
    results = {}

    try:
        print("Evaluating baseline (trained wiring)...")
        results["baseline (trained wiring)"] = evaluate(build(pre, post, weight, "base"))

        print("Evaluating node relabel (degree-preserving)...")
        rng = np.random.default_rng(7)
        perm = rng.permutation(n)
        results["node relabel (degree-preserving)"] = evaluate(
            build(perm[pre], perm[post], weight, "relabel", perm=perm))

        print("Evaluating zero transmission (feedforward)...")
        results["zero transmission (feedforward)"] = evaluate(
            build(pre, post, np.zeros_like(weight), "zero"))

        print("Evaluating sign flip (E/I inverted)...")
        results["sign flip (E/I inverted)"] = evaluate(
            build(pre, post, -weight, "flip"))

        if "dan_edge_pre" in packed:
            print("Evaluating zero dopamine gating (only fast ionotropic)...")
            results["zero dopamine gating"] = evaluate(
                build(pre, post, weight, "zero_dan", zero_dan=True))

        print()
        base = results["baseline (trained wiring)"][0]
        out_dict = {
            "ckpt": str(ckpt_path),
            "config": {
                "injection": cfg.get("injection", "broadcast"),
                "read_surface": cfg.get("read_surface", "all"),
                "leak": cfg.get("leak"),
                "threshold": cfg.get("threshold"),
            },
            "baseline_nll": base,
            "probes": {}
        }
        for name, (mean, sites) in results.items():
            delta = mean - base
            print(f"{name:36s} NLL {mean:8.4f} (delta {delta:+8.4f})  sites {['%.2f' % s for s in sites]}")
            out_dict["probes"][name] = {"nll": mean, "delta_vs_baseline": delta, "sites": sites}

        out_file.write_text(json.dumps(out_dict, indent=2), encoding="utf-8")
        print(f"\nSaved probe results to {out_file} (elapsed: {time.perf_counter() - started:.1f}s)")
    finally:
        for f in temp_files:
            f.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
