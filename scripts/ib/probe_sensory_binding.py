"""Wiring-perturbation probes on the sensory-injection arm: is the frozen
wiring causally alive now that routing is mandatory?"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from information_boltzmann.core.fly_reservoir import FlyReservoirLM

CKPT = ROOT / "results/q8_fly_reservoir_sensory_3000/BBest.pt"
GRAPH = ROOT / "data/malecns_v1/fly_reservoir_full.npz"
DATA = np.load(ROOT / "data/ib_owt_gpt2/validation.npy", mmap_mode="r")
SITES = (8192, 12288, 16384, 20480)
WARM, SCORE = 256, 128

saved = torch.load(CKPT, map_location="cuda", weights_only=False)
cfg = saved["config"]
io_state = {k: v for k, v in saved["model"].items()
            if k not in ("edge_index", "edge_pre", "edge_post", "edge_weight")}
packed = dict(np.load(GRAPH, allow_pickle=False))
pre, post = packed["edge_pre"].copy(), packed["edge_post"].copy()
weight = packed["edge_weight"].copy()
n = packed["neuron_body_ids"].shape[0]


def build(edge_pre, edge_post, edge_weight, tag="probe"):
    out = ROOT / f"data/malecns_v1/_probe_sensory_{tag}.npz"
    np.savez_compressed(out, edge_pre=edge_pre, edge_post=edge_post,
                        edge_weight=edge_weight,
                        neuron_body_ids=packed["neuron_body_ids"],
                        superclass_names=packed["superclass_names"],
                        superclass_id=packed["superclass_id"])
    model = FlyReservoirLM(out, vocab_size=cfg.get("vocab_size", 50257),
                           d_model=cfg["d_model"], leak=cfg["leak"],
                           threshold=cfg["threshold"], injection="sensory").cuda()
    model.load_state_dict(io_state)
    model.eval()
    return model


@torch.no_grad()
def evaluate(model):
    per_site = []
    for start in SITES:
        h = torch.zeros(1, model.n_neurons, device="cuda")
        total = 0.0
        for offset in range(WARM + SCORE):
            token = torch.tensor([int(DATA[start + offset])], device="cuda")
            h, _ = model.step(h, token)
            if offset >= WARM:
                target = torch.tensor([int(DATA[start + offset + 1])], device="cuda")
                logits = model.read(h)
                total += float(torch.nn.functional.cross_entropy(logits, target))
        per_site.append(total / SCORE)
    return sum(per_site) / len(per_site), per_site


results = {}
results["baseline (trained wiring)"] = evaluate(build(pre, post, weight, "base"))

rng = np.random.default_rng(7)
perm = rng.permutation(n)
results["node relabel (degree-preserving)"] = evaluate(
    build(perm[pre], perm[post], weight, "relabel"))

results["zero transmission (feedforward)"] = evaluate(
    build(pre, post, np.zeros_like(weight), "zero"))

results["sign flip (E/I inverted)"] = evaluate(
    build(pre, post, -weight, "flip"))

print()
for name, (mean, sites) in results.items():
    print(f"{name:36s} NLL {mean:8.4f}  sites {['%.2f' % s for s in sites]}")
base = results["baseline (trained wiring)"][0]
print()
for name, (mean, _) in results.items():
    print(f"delta vs baseline: {name:36s} {mean - base:+8.4f}")
