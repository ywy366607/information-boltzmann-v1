"""How tree-like are the trained body's representations? Gromov delta-hyperbolicity (2026-11-04).

Question (user, 11-04): should the latent transition table use Lorentz (hyperbolic) keys, or are
the representations Euclidean? Forward passes only, no training. For each run, the layer outputs
h (after each GDN layer and after the final norm) are taken every STRIDE bytes of unseen OWT
validation text, carrying the trained state from the run (never reset). For random subsets of N
points with a random base point w, the Gromov products A_xy = (d(x,w) + d(y,w) - d(x,y)) / 2 give
delta = max_xyz [min(A_xz, A_zy) - A_xy], reported as delta_rel = 2 delta / diameter
(Khrulkov et al. 2020): 0 for a tree metric, larger for flatter geometry.
Distances: Euclidean and angular (arccos of the cosine, the sphere metric). Baseline: Gaussian
points with the data's mean and covariance (same N and dimension), i.e. no structure beyond
second order. Representations learned in Euclidean coordinates can still be tree-like; a
delta_rel well below the Gaussian baseline says the data's structure is.
"""

from __future__ import annotations

import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
if sys.path and os.path.abspath(sys.path[0]) == script_dir:
    sys.path.pop(0)  # ib_local/types.py would shadow the standard library
ROOT_DIR = os.path.dirname(os.path.dirname(script_dir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from scripts.ib_local import gdn_fused  # noqa: E402
from scripts.ib_local.eval_recall import load  # noqa: E402
from scripts.ib_local.recall_task import owt_bytes  # noqa: E402

CHUNK = 512


@torch.no_grad()
def layer_outputs(model, state, data, stride):
    """h after every layer and after the final norm, every `stride` bytes: [L+1, P, d]."""
    base = model.base
    uniform = torch.full((1, CHUNK, 256), math.log(1 / 256), device="cuda")
    picks = [[] for _ in range(base.num_layers + 1)]
    for at in range(0, data.shape[1] - CHUNK + 1, CHUNK):
        ids = data[:, at:at + CHUNK]
        h = base.source.embedding(ids)
        states = []
        for index, layer in enumerate(base.layers):
            h, next_state, _ = layer(h, state[index])
            states.append(next_state)
            picks[index].append(h[0, stride - 1::stride].float())
        picks[-1].append(base.final_norm(h)[0, stride - 1::stride].float())
        state = torch.stack(states, 0)
        del uniform  # the table input is not used by --no-table runs
        uniform = None
    return [torch.cat(p) for p in picks]


def delta_rel(dist: torch.Tensor, base: int, block: int = 32) -> float:
    """Relative Gromov delta of a distance matrix with base point `base`."""
    row = dist[base]
    A = 0.5 * (row[:, None] + row[None, :] - dist)
    delta = torch.zeros((), device=dist.device)
    for start in range(0, len(A), block):
        maxmin = torch.minimum(A[start:start + block, :, None], A[None, :, :]).amax(dim=1)
        delta = torch.maximum(delta, (maxmin - A[start:start + block]).amax())
    return float(2 * delta / dist.max())


def distances(x: torch.Tensor, metric: str) -> torch.Tensor:
    if metric == "euclid":
        return torch.cdist(x, x)
    u = F.normalize(x, dim=-1)
    return torch.arccos((u @ u.T).clamp(-1, 1))


def summary(values):
    v = np.array(values)
    return dict(min=float(v.min()), median=float(np.median(v)), mean=float(v.mean()), max=float(v.max()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--bytes", type=int, default=262144)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--points", type=int, default=1500)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gdn_fused.enable()
    text = owt_bytes("validation", args.bytes)
    data = torch.from_numpy(text.astype(np.int64)).cuda()[None]
    rng = np.random.default_rng(0)
    report = {}
    for run in args.runs:
        model, state, _ = load(run)
        outs = layer_outputs(model, state.clone(), data, args.stride)
        rows = {}
        for index, x in enumerate(outs):
            name = f"layer{index}" if index < len(outs) - 1 else "final_norm"
            mean, cov = x.mean(0), torch.cov(x.T)
            gauss = torch.distributions.MultivariateNormal(
                mean, covariance_matrix=cov + 1e-5 * torch.eye(len(mean), device=x.device))
            row = {}
            for metric in ("euclid", "angle"):
                data_vals, gauss_vals = [], []
                for _ in range(args.repeats):
                    pick = torch.from_numpy(rng.choice(len(x), args.points, replace=False)).cuda()
                    base = int(rng.integers(args.points))
                    data_vals.append(delta_rel(distances(x[pick], metric), base))
                    g = gauss.sample((args.points,))
                    gauss_vals.append(delta_rel(distances(g, metric), base))
                row[metric] = dict(data=summary(data_vals), gaussian=summary(gauss_vals))
                print(f"{run.name} {name} {metric}: data {row[metric]['data']['median']:.3f} "
                      f"[{row[metric]['data']['min']:.3f}, {row[metric]['data']['max']:.3f}]  "
                      f"gaussian {row[metric]['gaussian']['median']:.3f}", flush=True)
            rows[name] = row
        report[run.name] = rows
        del model
        torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(config=vars(args) | {"runs": [str(r) for r in args.runs],
                                                                 "output": str(args.output)},
                                           report=report), indent=1))


if __name__ == "__main__":
    main()
