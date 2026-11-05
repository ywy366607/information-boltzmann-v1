"""Where is the mode bit held?  Per-head state contrast for runs of the mode task (2026-11-03).

The trained final state reads the evaluation mode stream twice: as is, and with every marker
letter flipped (eval_mode.flipped).  At each 512-byte chunk boundary the per-layer, per-head
memory matrices of the two reads are compared:
    contrast = |S_true - S_flip|_F / |S_true|_F,
binned by distance from the segment's marker (segments are 32768 bytes).  Also prints the
birth-formula latch ratio beta*eta*g / (1 - alpha(1-beta)) at a(x)=0 (analyze_feedback_birth).
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

import json  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from scripts.ib_local import gdn_fused  # noqa: E402
from scripts.ib_local.eval_mode import flipped  # noqa: E402
from scripts.ib_local.eval_recall import load  # noqa: E402
from scripts.ib_local.mode_task import EVAL_SEGMENT, eval_stream  # noqa: E402

CHUNK = 512
DISTANCES = (512, 2048, 4096, 8192, 16384, 32256)


def matrices(model, state):
    base = model.base
    layer = base.layers[0]
    flat = state.reshape(base.num_layers, -1)[:, :layer.matrix_size]
    return flat.reshape(base.num_layers, base.heads, layer.dk, layer.dv)


@torch.no_grad()
def states(model, state, stream):
    n = len(stream) // CHUNK * CHUNK
    data = torch.from_numpy(stream[:n].astype(np.int64)).cuda()[None]
    uniform = torch.full((1, CHUNK, 256), math.log(1 / 256), device="cuda")
    out = {}
    for at in range(0, n, CHUNK):
        _, state, _ = model.predict(data[:, at:at + CHUNK], state, uniform)
        out[at + CHUNK] = matrices(model, state).float().cpu()
    return out


def latch_ratio(model):
    rows = []
    for layer in model.base.layers:
        if not hasattr(layer, "feedback_logit"):
            return None
        alpha = layer._birth_retention().cpu()
        beta = torch.sigmoid(layer.beta_proj.bias.detach().float()).cpu()
        eta = torch.sigmoid(layer.feedback_logit.detach().float()).cpu()
        gain = F.softplus(layer.feedback_gain_raw.detach().float()).cpu()
        rows.append((beta * eta * gain / (1 - alpha * (1 - beta))).tolist())
    return rows


def main():
    gdn_fused.enable()
    runs = [Path(p) for p in sys.argv[1:]]
    stream, _, _ = eval_stream(12 * EVAL_SEGMENT)
    flip = flipped(stream)
    for run in runs:
        model, state0, _ = load(run)
        a, b = states(model, state0.clone(), stream), states(model, state0.clone(), flip)
        rows = {}
        for d in DISTANCES:
            keys = [k for k in a if k % EVAL_SEGMENT == d]
            c = torch.stack([(a[k] - b[k]).flatten(-2).norm(dim=-1)
                             / a[k].flatten(-2).norm(dim=-1).clamp_min(1e-9) for k in keys]).mean(0)
            rows[d] = c.tolist()
        ratio = latch_ratio(model)
        (run / "mode_heads.json").write_text(json.dumps(dict(contrast=rows, latch_ratio=ratio), indent=1))
        print(run.name, "latch ratio per layer/head:",
              None if ratio is None else [[round(x, 2) for x in r] for r in ratio])
        for d, c in rows.items():
            print(f"  d={d:>6}", " | ".join(" ".join(f"{x:.3f}" for x in layer) for layer in c))


if __name__ == "__main__":
    main()
