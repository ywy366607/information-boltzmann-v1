"""Calibrate the SOC input-edge target from E2 (2026-11-04).

E2 (results/cgb_mode_task_20261103/RESULT.md): the trained latch head L2H3, read along a
constant key with its real alpha, beta and the text's value v scaled by lambda, kept a bit for
32k events at lambda = 0.1 (separation 0.93-0.98) and mostly at 0.3 (0.56-0.72), and lost it at
lambda = 1. The disturbance D = beta (1 - eta) |v| scales with lambda while the tolerance T does
not, so the holding range in units of D/T is lambda * u1, where u1 is that head's D/T on the
same text. This prints u1 for the three seeds; the SOC target u* is chosen from it.
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
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from scripts.ib_local import gdn_fused  # noqa: E402
from scripts.ib_local.eval_recall import load  # noqa: E402
from scripts.ib_local.mode_task import EVAL_SEGMENT, eval_stream  # noqa: E402
from scripts.ib_local.soc import head_balances  # noqa: E402

CHUNK = 512


@torch.no_grad()
def main():
    gdn_fused.enable()
    stream, _, _ = eval_stream(2 * EVAL_SEGMENT)
    n = len(stream) // CHUNK * CHUNK
    data = torch.from_numpy(stream[:n].astype(np.int64)).cuda()[None]
    out = {}
    for seed in (11, 12, 13):
        run = Path(f"results/cgb_mode_task_20261103/latch_s{seed}")
        model, state, _ = load(run)
        uniform = torch.full((1, CHUNK, 256), np.log(1 / 256), device="cuda")
        ratios = []
        for at in range(0, n, CHUNK):
            ids = data[:, at:at + CHUNK]
            bal = head_balances(model, ids, state)
            ratios.append((bal["D"] / bal["T"].clamp_min(1e-6)).cpu())
            _, state, _ = model.predict(ids, state, uniform)
        u = torch.stack(ratios).median(0).values
        out[f"latch_s{seed}"] = dict(L2H3_u1=float(u[2, 3]), all=u.tolist())
        print(f"latch_s{seed}: L2H3 D/T at lambda=1 (median over windows) = {float(u[2, 3]):.3f};"
              f" holding range lambda 0.1-0.3 -> u = {0.1 * float(u[2, 3]):.3f}-{0.3 * float(u[2, 3]):.3f}",
              flush=True)
    Path("results/cgb_rich_world_20261104/input_edge_calibration.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
