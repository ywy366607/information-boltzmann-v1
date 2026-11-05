"""Where does long-range information stop mattering? Diagnostics on saved GDN checkpoints (2026-11-03).

Frozen weights, no training; the trained state in each checkpoint is not used (every
measurement starts from the initial state on validation bytes), so these are diagnostics,
not arm comparisons.

D1 history use: NLL on the same validation targets when the whole recurrent state
   (matrix and convolution cache) is reset every L bytes, against never resetting.
D2 influence by distance: with the full (untruncated) graph over 4,160 bytes, the gradient
   norm of the loss on the last 64 bytes with respect to the layer-0 input at each earlier
   position, binned by distance.  This is the credit a full-BPTT learner would receive.
D3 readback use (feedback checkpoints): NLL with eta set to zero in all heads, or only in
   the heads born above the latch threshold.
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
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from scripts.ib_local import gdn_fused  # noqa: E402
from scripts.ib_local.gated_deltanet import GatedDeltaNetLM  # noqa: E402

ROOT = Path(ROOT_DIR)
OUT = ROOT / "results/cgb_long_memory_diagnosis_20261103"
CHECKPOINTS = {
    "w32_sigmoid_control_s11": "results/cgb_feedback_pair_20261103_5000/decay_then_delta/last.pt",
    **{f"w512_logtau_control_s{s}": f"results/cgb_feedback_birth_20261103/seed{s}/decay_then_delta/last.pt"
       for s in (11, 12, 13)},
    **{f"w512_logtau_feedback_s{s}": f"results/cgb_feedback_birth_20261103/seed{s}/feedback_delta/last.pt"
       for s in (11, 12, 13)},
}
RESETS = (32, 128, 512, 2048, 8192, None)
CHUNK = 32
SCORE_FROM = 8192
N_BYTES = 65536 + SCORE_FROM
BINS = ((0, 32), (32, 128), (128, 512), (512, 1024), (1024, 2048), (2048, 4096))


def validation_bytes(n):
    data = ROOT / "data/ib_owt_10000"
    offset = json.loads((data / "tokenizer.json").read_text(encoding="utf-8"))["offset"]
    raw = np.asarray(np.load(data / "validation.npy", mmap_mode="r")[:3 * n])
    decoded = (raw[raw >= offset] - offset)[:n + 1].astype(np.int64)
    return torch.from_numpy(decoded).cuda()[None]


def load(path):
    snapshot = torch.load(ROOT / path, map_location="cuda", weights_only=False)
    model = GatedDeltaNetLM(**snapshot["config"]["model"]).cuda().eval()
    model.load_state_dict(snapshot["model"])
    return model


@torch.no_grad()
def history_nll(model, data):
    """NLL over targets from SCORE_FROM on, one batch row per reset interval in RESETS."""
    rows = len(RESETS)
    state = model.initial_state(rows, device="cuda")
    fresh = model.initial_state(1, device="cuda")
    total = torch.zeros(rows, device="cuda")
    count = 0
    for at in range(0, N_BYTES, CHUNK):
        for row, reset in enumerate(RESETS):
            if reset and at % reset == 0:
                state[:, row] = fresh[:, 0]
        ids = data[:, at:at + CHUNK].expand(rows, -1)
        targets = data[:, at + 1:at + CHUNK + 1].expand(rows, -1)
        logits, state, _ = model.predict_logits(ids, state)
        if at >= SCORE_FROM:
            nll = F.cross_entropy(logits.reshape(-1, 256), targets.reshape(-1), reduction="none")
            total += nll.view(rows, -1).sum(-1); count += CHUNK
    return {str(r): float(v) for r, v in zip(RESETS, (total / count).tolist())}


@torch.no_grad()
def stream_nll(model, data, chunk=512):
    """NLL over targets from SCORE_FROM on, never resetting."""
    state = model.initial_state(1, device="cuda")
    total = torch.zeros((), device="cuda")
    for at in range(0, N_BYTES, chunk):
        loss, state, _ = model(data[:, at:at + chunk], data[:, at + 1:at + chunk + 1], state)
        if at >= SCORE_FROM:
            total += loss * chunk
    return float(total) / (N_BYTES - SCORE_FROM)


def influence(model, data, segments=8, length=4096, scored=64):
    """Mean gradient norm of the last-`scored`-byte loss w.r.t. layer-0 input, by distance bin."""
    sums = np.zeros(len(BINS))
    starts = np.linspace(0, N_BYTES - length - scored - 1, segments).astype(int)
    for start in starts:
        ids = data[:, start:start + length + scored]
        targets = data[:, start + 1:start + length + scored + 1]
        h = model.source.embedding(ids).detach().requires_grad_()
        state = model.initial_state(1, device="cuda")
        x = h
        for index, layer in enumerate(model.layers):
            x, _, _ = layer(x, state[index])
        logits = model.decoder(model.final_norm(x))
        loss = F.cross_entropy(logits[0, -scored:], targets[0, -scored:])
        (grad,) = torch.autograd.grad(loss, h)
        norms = grad[0].norm(dim=-1).cpu().numpy()
        # Distance d >= 1 counts back from the first scored target position `length`.
        for b, (lo, hi) in enumerate(BINS):
            sums[b] += norms[length - hi:length - lo].mean() if lo else norms[length - hi:length].mean()
    return (sums / segments).tolist()


@torch.no_grad()
def readback_ablation(model, data):
    out = {"intact": stream_nll(model, data)}
    saved = [layer.feedback_logit.detach().clone() for layer in model.layers]
    for name, heads in (("no_readback", slice(None)), ("no_latch_heads", slice(2, None))):
        for layer in model.layers:
            layer.feedback_logit[heads] = -30.0
        out[name] = stream_nll(model, data)
        for layer, value in zip(model.layers, saved):
            layer.feedback_logit.copy_(value)
    return out


def main():
    gdn_fused.enable()
    OUT.mkdir(parents=True, exist_ok=True)
    data = validation_bytes(N_BYTES)
    report = {}
    for name, path in CHECKPOINTS.items():
        model = load(path)
        started = time.perf_counter()
        row = {"history": history_nll(model, data)}
        row["influence"] = influence(model, data)
        if model.recurrence == "feedback_delta":
            row["readback"] = readback_ablation(model, data)
        row["seconds"] = time.perf_counter() - started
        report[name] = row
        full = row["history"]["None"]
        print(name, "history cost of reset every L:",
              {k: round(v - full, 4) for k, v in row["history"].items() if k != "None"},
              "| influence by distance (rel. to 32-128):",
              [round(v / row["influence"][1], 4) for v in row["influence"]],
              "| readback:" if "readback" in row else "",
              {k: round(v - row["readback"]["intact"], 4) for k, v in row.get("readback", {}).items()},
              f"({row['seconds']:.0f} s)", flush=True)
        del model; torch.cuda.empty_cache()
    report["_bins"] = BINS
    (OUT / "diagnosis.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
