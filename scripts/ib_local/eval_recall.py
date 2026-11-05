"""Frozen-weight, live-state recall evaluation for runs of train_gdn_fast_table.py --stream recall.

For each gap G, a fresh stream (unseen text, fresh keys) is read from the run's final
trained state (and its saved table, if any); value bytes of each query are scored.
Reports per gap: mean NLL per value byte, the share of queries whose every value byte is
the argmax (exact recall), and the same for the first value byte.
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

from scripts.ib_local import gdn_fused  # noqa: E402
from scripts.ib_local.gated_deltanet import GatedDeltaNetLM  # noqa: E402
from scripts.ib_local.ngram_fast_table import ByteNgramFastTable  # noqa: E402
from scripts.ib_local.recall_task import EVAL_GAPS, eval_stream  # noqa: E402
from scripts.ib_local.train_gdn_fast_table import FLOOR, TableGDN  # noqa: E402

CHUNK = 512


def load(run: Path):
    snapshot = torch.load(run / "last.pt", map_location="cuda", weights_only=False)
    summary = snapshot["summary"]
    base = GatedDeltaNetLM(**summary["model_config"]).cuda()
    if summary.get("self_state") not in (None, False, "off"):
        from scripts.ib_local.self_state import add_self_state
        add_self_state(base, oracle=summary["self_state"] == "oracle",
                       latch=summary["self_state"] == "latch").cuda()
    if summary.get("latch_slots"):
        from scripts.ib_local.latch_slot import add_latch_slots
        add_latch_slots(base)
    model = TableGDN(base, summary["table"]).cuda().eval()
    model.load_state_dict(snapshot["model"])
    return model, snapshot["state"].cuda(), summary


def fresh_table(run: Path):
    saved = np.load(run / "table.npz")
    table = ByteNgramFastTable(slots_log2=int(saved["slots_log2"]))
    for name in ("rows", "count", "tag", "err"):
        getattr(table, name)[...] = saved[name]
    table.history = np.uint64(saved["history"])
    return table


@torch.no_grad()
def score(model, state, table, stream, spans):
    n = (len(stream) - 1) // CHUNK * CHUNK
    data = torch.from_numpy(stream[:n + 1].astype(np.int64)).cuda()[None]
    logp_all = np.empty((n, 256), dtype=np.float32)
    uniform = torch.full((1, CHUNK, 256), math.log(1 / 256), device="cuda")
    for at in range(0, n, CHUNK):
        if table is not None:
            _, probs = table.run(stream[at:at + CHUNK + 1], keep_probs=True)
            tab = torch.from_numpy(np.log(np.maximum(probs, FLOOR))).cuda()[None]
        else:
            tab = uniform
        logits, state, _ = model.predict(data[:, at:at + CHUNK], state, tab)
        logp_all[at:at + CHUNK] = torch.log_softmax(logits[0], -1).cpu().numpy()
    nll, exact, first = [], [], []
    for lo, hi, _ in spans:
        if hi > n:
            continue
        positions = np.arange(lo, hi)            # value bytes; predicted from position p-1
        rows = logp_all[positions - 1]
        target = stream[positions]
        nll.extend(-rows[np.arange(len(target)), target])
        hits = rows.argmax(-1) == target
        exact.append(bool(hits.all())); first.append(bool(hits[0]))
    return dict(value_byte_nll=float(np.mean(nll)), exact_recall=float(np.mean(exact)),
                first_byte_recall=float(np.mean(first)), queries=len(exact))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--source-start", type=int, default=3_000_000,
                        help="first decoded train byte of evaluation text; must exceed training bytes")
    args = parser.parse_args()
    gdn_fused.enable()
    for run in args.runs:
        model, state0, summary = load(run)
        rows = {}
        for gap in EVAL_GAPS:
            probes = int(min(128, 900_000 // (gap + 100)))
            stream, spans = eval_stream(gap, probes, source_start=args.source_start)
            table = fresh_table(run) if summary["table"] else None
            rows[str(gap)] = score(model, state0.clone(), table, stream, spans)
        (run / "recall_eval.json").write_text(json.dumps(rows, indent=1))
        print(run.name, {g: (round(r["value_byte_nll"], 3), round(r["exact_recall"], 3))
                         for g, r in rows.items()})


if __name__ == "__main__":
    main()
