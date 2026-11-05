"""Measure the byte n-gram fast table on the 11-03 pair protocol stream (2026-11-03).

Same bytes as results/cgb_feedback_birth_20261103 (OpenWebText bytes, 5000 x 512, phases
A_first / B_fresh / A_revisit / C_fresh).  Reports the table's own prequential NLL per phase,
throughput with RAM arrays and with np.memmap arrays on disk, and memory footprint, next to
the GDN controls of that run.
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
import shutil  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from scripts.ib_local.ngram_fast_table import ByteNgramFastTable  # noqa: E402
from scripts.ib_local.train_gated_deltanet import byte_protocol  # noqa: E402

ROOT = Path(ROOT_DIR)
OUT = ROOT / "results/cgb_ngram_fast_table_20261103"


def protocol_stream(steps=5000, tokens=512):
    data = ROOT / "data/ib_owt_10000"
    offset = json.loads((data / "tokenizer.json").read_text(encoding="utf-8"))["offset"]
    raw = np.load(data / "train.npy", mmap_mode="r")
    need = steps * tokens + 1
    prefix = np.asarray(raw[:2 * need])
    decoded = (prefix[prefix >= offset] - offset)[:need].astype(np.uint8)
    return byte_protocol(decoded, steps, tokens)


def phase_means(nll, phases, tokens=512):
    out = {}
    for p in phases:
        a, b = p["start_update"] * tokens, p["stop_update"] * tokens
        out[p["name"]] = float(nll[a:b].mean())
    return out


def gdn_controls():
    rows = {}
    for seed in ("seed11", "seed12", "seed13"):
        path = ROOT / f"results/cgb_feedback_birth_20261103/{seed}/training_comparison.json"
        arms = json.loads(path.read_text())["arms"]
        rows[seed] = {arm: {k: v["mean_nll"] for k, v in arms[arm]["phase_scores"].items()}
                      for arm in arms}
    return rows


def measure(stream, phases, storage=None, slots_log2=14, label="ram"):
    table = ByteNgramFastTable(slots_log2=slots_log2, storage=storage)
    warm = ByteNgramFastTable(slots_log2=4)
    warm.run(stream[:64])  # compile outside the timed pass
    started = time.perf_counter()
    nll, _ = table.run(stream)
    seconds = time.perf_counter() - started
    occupied = float((table.count > 0).mean())
    row = dict(label=label, slots_log2=slots_log2, table_mib=table.nbytes / 2**20,
               bytes_per_second=len(nll) / seconds, seconds=seconds,
               occupied_fraction=occupied, phases=phase_means(nll, phases))
    del table
    return row


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stream, phases = protocol_stream()
    results = [measure(stream, phases, slots_log2=s, label=f"ram_2^{s}") for s in (12, 14, 16)]
    disk = OUT / "memmap_scratch"
    results.append(measure(stream, phases, storage=disk, slots_log2=14, label="disk_memmap_2^14"))
    shutil.rmtree(disk, ignore_errors=True)
    report = dict(stream_bytes=len(stream), phases=phases, table=results, gdn_controls=gdn_controls())
    (OUT / "table_alone.json").write_text(json.dumps(report, indent=1))
    for r in results:
        ph = r["phases"]
        print(f"{r['label']:18s} {r['table_mib']:7.0f} MiB {r['bytes_per_second']:9.0f} B/s "
              f"occupied {r['occupied_fraction']:.2f}  " +
              "  ".join(f"{k} {v:.3f}" for k, v in ph.items()))
    for seed, arms in report["gdn_controls"].items():
        ph = arms["decay_then_delta"]
        print(f"GDN control {seed}  " + "  ".join(f"{k} {ph[k]:.3f}" for k in
                                                   ("A_first", "B_fresh", "A_revisit", "C_fresh")))


if __name__ == "__main__":
    main()
