"""One long pass over real text: does a finite body keep learning once it is full?

The gate reads about 390k bytes, far less than an infinite stream.  This reports
the online score per segment, with learning on and nothing reset, against the
same text read by online order-1.

Usage:
    python scripts/cgb_language_long_run.py context_cells --budget-kib 1024 --bytes 3000000 --json out.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from cgb_language_candidates import CANDIDATES  # noqa: E402
import cgb_language_gate as gate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("body")
    parser.add_argument("--budget-kib", type=int, default=1024)
    parser.add_argument("--bytes", type=int, default=3_000_000)
    parser.add_argument("--segment", type=int, default=250_000)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    # Past every interval the gate and its replications use.
    text = gate.stream("train", 20_000_000, 20_000_000 + int(args.bytes * 1.02))[:args.bytes]
    body = CANDIDATES[args.body](args.budget_kib)
    counts, totals = np.full((256, 256), 0.05), np.full(256, 12.8)
    rows, began, previous = [], time.time(), None
    for start in range(0, len(text), args.segment):
        nll, order1 = 0.0, 0.0
        segment = text[start:start + args.segment]
        for byte in segment:
            nll -= body.log_prob(byte)
            if previous is not None:
                order1 -= np.log(counts[previous, byte] / totals[previous])
                counts[previous, byte] += 1
                totals[previous] += 1
            body.observe(byte)
            previous = byte
        rows.append({"end": start + len(segment), "nll": nll / len(segment),
                     "order1": order1 / len(segment), "state_bytes": body.state_bytes()})
        print(rows[-1], f"{time.time() - began:.0f}s", flush=True)
    args.json.write_text(json.dumps({"body": args.body, "budget_kib": args.budget_kib, "rows": rows},
                                    indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
