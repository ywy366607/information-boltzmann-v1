"""Run the language gate on text no candidate was chosen on.

The gate's intervals fixed the search, so a pass there can be selection.  This
shifts every interval to unused text and rebuilds the yardsticks on it, with the
same thresholds.

Usage:
    python scripts/cgb_language_gate_replicate.py context_cells --shift 3000000 --json out.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import cgb_language_gate as gate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("body")
    parser.add_argument("--shift", type=int, default=3_000_000)
    parser.add_argument("--budget-kib", type=int, default=1024)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    original, other = gate.stream, gate.other_source
    gate.stream = lambda split, start, stop: original(
        split, start + (args.shift if split == "train" else 200_000),
        stop + (args.shift if split == "train" else 200_000))
    gate.other_source = lambda start, stop: other(start + args.shift, stop + args.shift)
    gate.REFERENCE = args.json.with_name(f"reference_shift{args.shift}.npz")

    reference = gate.load_reference()
    names, inventory = gate.load_inventory()
    evidence, candidate = gate.collect_live(args.body, args.budget_kib)
    report = gate.evaluate(args.body, evidence, candidate, args.budget_kib, reference, names, inventory)
    report["shift"] = args.shift
    for row in report["rows"]:
        print(f"{row['id']} [{row['status']:8s}] {row['title']}: {row['detail']}")
    print("ADMITTED" if report["admitted"] else "REJECTED", report["counts"], report["phases"], flush=True)
    print("yardsticks", {key: value for key, value in reference.items() if key in ("order0", "order1")},
          "table", float(reference["table_nll"].mean()), flush=True)
    args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
