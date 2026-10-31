"""CLI entry point to execute the four paired controls on real byte streams.

Usage:
    python scripts/run_cgb_small_life_harness.py --events 256 --real-data
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "information_boltzmann" / "core"))

from cgb_experimental_harness import (
    EmpiricalBaselines,
    format_stats,
    run_four_paired_controls,
)
from cgb_prequential_bridge import PackedTreePort


def fixture_rule(code, hidden, bit, context, depth, prefix):
    """Integer reference fixture for learner testing."""
    auxiliary = context[-1] if len(context) > 1 else 0
    return ((code + 9 if hidden else 1) * (3 if hidden == bit else 1)
            * (2 if hidden == ((context[0] >> depth) & 1) else 1)
            * (3 if hidden == ((auxiliary >> depth) & 1) else 1))


def create_reference_learner(seed: int = 20261025):
    return PackedTreePort(
        256,
        fixture_rule,
        context=(0,),
        seed=seed,
        learning=True,
        rule_id="harness-test-fixture-v1",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=int, default=256, help="Number of evaluated stream ticks")
    parser.add_argument("--real-data", action="store_true", help="Load real OpenWebText tokens")
    parser.add_argument("--output", type=Path, default=None, help="Path to write JSON results")
    parser.add_argument("--seed", type=int, default=20261025)
    args = parser.parse_args()

    if args.real_data:
        data_dir = ROOT / "data/ib_owt_10000"
        train_path = data_dir / "train.npy"
        val_path = data_dir / "validation.npy"
        if not train_path.exists() or not val_path.exists():
            raise FileNotFoundError("OWT dataset files missing in data/ib_owt_10000")
        import numpy as np
        train_bytes = np.load(train_path, mmap_mode="r")[:1_000_000].tobytes()
        val_bytes = np.load(val_path, mmap_mode="r")[:max(args.events + 64, 1024)].tobytes()
    else:
        # Synthetic deterministically structured stream for unit testing
        train_bytes = (b"The quick brown fox jumps over the lazy dog. " * 50)
        val_bytes = (b"The quick brown fox jumps over the lazy dog. 1234567890! " * 20)

    print(f"=== CGB Small Lifeform Experimental Harness ===")
    print(f"Stream events: {args.events}")
    print(f"Source size: {len(val_bytes)} bytes | Reference train size: {len(train_bytes)} bytes")

    results = run_four_paired_controls(
        learner_factory=lambda: create_reference_learner(args.seed),
        source_bytes=val_bytes,
        train_reference_bytes=train_bytes,
        events=args.events,
    )

    # Print clean summary table
    print("\n--- Summary of Paired Controls [Min | Median | Mean | Max] ---")
    modes = [
        ("Nominal (Active)", "nominal"),
        ("Frozen Learning", "frozen_learning"),
        ("History Disrupted", "history_disrupted"),
        ("Information Cut", "information_cut"),
        ("Resource Cut", "resource_cut"),
    ]
    for label, key in modes:
        entry = results[key]
        bits = entry["model_bits_per_byte"]["display"]
        gain_u = entry["gain_over_unigram_bits"]["display"]
        gain_b = entry["gain_over_bigram_bits"]["display"]
        surv = f"{entry['active_ticks']}/{entry['total_ticks']}"
        print(f"[{label:18}] Surv: {surv:>7} | Bits/Byte: {bits} | Gain vs Unigram: {gain_u} | Gain vs Bigram: {gain_b}")

    print("\n--- Pairwise Contrasts (Net Advantage of Nominal) ---")
    for contrast, val in results["pairwise_contrasts"].items():
        if isinstance(val, float):
            print(f"  {contrast:45}: {val:+.4f} bits/byte")
        else:
            print(f"  {contrast:45}: {val}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nSaved published artifact to: {args.output}")


if __name__ == "__main__":
    main()
