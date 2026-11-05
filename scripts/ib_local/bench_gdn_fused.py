"""Training throughput of the fused recurrence against the reference loop (2026-11-03).

Uses the real trainer (CUDA graph per window, grouped clipping, AdamW) and the model of
results/cgb_feedback_pair_20261103_5000.  Bytes come from a repository document, so the
numbers measure speed and agreement, not language ability.

1. Agreement: reference and fused runs from identical weights on identical bytes, window
   32, report the per-update loss difference.
2. Throughput and peak memory for several backpropagation windows.
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
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

from scripts.ib_local import gated_deltanet  # noqa: E402
from scripts.ib_local import gdn_fused  # noqa: E402
from scripts.ib_local.train_cbim_malecns_internal_time import TruncatedInternalTimeGraphTrainer  # noqa: E402

ROOT = Path(ROOT_DIR)


def build(recurrence: str, seed: int = 11):
    torch.manual_seed(seed)
    return gated_deltanet.GatedDeltaNetLM(vocab_size=256, d=128, layers=4, heads=4,
                                          recurrence=recurrence, mlp_hidden=256, conv_kernel=4,
                                          content_budget=32.0).cuda()


def stream(n: int) -> torch.Tensor:
    raw = (ROOT / "docs/information_boltzmann/CGB_LANGUAGE_LIFE_MODEL_V1_20261025.md").read_bytes()
    raw = (raw * (n // len(raw) + 2))[:n + 1]
    return torch.tensor(list(raw), dtype=torch.long, device="cuda")[None]


def run(recurrence, fused, tokens, chunk, steps, data, seed=11):
    (gdn_fused.enable if fused else gdn_fused.disable)()
    model = build(recurrence, seed)
    torch.cuda.reset_peak_memory_stats()
    runner = TruncatedInternalTimeGraphTrainer(model, tokens=tokens, chunk_tokens=chunk, lr=3e-4)
    losses, times = [], []
    for step in range(steps):
        ids = data[:, step * tokens:(step + 1) * tokens]
        targets = data[:, step * tokens + 1:(step + 1) * tokens + 1]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss, _, _ = runner.step(ids, targets)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        losses.append(float(loss))
    gdn_fused.disable()
    warm = times[2:] or times
    return {"recurrence": recurrence, "fused": fused, "tokens": tokens, "chunk": chunk,
            "steps": steps, "bytes_per_second": tokens * len(warm) / sum(warm),
            "peak_mib": torch.cuda.max_memory_allocated() / 2**20, "losses": losses}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/cgb_gdn_fused_20261103/bench.json")
    args = parser.parse_args()
    data = stream(2048 * 12)
    rows = []
    for recurrence in ("decay_then_delta", "feedback_delta"):
        ref = run(recurrence, False, 512, 32, 8, data)
        fus = run(recurrence, True, 512, 32, 8, data)
        diff = max(abs(a - b) for a, b in zip(ref["losses"], fus["losses"]))
        rows += [ref, fus]
        print(f"{recurrence}: max |loss diff| over 8 updates = {diff:.2e}; "
              f"reference {ref['bytes_per_second']:.0f} B/s, fused {fus['bytes_per_second']:.0f} B/s")
        rows[-1]["max_loss_diff_vs_reference"] = diff
    for tokens, chunk in ((512, 128), (512, 512), (2048, 512), (2048, 2048)):
        row = run("feedback_delta", True, tokens, chunk, 8, data)
        rows.append(row)
        print(f"fused feedback tokens={tokens} window={chunk}: "
              f"{row['bytes_per_second']:.0f} B/s, peak {row['peak_mib']:.0f} MiB")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"device": torch.cuda.get_device_name(),
                                       "torch": torch.__version__, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
