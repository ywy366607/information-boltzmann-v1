#!/usr/bin/env python3
"""Held-out spatial audit for the admitted token champion.

The token audit bank is a registered subset of the training bank, so the
0.822 I2T admission does not measure spatial generalization. This script
re-renders the same deterministic digit scenes at shifted integer offsets
and at novel full-resolution addresses, then evaluates the frozen champion
without any training. A large drop separates place-locked memorization from
a readout that survived because it reads the stroke.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fine_grain.omni_model import DualStreamOmni
from fine_grain.omni_tasks import _paint, equal_energy_ink
from fine_grain.ocr_1px import render_digit_mask
from fine_grain.pythia_bridge import capability_champion_kwargs
from fine_grain.token_tasks import counterfactual_token_samples
from fine_grain.vlm_data import COLORS, OCR_DIGITS
from scripts.train_pythia_capabilities import make_cycle_eval_bank
from scripts.train_pythia_tokens import evaluate_graph_decode, evaluate_token_case

DEFAULT_INIT = ROOT / "checkpoints" / "omni_d64_pythia_token_best.pt"

# Integer offsets applied to every cell's rendered start address. The audit
# grid keeps res=16, box=6, so valid starts span 0..(res-box); offsets are
# clamped per cell to stay inside the canvas.
JITTER_OFFSETS = (
    (1, 0), (-1, 0), (0, 1), (0, -1),
    (2, 1), (-2, -1), (1, 2),
)


def digit_box(res: int) -> int:
    return max(4, min(16, (int(res) + 2) // 3, int(res) - 2))


def base_starts(res: int, place: str) -> tuple[int, int]:
    box = digit_box(res)
    rows = {"top": 1, "middle": (int(res) - box) // 2, "bottom": int(res) - box - 1}
    cols = {"left": 1, "center": (int(res) - box) // 2, "right": int(res) - box - 1}
    row, col = str(place).split("_")
    return int(rows[row]), int(cols[col])


def shifted_scene(digit: str, color: str, res: int, y0: int, x0: int):
    box = digit_box(res)
    stroke = render_digit_mask(str(digit), int(res), box, int(y0), int(x0))
    mask = torch.from_numpy(stroke.astype(np.float32)).view(1, int(res), int(res))
    blank = torch.zeros(1, 3, int(res), int(res), dtype=torch.float32)
    rgb = _paint(blank, mask, equal_energy_ink(str(color)))
    return rgb


def build_shifted_bank(base_bank: list[dict], res: int, dy: int, dx: int, tag: str):
    out = []
    for sample in base_bank:
        y0, x0 = base_starts(res, sample["source_place"])
        y0 = int(np.clip(y0 + dy, 0, int(res) - digit_box(res)))
        x0 = int(np.clip(x0 + dx, 0, int(res) - digit_box(res)))
        item = dict(sample)
        item["image"] = shifted_scene(sample["digit"], sample["source_color"], res, y0, x0)
        item["source_place"] = f"{tag}_{y0}_{x0}"
        out.append(item)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", default=str(DEFAULT_INIT))
    parser.add_argument("--out", default=str(ROOT / "results" / "published" / "pythia_token_heldout_audit.json"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--decode-limit", type=int, default=-1)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(0)
    model = DualStreamOmni(**capability_champion_kwargs(
        language="pythia", lm_device=args.device,
    )).to(device)
    if model.lm is not None:
        model.lm.to(device)
    report = model.load_visual_champion(Path(args.init), skip_language_interface=False)
    tokenizer = model.lm_tok
    res = int(model.res)

    base = make_cycle_eval_bank(res, "image_to_current")
    control_train = make_cycle_eval_bank(res, "image_to_current")
    banks = {"control_grid": (base, control_train)}
    for dy, dx in JITTER_OFFSETS:
        shifted = build_shifted_bank(base, res, dy, dx, f"j{dy}_{dx}")
        banks[f"jitter_{dy:+d}_{dx:+d}"] = (shifted, shifted)

    results = {}
    for name, (bank, control) in banks.items():
        token = evaluate_token_case(model, tokenizer, bank, control, device, 16)
        decode = evaluate_graph_decode(model, tokenizer, bank, args.decode_limit)
        wrong = sorted(
            record["digit"] for record in token["records"] if record["accuracy"] < 1.0
        )
        results[name] = {
            "token_accuracy": token["token_accuracy"],
            "median_gap": token["median_gap"],
            "decode_exact": decode["exact"],
            "decode_n": decode["n"],
            "wrong_digits": wrong,
        }
        print(
            f"{name:>18s} acc={token['token_accuracy']:.3f} "
            f"gap={token['median_gap']:.3f} decode={decode['exact']:.3f} "
            f"wrong={len(wrong)}",
            flush=True,
        )

    record = {
        "schema": "pythia-token-heldout-spatial-audit",
        "init": str(args.init),
        "init_report": report,
        "language": model.language_meta(),
        "jitter_offsets": [list(offset) for offset in JITTER_OFFSETS],
        "banks": results,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
