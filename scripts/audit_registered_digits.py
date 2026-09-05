"""Keep legacy scores and audit identity with exact registered glyphs."""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fine_grain.capability_tasks import CAPABILITY_CASES, fixed_capability_bank
from fine_grain.gen_metrics import paired_digit_scores, digit_shift_scores
from fine_grain.omni_model import DualStreamOmni
from scripts.audit_attention_write import forward_batch


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--out", default="results/published/registered_digit_audit.json")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    report = {"metric": "exact-renderer paired-address identity; legacy sliding scores retained", "checkpoints": {}}
    for path in args.checkpoints:
        raw = torch.load(path, map_location="cpu")
        model = DualStreamOmni(**raw["config"]).to(args.device).eval()
        model.load_state_dict(raw["state_dict"])
        bank = fixed_capability_bank(model.res)
        rows = {}
        for case in CAPABILITY_CASES:
            samples = [s for s in bank if s["case"] == case]
            measured = []
            for start in range(0, len(samples), 24):
                part = samples[start:start + 24]
                output = forward_batch(model, part, args.device)
                for i, s in enumerate(part):
                    measured.append(paired_digit_scores(output["rgb"][i], s["digit"], s["target_color"], s["target_place"]))
            rows[case] = {k: sum(row[k] for row in measured) / len(measured) for k in measured[0]}
        oracle_bank = [s for s in bank if s["case"] == "text_to_both"]
        oracle = [paired_digit_scores(s["target_rgb"], s["digit"], s["target_color"], s["target_place"]) for s in oracle_bank]
        legacy = [digit_shift_scores(s["target_rgb"], s["digit"], s["target_color"])["digit_top1"] for s in oracle_bank]
        rows["oracle"] = {"paired_digit_top1": sum(v["paired_digit_top1"] for v in oracle) / len(oracle),
                          "legacy_digit_top1": sum(legacy) / len(legacy)}
        report["checkpoints"][path] = rows
        del model
        torch.cuda.empty_cache()
        print(json.dumps({path: rows}), flush=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
