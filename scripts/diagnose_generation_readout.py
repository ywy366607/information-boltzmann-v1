"""Fit diagnostic logit readouts of frozen fields, without changing a model.

Target-fitted outputs are probes, NOT generated images or candidate weights.
Least squares in logit space is not an upper bound on optimal RGB accuracy.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fine_grain.omni_model import DualStreamOmni
from fine_grain.real_capacity import build_real_bank, collate_real_capacity, forward_real_capacity
from scripts.train_real256_capacity import load_graph_weights, sha256
from scripts.train_sharegpt4o_t2i_overfit import edge_metrics


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sharegpt-manifest", required=True)
    parser.add_argument("--davis-manifest", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if Path(args.out).exists():
        parser.error("existing output is protected")
    torch.set_num_threads(4)
    digest = sha256(args.checkpoint)
    saved = torch.load(args.checkpoint, map_location="cpu")
    config = {**saved["config"], "lm_device": "cpu"}
    model = DualStreamOmni(**config).eval()
    load_graph_weights(model, saved["state_dict"])
    model.mot_stack.set_record_field_trace(True)
    report = {"checkpoint": args.checkpoint, "sha256": digest, "step": saved["step"],
              "scope": "target-fitted frozen-feature probes; not model outputs or RGB accuracy upper bounds", "rows": []}
    for resolution in (64, 256):
        samples = [s for s in build_real_bank(args.sharegpt_manifest, args.davis_manifest, resolution)
                   if s["task"] == "t2i"]
        for sample in samples:
            out = forward_real_capacity(model, collate_real_capacity(model.lm_tok, [sample], "cpu"))
            target = sample["target_rgb"].unsqueeze(0)
            y = torch.logit(target.clamp(.001, .999)).flatten(2).transpose(1, 2)[0].double()
            row = {"id": sample["id"], "resolution": resolution, "probes": []}
            for label, state in [("initial", model.mot_stack._last_X_steps[0]), ("terminal", out["X"])]:
                features = model.pix_head[0](state)[0].double()
                features = torch.cat([features, torch.ones(features.shape[0], 1)], dim=1)
                fit = torch.linalg.lstsq(features, y, driver="gelsd", rcond=1e-8)
                fitted_logits = features @ fit.solution
                pred = fitted_logits.sigmoid().float().T.reshape(1, 3, resolution, resolution)
                _, edge = edge_metrics(pred, target)
                row["probes"].append({"field": label, "feature_rank": int(fit.rank),
                                      "logit_relative_mse": float((fitted_logits-y).square().mean() / y.var(unbiased=False)),
                                      "psnr": -10*math.log10(float((pred-target).square().mean())),
                                      "edge_correlation": edge})
            _, edge = edge_metrics(out["rgb"], target)
            row["actual_output"] = {"psnr": -10*math.log10(float((out["rgb"]-target).square().mean())),
                                    "edge_correlation": edge}
            report["rows"].append(row)
            print(json.dumps(row), flush=True)
    if sha256(args.checkpoint) != digest:
        raise RuntimeError("checkpoint changed during the read-only diagnosis")
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
