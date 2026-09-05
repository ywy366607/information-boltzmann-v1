"""Separate measured foreground/background supervision in the real graph."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fine_grain.omni_model import DualStreamOmni
from fine_grain.real_capacity import build_real_bank, collate_real_capacity, forward_real_capacity
from fine_grain.unified_capacity import extend_unified_bank
from scripts.train_real256_capacity import load_graph_weights, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sharegpt-manifest", required=True)
    parser.add_argument("--davis-manifest", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if Path(args.out).exists():
        parser.error("existing diagnosis is protected")
    torch.set_num_threads(4)
    digest = sha256(args.checkpoint)
    raw = torch.load(args.checkpoint, map_location="cpu")
    if sha256(args.checkpoint) != digest:
        raise RuntimeError("checkpoint changed during loading")
    model = DualStreamOmni(**{**raw["config"], "lm_device": "cpu"}).eval()
    load_graph_weights(model, raw["state_dict"])
    report = {"checkpoint": args.checkpoint, "sha256_loaded": digest, "step": raw["step"],
              "device": "cpu", "scope": "one-checkpoint local gradient diagnosis; not a causal convergence result", "rows": []}
    for resolution in (64, 256):
        bank = extend_unified_bank(build_real_bank(args.sharegpt_manifest, args.davis_manifest, resolution), resolution)
        for sample in [s for s in bank if s["id"] in ("t2i1px-0", "t2i1px-7")]:
            batch = collate_real_capacity(model.lm_tok, [sample], "cpu")
            model.zero_grad(set_to_none=True)
            # Only the mean head derivative is needed. Retaining the full 256px
            # trunk's autograd graph needlessly consumes several GB of host RAM.
            with torch.no_grad():
                out = forward_real_capacity(model, batch)
                features = model.pix_head[0](out["X"]).detach()
                lv = out["rgb_lv"].detach().clamp(-6, 3)
            weight = model.pix_head[-1].weight
            pred = model._pts_to_img(F.linear(features, weight, model.pix_head[-1].bias)).sigmoid()
            target = batch["target_rgb"]
            mask = sample["target_seg"].bool().unsqueeze(0).unsqueeze(0).expand_as(pred)
            nll = .5 * (lv + (target - pred).square() * (-lv).exp())
            # Exact decomposition of the current spatially averaged likelihood.
            ink_loss, background_loss = (nll * mask).mean(), (nll * ~mask).mean()
            ink_grad = torch.autograd.grad(ink_loss, weight, retain_graph=True)[0]
            background_grad = torch.autograd.grad(background_loss, weight)[0]
            gradient = (pred.detach() - target) * (-lv.detach()).exp() / pred.numel()
            row = {"id": sample["id"], "resolution": resolution,
                   "ink_pixels": int(sample["target_seg"].sum()),
                   "ink_area_fraction": float(mask.float().mean()),
                   "ink_red_prediction_mean": float(pred.detach()[0, 0][sample["target_seg"].bool()].mean()),
                   "ink_abs_rgb_gradient_mass": float(gradient[mask].abs().sum()),
                   "background_abs_rgb_gradient_mass": float(gradient[~mask].abs().sum()),
                   "ink_mean_head_gradient_norm": float(ink_grad.norm()),
                   "background_mean_head_gradient_norm": float(background_grad.norm()),
                   "mean_head_gradient_cosine": float(F.cosine_similarity(ink_grad.flatten(), background_grad.flatten(), dim=0)),
                   "ink_pixel_nll_mean": float(nll.detach()[mask].mean()),
                   "background_pixel_nll_mean": float(nll.detach()[~mask].mean())}
            report["rows"].append(row)
            print(json.dumps(row), flush=True)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
