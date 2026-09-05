"""Read-only token-level diagnosis of caption selection in a saved shared graph."""
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
from scripts.train_real256_capacity import load_graph_weights, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sharegpt-manifest", required=True)
    parser.add_argument("--davis-manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if Path(args.out).exists():
        parser.error("existing diagnosis is protected; choose a new output")
    torch.set_num_threads(4)
    digest = sha256(args.checkpoint)
    raw = torch.load(args.checkpoint, map_location="cpu")
    if sha256(args.checkpoint) != digest:
        raise RuntimeError("checkpoint changed during loading")
    config = {**raw["config"], "lm_device": args.device}
    model = DualStreamOmni(**config).to(args.device).eval()
    load_graph_weights(model, raw["state_dict"])
    samples = [s for s in build_real_bank(args.sharegpt_manifest, args.davis_manifest, args.resolution)
               if s["task"] == "i2t"]
    answers = [model.lm_tok.encode(" " + s["answer"], add_special_tokens=False) for s in samples]
    common = 0
    while common < min(map(len, answers)) and answers[0][common] == answers[1][common]:
        common += 1
    if common == min(map(len, answers)):
        raise ValueError("diagnosis expects two captions that diverge before their end")
    report = {"checkpoint": args.checkpoint, "checkpoint_sha256": digest,
              "checkpoint_step_loaded": raw["step"], "device": args.device,
              "resolution": args.resolution, "common_answer_prefix_tokens": common,
              "branch_tokens": [model.lm_tok.decode([a[common]]) for a in answers], "samples": []}
    for index, sample in enumerate(samples):
        batch = collate_real_capacity(model.lm_tok, [sample], args.device)
        batch["image"].requires_grad_(True)
        model.zero_grad(set_to_none=True)
        out = forward_real_capacity(model, batch)
        positions = batch["labels"][0].ne(-100).nonzero().flatten()
        scores = out["token_logits"][0, out["n_vis_tokens"] + positions - 1].float()
        labels = batch["labels"][0, positions]
        nll = F.cross_entropy(scores, labels, reduction="none")
        probabilities = scores[common].softmax(-1)
        row = {"id": sample["id"], "mean_nll": float(nll.mean().detach()),
               "branch_nll": float(nll[common].detach()),
               "branch_fraction_of_summed_nll": float((nll[common] / nll.sum()).detach()),
               "branch_candidate_probabilities": [float(probabilities[a[common]].detach()) for a in answers],
               "token_losses": [{"token": model.lm_tok.decode([int(label)]), "nll": float(loss.detach())}
                                for label, loss in zip(labels, nll)]}
        nll[common].backward()
        row["branch_image_gradient_rms"] = float(batch["image"].grad.square().mean().sqrt())
        gradients = [(name, float(p.grad.square().mean().sqrt())) for name, p in model.named_parameters()
                     if p.grad is not None]
        row["largest_parameter_gradient_rms"] = sorted(gradients, key=lambda x: x[1], reverse=True)[:15]
        row["frozen_lm_has_gradient"] = any(p.grad is not None for p in model.lm.parameters())
        del out, scores, nll, probabilities
        changed = {**sample, "image": samples[1-index]["image"]}
        with torch.no_grad():
            control = forward_real_capacity(model, collate_real_capacity(model.lm_tok, [changed], args.device))
            logits = control["token_logits"][0, control["n_vis_tokens"] + positions[common] - 1].float()
            row["swapped_branch_candidate_probabilities"] = [float(logits.softmax(-1)[a[common]]) for a in answers]
        report["samples"].append(row)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({**report, "samples": [{k: v for k, v in row.items() if k != "token_losses"}
                                             for row in report["samples"]]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
