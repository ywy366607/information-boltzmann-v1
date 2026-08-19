#!/usr/bin/env python3
"""Is rank collapse a disease or a correct information bottleneck?

Measures:
  1. Prompt token inventory + embedding rank (vocab physics)
  2. Per-sequence H rank by layer (within one question)
  3. Last-token H rank within-task vs across-task (task semantic dim)
  4. Token energy share (does everything funnel to Answer: ?)
  5. H_ctx / S_hat rank vs H rank (prior cannot exceed span(H))
  6. Visual slice energy concentration (not assignment mass)

Usage:
  python scripts/measure_semantic_dof.py --steps 250 --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fine_grain.vlm_data import make_vqa_batch
from scripts.run_v0_surprise_eval import DualStreamVQAModel, eval_model


def erank(t: torch.Tensor) -> float:
    t = t.detach().float()
    if t.ndim != 2 or min(t.shape) == 0:
        return float("nan")
    s = torch.linalg.svdvals(t)
    p = (s * s).clamp_min(1e-12)
    p = p / p.sum()
    return float(torch.exp(-(p * p.log()).sum()))


def topk_energy(t: torch.Tensor, k: int = 4) -> float:
    """Fraction of ||rows||^2 sitting in the top-k rows."""
    t = t.detach().float()
    if t.ndim == 3:
        t = t[0]
    e = t.pow(2).sum(-1)
    e, _ = e.sort(descending=True)
    return float(e[:k].sum() / e.sum().clamp_min(1e-12))


def capture_layers(model, imgs, prompts, masks=None):
    store = [{"H": None, "S": None, "H_ctx": None, "S_hat": None} for _ in model.mot_stack.layers]
    hooks = []
    for i, layer in enumerate(model.mot_stack.layers):
        def mh(idx):
            def h(mod, inp, out):
                s2, h2, _ = out
                store[idx]["S"] = s2.detach()
                store[idx]["H"] = h2.detach()

            return h

        hooks.append(layer.mot.register_forward_hook(mh(i)))
    with torch.no_grad():
        model(imgs, prompts)
    for i, layer in enumerate(model.mot_stack.layers):
        g = layer.surprise_gate
        if getattr(g, "last_H_ctx", None) is not None:
            store[i]["H_ctx"] = g.last_H_ctx.detach()
            if hasattr(g, "lang_to_prior"):
                store[i]["S_hat"] = g.lang_to_prior(g.last_H_ctx).detach()
            elif hasattr(g, "prior_head"):
                store[i]["S_hat"] = g.prior_head(g.last_H_ctx).detach().chunk(2, dim=-1)[0]
    for h in hooks:
        h.remove()
    return store


def prompt_physics(model, device):
    prompts = {
        "ocr": "Question: What digit is drawn with the thin stroke? Answer:",
        "kinks": "Question: How many corners does the red polyline have? Answer:",
        "color": "Question: What color is the small square? Answer:",
    }
    rows = {}
    vocab_hits = {}
    for name, p in prompts.items():
        words = p.replace("?", " ?").split()
        ids = [model.vocab.get(w, 0) for w in words]
        unk = [w for w in words if w not in model.vocab]
        vocab_hits[name] = {
            "tokens": words,
            "n": len(words),
            "unique_ids": len(set(ids)),
            "unk": unk,
            "n_unk": len(unk),
        }
        tok = torch.tensor([ids], device=device)
        emb = model.embed(tok)[0]
        rows[name] = {
            "n_tokens": len(words),
            "unique_ids": len(set(ids)),
            "n_unk": len(unk),
            "unk": unk,
            "emb_erank": erank(emb),
            "emb_shape": list(emb.shape),
        }
    return rows, vocab_hits


def last_token_task_rank(model, device, n_per=48):
    """Batch-level rank of the last (Answer:) token, within and across tasks."""
    rng = np.random.default_rng(7)
    buckets = {"ocr": [], "kinks": [], "color": []}
    model.eval()
    with torch.no_grad():
        for task in buckets:
            left = n_per
            while left > 0:
                bsz = min(16, left)
                batch = make_vqa_batch(rng, batch=bsz, res=32, mix=[task])
                imgs = batch["image"].to(device)
                tok, mask = model.tokenize(batch["prompt"], device)
                emb = model.embed(tok)
                # run stack, keep last-layer H
                X = model.mot_stack.encode_X(imgs)
                H = model.mot_stack.text_in(emb)
                for layer in model.mot_stack.layers:
                    X, H, _ = layer(X, H, text_mask=mask, prompt_mask=mask)
                # last real token per row
                lengths = mask.sum(-1)
                last = H[torch.arange(H.shape[0], device=device), lengths - 1]
                buckets[task].append(last.cpu())
                left -= bsz
    packed = {k: torch.cat(v, 0) for k, v in buckets.items()}
    mixed = torch.cat(list(packed.values()), 0)
    out = {
        "n_per_task": n_per,
        "within": {k: erank(v) for k, v in packed.items()},
        "across_3tasks": erank(mixed),
        "pairwise_task_mean_cos": {},
    }
    # mean last-token per task, cosine between task centroids
    cents = {k: F.normalize(v.mean(0), dim=0) for k, v in packed.items()}
    names = list(cents)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            out["pairwise_task_mean_cos"][f"{a}-{b}"] = float((cents[a] * cents[b]).sum())
    return out


def sequence_ranks(model, device, n_batches=4):
    rng = np.random.default_rng(11)
    acc = {i: {"H": [], "H_last_share": [], "H_ctx": [], "S_hat": [], "S": [], "S_top4": []}
           for i in range(model.n_layers)}
    model.eval()
    with torch.no_grad():
        for _ in range(n_batches):
            batch = make_vqa_batch(rng, batch=16, res=32, mix=["ocr", "kinks", "color"])
            imgs = batch["image"].to(device)
            store = capture_layers(model, imgs, batch["prompt"])
            tok, mask = model.tokenize(batch["prompt"], device)
            lengths = mask.sum(-1)
            for i, box in enumerate(store):
                H = box["H"]
                if H is None:
                    continue
                for b in range(H.shape[0]):
                    L = int(lengths[b].item())
                    h = H[b, :L]
                    acc[i]["H"].append(erank(h))
                    e = h.pow(2).sum(-1)
                    acc[i]["H_last_share"].append(float(e[-1] / e.sum().clamp_min(1e-12)))
                if box["H_ctx"] is not None:
                    for b in range(box["H_ctx"].shape[0]):
                        acc[i]["H_ctx"].append(erank(box["H_ctx"][b]))
                if box["S_hat"] is not None:
                    for b in range(box["S_hat"].shape[0]):
                        acc[i]["S_hat"].append(erank(box["S_hat"][b]))
                if box["S"] is not None:
                    for b in range(box["S"].shape[0]):
                        acc[i]["S"].append(erank(box["S"][b]))
                        acc[i]["S_top4"].append(topk_energy(box["S"][b], 4))
    return {
        i: {k: float(np.mean(v)) if v else float("nan") for k, v in d.items()}
        for i, d in acc.items()
    }


def train(mode, steps, coef, device, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = DualStreamVQAModel(
        d_model=128, n_slices=32, n_layers=4, res=32,
        surprise_mode=mode, surprise_beta=1.5, prior_loss_coef=coef,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for step in range(1, steps + 1):
        model.train()
        b = make_vqa_batch(rng, batch=16, res=32, mix=["ocr", "kinks", "color"])
        imgs = b["image"].to(device)
        tgt = torch.tensor([model.ans_to_idx[a] for a in b["answer"]], device=device)
        opt.zero_grad()
        out = model(imgs, b["prompt"])
        model.task_loss(out, tgt).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % 50 == 0:
            print(f"  [{mode} λ={coef}] step {step}/{steps} pred={float(out['pred_loss'].detach()):.3f}",
                  flush=True)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/published/semantic_dof.json")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"device={device}", flush=True)

    scratch = DualStreamVQAModel(
        d_model=128, n_slices=32, n_layers=4, res=32, surprise_mode="v0_jepa",
    ).to(device)
    phys, vocab = prompt_physics(scratch, device)
    print("\n=== Prompt physics (word-piece / embed) ===")
    for k, v in phys.items():
        print(f"  {k}: T={v['n_tokens']} unique_ids={v['unique_ids']} unk={v['unk']} emb_erank={v['emb_erank']:.2f}")

    report = {"prompt_physics": phys, "arms": {}}

    for mode, coef, tag in (
        ("v0_jepa", 0.0, "jepa_lambda0"),
        ("v0_jepa", 0.1, "jepa_lambda01"),
        ("v1_bayes", 0.1, "bayes_lambda01"),
    ):
        print(f"\n======= train {tag} {args.steps} steps =======", flush=True)
        model = train(mode, args.steps, coef, device)
        rng_val = np.random.default_rng(123)
        val = eval_model(model, rng_val, val_batches=8, batch_size=16, res=32, device=device)
        seq = sequence_ranks(model, device)
        last = last_token_task_rank(model, device)
        print(f"  val_acc={val['acc']*100:.1f}% tasks="
              f"{ {k: round(v*100,1) for k,v in val['task_accs'].items()} }")
        print("  per-seq ranks (mean over mixed batch):")
        for i, r in seq.items():
            print(
                f"    L{i}  H={r['H']:.2f}  last_share={r['H_last_share']:.2f}  "
                f"Hctx={r['H_ctx']:.2f}  Shat={r['S_hat']:.2f}  "
                f"S={r['S']:.2f}  S_top4E={r['S_top4']:.2f}"
            )
        print("  last-token H rank  within:", {k: f"{v:.2f}" for k, v in last['within'].items()},
              f"  across3={last['across_3tasks']:.2f}")
        print("  task-centroid cos:", last["pairwise_task_mean_cos"])
        report["arms"][tag] = {
            "val": {"acc": val["acc"], "loss": val["loss"], "tasks": val["task_accs"]},
            "seq": seq,
            "last_token": last,
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # 2000-step published Bayes: load weights, force old uniform readout for faithful H
    ckpt = ROOT / "checkpoints" / "v1_bayes_2000step_run1_best.pt"
    if ckpt.exists():
        print("\n======= published 2000-step Bayes (legacy uniform readout) =======", flush=True)
        model = DualStreamVQAModel(
            d_model=128, n_slices=32, n_layers=4, res=32,
            surprise_mode="v1_bayes", prior_loss_coef=0.0,
        ).to(device)
        sd = torch.load(ckpt, map_location=device)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"  load missing={len(missing)} unexpected={len(unexpected)}")

        def _uniform_readout(self, H, text_mask=None):
            # recreate the *0.02 + no-norm pathology: softmax ~ uniform
            B, T, d = H.shape
            w = torch.ones(B, self.n_slices, T, device=H.device, dtype=H.dtype)
            if text_mask is not None:
                w = w.masked_fill(~text_mask.bool().unsqueeze(1), 0.0)
            w = w / w.sum(-1, keepdim=True).clamp_min(1e-8)
            self.last_attn = w.detach()
            return torch.matmul(w, H)

        for layer in model.mot_stack.layers:
            layer.surprise_gate._predict_prior_from_h = _uniform_readout.__get__(
                layer.surprise_gate, type(layer.surprise_gate)
            )
        rng_val = np.random.default_rng(123)
        val = eval_model(model, rng_val, val_batches=8, batch_size=16, res=32, device=device)
        seq = sequence_ranks(model, device)
        last = last_token_task_rank(model, device)
        print(f"  val_acc={val['acc']*100:.1f}% (this eval seed; published mean 84.6%)")
        for i, r in seq.items():
            print(
                f"    L{i}  H={r['H']:.2f}  last_share={r['H_last_share']:.2f}  "
                f"Hctx={r['H_ctx']:.2f}  Shat={r['S_hat']:.2f}  "
                f"S={r['S']:.2f}  S_top4E={r['S_top4']:.2f}"
            )
        print("  last-token H rank  within:", {k: f"{v:.2f}" for k, v in last['within'].items()},
              f"  across3={last['across_3tasks']:.2f}")
        print("  task-centroid cos:", last["pairwise_task_mean_cos"])
        report["arms"]["bayes_2000_legacy"] = {
            "val": {"acc": val["acc"], "loss": val["loss"], "tasks": val["task_accs"]},
            "seq": seq,
            "last_token": last,
        }

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
