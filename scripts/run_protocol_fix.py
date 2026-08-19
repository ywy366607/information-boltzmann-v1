#!/usr/bin/env python3
"""Compare full-text CE vs answer-only loss; free-gen vs constrained ranking.

Runs key cells (default B@16 seeds) and writes a real measured table.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("ML_CACHE_ROOT", r"D:\ml_cache")
os.environ["HF_HOME"] = str(Path(os.environ["ML_CACHE_ROOT"]) / "huggingface")
os.environ["MODELSCOPE_CACHE"] = str(Path(os.environ["ML_CACHE_ROOT"]) / "modelscope")
os.environ["TORCH_HOME"] = str(Path(os.environ["ML_CACHE_ROOT"]) / "torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fine_grain.frontends import build_frontend  # noqa: E402
from fine_grain.llm_backend import cache_root, load_frozen_lm  # noqa: E402
from fine_grain.vlm_data import (  # noqa: E402
    answer_only_labels,
    exact_match,
    make_vqa_batch,
    tokenize_captions,
)
from scripts.train_vlm_frontends import (  # noqa: E402
    VLMBridge,
    _greedy_answer,
    _trainable,
    constrained_rank_answer,
)


def train_bridge(kind, T, args, llm, tokenizer, d_llm, device, seed, answer_only: bool):
    torch.manual_seed(seed)
    fe = build_frontend(
        kind, d_llm, res=args.res, T=T, patch=args.patch,
        dim=args.dim, depth=args.depth, deslice_topk=args.topk,
        projector=args.projector,
    ).to(device)
    bridge = VLMBridge(fe, llm).to(device)
    opt = torch.optim.AdamW(_trainable(bridge), lr=args.lr, weight_decay=0.01)
    rng = np.random.default_rng(seed + 7)
    bridge.train()
    last = float("nan")
    t0 = time.time()
    for step in range(1, args.steps + 1):
        data = make_vqa_batch(rng, args.batch, res=args.res, mix=("color", "kinks"))
        img = data["image"].to(device)
        if answer_only:
            ids, mask, text_lab = answer_only_labels(
                tokenizer, data["prompt"], data["text"], max_length=args.max_len,
            )
            ids, mask, text_lab = ids.to(device), mask.to(device), text_lab.to(device)
            loss, meta = bridge(img, ids, mask, text_labels=text_lab)
        else:
            ids, mask = tokenize_captions(tokenizer, data["text"], max_length=args.max_len)
            ids, mask = ids.to(device), mask.to(device)
            loss, meta = bridge(img, ids, mask, text_labels=None)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last = float(loss.detach().float().item())
        if step % args.log_every == 0 or step in (1, args.steps):
            print(
                f"  [{kind} T={meta.get('T', T)} seed={seed} "
                f"ao={answer_only}] step {step} loss={last:.4f}",
                flush=True,
            )
    return bridge, last, time.time() - t0


@torch.no_grad()
def eval_both(bridge, tokenizer, device, res, n, batch, val_seed, max_new):
    bridge.eval()
    rng = np.random.default_rng(val_seed)
    free = defaultdict(lambda: {"hit": 0, "tot": 0})
    cons = defaultdict(lambda: {"hit": 0, "tot": 0})
    free_ex, cons_ex = [], []
    left = n
    while left > 0:
        b = min(batch, left)
        data = make_vqa_batch(rng, b, res=res, mix=("color", "kinks"))
        img = data["image"].to(device)
        free_preds = _greedy_answer(
            bridge, tokenizer, img, data["prompt"], max_new=max_new, early_stop=True,
        )
        cons_preds = constrained_rank_answer(
            bridge, tokenizer, img, data["prompt"], data["probe"],
        )
        for i in range(b):
            kind = data["probe"][i]
            gold = data["answer"][i]
            for mode, pred, bucket, ex in (
                ("free", free_preds[i], free, free_ex),
                ("cons", cons_preds[i], cons, cons_ex),
            ):
                ok = exact_match(pred, gold)
                bucket[kind]["tot"] += 1
                bucket[kind]["hit"] += int(ok)
                if len(ex) < 6 and not ok:
                    ex.append({"task": kind, "gold": gold, "pred": pred[:60], "mode": mode})
        left -= b
    bridge.train()

    def pack(bucket):
        out = {}
        hits = tots = 0
        for k, v in bucket.items():
            out[f"acc_{k}"] = v["hit"] / max(v["tot"], 1)
            out[f"n_{k}"] = v["tot"]
            hits += v["hit"]
            tots += v["tot"]
        out["acc_overall"] = hits / max(tots, 1)
        out["n_overall"] = tots
        return out

    return {
        "free_gen": pack(free),
        "constrained": pack(cons),
        "free_fail_ex": free_ex,
        "cons_fail_ex": cons_ex,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="*", default=["B:16:0", "B:16:1", "B:64:1", "A:16:1"])
    ap.add_argument("--modes", nargs="*", default=["full", "answer_only"],
                    help="full = CE on whole Q+A; answer_only = CE on answer span")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--res", type=int, default=32)
    ap.add_argument("--patch", type=int, default=4)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--projector", default="mlp")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max_len", type=int, default=48)
    ap.add_argument("--probe_n", type=int, default=64)
    ap.add_argument("--val_seed", type=int, default=90_001)
    ap.add_argument("--max_new", type=int, default=6)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--prefer", default="gemma")
    ap.add_argument("--out", default="results/published/vqa_protocol_fix.json")
    ap.add_argument("--md", default="results/published/vqa_protocol_fix.md")
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"cache={cache_root()}", flush=True)
    llm, tokenizer, d_llm, note = load_frozen_lm(prefer=args.prefer, device=str(device))
    print(f"backend={note}", flush=True)

    # unit-ish check: answer-only masks some tokens
    demo = make_vqa_batch(np.random.default_rng(0), 2, res=args.res)
    ids, mask, lab = answer_only_labels(tokenizer, demo["prompt"], demo["text"], 48)
    assert (lab != -100).any(), "answer-only labels empty"
    assert (lab == -100).any(), "answer-only did not mask prompt"
    print(
        f"answer-only check: supervised_frac="
        f"{float((lab != -100).sum() / mask.sum()):.3f}",
        flush=True,
    )

    rows = []
    for cell in args.cells:
        kind, T, seed = cell.split(":")
        kind, T, seed = kind.upper(), int(T), int(seed)
        for mode in args.modes:
            ao = mode in ("answer_only", "ao", "answer")
            print(f"\n===== {kind} T={T} seed={seed} train={mode} =====", flush=True)
            bridge, loss, secs = train_bridge(
                kind, T, args, llm, tokenizer, d_llm, device, seed, answer_only=ao,
            )
            metrics = eval_both(
                bridge, tokenizer, device, args.res,
                n=args.probe_n, batch=min(8, max(args.batch, 4)),
                val_seed=args.val_seed, max_new=args.max_new,
            )
            row = {
                "kind": kind,
                "T": T,
                "seed": seed,
                "train": "answer_only" if ao else "full_text",
                "final_loss": loss,
                "seconds": secs,
                **{f"free_{k}": v for k, v in metrics["free_gen"].items()},
                **{f"cons_{k}": v for k, v in metrics["constrained"].items()},
                "free_fail_ex": metrics["free_fail_ex"],
                "cons_fail_ex": metrics["cons_fail_ex"],
            }
            rows.append(row)
            print(
                f"  free overall={row['free_acc_overall']:.3f} "
                f"color={row['free_acc_color']:.3f} kinks={row['free_acc_kinks']:.3f} | "
                f"cons overall={row['cons_acc_overall']:.3f} "
                f"color={row['cons_acc_color']:.3f} kinks={row['cons_acc_kinks']:.3f}",
                flush=True,
            )
            del bridge
            if device.type == "cuda":
                torch.cuda.empty_cache()

    payload = {
        "backend": note,
        "projector": args.projector,
        "steps": args.steps,
        "val_seed": args.val_seed,
        "table": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    lines = [
        "# VQA protocol fix: answer-only loss + constrained ranking",
        "",
        f"- Backend: `{note}`",
        f"- projector={args.projector} steps={args.steps} val_seed={args.val_seed}",
        "- **full_text**: CE on entire `Question:… Answer:…` (old)",
        "- **answer_only**: CE only on answer token span",
        "- **free**: greedy free-gen exact-match (early-stop heuristics)",
        "- **cons**: constrained ranking over {red,green,blue,yellow} / {5,6,7,8}",
        "",
        "| kind | T | seed | train | free_acc | free_color | free_kinks | cons_acc | cons_color | cons_kinks | loss |",
        "|------|---|------|-------|----------|------------|------------|----------|------------|------------|------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['kind']} | {r['T']} | {r['seed']} | {r['train']} | "
            f"{r['free_acc_overall']:.3f} | {r['free_acc_color']:.3f} | {r['free_acc_kinks']:.3f} | "
            f"{r['cons_acc_overall']:.3f} | {r['cons_acc_color']:.3f} | {r['cons_acc_kinks']:.3f} | "
            f"{r['final_loss']:.3f} |"
        )
    lines.extend(["", "## Takeaways", ""])
    # pairwise full vs ao for free and cons
    by_key = {}
    for r in rows:
        by_key[(r["kind"], r["T"], r["seed"], r["train"])] = r
    for cell in args.cells:
        kind, T, seed = cell.split(":")
        kind, T, seed = kind.upper(), int(T), int(seed)
        a = by_key.get((kind, T, seed, "full_text"))
        b = by_key.get((kind, T, seed, "answer_only"))
        if a and b:
            lines.append(
                f"- **{kind}@T{T} s{seed}**: free {a['free_acc_overall']:.3f}→{b['free_acc_overall']:.3f} "
                f"(color {a['free_acc_color']:.3f}→{b['free_acc_color']:.3f}); "
                f"cons {a['cons_acc_overall']:.3f}→{b['cons_acc_overall']:.3f} "
                f"(color {a['cons_acc_color']:.3f}→{b['cons_acc_color']:.3f})"
            )
    # highlight constrained lift under same train
    for r in rows:
        lift = r["cons_acc_overall"] - r["free_acc_overall"]
        if abs(lift) >= 0.05:
            lines.append(
                f"- {r['kind']}@T{r['T']} s{r['seed']} {r['train']}: "
                f"cons−free overall = {lift:+.3f}"
            )
    lines.append("")
    Path(args.md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"json → {args.out}", flush=True)
    print(f"md → {args.md}", flush=True)


if __name__ == "__main__":
    main()
