#!/usr/bin/env python3
"""Diagnose free-gen VQA failures for A/B/C (why probe≠VQA).

Retrains a short list of cells, dumps raw generations + failure taxonomy.
No invented metrics — all from live greedy decode.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
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
    COLORS,
    KINK_KS,
    exact_match,
    make_vqa_batch,
    normalize_answer,
    tokenize_captions,
)
from scripts.train_vlm_frontends import VLMBridge, _greedy_answer, _trainable  # noqa: E402

COLOR_SET = set(COLORS)
KINK_SET = {str(k) for k in KINK_KS}


def classify_failure(pred: str, gold: str, kind: str) -> str:
    """Taxonomy for a single mismatch (or ok)."""
    if exact_match(pred, gold):
        return "ok"
    raw = pred if pred is not None else ""
    p = normalize_answer(raw)
    g = normalize_answer(gold)
    if not raw.strip():
        return "empty_raw"
    if not p:
        return "empty_after_norm"
    # valid color word but wrong
    if kind == "color":
        # extract first color-like token
        tokens = p.replace("/", " ").split()
        hit_colors = [t for t in tokens if t in COLOR_SET]
        if hit_colors:
            if hit_colors[0] == g:
                return "format_extra_but_gold_first"  # should have matched; defensive
            return "wrong_color_word"
        # digit-like answer on color question
        if any(ch.isdigit() for ch in p):
            return "color_but_emitted_number"
        if p in ("yes", "no", "true", "false"):
            return "color_yesno_garbage"
        return "color_non_vocab_garbage"
    if kind == "kinks":
        # any digit in pred
        digits = "".join(ch if ch.isdigit() else " " for ch in p).split()
        if digits:
            if digits[0] == g:
                return "format_extra_but_gold_digit"
            if digits[0] in KINK_SET:
                return "wrong_kink_count"
            return "kinks_digit_out_of_range"
        if any(t in COLOR_SET for t in p.split()):
            return "kinks_but_emitted_color"
        return "kinks_non_numeric_garbage"
    return "other"


@torch.no_grad()
def eval_detailed(bridge, tokenizer, device, res, n=64, batch=8, val_seed=90_001, max_new=12):
    bridge.eval()
    rng = np.random.default_rng(val_seed)
    rows = []
    left = n
    while left > 0:
        b = min(batch, left)
        data = make_vqa_batch(rng, b, res=res, mix=("color", "kinks"))
        img = data["image"].to(device)
        preds = _greedy_answer(bridge, tokenizer, img, data["prompt"], max_new=max_new)
        for i in range(b):
            pred, gold, kind = preds[i], data["answer"][i], data["probe"][i]
            tag = classify_failure(pred, gold, kind)
            rows.append({
                "kind_task": kind,
                "gold": gold,
                "pred_raw": pred,
                "pred_norm": normalize_answer(pred),
                "ok": tag == "ok",
                "fail_tag": tag,
                "prompt": data["prompt"][i],
            })
        left -= b
    bridge.train()
    return rows


def train_cell(kind, T, args, llm, tokenizer, d_llm, device, seed):
    torch.manual_seed(seed)
    fe = build_frontend(
        kind, d_llm, res=args.res, T=T, patch=args.patch,
        dim=args.dim, depth=args.depth, deslice_topk=args.topk,
        projector=args.projector,
    ).to(device)
    bridge = VLMBridge(fe, llm).to(device)
    opt = torch.optim.AdamW(_trainable(bridge), lr=args.lr, weight_decay=0.01)
    rng = np.random.default_rng(seed + 7)
    t0 = time.time()
    bridge.train()
    last_loss = float("nan")
    for step in range(1, args.steps + 1):
        data = make_vqa_batch(rng, args.batch, res=args.res, mix=("color", "kinks"))
        img = data["image"].to(device)
        ids, mask = tokenize_captions(tokenizer, data["text"], max_length=args.max_len)
        ids, mask = ids.to(device), mask.to(device)
        loss, meta = bridge(img, ids, mask)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last_loss = float(loss.detach().float().item())
        if step % args.log_every == 0 or step == 1 or step == args.steps:
            print(
                f"  [{kind} T={meta.get('T', T)} seed={seed}] "
                f"step {step} loss={last_loss:.4f}",
                flush=True,
            )
    return bridge, last_loss, time.time() - t0


def summarize(rows):
    by_task = defaultdict(list)
    for r in rows:
        by_task[r["kind_task"]].append(r)
    out = {"overall": {}, "by_task": {}, "fail_tag_counts": {}, "examples": {}}
    for task, rs in by_task.items():
        n = len(rs)
        n_ok = sum(1 for r in rs if r["ok"])
        tags = Counter(r["fail_tag"] for r in rs)
        out["by_task"][task] = {
            "n": n,
            "acc": n_ok / max(n, 1),
            "fail_tags": dict(tags),
        }
    n_all = len(rows)
    n_ok = sum(1 for r in rows if r["ok"])
    out["overall"] = {"n": n_all, "acc": n_ok / max(n_all, 1)}
    out["fail_tag_counts"] = dict(Counter(r["fail_tag"] for r in rows))
    # examples per tag (up to 3)
    by_tag = defaultdict(list)
    for r in rows:
        if r["fail_tag"] != "ok" and len(by_tag[r["fail_tag"]]) < 4:
            by_tag[r["fail_tag"]].append({
                "task": r["kind_task"],
                "gold": r["gold"],
                "pred_raw": r["pred_raw"][:80],
                "pred_norm": r["pred_norm"][:80],
            })
    out["examples"] = dict(by_tag)
    # top predicted strings overall
    out["top_pred_norm"] = Counter(r["pred_norm"] for r in rows).most_common(12)
    out["top_pred_raw"] = Counter(r["pred_raw"] for r in rows).most_common(12)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="*", default=[
        "B:16:0",   # smoking gun: probe color 0.88, VQA ~0
        "B:16:1",   # same T, better seed
        "B:64:1",   # VQA relatively good for B
        "A:16:1",   # A peak-ish
    ])
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
    ap.add_argument("--max_new", type=int, default=12)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--prefer", default="gemma")
    ap.add_argument("--out", default="results/published/vqa_failure_diagnosis.json")
    ap.add_argument("--md", default="results/published/vqa_failure_diagnosis.md")
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"cache={cache_root()}", flush=True)
    llm, tokenizer, d_llm, note = load_frozen_lm(prefer=args.prefer, device=str(device))
    print(f"backend={note}", flush=True)

    report = {
        "backend": note,
        "projector": args.projector,
        "steps": args.steps,
        "val_seed": args.val_seed,
        "max_new": args.max_new,
        "cells": [],
    }

    for cell in args.cells:
        parts = cell.split(":")
        kind, T, seed = parts[0].upper(), int(parts[1]), int(parts[2])
        print(f"\n===== DIAG {kind} T={T} seed={seed} =====", flush=True)
        bridge, loss, secs = train_cell(
            kind, T, args, llm, tokenizer, d_llm, device, seed,
        )
        rows = eval_detailed(
            bridge, tokenizer, device, args.res,
            n=args.probe_n, batch=min(8, args.batch),
            val_seed=args.val_seed, max_new=args.max_new,
        )
        summ = summarize(rows)
        cell_rep = {
            "kind": kind,
            "T": T,
            "seed": seed,
            "final_loss": loss,
            "seconds": secs,
            "summary": summ,
            "all_rows": rows,
        }
        report["cells"].append(cell_rep)
        print(json.dumps({
            "cell": f"{kind}:T{T}:s{seed}",
            "acc": summ["overall"]["acc"],
            "by_task": {k: v["acc"] for k, v in summ["by_task"].items()},
            "fail_tags": summ["fail_tag_counts"],
            "top_pred_norm": summ["top_pred_norm"][:6],
        }, indent=2), flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        del bridge

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # markdown
    lines = [
        "# VQA free-gen failure diagnosis",
        "",
        f"- Backend: `{note}`",
        f"- projector={args.projector} steps={args.steps} val_seed={args.val_seed} max_new={args.max_new}",
        f"- Cells: {', '.join(args.cells)}",
        "",
    ]
    for c in report["cells"]:
        s = c["summary"]
        lines.append(f"## {c['kind']} T={c['T']} seed={c['seed']}")
        lines.append("")
        lines.append(
            f"- final_loss={c['final_loss']:.4f} seconds={c['seconds']:.1f} "
            f"overall_acc={s['overall']['acc']:.3f} n={s['overall']['n']}"
        )
        for task, info in s["by_task"].items():
            lines.append(
                f"- **{task}** acc={info['acc']:.3f} (n={info['n']}) tags={info['fail_tags']}"
            )
        lines.append(f"- fail_tag totals: `{s['fail_tag_counts']}`")
        lines.append(f"- top pred_norm: `{s['top_pred_norm'][:8]}`")
        lines.append("")
        lines.append("| fail_tag | task | gold | pred_raw |")
        lines.append("|----------|------|------|----------|")
        for tag, exs in s["examples"].items():
            for ex in exs[:2]:
                pr = (ex["pred_raw"] or "").replace("|", "\\|")[:40]
                lines.append(f"| {tag} | {ex['task']} | {ex['gold']} | `{pr}` |")
        lines.append("")

    lines.extend([
        "## How to read tags",
        "",
        "- `wrong_color_word`: emitted a valid color ≠ gold (content error).",
        "- `color_non_vocab_garbage` / `kinks_non_numeric_garbage`: not in answer vocab (format/LM garbage).",
        "- `color_but_emitted_number` / `kinks_but_emitted_color`: **task confusion**.",
        "- `empty_*`: model produced nothing useful.",
        "- `wrong_kink_count`: valid digit in 5–8 but wrong count.",
        "",
    ])
    Path(args.md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"json → {args.out}", flush=True)
    print(f"md → {args.md}", flush=True)


if __name__ == "__main__":
    main()
