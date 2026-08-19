#!/usr/bin/env python3
"""Train A/B/C frontends on synthetic 1px OCR → frozen Gemma (LLM).

Large on-the-fly stream (or optional disk dataset). Answer-only CE + free-gen
and constrained ranking metrics.

Example:
  set ML_CACHE_ROOT=D:\\ml_cache
  python scripts/train_ocr_vlm.py --steps 600 --T_list 64 32 16 --seeds 0 1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("ML_CACHE_ROOT", r"D:\ml_cache")
os.environ["HF_HOME"] = str(Path(os.environ["ML_CACHE_ROOT"]) / "huggingface")
os.environ["MODELSCOPE_CACHE"] = str(Path(os.environ["ML_CACHE_ROOT"]) / "modelscope")
os.environ["TORCH_HOME"] = str(Path(os.environ["ML_CACHE_ROOT"]) / "torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fine_grain.frontends import build_frontend, patch_token_count  # noqa: E402
from fine_grain.llm_backend import cache_root, load_frozen_lm  # noqa: E402
from fine_grain.ocr_1px import OCR_DIGITS  # noqa: E402
from fine_grain.vlm_data import (  # noqa: E402
    answer_only_labels,
    exact_match,
    make_ocr_vqa_batch,
    tokenize_captions,
)
from scripts.train_vlm_frontends import (  # noqa: E402
    VLMBridge,
    _greedy_answer,
    _trainable,
    constrained_rank_answer,
)


def train_one(kind, T, args, llm, tokenizer, d_llm, device, seed: int):
    torch.manual_seed(seed)
    fe = build_frontend(
        kind, d_llm, res=args.res, T=T, patch=args.patch,
        dim=args.dim, depth=args.depth, deslice_topk=args.topk,
        projector=args.projector,
    ).to(device)
    bridge = VLMBridge(fe, llm).to(device)
    opt = torch.optim.AdamW(_trainable(bridge), lr=args.lr, weight_decay=0.01)
    rng = np.random.default_rng(seed + 11)
    bridge.train()
    hist = []
    t0 = time.time()
    last_loss = float("nan")
    for step in range(1, args.steps + 1):
        data = make_ocr_vqa_batch(
            rng, args.batch, res=args.res,
            hard_frac=args.hard_frac, hard_box=args.hard_box,
            box_min=args.box_min, box_max=args.box_max,
        )
        img = data["image"].to(device)
        if args.answer_only:
            ids, mask, lab = answer_only_labels(
                tokenizer, data["prompt"], data["text"], max_length=args.max_len,
            )
            ids, mask, lab = ids.to(device), mask.to(device), lab.to(device)
            loss, meta = bridge(img, ids, mask, text_labels=lab)
        else:
            ids, mask = tokenize_captions(tokenizer, data["text"], max_length=args.max_len)
            ids, mask = ids.to(device), mask.to(device)
            loss, meta = bridge(img, ids, mask)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last_loss = float(loss.detach().float().item())
        if step % args.log_every == 0 or step in (1, args.steps):
            hist.append({"step": step, "loss": last_loss, "T": int(meta.get("T", T))})
            print(
                f"  [ocr {kind} T={meta.get('T', T)} seed={seed}] "
                f"step {step:4d} loss={last_loss:.4f}",
                flush=True,
            )
    metrics = eval_ocr(
        bridge, tokenizer, device, args.res,
        n=args.probe_n, batch=min(8, max(args.batch, 4)),
        val_seed=args.val_seed, max_new=args.max_new,
        hard_frac=args.hard_frac, hard_box=args.hard_box,
        box_min=args.box_min, box_max=args.box_max,
    )
    return {
        "kind": kind,
        "T": int(hist[-1]["T"]) if hist else T,
        "requested_T": T,
        "seed": seed,
        "final_loss": last_loss,
        "seconds": time.time() - t0,
        "status": "ok",
        "history": hist,
        **metrics,
    }


@torch.no_grad()
def eval_ocr(bridge, tokenizer, device, res, n, batch, val_seed, max_new,
             hard_frac, hard_box, box_min, box_max):
    bridge.eval()
    rng = np.random.default_rng(val_seed)
    free_hit = free_tot = 0
    cons_hit = cons_tot = 0
    by_digit_free = {d: {"h": 0, "t": 0} for d in OCR_DIGITS}
    by_digit_cons = {d: {"h": 0, "t": 0} for d in OCR_DIGITS}
    fails = []
    left = n
    while left > 0:
        b = min(batch, left)
        data = make_ocr_vqa_batch(
            rng, b, res=res,
            hard_frac=hard_frac, hard_box=hard_box,
            box_min=box_min, box_max=box_max,
        )
        img = data["image"].to(device)
        free_p = _greedy_answer(
            bridge, tokenizer, img, data["prompt"], max_new=max_new, early_stop=True,
        )
        cons_p = constrained_rank_answer(
            bridge, tokenizer, img, data["prompt"], data["probe"],
        )
        for i in range(b):
            gold = data["answer"][i]
            ok_f = exact_match(free_p[i], gold)
            ok_c = exact_match(cons_p[i], gold)
            free_hit += int(ok_f)
            free_tot += 1
            cons_hit += int(ok_c)
            cons_tot += 1
            by_digit_free[gold]["t"] += 1
            by_digit_free[gold]["h"] += int(ok_f)
            by_digit_cons[gold]["t"] += 1
            by_digit_cons[gold]["h"] += int(ok_c)
            if not ok_f and len(fails) < 12:
                fails.append({
                    "gold": gold,
                    "free": free_p[i][:50],
                    "cons": cons_p[i],
                })
        left -= b
    bridge.train()
    return {
        "free_acc": free_hit / max(free_tot, 1),
        "cons_acc": cons_hit / max(cons_tot, 1),
        "n_eval": free_tot,
        "free_by_digit": {
            d: by_digit_free[d]["h"] / max(by_digit_free[d]["t"], 1)
            for d in OCR_DIGITS
        },
        "cons_by_digit": {
            d: by_digit_cons[d]["h"] / max(by_digit_cons[d]["t"], 1)
            for d in OCR_DIGITS
        },
        "fail_examples": fails,
        # chance for 10-way
        "chance": 0.1,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--res", type=int, default=32)
    ap.add_argument("--patch", type=int, default=4)
    ap.add_argument("--T_list", type=int, nargs="*", default=[64, 32, 16])
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--projector", default="mlp", choices=["mlp", "linear"])
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max_len", type=int, default=48)
    ap.add_argument("--probe_n", type=int, default=200)
    ap.add_argument("--val_seed", type=int, default=90_001)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0],
                    help="train seeds (default: single seed 0)")
    ap.add_argument("--kinds", nargs="*", default=["A", "B", "C"])
    ap.add_argument("--answer_only", action="store_true", default=True)
    ap.add_argument("--full_text", action="store_true",
                    help="use full Q+A CE instead of answer-only")
    ap.add_argument("--max_new", type=int, default=4)
    ap.add_argument("--hard_frac", type=float, default=0.35)
    ap.add_argument("--hard_box", type=int, default=14)
    ap.add_argument("--box_min", type=int, default=12)
    ap.add_argument("--box_max", type=int, default=22)
    ap.add_argument("--prefer", default="gemma")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results/published/ocr1px_vlm_table.json")
    ap.add_argument("--conclusion", default="results/published/ocr1px_vlm_conclusion.md")
    ap.add_argument(
        "--materialize", type=int, default=0,
        help="if >0, also write data/ocr1px with this many train samples before train",
    )
    args = ap.parse_args()
    if args.full_text:
        args.answer_only = False

    device = torch.device(args.device)
    print(f"cache={cache_root()}", flush=True)
    assert str(cache_root()).upper().startswith("D:")

    if args.materialize and args.materialize > 0:
        from scripts.build_ocr_dataset import build_split
        out_root = Path("data/ocr1px")
        print(f"materializing train n={args.materialize} → {out_root}", flush=True)
        build_split(str(out_root / "train"), args.materialize, args.res, 0, args.hard_frac)
        build_split(str(out_root / "val"), min(5000, max(1000, args.materialize // 10)),
                    args.res, args.val_seed, args.hard_frac)
        meta = {
            "task": "ocr_1px_digits",
            "n_train": args.materialize,
            "res": args.res,
            "note": "on-the-fly training still used; disk copy for inspection",
        }
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"loading LLM prefer={args.prefer} …", flush=True)
    llm, tokenizer, d_llm, note = load_frozen_lm(prefer=args.prefer, device=str(device))
    print(f"backend={note}", flush=True)
    print(
        f"OCR 1px digits → LLM | answer_only={args.answer_only} "
        f"steps={args.steps} T={args.T_list} seeds={args.seeds}",
        flush=True,
    )

    # effective train stream size for logging
    stream_n = args.steps * args.batch
    # resume: keep completed cells from existing table
    rows = []
    done_keys = set()
    out_path = Path(args.out)
    if out_path.is_file():
        try:
            prev = json.loads(out_path.read_text(encoding="utf-8"))
            for r in prev.get("table") or []:
                if r.get("status") == "ok" and "free_acc" in r and "cons_acc" in r:
                    rows.append(r)
                    done_keys.add((r.get("kind"), r.get("T"), r.get("seed")))
            if done_keys:
                print(f"resume: skip {len(done_keys)} completed cells {sorted(done_keys)}", flush=True)
        except Exception as e:
            print(f"resume: could not load {out_path}: {e}", flush=True)

    native = patch_token_count(args.res, args.patch)
    for seed in args.seeds:
        for T in args.T_list:
            for kind in args.kinds:
                if kind == "A" and T > native:
                    continue
                key = (kind, T, seed)
                # Skip only exact completed keys. For C, stored T may be split
                # total (e.g. 48 for budget 64) — match requested budget via meta.
                def _already(r, kind=kind, T=T, seed=seed):
                    if r.get("status") != "ok" or "free_acc" not in r:
                        return False
                    if r.get("kind") != kind or r.get("seed") != seed:
                        return False
                    if r.get("T") == T:
                        return True
                    # optional: requested_T field if present
                    if r.get("requested_T") == T:
                        return True
                    return False

                if key in done_keys or any(_already(r) for r in rows):
                    print(f"\n=== {kind} T={T} seed={seed} SKIP (have results) ===", flush=True)
                    continue
                print(f"\n=== {kind} T={T} seed={seed} ===", flush=True)
                try:
                    row = train_one(kind, T, args, llm, tokenizer, d_llm, device, seed)
                    rows.append(row)
                    done_keys.add((row.get("kind"), row.get("T"), row.get("seed")))
                    print(
                        f"  → free={row.get('free_acc', float('nan')):.3f} "
                        f"cons={row.get('cons_acc', float('nan')):.3f}",
                        flush=True,
                    )
                except Exception as e:
                    err = "OOM" if "out of memory" in str(e).lower() else str(e)
                    rows.append({
                        "kind": kind, "T": T, "seed": seed,
                        "status": "error", "error": err,
                    })
                    print(f"  FAIL {e}", flush=True)
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                write_ocr_artifacts(rows, args, note, stream_n)

    write_ocr_artifacts(rows, args, note, stream_n)
    print(f"json → {args.out}", flush=True)


def write_ocr_artifacts(rows, args, note, stream_n: int) -> None:
    """Write/overwrite published JSON + markdown (safe after each cell)."""
    payload = {
        "task": "ocr_1px_digits_vlm",
        "backend": note,
        "cache_root": str(cache_root()),
        "res": args.res,
        "patch": args.patch,
        "steps": args.steps,
        "batch": args.batch,
        "stream_samples": stream_n,
        "answer_only": args.answer_only,
        "projector": args.projector,
        "topk": args.topk,
        "alphabet": list(OCR_DIGITS),
        "val_seed": args.val_seed,
        "probe_n": args.probe_n,
        "chance": 0.1,
        "T_list": args.T_list,
        "seeds": args.seeds,
        "table": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    lines = [
        "# 1px OCR digits → frozen LLM (A/B/C)",
        "",
        f"- Backend: `{note}`",
        f"- Task: single stick-digit **0–9**, **1px Bresenham** strokes on noisy canvas",
        f"- res={args.res} projector={args.projector} topk={args.topk} "
        f"answer_only={args.answer_only}",
        f"- Train stream ≈ **{stream_n}** samples/cell (steps×batch); eval n={args.probe_n}",
        f"- Metrics: free-gen exact-match + constrained ranking over digits",
        f"- Seeds: {args.seeds} (single-seed default)",
        f"- Chance = 0.10 (10-way)",
        "",
        "| kind | T | seed | loss | free_acc | cons_acc | notes |",
        "|------|---|------|------|----------|----------|-------|",
    ]
    ok = [r for r in rows if r.get("status") == "ok"]
    for r in rows:
        if r.get("status") != "ok":
            lines.append(
                f"| {r.get('kind')} | {r.get('T')} | {r.get('seed')} | — | — | — | {r.get('error')} |"
            )
        else:
            lines.append(
                f"| {r['kind']} | {r['T']} | {r['seed']} | {r['final_loss']:.3f} | "
                f"{r['free_acc']:.3f} | {r['cons_acc']:.3f} | |"
            )
    lines.extend(["", "## Conclusion", ""])
    if ok:
        best_free = max(ok, key=lambda x: x["free_acc"])
        best_cons = max(ok, key=lambda x: x["cons_acc"])
        lines.append(
            f"- Best **free-gen**: {best_free['kind']}@T{best_free['T']} "
            f"s{best_free['seed']} acc={best_free['free_acc']:.3f}"
        )
        lines.append(
            f"- Best **constrained**: {best_cons['kind']}@T{best_cons['T']} "
            f"s{best_cons['seed']} acc={best_cons['cons_acc']:.3f}"
        )
        by_kind = {}
        for r in ok:
            by_kind.setdefault(r["kind"], []).append(r)
        for k, vs in sorted(by_kind.items()):
            mf = max(vs, key=lambda x: x["free_acc"])
            mc = max(vs, key=lambda x: x["cons_acc"])
            lines.append(
                f"- **{k}**: best free={mf['free_acc']:.3f} (T{mf['T']}), "
                f"best cons={mc['cons_acc']:.3f} (T{mc['T']})"
            )
        b_cons = [r for r in ok if r["kind"] == "B"]
        a_cons = [r for r in ok if r["kind"] == "A"]
        if b_cons and a_cons:
            bb = max(b_cons, key=lambda x: x["cons_acc"])
            aa = max(a_cons, key=lambda x: x["cons_acc"])
            lines.append(
                f"- B vs A (best cons): B={bb['cons_acc']:.3f} vs A={aa['cons_acc']:.3f} "
                f"(Δ={bb['cons_acc']-aa['cons_acc']:+.3f})"
            )
        mean_cons = float(np.mean([r["cons_acc"] for r in ok]))
        mean_free = float(np.mean([r["free_acc"] for r in ok]))
        lines.append(
            f"- Mean free={mean_free:.3f}, mean cons={mean_cons:.3f} "
            f"(chance=0.10); cons−free≈{mean_cons-mean_free:+.3f}"
        )
    else:
        lines.append("- No successful runs yet (partial run).")
    Path(args.conclusion).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()

