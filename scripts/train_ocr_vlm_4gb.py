#!/usr/bin/env python3
"""4GB-friendly VLM-OCR frontend training (frozen small LM).

Defaults tuned for ~4GB VRAM:
  - batch_size=1, grad_accum=8  (effective batch 8)
  - train frontend + MLP only; LLM frozen
  - multi-digit string OCR (not single-class toy only)
  - answer-only CE, free-gen CER / exact string match
  - optional AMP fp16 (falls back if unstable)
  - single seed, A vs B (C optional)

Example:
  set ML_CACHE_ROOT=D:\\ml_cache
  python scripts/train_ocr_vlm_4gb.py --prefer gemma --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

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
from fine_grain.vlm_data import (  # noqa: E402
    answer_only_labels,
    char_error_rate,
    exact_string_match,
    make_ocr_string_vqa_batch,
    make_ocr_vqa_batch,
)
from scripts.train_vlm_frontends import (  # noqa: E402
    VLMBridge,
    _greedy_answer,
    _trainable,
)


def _vram_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def train_one(kind: str, T: int, args, llm, tokenizer, d_llm, device, seed: int) -> dict:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    fe = build_frontend(
        kind, d_llm, res=args.res, T=T, patch=args.patch,
        dim=args.dim, depth=args.depth, deslice_topk=args.topk,
        projector=args.projector,
    ).to(device)
    bridge = VLMBridge(fe, llm).to(device)
    # freeze LLM explicitly
    for p in bridge.llm.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(_trainable(bridge), lr=args.lr, weight_decay=0.01)

    use_amp = bool(args.amp and device.type == "cuda")
    # fp16 GradScaler; bf16 usually no scaler
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and not args.bf16)
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16

    rng = np.random.default_rng(seed + 17)
    bridge.train()
    hist = []
    t0 = time.time()
    last_loss = float("nan")
    accum = max(1, int(args.grad_accum))
    opt.zero_grad(set_to_none=True)

    for step in range(1, args.steps + 1):
        if args.task == "string":
            data = make_ocr_string_vqa_batch(
                rng, args.batch, res=args.res,
                min_len=args.min_len, max_len=args.str_max_len,
                char_box=args.char_box, gap=args.gap,
                hard_frac=args.hard_frac,
            )
        else:
            data = make_ocr_vqa_batch(
                rng, args.batch, res=args.res,
                hard_frac=args.hard_frac,
            )
        img = data["image"].to(device, non_blocking=True)
        ids, mask, lab = answer_only_labels(
            tokenizer, data["prompt"], data["text"], max_length=args.max_len,
        )
        ids, mask, lab = ids.to(device), mask.to(device), lab.to(device)

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            loss, meta = bridge(img, ids, mask, text_labels=lab)
            loss = loss / accum

        if use_amp and not args.bf16:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if step % accum == 0 or step == args.steps:
            if use_amp and not args.bf16:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(_trainable(bridge), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(_trainable(bridge), 1.0)
                opt.step()
            opt.zero_grad(set_to_none=True)

        last_loss = float(loss.detach().float().item() * accum)
        if step % args.log_every == 0 or step in (1, args.steps):
            hist.append({
                "step": step,
                "loss": last_loss,
                "T": int(meta.get("T", T)),
                "vram_mb": _vram_mb(),
            })
            print(
                f"  [{kind} T={meta.get('T', T)} s{seed}] step {step:4d} "
                f"loss={last_loss:.4f} vram_peak={_vram_mb():.0f}MB",
                flush=True,
            )

    metrics = eval_string_ocr(
        bridge, tokenizer, device, args,
        n=args.probe_n, val_seed=args.val_seed,
    )
    return {
        "kind": kind,
        "T": int(hist[-1]["T"]) if hist else T,
        "requested_T": T,
        "seed": seed,
        "final_loss": last_loss,
        "seconds": time.time() - t0,
        "peak_vram_mb": _vram_mb(),
        "status": "ok",
        "history": hist,
        **metrics,
    }


@torch.no_grad()
def eval_string_ocr(bridge, tokenizer, device, args, n: int, val_seed: int) -> dict:
    bridge.eval()
    rng = np.random.default_rng(val_seed)
    exact_hit = 0
    cer_sum = 0.0
    tot = 0
    fails: List[dict] = []
    left = n
    bs = 1  # eval always batch=1 for 4GB
    while left > 0:
        b = min(bs, left)
        if args.task == "string":
            data = make_ocr_string_vqa_batch(
                rng, b, res=args.res,
                min_len=args.min_len, max_len=args.str_max_len,
                char_box=args.char_box, gap=args.gap,
                hard_frac=args.hard_frac,
            )
        else:
            data = make_ocr_vqa_batch(rng, b, res=args.res, hard_frac=args.hard_frac)
        img = data["image"].to(device)
        preds = _greedy_answer(
            bridge, tokenizer, img, data["prompt"],
            max_new=args.max_new, early_stop=True,
        )
        for i in range(b):
            gold = data["answer"][i]
            pred = preds[i]
            ok = exact_string_match(pred, gold)
            cer = char_error_rate(pred, gold)
            exact_hit += int(ok)
            cer_sum += cer
            tot += 1
            if not ok and len(fails) < 10:
                fails.append({"gold": gold, "pred": pred[:40], "cer": cer})
        left -= b
    bridge.train()
    return {
        "exact_acc": exact_hit / max(tot, 1),
        "cer": cer_sum / max(tot, 1),
        "n_eval": tot,
        "fail_examples": fails,
    }


def write_artifacts(rows, args, note, stream_n: int) -> None:
    payload = {
        "task": f"ocr_vlm_4gb_{args.task}",
        "backend": note,
        "cache_root": str(cache_root()),
        "profile": "4gb",
        "res": args.res,
        "batch": args.batch,
        "grad_accum": args.grad_accum,
        "effective_batch": args.batch * args.grad_accum,
        "steps": args.steps,
        "stream_samples": stream_n,
        "amp": args.amp,
        "bf16": args.bf16,
        "answer_only": True,
        "projector": args.projector,
        "prefer": args.prefer,
        "min_len": args.min_len,
        "str_max_len": args.str_max_len,
        "T_list": args.T_list,
        "seeds": args.seeds,
        "kinds": args.kinds,
        "table": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    lines = [
        "# VLM-OCR frontend @ 4GB (frozen small LM)",
        "",
        f"- Backend: `{note}`",
        f"- Profile: batch={args.batch} accum={args.grad_accum} "
        f"eff_batch={args.batch * args.grad_accum} amp={args.amp} bf16={args.bf16}",
        f"- Task: **{args.task}** 1px strokes; "
        f"{'multi-digit strings len ' + str(args.min_len) + '-' + str(args.str_max_len) if args.task == 'string' else 'single digit'}",
        f"- Train: frontend+MLP only, LLM frozen, answer-only CE",
        f"- Eval: free-gen exact string match + CER (n={args.probe_n})",
        f"- steps={args.steps} res={args.res} projector={args.projector}",
        "",
        "| kind | T | seed | loss | exact_acc | CER | peak_VRAM_MB |",
        "|------|---|------|------|-----------|-----|--------------|",
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
                f"{r['exact_acc']:.3f} | {r['cer']:.3f} | {r.get('peak_vram_mb', 0):.0f} |"
            )
    lines.extend(["", "## Conclusion", ""])
    if ok:
        best = min(ok, key=lambda x: x["cer"])
        lines.append(
            f"- Best CER: {best['kind']}@T{best['T']} CER={best['cer']:.3f} "
            f"exact={best['exact_acc']:.3f}"
        )
        for k in sorted({r["kind"] for r in ok}):
            sub = [r for r in ok if r["kind"] == k]
            b = min(sub, key=lambda x: x["cer"])
            lines.append(
                f"- **{k}**: best CER={b['cer']:.3f} exact={b['exact_acc']:.3f} (T{b['T']})"
            )
        a = [r for r in ok if r["kind"] == "A"]
        b = [r for r in ok if r["kind"] == "B"]
        if a and b:
            aa = min(a, key=lambda x: x["cer"])
            bb = min(b, key=lambda x: x["cer"])
            lines.append(
                f"- B vs A (best CER): B={bb['cer']:.3f} vs A={aa['cer']:.3f} "
                f"(ΔCER={bb['cer']-aa['cer']:+.3f}; lower better)"
            )
        lines.append(
            f"- Peak VRAM observed: {max(r.get('peak_vram_mb', 0) for r in ok):.0f} MB"
        )
        lines.append(
            "- 4GB profile: frozen LM + train vision only; string OCR = real short-sequence "
            "readout metric (CER), not 10-way classification."
        )
    else:
        lines.append("- No successful runs.")
    Path(args.conclusion).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    # 4GB defaults
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--res", type=int, default=32)
    ap.add_argument("--patch", type=int, default=4)
    ap.add_argument("--T_list", type=int, nargs="*", default=[32, 16])
    ap.add_argument("--kinds", nargs="*", default=["A", "B"])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0])
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--projector", default="mlp", choices=["mlp", "linear"])
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max_len", type=int, default=64, help="tokenizer max length")
    ap.add_argument("--max_new", type=int, default=8)
    ap.add_argument("--probe_n", type=int, default=64)
    ap.add_argument("--val_seed", type=int, default=90_001)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--task", choices=["string", "digit"], default="string")
    ap.add_argument("--min_len", type=int, default=2, help="min OCR string length")
    ap.add_argument("--str_max_len", type=int, default=4, help="max OCR string length")
    ap.add_argument("--char_box", type=int, default=10)
    ap.add_argument("--gap", type=int, default=2)
    ap.add_argument("--hard_frac", type=float, default=0.3)
    ap.add_argument("--prefer", default="gemma")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results/published/ocr_vlm_4gb_table.json")
    ap.add_argument("--conclusion", default="results/published/ocr_vlm_4gb_conclusion.md")
    args = ap.parse_args()
    if args.no_amp:
        args.amp = False

    device = torch.device(args.device)
    print(f"cache={cache_root()} device={device}", flush=True)
    assert str(cache_root()).upper().startswith("D:")

    print(
        f"4GB profile: batch={args.batch} accum={args.grad_accum} "
        f"eff={args.batch * args.grad_accum} amp={args.amp} task={args.task}",
        flush=True,
    )
    print(f"loading LLM prefer={args.prefer} …", flush=True)
    llm, tokenizer, d_llm, note = load_frozen_lm(prefer=args.prefer, device=str(device))
    print(f"backend={note} d_llm={d_llm}", flush=True)

    stream_n = args.steps * args.batch  # micro-batch samples
    rows = []
    done = set()
    out_path = Path(args.out)
    if out_path.is_file():
        try:
            prev = json.loads(out_path.read_text(encoding="utf-8"))
            for r in prev.get("table") or []:
                if r.get("status") == "ok" and "cer" in r:
                    rows.append(r)
                    done.add((r.get("kind"), r.get("requested_T", r.get("T")), r.get("seed")))
            if done:
                print(f"resume skip {sorted(done)}", flush=True)
        except Exception as e:
            print(f"resume load fail: {e}", flush=True)

    native = patch_token_count(args.res, args.patch)
    for seed in args.seeds:
        for T in args.T_list:
            for kind in args.kinds:
                if kind == "A" and T > native:
                    continue
                if (kind, T, seed) in done:
                    print(f"=== {kind} T={T} s{seed} SKIP ===", flush=True)
                    continue
                print(f"\n=== {kind} T={T} seed={seed} ===", flush=True)
                try:
                    row = train_one(kind, T, args, llm, tokenizer, d_llm, device, seed)
                    rows.append(row)
                    done.add((kind, T, seed))
                    print(
                        f"  → exact={row['exact_acc']:.3f} CER={row['cer']:.3f} "
                        f"vram={row['peak_vram_mb']:.0f}MB",
                        flush=True,
                    )
                except RuntimeError as e:
                    msg = str(e)
                    err = "OOM" if "out of memory" in msg.lower() else msg[:200]
                    rows.append({
                        "kind": kind, "T": T, "requested_T": T, "seed": seed,
                        "status": "error", "error": err,
                    })
                    print(f"  FAIL {err}", flush=True)
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                except Exception as e:
                    rows.append({
                        "kind": kind, "T": T, "requested_T": T, "seed": seed,
                        "status": "error", "error": str(e)[:200],
                    })
                    print(f"  FAIL {e}", flush=True)
                write_artifacts(rows, args, note, stream_n)

    write_artifacts(rows, args, note, stream_n)
    print(f"json → {args.out}", flush=True)


if __name__ == "__main__":
    main()
