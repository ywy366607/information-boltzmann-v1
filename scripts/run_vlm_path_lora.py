#!/usr/bin/env python3
"""Phase 3: frozen LM vs LoRA (trainable language) on VLM-OCR path.

Compares A (patch) / B (slice) under:
  - frozen: train frontend+projector only
  - lora:   train frontend+projector + LoRA on last-N LLM layers

Metrics: TF answer-span acc + digit-constrained CER/exact (not open free-gen).

Example (4GB):
  set ML_CACHE_ROOT=D:\\ml_cache
  python scripts/run_vlm_path_lora.py --steps 400 --probe_n 48 --amp
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

from fine_grain.frontends import build_frontend  # noqa: E402
from fine_grain.llm_backend import cache_root, load_frozen_lm  # noqa: E402
from fine_grain.lora_llm import (  # noqa: E402
    apply_lora,
    enable_lora_grads,
    peft_available,
)
from fine_grain.vlm_data import (  # noqa: E402
    answer_only_labels,
    char_error_rate,
    exact_string_match,
    make_ocr_string_vqa_batch,
)
from fine_grain.visual_latent_cot import greedy_digit_string  # noqa: E402
from scripts.train_vlm_frontends import (  # noqa: E402
    VLMBridge,
    _trainable,
    bridge_tf_logits,
    tf_answer_span_accuracy,
)


def _vram_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


@torch.no_grad()
def eval_tf_and_constrained(bridge, tokenizer, device, args) -> dict:
    bridge.eval()
    rng = np.random.default_rng(args.val_seed)
    tf_hit = tf_tot = 0
    exact_hit = 0
    cer_sum = 0.0
    n = 0
    fails = []
    for _ in range(args.probe_n):
        data = make_ocr_string_vqa_batch(
            rng, 1, res=args.res,
            min_len=args.min_len, max_len=args.str_max_len,
            char_box=args.char_box, hard_frac=args.hard_frac,
        )
        img = data["image"].to(device)
        ids, mask, lab = answer_only_labels(
            tokenizer, data["prompt"], data["text"], max_length=args.max_len,
        )
        ids, mask, lab = ids.to(device), mask.to(device), lab.to(device)
        logits, Tvis = bridge_tf_logits(bridge, img, ids, mask)
        _, n_hit, n_ans = tf_answer_span_accuracy(logits, Tvis, ids, lab)
        tf_hit += n_hit
        tf_tot += n_ans
        gold = data["answer"][0]
        max_new = min(args.max_new, max(2, len(gold) + 1))
        pred = greedy_digit_string(
            bridge, tokenizer, img, data["prompt"],
            max_new=max_new, refine_every=0,
        )[0]
        ok = exact_string_match(pred, gold)
        cer = char_error_rate(pred, gold)
        exact_hit += int(ok)
        cer_sum += min(cer, 3.0)
        n += 1
        if not ok and len(fails) < 8:
            fails.append({"gold": gold, "pred": pred[:32], "cer": round(cer, 3)})
    bridge.train()
    return {
        "tf_acc": tf_hit / max(tf_tot, 1),
        "tf_n_tokens": tf_tot,
        "exact_acc": exact_hit / max(n, 1),
        "cer": cer_sum / max(n, 1),
        "n_eval": n,
        "fail_examples": fails,
        "decode": "digit-constrained",
    }


def train_one(
    kind: str,
    mode: str,
    args,
    llm,
    tokenizer,
    d_llm,
    device,
    seed: int,
    lora_meta: dict | None,
) -> dict:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    fe = build_frontend(
        kind, d_llm, res=args.res, T=args.T, patch=args.patch,
        dim=args.dim, depth=args.depth, deslice_topk=args.topk,
        projector=args.projector,
    )
    bridge = VLMBridge(fe, llm).to(device)
    # VLMBridge freezes all llm; re-enable LoRA if present
    if mode == "lora":
        n_lora = enable_lora_grads(bridge.llm)
        print(f"  re-enabled LoRA grads: {n_lora} params", flush=True)
    else:
        for p in bridge.llm.parameters():
            p.requires_grad_(False)

    train_params = _trainable(bridge)
    n_train = sum(p.numel() for p in train_params)
    print(f"  trainable tensors={len(train_params)} params={n_train}", flush=True)
    opt = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=0.01)
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    rng = np.random.default_rng(seed + 41)
    bridge.train()
    accum = max(1, args.grad_accum)
    opt.zero_grad(set_to_none=True)
    last = float("nan")
    t0 = time.time()
    hist = []
    meta = {}

    for step in range(1, args.steps + 1):
        data = make_ocr_string_vqa_batch(
            rng, args.batch, res=args.res,
            min_len=args.min_len, max_len=args.str_max_len,
            char_box=args.char_box, hard_frac=args.hard_frac,
        )
        img = data["image"].to(device)
        ids, mask, lab = answer_only_labels(
            tokenizer, data["prompt"], data["text"], max_length=args.max_len,
        )
        ids, mask, lab = ids.to(device), mask.to(device), lab.to(device)
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
            loss, meta = bridge(img, ids, mask, text_labels=lab)
            loss = loss / accum
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if step % accum == 0 or step == args.steps:
            if use_amp:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(train_params, 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(train_params, 1.0)
                opt.step()
            opt.zero_grad(set_to_none=True)
        last = float(loss.detach().float().item() * accum)
        if step % args.log_every == 0 or step in (1, args.steps):
            hist.append({"step": step, "loss": last, "vram_mb": _vram_mb()})
            print(
                f"  [{kind}/{mode}] step {step:4d} loss={last:.4f} "
                f"vram={_vram_mb():.0f}MB",
                flush=True,
            )

    metrics = eval_tf_and_constrained(bridge, tokenizer, device, args)
    row = {
        "kind": kind,
        "mode": mode,
        "T": int(meta.get("T", args.T)),
        "seed": seed,
        "final_loss": last,
        "seconds": time.time() - t0,
        "peak_vram_mb": _vram_mb(),
        "trainable_params": n_train,
        "status": "ok",
        "history": hist,
        **metrics,
    }
    if lora_meta:
        row["lora"] = lora_meta
    return row


def write_artifacts(rows, args, note: str) -> None:
    payload = {
        "task": "vlm_path_lora_phase3",
        "backend": note,
        "cache_root": str(cache_root()),
        "profile": "4gb",
        "res": args.res,
        "batch": args.batch,
        "grad_accum": args.grad_accum,
        "steps": args.steps,
        "T": args.T,
        "projector": args.projector,
        "amp": args.amp,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_last_n": args.lora_last_n,
        "min_len": args.min_len,
        "str_max_len": args.str_max_len,
        "probe_n": args.probe_n,
        "kinds": args.kinds,
        "modes": args.modes,
        "metrics": ["tf_acc", "exact_acc", "cer"],
        "decode": "digit-constrained",
        "table": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    lines = [
        "# Phase 3: frozen LM vs LoRA (trainable language side)",
        "",
        f"- Backend: `{note}`",
        f"- LoRA: r={args.lora_r} α={args.lora_alpha} last_n={args.lora_last_n} "
        f"targets=q_proj,v_proj",
        f"- Profile: batch={args.batch} accum={args.grad_accum} steps={args.steps} "
        f"res={args.res} T={args.T} amp={args.amp}",
        f"- Task: digit strings len {args.min_len}-{args.str_max_len}; "
        f"eval n={args.probe_n}",
        f"- Metrics: **TF answer-token acc** + **digit-constrained** CER/exact",
        "",
        "| kind | mode | tf_acc | exact | CER | loss | train_params | VRAM | status |",
        "|------|------|--------|-------|-----|------|--------------|------|--------|",
    ]
    for r in rows:
        if r.get("status") != "ok":
            lines.append(
                f"| {r.get('kind')} | {r.get('mode')} | — | — | — | — | — | — | "
                f"{r.get('error')} |"
            )
        else:
            lines.append(
                f"| {r['kind']} | {r['mode']} | {r['tf_acc']:.3f} | "
                f"{r['exact_acc']:.3f} | {r['cer']:.3f} | {r['final_loss']:.3f} | "
                f"{r.get('trainable_params', 0)} | {r.get('peak_vram_mb', 0):.0f} | ok |"
            )

    lines.extend(["", "## Comparisons", ""])
    ok = [r for r in rows if r.get("status") == "ok"]
    by = {(r["kind"], r["mode"]): r for r in ok}

    def _pair(k: str):
        fr, lo = by.get((k, "frozen")), by.get((k, "lora"))
        if fr and lo:
            lines.append(f"### {k}: LoRA vs frozen")
            lines.append(
                f"- TF: LoRA={lo['tf_acc']:.3f} vs frozen={fr['tf_acc']:.3f} "
                f"(Δ={lo['tf_acc']-fr['tf_acc']:+.3f})"
            )
            lines.append(
                f"- CER: LoRA={lo['cer']:.3f} vs frozen={fr['cer']:.3f} "
                f"(Δ={lo['cer']-fr['cer']:+.3f}; lower better)"
            )
            lines.append(
                f"- exact: LoRA={lo['exact_acc']:.3f} vs frozen={fr['exact_acc']:.3f} "
                f"(Δ={lo['exact_acc']-fr['exact_acc']:+.3f})"
            )
            win = (
                lo["tf_acc"] > fr["tf_acc"] + 0.03
                or lo["cer"] < fr["cer"] - 0.05
                or lo["exact_acc"] > fr["exact_acc"] + 0.02
            )
            if win:
                lines.append(f"- **LoRA helps {k}** under thresholds.")
            else:
                lines.append(f"- No clear LoRA win for {k} under thresholds.")

    for k in args.kinds:
        _pair(k)

    # B vs A under best mode
    for mode in args.modes:
        a, b = by.get(("A", mode)), by.get(("B", mode))
        if a and b:
            lines.append(f"### B vs A under **{mode}**")
            lines.append(
                f"- TF Δ(B−A)={b['tf_acc']-a['tf_acc']:+.3f}; "
                f"CER Δ={b['cer']-a['cer']:+.3f}; "
                f"exact Δ={b['exact_acc']-a['exact_acc']:+.3f}"
            )

    if ok:
        best_ex = max(ok, key=lambda x: (x["exact_acc"], -x["cer"], x["tf_acc"]))
        lines.append("")
        lines.append(
            f"- Best cell by exact then CER: **{best_ex['kind']}/{best_ex['mode']}** "
            f"exact={best_ex['exact_acc']:.3f} CER={best_ex['cer']:.3f} "
            f"tf={best_ex['tf_acc']:.3f}"
        )
        lora_rows = [r for r in ok if r["mode"] == "lora"]
        fr_rows = [r for r in ok if r["mode"] == "frozen"]
        if lora_rows and fr_rows:
            mean = lambda xs, k: sum(r[k] for r in xs) / len(xs)
            lines.append(
                f"- Mean LoRA vs frozen: TF {mean(lora_rows,'tf_acc'):.3f} vs "
                f"{mean(fr_rows,'tf_acc'):.3f}; "
                f"CER {mean(lora_rows,'cer'):.3f} vs {mean(fr_rows,'cer'):.3f}; "
                f"exact {mean(lora_rows,'exact_acc'):.3f} vs {mean(fr_rows,'exact_acc'):.3f}"
            )
            if mean(lora_rows, "exact_acc") > mean(fr_rows, "exact_acc") + 0.02 or (
                mean(lora_rows, "cer") < mean(fr_rows, "cer") - 0.05
            ) or mean(lora_rows, "tf_acc") > mean(fr_rows, "tf_acc") + 0.03:
                lines.append(
                    "- **Claim:** trainable language (LoRA) unlocks better VLM-path "
                    "readout than frozen LM under this matched budget."
                )
            else:
                lines.append(
                    "- **Honest null/weak:** LoRA does not clearly beat frozen under "
                    "this short budget (or absolute OCR still weak)."
                )

    lines.append("")
    lines.append("Non-claim: open free-gen alone is not product OCR proof.")
    Path(args.conclusion).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"json → {args.out}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--res", type=int, default=32)
    ap.add_argument("--patch", type=int, default=4)
    ap.add_argument("--T", type=int, default=32)
    ap.add_argument("--kinds", nargs="*", default=["A", "B"])
    ap.add_argument("--modes", nargs="*", default=["frozen", "lora"])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0])
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--projector", default="mlp")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--lora_last_n", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--max_new", type=int, default=5)
    ap.add_argument("--probe_n", type=int, default=48)
    ap.add_argument("--val_seed", type=int, default=90_001)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--min_len", type=int, default=2)
    ap.add_argument("--str_max_len", type=int, default=3)
    ap.add_argument("--char_box", type=int, default=10)
    ap.add_argument("--hard_frac", type=float, default=0.25)
    ap.add_argument("--prefer", default="gemma")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results/published/vlm_path_lora_table.json")
    ap.add_argument(
        "--conclusion", default="results/published/vlm_path_lora_conclusion.md",
    )
    args = ap.parse_args()
    if args.no_amp:
        args.amp = False

    if "lora" in args.modes and not peft_available():
        raise SystemExit("peft not installed; run scripts/pip_get.py peft accelerate")

    device = torch.device(args.device)
    print(f"cache={cache_root()} device={device} Phase3 LoRA", flush=True)
    assert str(cache_root()).upper().startswith("D:")

    rows = []
    note = ""
    for seed in args.seeds:
        for mode in args.modes:
            for kind in args.kinds:
                print(f"\n=== {kind} mode={mode} seed={seed} ===", flush=True)
                lora_meta = None
                try:
                    # Fresh LM each cell so LoRA state never leaks across cells
                    llm, tokenizer, d_llm, note = load_frozen_lm(
                        prefer=args.prefer, device=str(device),
                    )
                    if mode == "lora":
                        llm, lora_meta = apply_lora(
                            llm,
                            r=args.lora_r,
                            alpha=args.lora_alpha,
                            dropout=args.lora_dropout,
                            last_n_layers=args.lora_last_n,
                        )
                        print(
                            f"  LoRA trainable%={lora_meta['trainable_pct']:.4f} "
                            f"params={lora_meta['trainable_params']} "
                            f"layers={lora_meta['layer_range']}",
                            flush=True,
                        )
                    row = train_one(
                        kind, mode, args, llm, tokenizer, d_llm, device, seed, lora_meta,
                    )
                    rows.append(row)
                    print(
                        f"  → tf={row['tf_acc']:.3f} exact={row['exact_acc']:.3f} "
                        f"CER={row['cer']:.3f} vram={row['peak_vram_mb']:.0f}MB",
                        flush=True,
                    )
                except RuntimeError as e:
                    msg = str(e)
                    err = "OOM" if "out of memory" in msg.lower() else msg[:220]
                    rows.append({
                        "kind": kind, "mode": mode, "seed": seed,
                        "status": "error", "error": err,
                    })
                    print(f"  FAIL {err}", flush=True)
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                except Exception as e:
                    rows.append({
                        "kind": kind, "mode": mode, "seed": seed,
                        "status": "error", "error": str(e)[:220],
                    })
                    print(f"  FAIL {e}", flush=True)
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                finally:
                    # free LLM between cells
                    try:
                        del llm  # noqa: F821
                    except Exception:
                        pass
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                write_artifacts(rows, args, note or "unknown")

    write_artifacts(rows, args, note or "unknown")


if __name__ == "__main__":
    main()
