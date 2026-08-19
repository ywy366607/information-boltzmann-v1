#!/usr/bin/env python3
"""Phase 2: teacher-forced answer-span acc + digit-constrained CER (A/B/B_vlcot).

Frozen small local LM; train frontend (+ projector / VL-CoT) only.
Primary free-path KPI = digit-constrained CER/exact (not open free-gen).

Example (4GB):
  set ML_CACHE_ROOT=D:\\ml_cache
  python scripts/run_vlm_path_tf_cer.py --steps 400 --probe_n 48 --amp
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
from fine_grain.visual_latent_cot import (  # noqa: E402
    VLCoTBridge,
    VisualLatentCoTSlice,
    greedy_digit_string,
)
from fine_grain.vlm_data import (  # noqa: E402
    answer_only_labels,
    char_error_rate,
    exact_string_match,
    make_ocr_string_vqa_batch,
)
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


def build_bridge(kind: str, d_llm: int, args, llm):
    k = kind.upper()
    if k == "A":
        fe = build_frontend(
            "A", d_llm, res=args.res, T=args.T, patch=args.patch,
            dim=args.dim, depth=args.depth, deslice_topk=args.topk,
            projector=args.projector,
        )
        return VLMBridge(fe, llm), False
    if k == "B":
        fe = build_frontend(
            "B", d_llm, res=args.res, T=args.T, patch=args.patch,
            dim=args.dim, depth=args.depth, deslice_topk=args.topk,
            projector=args.projector,
        )
        return VLMBridge(fe, llm), False
    if k in ("B_VLCOT", "VLCOT", "B_VL"):
        fe = VisualLatentCoTSlice(
            d_llm, res=args.res, T=args.T, dim=args.dim, depth=args.depth,
            deslice_topk=args.topk, projector=args.projector,
            latent_steps=args.latent_steps, beta=args.beta, gamma=args.gamma,
        )
        return VLCoTBridge(fe, llm, pass1_ce_weight=args.pass1_w), True
    raise ValueError(kind)


@torch.no_grad()
def eval_tf_and_constrained(bridge, tokenizer, device, args, two_look: bool) -> dict:
    """Measured TF answer-span acc + digit-constrained CER/exact."""
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
            max_new=max_new,
            refine_every=(1 if two_look else 0),
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


def train_one(kind: str, args, llm, tokenizer, d_llm, device, seed: int) -> dict:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    bridge, two_look = build_bridge(kind, d_llm, args, llm)
    bridge = bridge.to(device)
    for p in bridge.llm.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(_trainable(bridge), lr=args.lr, weight_decay=0.01)
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    rng = np.random.default_rng(seed + 31)
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
            if two_look:
                loss, meta = bridge(
                    img, ids, mask, text_labels=lab, two_look=True,
                )
            else:
                loss, meta = bridge(img, ids, mask, text_labels=lab)
            loss = loss / accum
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if step % accum == 0 or step == args.steps:
            if use_amp:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(_trainable(bridge), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(_trainable(bridge), 1.0)
                opt.step()
            opt.zero_grad(set_to_none=True)
        last = float(loss.detach().float().item() * accum)
        if step % args.log_every == 0 or step in (1, args.steps):
            hist.append({"step": step, "loss": last, "vram_mb": _vram_mb()})
            print(
                f"  [{kind}] step {step:4d} loss={last:.4f} "
                f"vram={_vram_mb():.0f}MB",
                flush=True,
            )

    metrics = eval_tf_and_constrained(bridge, tokenizer, device, args, two_look)
    return {
        "kind": kind,
        "T": int(meta.get("T", args.T)),
        "requested_T": args.T,
        "two_look": two_look,
        "latent_steps": args.latent_steps if two_look else 1,
        "seed": seed,
        "final_loss": last,
        "seconds": time.time() - t0,
        "peak_vram_mb": _vram_mb(),
        "status": "ok",
        "history": hist,
        **metrics,
    }


def write_artifacts(rows, args, note: str) -> None:
    payload = {
        "task": "vlm_path_tf_cer",
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
        "min_len": args.min_len,
        "str_max_len": args.str_max_len,
        "probe_n": args.probe_n,
        "kinds": args.kinds,
        "seeds": args.seeds,
        "metrics": ["tf_acc", "exact_acc", "cer"],
        "decode": "digit-constrained",
        "table": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    lines = [
        "# VLM-path Phase 2: TF answer-span + digit-constrained CER",
        "",
        f"- Backend: `{note}`",
        f"- Train: frontend (+ VL-CoT if B_vlcot) + projector; **LLM frozen**",
        f"- Task: multi-digit 1px strings len {args.min_len}-{args.str_max_len}",
        f"- Profile: batch={args.batch} accum={args.grad_accum} steps={args.steps} "
        f"res={args.res} T={args.T} amp={args.amp}",
        f"- Eval n={args.probe_n}: **TF answer-token acc** + **digit-constrained** "
        f"CER/exact (not open free-gen as primary)",
        "",
        "| kind | T | two_look | tf_acc | exact | CER | loss | VRAM | status |",
        "|------|---|---------|--------|-------|-----|------|------|--------|",
    ]
    for r in rows:
        if r.get("status") != "ok":
            lines.append(
                f"| {r.get('kind')} | {r.get('T')} | — | — | — | — | — | — | "
                f"{r.get('error')} |"
            )
        else:
            lines.append(
                f"| {r['kind']} | {r['T']} | {r.get('two_look')} | "
                f"{r['tf_acc']:.3f} | {r['exact_acc']:.3f} | {r['cer']:.3f} | "
                f"{r['final_loss']:.3f} | {r.get('peak_vram_mb', 0):.0f} | ok |"
            )

    lines.extend(["", "## Relative A vs B", ""])
    ok = [r for r in rows if r.get("status") == "ok"]
    by = {r["kind"]: r for r in ok}
    a = by.get("A")
    b = by.get("B")
    bv = by.get("B_vlcot") or by.get("B_VLCOT")
    if a and b:
        lines.append(
            f"- TF: B={b['tf_acc']:.3f} vs A={a['tf_acc']:.3f} "
            f"(Δ={b['tf_acc']-a['tf_acc']:+.3f})"
        )
        lines.append(
            f"- constrained CER: B={b['cer']:.3f} vs A={a['cer']:.3f} "
            f"(Δ={b['cer']-a['cer']:+.3f}; lower better)"
        )
        lines.append(
            f"- constrained exact: B={b['exact_acc']:.3f} vs A={a['exact_acc']:.3f} "
            f"(Δ={b['exact_acc']-a['exact_acc']:+.3f})"
        )
        # Meaningful win thresholds (avoid noise-scale CER claims)
        b_wins_tf = b["tf_acc"] > a["tf_acc"] + 0.03
        b_wins_cer = b["cer"] < a["cer"] - 0.05
        b_wins_ex = b["exact_acc"] > a["exact_acc"] + 0.02
        if b_wins_tf or b_wins_cer or b_wins_ex:
            parts = []
            if b_wins_tf:
                parts.append("TF")
            if b_wins_cer:
                parts.append("CER")
            if b_wins_ex:
                parts.append("exact")
            lines.append(f"- **B beats A** on {', '.join(parts)}")
        else:
            lines.append(
                "- **No clear B>A** on TF / constrained CER / exact under frozen "
                "small LM (margins below noise thresholds)."
            )
    if a and bv:
        lines.append(
            f"- B_vlcot vs A: TF Δ={bv['tf_acc']-a['tf_acc']:+.3f}, "
            f"CER Δ={bv['cer']-a['cer']:+.3f}, exact Δ={bv['exact_acc']-a['exact_acc']:+.3f}"
        )
    stuck = ok and all(r["exact_acc"] < 0.05 for r in ok) and all(
        r["cer"] > 0.7 for r in ok
    )
    if stuck:
        lines.append(
            "- **Honest stuck:** under frozen small LM, constrained exact≈0 and "
            "CER remains high for all arms — Phase 3 (trainable LM / LoRA) required "
            "for product-positive OCR, not longer open free-gen."
        )
    lines.append("")
    lines.append(
        "Non-claim: open free-gen exact≈0 is not product OCR success."
    )
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
    ap.add_argument("--kinds", nargs="*", default=["A", "B", "B_vlcot"])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0])
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--projector", default="mlp")
    ap.add_argument("--latent_steps", type=int, default=2)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--pass1_w", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=3e-4)
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
    ap.add_argument("--out", default="results/published/vlm_path_tf_cer_table.json")
    ap.add_argument(
        "--conclusion", default="results/published/vlm_path_tf_cer_conclusion.md",
    )
    args = ap.parse_args()
    if args.no_amp:
        args.amp = False

    device = torch.device(args.device)
    print(f"cache={cache_root()} device={device} Phase2 TF+CER", flush=True)
    assert str(cache_root()).upper().startswith("D:")
    print(f"loading LLM prefer={args.prefer} …", flush=True)
    llm, tokenizer, d_llm, note = load_frozen_lm(prefer=args.prefer, device=str(device))
    print(f"backend={note} d_llm={d_llm}", flush=True)

    rows = []
    for seed in args.seeds:
        for kind in args.kinds:
            print(f"\n=== {kind} T={args.T} seed={seed} ===", flush=True)
            try:
                row = train_one(kind, args, llm, tokenizer, d_llm, device, seed)
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
                    "kind": kind, "T": args.T, "seed": seed,
                    "status": "error", "error": err,
                })
                print(f"  FAIL {err}", flush=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            except Exception as e:
                rows.append({
                    "kind": kind, "T": args.T, "seed": seed,
                    "status": "error", "error": str(e)[:220],
                })
                print(f"  FAIL {e}", flush=True)
            write_artifacts(rows, args, note)

    write_artifacts(rows, args, note)


if __name__ == "__main__":
    main()
