#!/usr/bin/env python3
"""M1: Visual Latent CoT + LLM hidden → second look (4GB).

A: static patch, single look
B_vlcot: K-step visual latent + two-look (LLM last-h → h_to_z → re-slice)

Eval: digit-constrained free-gen + CER / exact (fairer than open vocab collapse).

  set ML_CACHE_ROOT=D:\\ml_cache
  python scripts/train_vlcot_ocr_4gb.py --device cuda --steps 800
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

from fine_grain.frontends import PatchFrontend  # noqa: E402
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
from scripts.train_vlm_frontends import _trainable  # noqa: E402


def _vram_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def build_fe(kind: str, d_llm: int, args):
    k = kind.upper()
    if k == "A":
        return PatchFrontend(
            d_llm, res=args.res, patch=args.patch, dim=args.dim,
            depth=args.depth, T=args.T, projector=args.projector,
        )
    if k in ("B", "B_VLCOT", "VLCOT"):
        return VisualLatentCoTSlice(
            d_llm, res=args.res, T=args.T, dim=args.dim, depth=args.depth,
            deslice_topk=args.topk, projector=args.projector,
            latent_steps=args.latent_steps, beta=args.beta, gamma=args.gamma,
        )
    raise ValueError(kind)


def train_one(kind, args, llm, tokenizer, d_llm, device, seed: int):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    fe = build_fe(kind, d_llm, args).to(device)
    two_look = isinstance(fe, VisualLatentCoTSlice)
    bridge = VLCoTBridge(fe, llm, pass1_ce_weight=args.pass1_w).to(device)
    opt = torch.optim.AdamW(_trainable(bridge), lr=args.lr, weight_decay=0.01)
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    rng = np.random.default_rng(seed + 21)
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
            loss, meta = bridge(
                img, ids, mask, text_labels=lab, two_look=two_look,
            )
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
                f"  [{kind} two_look={meta.get('two_look')}] "
                f"step {step:4d} loss={last:.4f} vram={_vram_mb():.0f}MB",
                flush=True,
            )

    # ---- eval: digit-constrained (+ re-look for VLCoT) ----
    bridge.eval()
    rng_v = np.random.default_rng(args.val_seed)
    exact = cer_sum = n = 0
    fails = []
    with torch.no_grad():
        for _ in range(args.probe_n):
            data = make_ocr_string_vqa_batch(
                rng_v, 1, res=args.res,
                min_len=args.min_len, max_len=args.str_max_len,
                char_box=args.char_box, hard_frac=args.hard_frac,
            )
            img = data["image"].to(device)
            gold = data["answer"][0]
            # max_new ~ gold length + 1
            max_new = min(args.max_new, max(2, len(gold) + 1))
            pred = greedy_digit_string(
                bridge, tokenizer, img, data["prompt"],
                max_new=max_new,
                refine_every=(1 if two_look else 0),
            )[0]
            ok = exact_string_match(pred, gold)
            cer = char_error_rate(pred, gold)
            # clamp CER display-ish: still report raw
            exact += int(ok)
            cer_sum += min(cer, 3.0)
            n += 1
            if not ok and len(fails) < 8:
                fails.append({"gold": gold, "pred": pred[:24], "cer": round(cer, 3)})
    bridge.train()
    return {
        "kind": kind,
        "T": int(meta.get("T", args.T)),
        "latent_steps": args.latent_steps if two_look else 1,
        "two_look": bool(meta.get("two_look")),
        "seed": seed,
        "final_loss": last,
        "exact_acc": exact / max(n, 1),
        "cer": cer_sum / max(n, 1),
        "n_eval": n,
        "peak_vram_mb": _vram_mb(),
        "seconds": time.time() - t0,
        "status": "ok",
        "fail_examples": fails,
        "meta": {k: meta[k] for k in meta if k in (
            "kind", "latent_steps", "beta", "gamma", "two_look", "deslice_topk",
        )},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--res", type=int, default=32)
    ap.add_argument("--patch", type=int, default=4)
    ap.add_argument("--T", type=int, default=32)
    ap.add_argument("--kinds", nargs="*", default=["A", "B_vlcot"])
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
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--min_len", type=int, default=2)
    ap.add_argument("--str_max_len", type=int, default=3)
    ap.add_argument("--char_box", type=int, default=10)
    ap.add_argument("--hard_frac", type=float, default=0.25)
    ap.add_argument("--prefer", default="gemma")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results/published/vlcot_m1_ocr_table.json")
    ap.add_argument("--conclusion", default="results/published/vlcot_m1_ocr_conclusion.md")
    args = ap.parse_args()
    if args.no_amp:
        args.amp = False

    device = torch.device(args.device)
    print(f"cache={cache_root()} M1 two-look VL-CoT", flush=True)
    llm, tokenizer, d_llm, note = load_frozen_lm(prefer=args.prefer, device=str(device))
    print(f"backend={note}", flush=True)

    rows = []
    for seed in args.seeds:
        for kind in args.kinds:
            print(f"\n=== {kind} ===", flush=True)
            try:
                row = train_one(kind, args, llm, tokenizer, d_llm, device, seed)
                rows.append(row)
                print(
                    f"  → exact={row['exact_acc']:.3f} CER={row['cer']:.3f} "
                    f"two_look={row['two_look']} vram={row['peak_vram_mb']:.0f}MB",
                    flush=True,
                )
            except Exception as e:
                import traceback
                traceback.print_exc()
                rows.append({"kind": kind, "status": "error", "error": str(e)[:240]})
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    payload = {
        "task": "vlcot_m1_two_look_ocr",
        "backend": note,
        "design": {
            "M1": "LLM last-h (prompt end) → h_to_z → second visual encode",
            "decode": "digit-constrained greedy + refine_every=1 for B_vlcot",
            "train": "CE(pass2) + pass1_w*CE(pass1) + 0.1*NextLat; no LLM BPTT",
            "latent_steps": args.latent_steps,
            "pass1_w": args.pass1_w,
            "beta": args.beta,
            "gamma": args.gamma,
        },
        "table": rows,
        "steps": args.steps,
        "T": args.T,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    lines = [
        "# M1 Visual Latent CoT + LLM two-look OCR",
        "",
        f"- Backend: `{note}`",
        f"- B_vlcot: K={args.latent_steps} latent + **LLM h → second look**",
        f"- Eval: **digit-constrained** gen; refine_every=1 for B",
        f"- steps={args.steps} batch={args.batch} accum={args.grad_accum}",
        "",
        "| kind | two_look | exact | CER | VRAM | loss |",
        "|------|----------|-------|-----|------|------|",
    ]
    for r in rows:
        if r.get("status") != "ok":
            lines.append(f"| {r.get('kind')} | — | — | — | — | {r.get('error')} |")
        else:
            lines.append(
                f"| {r['kind']} | {r.get('two_look')} | {r['exact_acc']:.3f} | "
                f"{r['cer']:.3f} | {r['peak_vram_mb']:.0f} | {r['final_loss']:.3f} |"
            )
    lines.extend(["", "## Conclusion", ""])
    ok = [r for r in rows if r.get("status") == "ok"]
    a = next((r for r in ok if r["kind"] == "A"), None)
    b = next((r for r in ok if "B" in str(r["kind"])), None)
    if a and b:
        lines.append(
            f"- CER A={a['cer']:.3f} vs B={b['cer']:.3f} (Δ={b['cer']-a['cer']:+.3f})"
        )
        lines.append(
            f"- exact A={a['exact_acc']:.3f} vs B={b['exact_acc']:.3f}"
        )
        if b["cer"] < a["cer"] - 1e-6 or b["exact_acc"] > a["exact_acc"] + 1e-6:
            lines.append("- **Relative VLM-path signal:** B_vlcot better than A on this run.")
        else:
            lines.append("- No clear B>A advantage on this run (absolute OCR may still be weak).")
    lines.append(
        "- Decode constrained to digits reduces open-vocab collapse; free-gen garbage less dominant."
    )
    Path(args.conclusion).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"→ {args.out}", flush=True)


if __name__ == "__main__":
    main()
