#!/usr/bin/env python3
"""Train Transolver3 native multimodal VLM-OCR (vision–language co-evolution).

Product path: full-res field X + shared U + H co-evolve; LLM interface tokens only.
Compares A (static patch tokens) vs B_xmodal (Transolver3).

  set ML_CACHE_ROOT=D:\\ml_cache
  python scripts/train_transolver3_ocr.py --steps 400 --amp --kinds A B_xmodal
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

from fine_grain.llm_backend import cache_root, load_frozen_lm  # noqa: E402
from fine_grain.transolver3_vlm import (  # noqa: E402
    MoTOCRBridge,
    Transolver3OCRBridge,
    build_ocr_bridge,
)
from fine_grain.vlm_data import (  # noqa: E402
    answer_only_labels,
    char_error_rate,
    exact_string_match,
    make_ocr_string_vqa_batch,
)
from fine_grain.visual_latent_cot import greedy_digit_string  # noqa: E402
from scripts.train_vlm_frontends import (  # noqa: E402
    _trainable,
    bridge_tf_logits,
    tf_answer_span_accuracy,
)


def _vram_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


@torch.no_grad()
def _greedy_digit_mot(bridge: MoTOCRBridge, tokenizer, images, prompts, max_new: int = 6):
    """Digit-constrained gen after one MoT pass on the prompt (joint slice‖text)."""
    from fine_grain.vlm_data import tokenize_captions

    device = images.device
    dtype = next(bridge.llm.parameters()).dtype
    allowed = set()
    for d in "0123456789":
        for variant in (d, " " + d):
            ids_ = tokenizer.encode(variant, add_special_tokens=False)
            if ids_:
                allowed.add(int(ids_[0]))
    allowed = sorted(allowed) if allowed else None

    ids, mask = tokenize_captions(tokenizer, list(prompts), max_length=48)
    ids, mask = ids.to(device), mask.to(device)
    emb = bridge.llm.get_input_embeddings()(ids)
    _, text_llm, slices, _ = bridge.frontend.forward_mot(
        images, emb.float(), text_mask=mask,
    )
    v = bridge.frontend.proj(slices).to(dtype=dtype)
    text_llm = text_llm.to(dtype=dtype)
    inputs = torch.cat([v, text_llm], dim=1)
    B, Tvis = v.shape[0], v.shape[1]
    attn = torch.cat([
        torch.ones(B, Tvis, device=device, dtype=mask.dtype), mask,
    ], dim=1)
    gen = ids
    for _ in range(max_new):
        out = bridge.llm(inputs_embeds=inputs, attention_mask=attn, use_cache=False)
        logits = out.logits[:, -1, :]
        if allowed is not None:
            m = torch.full_like(logits, -1e9)
            idx = torch.tensor(allowed, device=device, dtype=torch.long)
            m[:, idx] = logits[:, idx]
            logits = m
        next_id = logits.argmax(dim=-1, keepdim=True)
        gen = torch.cat([gen, next_id], dim=1)
        next_emb = bridge.llm.get_input_embeddings()(next_id).to(dtype=dtype)
        inputs = torch.cat([inputs, next_emb], dim=1)
        attn = torch.cat([
            attn, torch.ones(B, 1, device=device, dtype=attn.dtype),
        ], dim=1)
    texts = []
    plens = mask.sum(dim=1).tolist()
    for i in range(B):
        pl = int(plens[i])
        texts.append(tokenizer.decode(gen[i, pl:].tolist(), skip_special_tokens=True))
    return texts


@torch.no_grad()
def eval_tf_cer(bridge, tokenizer, device, args) -> dict:
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

        # TF via real bridge path
        dtype = next(bridge.llm.parameters()).dtype
        if isinstance(bridge, Transolver3OCRBridge):
            h_seed = bridge._prompt_seed_h(ids, mask, lab)
            tok, h, _ = bridge.evolve_field(img, h_seed=h_seed)
            v = bridge._pack_interface(tok, h).to(dtype=dtype)
            emb = bridge.llm.get_input_embeddings()(ids)
            inputs = torch.cat([v, emb], dim=1)
            Tvis = v.shape[1]
            B = ids.shape[0]
            attn = torch.cat([
                torch.ones(B, Tvis, device=device, dtype=mask.dtype), mask,
            ], dim=1)
            out = bridge.llm(inputs_embeds=inputs, attention_mask=attn, use_cache=False)
            _, n_hit, n_ans = tf_answer_span_accuracy(out.logits, Tvis, ids, lab)
        elif isinstance(bridge, MoTOCRBridge):
            emb = bridge.llm.get_input_embeddings()(ids)
            _, text_llm, slices, _ = bridge.frontend.forward_mot(
                img, emb.float(), text_mask=mask,
            )
            v = bridge.frontend.proj(slices).to(dtype=dtype)
            text_llm = text_llm.to(dtype=dtype)
            inputs = torch.cat([v, text_llm], dim=1)
            Tvis = v.shape[1]
            B = ids.shape[0]
            attn = torch.cat([
                torch.ones(B, Tvis, device=device, dtype=mask.dtype), mask,
            ], dim=1)
            out = bridge.llm(inputs_embeds=inputs, attention_mask=attn, use_cache=False)
            _, n_hit, n_ans = tf_answer_span_accuracy(out.logits, Tvis, ids, lab)
        else:
            logits, Tvis = bridge_tf_logits(bridge, img, ids, mask)
            _, n_hit, n_ans = tf_answer_span_accuracy(logits, Tvis, ids, lab)
        tf_hit += n_hit
        tf_tot += n_ans

        gold = data["answer"][0]
        max_new = min(args.max_new, max(2, len(gold) + 1))
        if isinstance(bridge, MoTOCRBridge):
            pred = _greedy_digit_mot(
                bridge, tokenizer, img, data["prompt"], max_new=max_new,
            )[0]
        else:
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


def train_one(kind: str, args, llm, tokenizer, d_llm, device, seed: int) -> dict:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    freeze = bool(getattr(args, "freeze_llm", True))
    last_n = int(getattr(args, "llm_last_n", 0) or 0)
    use_lora = bool(getattr(args, "joint_lora", False)) and not freeze
    lang_cfg = {"mode": "frozen", "n_trainable": 0}

    # Apply LoRA on LLM *before* building bridge so bridge.llm is PeftModel
    if use_lora:
        from fine_grain.lora_llm import apply_lora, enable_lora_grads, peft_available
        if not peft_available():
            raise RuntimeError("joint LoRA needs peft; pip install peft or use scripts/pip_get.py peft")
        llm, lora_meta = apply_lora(
            llm,
            r=int(getattr(args, "lora_r", 8)),
            alpha=int(getattr(args, "lora_alpha", 16)),
            dropout=0.05,
            last_n_layers=max(1, last_n or 4),
        )
        enable_lora_grads(llm)
        lang_cfg = {
            "mode": f"lora_r{lora_meta['lora_r']}_layers_{lora_meta['layer_range']}",
            "n_trainable": lora_meta["trainable_params"],
            **lora_meta,
        }
        print(f"  LoRA joint: {lang_cfg['mode']} params={lang_cfg['n_trainable']}", flush=True)

    bridge = build_ocr_bridge(kind, d_llm, llm, args).to(device)
    if use_lora:
        # base frozen; only LoRA + vision train
        from fine_grain.lora_llm import enable_lora_grads
        for p in bridge.llm.parameters():
            p.requires_grad_(False)
        enable_lora_grads(bridge.llm)
        if hasattr(bridge, "freeze_llm"):
            # LM backbone frozen; adapters train — treat as joint language side
            bridge.freeze_llm = True  # no full LM graph surprises in two-look
    else:
        from fine_grain.transolver3_vlm import configure_llm_trainability
        lang_cfg = configure_llm_trainability(
            bridge.llm, freeze_llm=freeze, last_n_layers=0 if freeze else last_n,
        )
        if hasattr(bridge, "freeze_llm"):
            bridge.freeze_llm = freeze

    if (not freeze or use_lora) and hasattr(bridge.llm, "gradient_checkpointing_enable"):
        try:
            bridge.llm.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            print("  gradient_checkpointing ON", flush=True)
        except TypeError:
            try:
                bridge.llm.gradient_checkpointing_enable()
                print("  gradient_checkpointing ON", flush=True)
            except Exception as e:
                print(f"  gradient_checkpointing skip: {e}", flush=True)
        except Exception as e:
            print(f"  gradient_checkpointing skip: {e}", flush=True)
    # Dual LR: vision/interface vs language
    vis, lang = [], []
    for name, p in bridge.named_parameters():
        if not p.requires_grad:
            continue
        if name == "llm" or name.startswith("llm."):
            lang.append(p)
        else:
            vis.append(p)
    n_vis = sum(p.numel() for p in vis)
    n_lang = sum(p.numel() for p in lang)
    print(
        f"  trainable vis={n_vis} lang={n_lang} freeze_llm={freeze} "
        f"lang_mode={lang_cfg.get('mode')} lr_vis={args.lr} lr_lang={args.llm_lr}",
        flush=True,
    )
    if n_lang > 20_000_000:
        print(
            "  WARN: large language trainable set — AdamW may spill to Windows "
            "shared RAM on 4GB. Use --joint (LoRA) not --joint_full / last-n full weights.",
            flush=True,
        )
    param_groups = [{"params": vis, "lr": args.lr}]
    if lang:
        param_groups.append({"params": lang, "lr": args.llm_lr})
    opt = torch.optim.AdamW(param_groups, weight_decay=0.01)
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    rng = np.random.default_rng(seed + 51)
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
                f"Tif={meta.get('T_interface', meta.get('T'))} "
                f"two_look={meta.get('two_look')} vram={_vram_mb():.0f}MB",
                flush=True,
            )
        # joint: free fragmentation so Windows "shared memory" spill is rarer
        if not freeze and device.type == "cuda" and step % 10 == 0:
            torch.cuda.empty_cache()

    metrics = eval_tf_cer(bridge, tokenizer, device, args)
    return {
        "kind": kind,
        "transolver3": kind.upper() not in ("A", "PATCH"),
        "mot": "MOT" in kind.upper(),
        "freeze_llm": bool(getattr(args, "freeze_llm", True)),
        "joint": not bool(getattr(args, "freeze_llm", True)),
        "llm_last_n": int(getattr(args, "llm_last_n", 0) or 0),
        "lang_mode": lang_cfg.get("mode"),
        "T": int(meta.get("T", args.T)),
        "n_layers": args.n_layers if kind.upper() not in ("A", "PATCH") else 0,
        "coevolve_rounds": args.coevolve_rounds if kind.upper() not in ("A", "PATCH") else 1,
        "seed": seed,
        "final_loss": last,
        "seconds": time.time() - t0,
        "peak_vram_mb": _vram_mb(),
        "n_trainable_vis": n_vis,
        "n_trainable_lang": n_lang,
        "status": "ok",
        "history": hist,
        **metrics,
    }


def write_artifacts(rows, args, note: str) -> None:
    payload = {
        "task": "transolver3_native_multimodal_ocr",
        "backend": note,
        "thesis": "vision-language co-evolve; full-res X; U workspace; not slice tokens",
        "cache_root": str(cache_root()),
        "res": args.res,
        "steps": args.steps,
        "batch": args.batch,
        "grad_accum": args.grad_accum,
        "T": args.T,
        "n_layers": args.n_layers,
        "coevolve_rounds": args.coevolve_rounds,
        "freeze_llm": bool(getattr(args, "freeze_llm", True)),
        "joint": not bool(getattr(args, "freeze_llm", True)),
        "lr": args.lr,
        "llm_lr": args.llm_lr,
        "kinds": args.kinds,
        "table": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    lines = [
        "# Transolver3 native multimodal VLM-OCR (vision–language co-evolution)",
        "",
        f"- Backend: `{note}`",
        f"- Thesis: full-res field X + shared U; **not** permanent vision slice tokens",
        f"- Profile: steps={args.steps} batch={args.batch} accum={args.grad_accum} "
        f"res={args.res} T={args.T} n_layers={args.n_layers} "
        f"coevolve_rounds={args.coevolve_rounds}",
        f"- **joint train** freeze_llm={getattr(args, 'freeze_llm', True)} "
        f"lr_vis={args.lr} lr_lang={args.llm_lr}",
        f"- Metrics: TF answer-span + digit-constrained CER/exact (n={args.probe_n})",
        "",
        "| kind | joint | mot | tf_acc | exact | CER | loss | VRAM | status |",
        "|------|-------|-----|--------|-------|-----|------|------|--------|",
    ]
    for r in rows:
        if r.get("status") != "ok":
            lines.append(
                f"| {r.get('kind')} | — | — | — | — | — | — | — | {r.get('error')} |"
            )
        else:
            lines.append(
                f"| {r['kind']} | {r.get('joint')} | {r.get('mot')} | {r['tf_acc']:.3f} | "
                f"{r['exact_acc']:.3f} | {r['cer']:.3f} | {r['final_loss']:.3f} | "
                f"{r.get('peak_vram_mb', 0):.0f} | ok |"
            )
    lines.extend(["", "## Comparison (vs A)", ""])
    ok = [r for r in rows if r.get("status") == "ok"]
    by = {r["kind"]: r for r in ok}
    a = by.get("A")
    if a:
        for k, r in by.items():
            if k == "A":
                continue
            lines.append(
                f"- **{k}** vs A: TF Δ={r['tf_acc']-a['tf_acc']:+.3f}, "
                f"CER Δ={r['cer']-a['cer']:+.3f}, exact Δ={r['exact_acc']-a['exact_acc']:+.3f}"
            )
        mot = by.get("B_mot")
        hx = by.get("B_xmodal")
        if mot and hx:
            lines.append("")
            lines.append("## MoT (joint self-attn) vs H-path (B_xmodal)")
            lines.append(
                f"- TF: MoT={mot['tf_acc']:.3f} vs H-path={hx['tf_acc']:.3f} "
                f"(Δ={mot['tf_acc']-hx['tf_acc']:+.3f})"
            )
            lines.append(
                f"- CER: MoT={mot['cer']:.3f} vs H-path={hx['cer']:.3f} "
                f"(Δ={mot['cer']-hx['cer']:+.3f})"
            )
            if mot["cer"] > hx["cer"] + 0.05 and mot["tf_acc"] < hx["tf_acc"] - 0.02:
                lines.append(
                    "- **Rollback hint:** MoT underperforms H-path on this budget; "
                    "keep B_xmodal as default until MoT is tuned."
                )
            elif mot["cer"] < hx["cer"] - 0.02 or mot["tf_acc"] > hx["tf_acc"] + 0.02:
                lines.append("- **MoT preferred** over continuous-H path on this run.")
            else:
                lines.append("- MoT ≈ H-path under short budget (architecture comparison).")
    lines.append("")
    lines.append(
        "MoT = slices‖text in **one** self-attn + deslice to point field. "
        "Non-claim: open free-gen alone is not product proof."
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
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--coevolve_rounds", type=int, default=2)
    ap.add_argument("--pass1_w", type=float, default=0.25)
    ap.add_argument("--use_h_token", action="store_true", default=True)
    ap.add_argument("--no_h_token", action="store_true")
    ap.add_argument(
        "--kinds", nargs="*", default=["A", "B_xmodal", "B_mot"],
        help="A=patch; B_xmodal=H-path; B_mot=joint self-attn MoT (preferred co-evolve)",
    )
    ap.add_argument(
        "--joint", action="store_true",
        help="Joint vision+language via LoRA (4GB-safe). Not full-weight unfreeze.",
    )
    ap.add_argument(
        "--joint_full", action="store_true",
        help="Unfreeze full LLM weights (needs >>4GB; spills to shared RAM on 1650)",
    )
    ap.add_argument(
        "--joint_lastn", action="store_true",
        help="Joint by unfreezing last --llm_last_n full layers (heavier than LoRA)",
    )
    ap.add_argument(
        "--freeze_llm", action="store_true", default=None,
        help="Force freeze LLM (default True unless --joint*)",
    )
    ap.add_argument(
        "--llm_last_n", type=int, default=4,
        help="Last N layers for LoRA or --joint_lastn",
    )
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument(
        "--llm_lr", type=float, default=1e-4,
        help="LR for language/LoRA params when joint",
    )
    ap.add_argument(
        "--llm_dtype", default="auto",
        choices=["auto", "fp32", "fp16"],
        help="LLM weight dtype. joint+auto → fp16 on CUDA",
    )
    ap.add_argument("--seeds", type=int, nargs="*", default=[0])
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--projector", default="mlp")
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
    ap.add_argument("--out", default="results/published/transolver3_ocr_table.json")
    ap.add_argument(
        "--conclusion", default="results/published/transolver3_ocr_conclusion.md",
    )
    args = ap.parse_args()
    if args.no_amp:
        args.amp = False
    if args.no_h_token:
        args.use_h_token = False
    # Joint modes:
    #   --joint       → LoRA (default 4GB-safe "language not frozen")
    #   --joint_lastn → full weights last N layers
    #   --joint_full  → all LLM weights (will OOM/shared-RAM on 4GB)
    args.joint_lora = False
    if args.joint_full:
        args.freeze_llm = False
        args.llm_last_n = 0
        args.joint_lora = False
        print("  WARN: --joint_full almost always spills to shared RAM on 4GB", flush=True)
    elif args.joint_lastn:
        args.freeze_llm = False
        args.joint_lora = False
        if args.llm_last_n <= 0:
            args.llm_last_n = 2
    elif args.joint:
        args.freeze_llm = False
        args.joint_lora = True  # LoRA is the real joint path on 4GB
        if args.llm_last_n <= 0:
            args.llm_last_n = 4
    elif args.freeze_llm is None:
        args.freeze_llm = True

    if (args.joint or args.joint_lastn or args.joint_full) and args.coevolve_rounds > 1:
        print(f"  joint: coevolve_rounds {args.coevolve_rounds} → 1", flush=True)
        args.coevolve_rounds = 1

    device = torch.device(args.device)
    load_dtype = None
    if args.llm_dtype == "fp16":
        load_dtype = torch.float16
    elif args.llm_dtype == "fp32":
        load_dtype = torch.float32
    elif not args.freeze_llm and device.type == "cuda":
        # LoRA joint: fp32 backbone often more stable; weights frozen so OK
        # lastn/full: prefer fp16
        if args.joint_lora:
            load_dtype = torch.float32
        else:
            load_dtype = torch.float16
            print("  joint auto: LLM fp16", flush=True)

    print(
        f"cache={cache_root()} Transolver3 OCR joint_lora={args.joint_lora} "
        f"freeze_llm={args.freeze_llm} llm_last_n={args.llm_last_n}",
        flush=True,
    )
    assert str(cache_root()).upper().startswith("D:")
    llm, tokenizer, d_llm, note = load_frozen_lm(
        prefer=args.prefer, device=str(device), dtype=load_dtype,
    )
    print(f"backend={note} d_llm={d_llm}", flush=True)

    rows = []
    for seed in args.seeds:
        for kind in args.kinds:
            print(f"\n=== {kind} seed={seed} ===", flush=True)
            try:
                # fresh LLM weights per cell to avoid any state leak
                if seed != args.seeds[0] or kind != args.kinds[0]:
                    llm, tokenizer, d_llm, note = load_frozen_lm(
                        prefer=args.prefer, device=str(device), dtype=load_dtype,
                    )
                row = train_one(kind, args, llm, tokenizer, d_llm, device, seed)
                rows.append(row)
                print(
                    f"  → tf={row['tf_acc']:.3f} exact={row['exact_acc']:.3f} "
                    f"CER={row['cer']:.3f} vram={row['peak_vram_mb']:.0f}MB",
                    flush=True,
                )
            except RuntimeError as e:
                msg = str(e)
                err = "OOM" if "out of memory" in msg.lower() else msg[:240]
                rows.append({"kind": kind, "seed": seed, "status": "error", "error": err})
                print(f"  FAIL {err}", flush=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            except Exception as e:
                rows.append({
                    "kind": kind, "seed": seed, "status": "error", "error": str(e)[:240],
                })
                print(f"  FAIL {e}", flush=True)
            write_artifacts(rows, args, note)
            if device.type == "cuda":
                torch.cuda.empty_cache()

    write_artifacts(rows, args, note)


if __name__ == "__main__":
    main()
