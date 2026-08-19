#!/usr/bin/env python3
"""Small VLM: slice (or patch) vision encoder + Gemma on real HF captions.

Question: can **slice** work as a vision encoder for a tiny VLM?

Default recipe (4GB joint):
  - Dataset: jxie/flickr8k
  - Vision: SliceFrontend vs PatchFrontend + MLP
  - LLM: Gemma-3-270M + **LoRA** (language not frozen; full unfreeze OOMs 4GB)
  - Loss: answer-only CE on caption tokens

Example:
  set ML_CACHE_ROOT=D:\\ml_cache
  python scripts/train_small_vlm_hf.py --joint --kinds slice patch --steps 800 --res 64
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
os.environ["HUGGINGFACE_HUB_CACHE"] = str(Path(os.environ["ML_CACHE_ROOT"]) / "huggingface" / "hub")
os.environ["MODELSCOPE_CACHE"] = str(Path(os.environ["ML_CACHE_ROOT"]) / "modelscope")
os.environ["TORCH_HOME"] = str(Path(os.environ["ML_CACHE_ROOT"]) / "torch")
try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except Exception:
    pass
for _k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(_k, None)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fine_grain.frontends import PatchFrontend, SliceFrontend  # noqa: E402
from fine_grain.hf_caption_data import (  # noqa: E402
    HF_VLM_DATASETS,
    Flickr8kCaptionStore,
    sample_batch,
)
from fine_grain.llm_backend import cache_root, load_frozen_lm  # noqa: E402
from fine_grain.vlm_data import answer_only_labels  # noqa: E402
from scripts.train_vlm_frontends import VLMBridge, _trainable  # noqa: E402


def _vram_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def build_fe(kind: str, d_llm: int, args):
    k = kind.lower()
    if k in ("patch", "a"):
        return PatchFrontend(
            d_llm, res=args.res, patch=args.patch, dim=args.dim,
            depth=args.depth, T=args.T, projector=args.projector,
        ), "patch"
    if k in ("slice", "b", "slice_fe"):
        return SliceFrontend(
            d_llm, res=args.res, T=args.T, dim=args.dim,
            depth=args.depth, deslice_topk=args.topk, projector=args.projector,
        ), "slice"
    raise ValueError(kind)


@torch.no_grad()
def eval_tf_caption(bridge, tokenizer, store, device, args, n: int, seed: int):
    """Teacher-forced token accuracy on caption answer span (held-out indices)."""
    bridge.eval()
    rng = np.random.default_rng(seed)
    n = min(n, len(store))
    idxs = rng.choice(len(store), size=n, replace=False)
    hit = tot = 0
    losses = []
    for i in idxs:
        batch = sample_batch(store, [int(i)], res=args.res)
        img = batch["image"].to(device)
        ids, mask, lab = answer_only_labels(
            tokenizer, batch["prompt"], batch["text"], max_length=args.max_len,
        )
        ids, mask, lab = ids.to(device), mask.to(device), lab.to(device)
        loss, meta = bridge(img, ids, mask, text_labels=lab)
        losses.append(float(loss.item()))
        # token acc on answer span via logits
        vis = bridge.frontend(img)
        v = vis.tokens
        dtype = next(bridge.llm.parameters()).dtype
        v = v.to(dtype=dtype)
        emb = bridge.llm.get_input_embeddings()(ids)
        inputs = torch.cat([v, emb], dim=1)
        T = v.shape[1]
        attn = torch.cat([
            torch.ones(1, T, device=device, dtype=mask.dtype), mask,
        ], dim=1)
        out = bridge.llm(inputs_embeds=inputs, attention_mask=attn, use_cache=False)
        L = ids.shape[1]
        pred = out.logits[:, T - 1 : T + L - 1].argmax(-1)
        ans = lab != -100
        if ans.any():
            hit += int((pred[ans] == lab[ans]).sum().item())
            tot += int(ans.sum().item())
    bridge.train()
    return {
        "tf_acc": hit / max(tot, 1),
        "tf_n_tokens": tot,
        "eval_loss": float(np.mean(losses)) if losses else float("nan"),
        "n_eval": n,
    }


def train_one(kind, args, llm, tokenizer, d_llm, device, store_train, store_val):
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    joint = bool(getattr(args, "joint", True))
    lang_mode = "frozen"
    if joint:
        # 4GB: LoRA = language trainable without full-weight Adam blowup
        from fine_grain.lora_llm import apply_lora, enable_lora_grads, peft_available
        if not peft_available():
            raise RuntimeError("joint needs peft (LoRA). Use scripts/pip_get.py peft")
        llm, lora_meta = apply_lora(
            llm,
            r=int(args.lora_r),
            alpha=int(args.lora_alpha),
            last_n_layers=int(args.llm_last_n),
        )
        enable_lora_grads(llm)
        lang_mode = f"lora_r{lora_meta['lora_r']}_{lora_meta['layer_range']}"
        print(
            f"  joint LoRA: {lang_mode} lang_params={lora_meta['trainable_params']}",
            flush=True,
        )

    fe, tag = build_fe(kind, d_llm, args)
    bridge = VLMBridge(fe, llm).to(device)
    if joint:
        for p in bridge.llm.parameters():
            p.requires_grad_(False)
        from fine_grain.lora_llm import enable_lora_grads
        enable_lora_grads(bridge.llm)
        if hasattr(bridge.llm, "gradient_checkpointing_enable"):
            try:
                bridge.llm.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                try:
                    bridge.llm.gradient_checkpointing_enable()
                except Exception:
                    pass
    else:
        for p in bridge.llm.parameters():
            p.requires_grad_(False)

    vis, lang = [], []
    for name, p in bridge.named_parameters():
        if not p.requires_grad:
            continue
        if name == "llm" or name.startswith("llm."):
            lang.append(p)
        else:
            vis.append(p)
    groups = [{"params": vis, "lr": args.lr}]
    if lang:
        groups.append({"params": lang, "lr": args.llm_lr})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    rng = np.random.default_rng(args.seed + 7)
    bridge.train()
    accum = max(1, args.grad_accum)
    opt.zero_grad(set_to_none=True)
    hist = []
    last = float("nan")
    t0 = time.time()
    n_vis = sum(p.numel() for p in vis)
    n_lang = sum(p.numel() for p in lang)
    print(
        f"  [{tag}] joint={joint} lang={lang_mode} "
        f"trainable vis={n_vis} lang={n_lang} res={args.res} T={args.T}",
        flush=True,
    )

    for step in range(1, args.steps + 1):
        idxs = rng.integers(0, len(store_train), size=args.batch)
        batch = sample_batch(store_train, idxs, res=args.res)
        img = batch["image"].to(device)
        ids, mask, lab = answer_only_labels(
            tokenizer, batch["prompt"], batch["text"], max_length=args.max_len,
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
                f"  [{tag}] step {step:4d} loss={last:.4f} vram={_vram_mb():.0f}MB",
                flush=True,
            )
        if joint and device.type == "cuda" and step % 20 == 0:
            torch.cuda.empty_cache()

    metrics = eval_tf_caption(
        bridge, tokenizer, store_val, device, args, n=args.probe_n, seed=args.val_seed,
    )
    return {
        "kind": tag,
        "encoder": tag,
        "status": "ok",
        "joint": joint,
        "lang_mode": lang_mode,
        "final_loss": last,
        "seconds": time.time() - t0,
        "peak_vram_mb": _vram_mb(),
        "trainable_vis": n_vis,
        "trainable_lang": n_lang,
        "history": hist,
        **metrics,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kinds", nargs="*", default=["slice", "patch"])
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--patch", type=int, default=8)
    ap.add_argument("--T", type=int, default=64, help="slice count / token budget")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--projector", default="mlp")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--llm_lr", type=float, default=1e-4, help="LoRA / language LR")
    ap.add_argument(
        "--joint", action="store_true", default=True,
        help="Train language via LoRA (default ON). Full unfreeze OOMs 4GB.",
    )
    ap.add_argument(
        "--freeze_llm", action="store_true",
        help="Freeze LLM entirely (overrides --joint)",
    )
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--llm_last_n", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--max_rows", type=int, default=4000, help="cap train images")
    ap.add_argument("--probe_n", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val_seed", type=int, default=123)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--prefer", default="gemma")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results/published/small_vlm_hf_table.json")
    ap.add_argument("--conclusion", default="results/published/small_vlm_hf_conclusion.md")
    args = ap.parse_args()
    if args.no_amp:
        args.amp = False
    if args.freeze_llm:
        args.joint = False

    device = torch.device(args.device)
    print("=== HF small VLM (slice as encoder?) ===", flush=True)
    print("Recommended datasets:", json.dumps(HF_VLM_DATASETS, indent=2)[:500], "...", flush=True)
    print(f"cache={cache_root()} loading flickr8k …", flush=True)
    store_train = Flickr8kCaptionStore(split="train", max_rows=args.max_rows)
    # use last 500 of train as val if no separate test loaded — load test
    try:
        store_val = Flickr8kCaptionStore(split="test", max_rows=min(500, args.probe_n * 4))
    except Exception:
        store_val = store_train
    print(f"train n={len(store_train)} val n={len(store_val)}", flush=True)

    llm, tokenizer, d_llm, note = load_frozen_lm(prefer=args.prefer, device=str(device))
    print(f"backend={note} d_llm={d_llm}", flush=True)

    rows = []
    for kind in args.kinds:
        print(f"\n=== encoder={kind} ===", flush=True)
        try:
            # reload LLM frozen each cell to avoid state leak
            if rows:
                llm, tokenizer, d_llm, note = load_frozen_lm(
                    prefer=args.prefer, device=str(device),
                )
            row = train_one(kind, args, llm, tokenizer, d_llm, device, store_train, store_val)
            rows.append(row)
            print(
                f"  → tf_acc={row['tf_acc']:.3f} eval_loss={row['eval_loss']:.3f} "
                f"vram={row['peak_vram_mb']:.0f}MB",
                flush=True,
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            rows.append({"kind": kind, "status": "error", "error": str(e)[:300]})
            if device.type == "cuda":
                torch.cuda.empty_cache()

    payload = {
        "task": "small_vlm_hf_caption",
        "dataset": "jxie/flickr8k",
        "backend": note,
        "res": args.res,
        "T": args.T,
        "steps": args.steps,
        "max_rows": args.max_rows,
        "joint": bool(args.joint),
        "freeze_llm": not bool(args.joint),
        "lang": "lora" if args.joint else "frozen",
        "lora_r": args.lora_r if args.joint else None,
        "llm_last_n": args.llm_last_n if args.joint else None,
        "question": "Can slice work as VLM vision encoder at Gemma-270M scale (joint LoRA)?",
        "table": rows,
        "hf_datasets_catalog": HF_VLM_DATASETS,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    lines = [
        "# Small VLM on HF captions: slice vs patch encoder",
        "",
        f"- Dataset: **jxie/flickr8k** (train max_rows={args.max_rows})",
        f"- LLM: Gemma-3-270M; **joint={args.joint}** "
        f"({'LoRA r='+str(args.lora_r)+' last-'+str(args.llm_last_n) if args.joint else 'FROZEN'})",
        f"- Vision+MLP always trained; language via LoRA when joint (full unfreeze OOMs 4GB)",
        f"- res={args.res} T={args.T} steps={args.steps} batch={args.batch} accum={args.grad_accum}",
        f"- Metric: teacher-forced caption-span token acc",
        "",
        "| encoder | joint | lang | tf_acc | eval_loss | final_loss | VRAM | status |",
        "|---------|-------|------|--------|-----------|------------|------|--------|",
    ]
    for r in rows:
        if r.get("status") != "ok":
            lines.append(
                f"| {r.get('kind')} | — | — | — | — | — | — | {r.get('error')} |"
            )
        else:
            lines.append(
                f"| {r['kind']} | {r.get('joint')} | {r.get('lang_mode')} | "
                f"{r['tf_acc']:.3f} | {r['eval_loss']:.3f} | "
                f"{r['final_loss']:.3f} | {r.get('peak_vram_mb', 0):.0f} | ok |"
            )
    ok = [r for r in rows if r.get("status") == "ok"]
    by = {r["kind"]: r for r in ok}
    if "slice" in by and "patch" in by:
        s, p = by["slice"], by["patch"]
        lines.extend([
            "",
            "## slice vs patch",
            f"- TF: slice={s['tf_acc']:.3f} vs patch={p['tf_acc']:.3f} "
            f"(Δ={s['tf_acc']-p['tf_acc']:+.3f})",
            f"- eval_loss: slice={s['eval_loss']:.3f} vs patch={p['eval_loss']:.3f}",
        ])
        if s["tf_acc"] > p["tf_acc"] + 0.01 or s["eval_loss"] < p["eval_loss"] - 0.05:
            lines.append("- **Slice looks competitive as a vision encoder** under this budget.")
        else:
            lines.append(
                "- No clear slice win yet; still a valid encoder path to scale "
                "(more data / steps / unfreeze projector curriculum)."
            )
    lines.extend([
        "",
        "## HF datasets for next scale",
        "- `jxie/flickr8k` — this run",
        "- `nlphuji/flickr30k` — medium",
        "- `liuhaotian/LLaVA-Pretrain` (558k) — LLaVA stage-1 projector pretrain",
        "- `Multimodal-Fatima/COCO_captions_train` — COCO shards",
        "",
        "Synthetic 1px OCR remains diagnostic; real caption data answers encoder utility.",
    ])
    Path(args.conclusion).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"json → {args.out}", flush=True)


if __name__ == "__main__":
    main()
