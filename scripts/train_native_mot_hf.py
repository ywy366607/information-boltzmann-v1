#!/usr/bin/env python3
"""Train native MoT stack (spec) on HF captions — small VLM pretrain path.

Architecture (per layer):
  X ─SliceRead─► S ─► Visual expert ─┐
                      shared K/V space ├─► S', H'
  H ───────────────► Language expert ─┘
  X' = LocalVisual(X + Deslice(S'))

Validation eligibility (documented): data tokens ≥ 1 × model parameters.

4GB recipe: d_x=128, d=256 or 512, n_layers=2–4, freeze LLM + train stack+proj
or LoRA joint.

Example:
  set ML_CACHE_ROOT=D:\\ml_cache
  python scripts/train_native_mot_hf.py --steps 400 --res 32 --d 256 --n_layers 2 --joint
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
import torch.nn as nn

os.environ.setdefault("ML_CACHE_ROOT", r"D:\ml_cache")
os.environ["HF_HOME"] = str(Path(os.environ["ML_CACHE_ROOT"]) / "huggingface")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(Path(os.environ["ML_CACHE_ROOT"]) / "huggingface" / "hub")
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

from fine_grain.hf_caption_data import Flickr8kCaptionStore, sample_batch  # noqa: E402
from fine_grain.llm_backend import cache_root, load_frozen_lm  # noqa: E402
from fine_grain.native_mot import (  # noqa: E402
    NativeMoTStack,
    estimate_pretrain_tokens_needed,
)
from fine_grain.vlm_data import answer_only_labels  # noqa: E402
from scripts.train_vlm_frontends import _trainable  # noqa: E402


def _vram() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


class NativeMoTVLM(nn.Module):
    """Native MoT stack + LLM; interface tokens prepended; optional LoRA language."""

    def __init__(self, stack: NativeMoTStack, llm: nn.Module):
        super().__init__()
        self.stack = stack
        self.llm = llm

    def forward(self, images, input_ids, attention_mask, text_labels=None):
        emb = self.llm.get_input_embeddings()(input_ids)
        dtype = next(self.llm.parameters()).dtype
        X, H_llm, tok, traces = self.stack.forward_native(
            images, emb.float(), text_mask=attention_mask,
        )
        v = tok.to(dtype=dtype)
        text = H_llm.to(dtype=dtype)
        inputs = torch.cat([v, text], dim=1)
        T = v.shape[1]
        B, L = input_ids.shape
        vis_m = torch.ones(B, T, device=images.device, dtype=attention_mask.dtype)
        attn = torch.cat([vis_m, attention_mask], dim=1)
        ignore = torch.full((B, T), -100, device=images.device, dtype=input_ids.dtype)
        if text_labels is None:
            text_lab = input_ids.clone().masked_fill(attention_mask == 0, -100)
        else:
            text_lab = text_labels
        labels = torch.cat([ignore, text_lab], dim=1)
        out = self.llm(
            inputs_embeds=inputs, attention_mask=attn, labels=labels, use_cache=False,
        )
        meta = {
            "T_interface": T,
            "n_layers": self.stack.n_layers,
            "traces": [
                {"layer": t.layer, "x_delta": t.x_delta, "h_delta": t.h_delta}
                for t in traces
            ],
        }
        return out.loss, meta


@torch.no_grad()
def eval_tf(bridge, tokenizer, store, device, args, n, seed):
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
        loss, _ = bridge(img, ids, mask, text_labels=lab)
        losses.append(float(loss.item()))
        emb = bridge.llm.get_input_embeddings()(ids)
        dtype = next(bridge.llm.parameters()).dtype
        _, H_llm, tok, _ = bridge.stack.forward_native(img, emb.float(), mask)
        v = tok.to(dtype=dtype)
        text = H_llm.to(dtype=dtype)
        inputs = torch.cat([v, text], dim=1)
        T = v.shape[1]
        attn = torch.cat([torch.ones(1, T, device=device, dtype=mask.dtype), mask], 1)
        out = bridge.llm(inputs_embeds=inputs, attention_mask=attn, use_cache=False)
        L = ids.shape[1]
        pred = out.logits[:, T - 1 : T + L - 1].argmax(-1)
        ans = lab != -100
        if ans.any():
            hit += int((pred[ans] == lab[ans]).sum())
            tot += int(ans.sum())
    bridge.train()
    return {
        "tf_acc": hit / max(tot, 1),
        "tf_n_tokens": tot,
        "eval_loss": float(np.mean(losses)) if losses else float("nan"),
        "n_eval": n,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--res", type=int, default=32)
    ap.add_argument("--d_x", type=int, default=128)
    ap.add_argument("--d", type=int, default=256, help="slice/lang width (512 full spec)")
    ap.add_argument("--n_slices", type=int, default=32)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--llm_lr", type=float, default=1e-4)
    ap.add_argument("--joint", action="store_true", default=True)
    ap.add_argument("--freeze_llm", action="store_true")
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--llm_last_n", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=48)
    ap.add_argument("--max_rows", type=int, default=2000)
    ap.add_argument("--probe_n", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val_seed", type=int, default=99)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--prefer", default="gemma")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results/published/native_mot_hf_table.json")
    ap.add_argument("--conclusion", default="results/published/native_mot_hf_conclusion.md")
    args = ap.parse_args()
    if args.no_amp:
        args.amp = False
    if args.freeze_llm:
        args.joint = False

    device = torch.device(args.device)
    print("=== Native MoT pretrain (spec) on Flickr8k ===", flush=True)
    store_tr = Flickr8kCaptionStore("train", max_rows=args.max_rows)
    try:
        store_va = Flickr8kCaptionStore("test", max_rows=min(500, args.probe_n * 4))
    except Exception:
        store_va = store_tr
    print(f"train={len(store_tr)} val={len(store_va)}", flush=True)

    llm, tokenizer, d_llm, note = load_frozen_lm(prefer=args.prefer, device=str(device))
    print(f"backend={note}", flush=True)

    lang_mode = "frozen"
    if args.joint:
        from fine_grain.lora_llm import apply_lora, enable_lora_grads, peft_available
        if not peft_available():
            raise SystemExit("need peft for --joint")
        llm, meta = apply_lora(
            llm, r=args.lora_r, alpha=args.lora_alpha, last_n_layers=args.llm_last_n,
        )
        enable_lora_grads(llm)
        lang_mode = f"lora_r{meta['lora_r']}_{meta['layer_range']}"
        print(f"joint {lang_mode} params={meta['trainable_params']}", flush=True)

    stack = NativeMoTStack(
        d_llm=d_llm, res=args.res, d_x=args.d_x, d=args.d,
        n_slices=args.n_slices, n_layers=args.n_layers, n_heads=args.n_heads,
        deslice_topk=args.topk, projector="mlp",
    )
    n_stack = stack.count_params()
    need_tok = estimate_pretrain_tokens_needed(n_stack + (268_000_000 if not args.joint else 100_000), 1.0)
    # rough flickr tokens: rows * avg_cap_len
    approx_data_tok = len(store_tr) * 20 * max(1, args.steps // max(1, len(store_tr)))
    print(
        f"stack_params={n_stack} need_tokens≥{need_tok} "
        f"approx_seen_tokens~{approx_data_tok} "
        f"eligible={approx_data_tok >= need_tok}",
        flush=True,
    )

    bridge = NativeMoTVLM(stack, llm).to(device)
    if args.joint:
        for p in bridge.llm.parameters():
            p.requires_grad_(False)
        from fine_grain.lora_llm import enable_lora_grads
        enable_lora_grads(bridge.llm)
    else:
        for p in bridge.llm.parameters():
            p.requires_grad_(False)

    vis, lang = [], []
    for n, p in bridge.named_parameters():
        if not p.requires_grad:
            continue
        (lang if n.startswith("llm") else vis).append(p)
    opt = torch.optim.AdamW(
        [{"params": vis, "lr": args.lr}]
        + ([{"params": lang, "lr": args.llm_lr}] if lang else []),
        weight_decay=0.01,
    )
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    rng = np.random.default_rng(args.seed)
    bridge.train()
    accum = max(1, args.grad_accum)
    opt.zero_grad(set_to_none=True)
    hist = []
    last = float("nan")
    t0 = time.time()
    print(
        f"trainable vis={sum(p.numel() for p in vis)} lang={sum(p.numel() for p in lang)}",
        flush=True,
    )

    for step in range(1, args.steps + 1):
        idxs = rng.integers(0, len(store_tr), size=args.batch)
        batch = sample_batch(store_tr, idxs, res=args.res)
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
            hist.append({"step": step, "loss": last, "vram": _vram()})
            print(f"  step {step:4d} loss={last:.4f} vram={_vram():.0f}MB", flush=True)
        if device.type == "cuda" and step % 20 == 0:
            torch.cuda.empty_cache()

    metrics = eval_tf(bridge, tokenizer, store_va, device, args, args.probe_n, args.val_seed)
    row = {
        "kind": "native_mot",
        "status": "ok",
        "joint": args.joint,
        "lang_mode": lang_mode,
        "d_x": args.d_x,
        "d": args.d,
        "n_slices": args.n_slices,
        "n_layers": args.n_layers,
        "stack_params": n_stack,
        "tokens_needed_1x": need_tok,
        "approx_tokens_seen": approx_data_tok,
        "validation_eligible_1x": approx_data_tok >= need_tok,
        "final_loss": last,
        "peak_vram_mb": _vram(),
        "seconds": time.time() - t0,
        "history": hist,
        **metrics,
    }
    payload = {
        "task": "native_mot_hf_pretrain_smoke",
        "dataset": "jxie/flickr8k",
        "backend": note,
        "spec": "modal_private_qkv_ffn_shared_kv_space_slice_ephemeral",
        "table": [row],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    lines = [
        "# Native MoT (spec) on Flickr8k",
        "",
        f"- Spec: private QKV/FFN per modality; K/V concat shared attention space; "
        f"SliceRead→MoT→Deslice→LocalVisual",
        f"- Dims: d_x={args.d_x} d={args.d} M={args.n_slices} L={args.n_layers} res={args.res}",
        f"- LLM: {lang_mode}; stack_params={n_stack}",
        f"- Token gate (1× params): need ≥{need_tok}, approx_seen~{approx_data_tok}, "
        f"**eligible={row['validation_eligible_1x']}**",
        f"- steps={args.steps} max_rows={args.max_rows}",
        "",
        f"| kind | tf_acc | eval_loss | CER-N/A | VRAM | eligible |",
        f"|------|--------|-----------|---------|------|----------|",
        f"| native_mot | {row['tf_acc']:.3f} | {row['eval_loss']:.3f} | — | "
        f"{row['peak_vram_mb']:.0f} | {row['validation_eligible_1x']} |",
        "",
        "## Notes",
        "- This smoke proves the **architecture path** on real captions.",
        "- Full **validation eligibility** needs open multimodal data with "
        "token count ≥ parameter count (scale Flickr30k / COCO shards / LLaVA-558k).",
        "- Full unfreeze of 270M on 4GB spills to shared RAM; joint uses LoRA.",
    ]
    Path(args.conclusion).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
