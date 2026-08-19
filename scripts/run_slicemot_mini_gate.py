#!/usr/bin/env python3
"""SliceMoT mini mechanism gate + residual ablation arms.

Registered mini config (locked):
  layers=2, M=32, d=256, d_x=256  (X width aligned with S/MoT width)
  Gemma-3-270M + LoRA + Gemma tokenizer
  Native MoT (optional dual_ps: PatchEmbed on point field + Slice into MoT)
  Per-layer independent SliceRead/Deslice/MoT (no cross-layer tying)
  **Must consume loss-bearing tokens ≥ n_params** before efficacy claims.

n_params for 1× gate = NativeMoT stack + LoRA adapters
(Gemma backbone frozen weights excluded).

Default stack knobs (locked from cross-machine collapse study; not ablated here):
  **Stiefel ON**  — primary anti-collapse (slice direction NS)
  **Ada-Temp ON** — pointwise adaptive temperature (mild positive)
  Gumbel OFF; soft deslice default; LocalVisual dw3 default

Arms (relative to base = ST + Ada-Temp):
  base    — ST+Ada-Temp, soft deslice, LocalVisual dw3
  topk2   — base + deslice_topk=2
  gumbel  — base + Gumbel (optional; early-stop negative)
  nolocal  — base + LocalVisual off
  cnx7     — base + LocalVisual ConvNeXt-ish DW7
  dual_ps  — PatchEmbed(X)+MoT(P,S,H)+Deslice(S')+Unpatch(P'); no extra LocalVisual

Data: jxie/flickr8k multi-epoch until token budget.

Example:
  set ML_CACHE_ROOT=D:\\ml_cache
  python scripts/run_slicemot_mini_gate.py --arm base
  python scripts/run_slicemot_mini_gate.py --arm topk2
  python scripts/run_slicemot_mini_gate.py --arm gumbel
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import List, Sequence

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
from fine_grain.lora_llm import apply_lora, enable_lora_grads, peft_available  # noqa: E402
from fine_grain.native_mot import NativeMoTStack  # noqa: E402
from fine_grain.vlm_data import answer_only_labels  # noqa: E402
from scripts.train_vlm_frontends import _trainable  # noqa: E402


# ---- registered mini hyperparams (do not change without new experiment id) ----
MINI = dict(
    name="SliceMoT-Mini-2L",
    n_layers=2,
    n_slices=32,
    d=256,
    d_x=256,  # aligned with d (user lock)
    n_heads=8,
    res=32,
    lora_r=8,
    lora_alpha=16,
    llm_last_n=4,
    layer_param_tying=False,
)

# Locked defaults: Stiefel + Ada-Temp; Gumbel off; soft deslice; LocalVisual dw3.
# Attribution: Ada-Temp/Gumbel ← Transolver++; Stiefel/topk/local ← this repo.
_BASE = dict(
    use_ada_temp=True, use_stiefel=True, use_gumbel=False,
    deslice_topk=0, local_kind="dw3",
    dual_patch=False, patch_size=4, use_unpatch=True,
)
ARMS = {
    "base": dict(
        **_BASE,
        note="DEFAULT: ST+Ada-Temp; soft deslice; LocalVisual dw3",
    ),
    "topk2": dict(
        **{**_BASE, "deslice_topk": 2},
        note="base + sparse deslice topk=2",
    ),
    "gumbel": dict(
        **{**_BASE, "use_gumbel": True},
        note="OPTIONAL; early-stop negative — default Gumbel OFF",
    ),
    "nolocal": dict(
        **{**_BASE, "local_kind": "none"},
        note="base + LocalVisual OFF (identity after Deslice)",
    ),
    "cnx7": dict(
        **{**_BASE, "local_kind": "cnx7"},
        note="base + LocalVisual ConvNeXt-ish DW7",
    ),
    "dual_ps": dict(
        **{
            **_BASE,
            "dual_patch": True,
            "patch_size": 4,
            "use_unpatch": True,
            "local_kind": "none",
        },
        note="PatchEmbed(X)+MoT(P,S,H)+Deslice(S')+Unpatch(P'); ST+Ada-Temp",
    ),
}


class SliceMoTMiniVLM(nn.Module):
    def __init__(self, stack: NativeMoTStack, llm: nn.Module):
        super().__init__()
        self.stack = stack
        self.llm = llm

    def forward(self, images, input_ids, attention_mask, text_labels=None):
        emb = self.llm.get_input_embeddings()(input_ids)
        dtype = next(self.llm.parameters()).dtype
        if text_labels is not None:
            prompt_mask = (text_labels < 0) & (attention_mask > 0)
        else:
            prompt_mask = attention_mask > 0
        _, H_llm, tok, traces = self.stack.forward_native(
            images, emb.float(),
            text_mask=attention_mask > 0,
            prompt_mask=prompt_mask,
        )
        v = tok.to(dtype=dtype)
        text = H_llm.to(dtype=dtype)
        inputs = torch.cat([v, text], dim=1)
        Tvis = v.shape[1]
        B = input_ids.shape[0]
        vis_m = torch.ones(B, Tvis, device=images.device, dtype=attention_mask.dtype)
        attn = torch.cat([vis_m, attention_mask], dim=1)
        ignore = torch.full((B, Tvis), -100, device=images.device, dtype=input_ids.dtype)
        if text_labels is None:
            text_lab = input_ids.clone().masked_fill(attention_mask == 0, -100)
        else:
            text_lab = text_labels
        labels = torch.cat([ignore, text_lab], dim=1)
        out = self.llm(
            inputs_embeds=inputs, attention_mask=attn, labels=labels, use_cache=False,
        )
        n_tok = int((text_lab != -100).sum().item())
        meta = {
            "n_loss_tokens": n_tok,
            "traces": [
                {"layer": t.layer, "x_delta": t.x_delta, "h_delta": t.h_delta}
                for t in traces
            ],
        }
        return out.loss, meta


def count_loss_tokens(lab: torch.Tensor) -> int:
    return int((lab != -100).sum().item())


def warmup_cosine_factor(
    step: int,
    total: int,
    warmup_ratio: float = 0.03,
    min_lr_ratio: float = 0.1,
) -> float:
    """Linear warmup then cosine decay to min_lr_ratio * peak.

    step/total are 1-based micro-steps (same counter as the training loop).
    """
    total = max(int(total), 1)
    step = max(int(step), 0)
    warm = max(1, int(total * float(warmup_ratio)))
    if step <= warm:
        return float(step) / float(warm)
    t = (step - warm) / max(total - warm, 1)
    t = min(1.0, max(0.0, t))
    cos = 0.5 * (1.0 + math.cos(math.pi * t))
    m = float(min_lr_ratio)
    return m + (1.0 - m) * cos


def apply_param_group_lrs(opt: torch.optim.Optimizer, base_lrs: Sequence[float], factor: float) -> List[float]:
    lrs = []
    for g, base in zip(opt.param_groups, base_lrs):
        lr = float(base) * float(factor)
        g["lr"] = lr
        lrs.append(lr)
    return lrs


@torch.no_grad()
def eval_matched_vs_shuffle(bridge, tokenizer, store, device, res, max_len, n=32, seed=7):
    """Lite Gate A probes: matched CE vs cyclically shuffled images."""
    bridge.eval()
    rng = np.random.default_rng(seed)
    n = min(n, len(store))
    idxs = rng.choice(len(store), size=n, replace=False)
    matched, shuffled = [], []
    hit = tot = 0
    for k, i in enumerate(idxs):
        batch = sample_batch(store, [int(i)], res=res)
        img = batch["image"].to(device)
        ids, mask, lab = answer_only_labels(
            tokenizer, batch["prompt"], batch["text"], max_length=max_len,
        )
        ids, mask, lab = ids.to(device), mask.to(device), lab.to(device)
        loss_m, _ = bridge(img, ids, mask, text_labels=lab)
        matched.append(float(loss_m.item()))
        j = int(idxs[(k + n // 3) % n])
        img_s = sample_batch(store, [j], res=res)["image"].to(device)
        loss_s, _ = bridge(img_s, ids, mask, text_labels=lab)
        shuffled.append(float(loss_s.item()))
        emb = bridge.llm.get_input_embeddings()(ids)
        dtype = next(bridge.llm.parameters()).dtype
        prompt_mask = (lab < 0) & (mask > 0)
        _, H_llm, tok, _ = bridge.stack.forward_native(
            img, emb.float(), text_mask=mask > 0, prompt_mask=prompt_mask,
        )
        v = tok.to(dtype=dtype)
        text = H_llm.to(dtype=dtype)
        inputs = torch.cat([v, text], dim=1)
        Tvis = v.shape[1]
        attn = torch.cat([torch.ones(1, Tvis, device=device, dtype=mask.dtype), mask], 1)
        out = bridge.llm(inputs_embeds=inputs, attention_mask=attn, use_cache=False)
        L = ids.shape[1]
        pred = out.logits[:, Tvis - 1 : Tvis + L - 1].argmax(-1)
        ans = lab != -100
        if ans.any():
            hit += int((pred[ans] == lab[ans]).sum())
            tot += int(ans.sum())
    bridge.train()
    m = np.array(matched)
    s = np.array(shuffled)
    return {
        "matched_nll": float(m.mean()),
        "shuffled_nll": float(s.mean()),
        "delta_nll_shuffle_minus_matched": float((s - m).mean()),
        "frac_matched_lower_nll": float((m < s).mean()),
        "tf_acc": hit / max(tot, 1),
        "n_eval": n,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--arm", default="base", choices=sorted(ARMS.keys()),
        help="arm under ST+Ada-Temp: base|topk2|gumbel|nolocal|cnx7|dual_ps",
    )
    ap.add_argument("--max_steps", type=int, default=100_000, help="safety cap")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4, help="peak lr for MoT stack")
    ap.add_argument("--llm_lr", type=float, default=1e-4, help="peak lr for LoRA")
    ap.add_argument(
        "--warmup_ratio", type=float, default=0.03,
        help="linear warmup fraction of max_steps (schedule horizon)",
    )
    ap.add_argument(
        "--min_lr_ratio", type=float, default=0.1,
        help="cosine floor as fraction of peak (0.1 = decay to 10% peak)",
    )
    ap.add_argument(
        "--no_schedule", action="store_true",
        help="disable warmup+cosine; use constant peak lrs",
    )
    ap.add_argument("--max_len", type=int, default=48)
    ap.add_argument("--max_rows", type=int, default=None, help="default=all train")
    ap.add_argument("--token_ratio", type=float, default=1.0, help="need tokens >= ratio*params")
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--probe_n", type=int, default=48)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--prefer", default="gemma")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out", default=None,
        help="default: results/published/slicemot_mini_{arm}_table.json",
    )
    ap.add_argument(
        "--conclusion", default=None,
        help="default: results/published/slicemot_mini_{arm}_conclusion.md",
    )
    args = ap.parse_args()
    if args.no_amp:
        args.amp = False
    if args.out is None:
        args.out = f"results/published/slicemot_mini_{args.arm}_table.json"
    if args.conclusion is None:
        args.conclusion = f"results/published/slicemot_mini_{args.arm}_conclusion.md"

    arm = ARMS[args.arm]
    flag_keys = (
        "use_ada_temp", "use_gumbel", "deslice_topk", "use_stiefel", "local_kind",
        "dual_patch", "patch_size", "use_unpatch",
    )
    cfg = {**MINI, **{k: arm[k] for k in flag_keys}, "arm": args.arm, "arm_note": arm["note"]}

    device = torch.device(args.device)
    print(f"=== {cfg['name']} arm={args.arm} ===", flush=True)
    print(json.dumps(cfg, indent=2), flush=True)
    print(f"cache={cache_root()}", flush=True)

    store = Flickr8kCaptionStore("train", max_rows=args.max_rows)
    try:
        store_val = Flickr8kCaptionStore("test", max_rows=min(400, args.probe_n * 4))
    except Exception:
        store_val = store
    print(f"data train={len(store)} val={len(store_val)}", flush=True)

    if not peft_available():
        raise SystemExit("need peft for Gemma+LoRA mini gate")

    llm, tokenizer, d_llm, note = load_frozen_lm(prefer=args.prefer, device=str(device))
    llm, lora_meta = apply_lora(
        llm, r=cfg["lora_r"], alpha=cfg["lora_alpha"], last_n_layers=cfg["llm_last_n"],
    )
    enable_lora_grads(llm)

    stack = NativeMoTStack(
        d_llm=d_llm,
        res=cfg["res"],
        d_x=cfg["d_x"],
        d=cfg["d"],
        n_slices=cfg["n_slices"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        deslice_topk=cfg["deslice_topk"],
        use_ada_temp=cfg["use_ada_temp"],
        use_gumbel=cfg["use_gumbel"],
        use_stiefel=cfg["use_stiefel"],
        local_kind=cfg.get("local_kind", "dw3"),
        dual_patch=bool(cfg.get("dual_patch", False)),
        patch_size=int(cfg.get("patch_size", 4)),
        use_unpatch=bool(cfg.get("use_unpatch", True)),
        projector="mlp",
    )
    # assert per-layer independence
    if cfg["n_layers"] >= 2:
        assert stack.layers[0].read is not stack.layers[1].read

    n_stack = stack.count_params()
    n_lora = int(lora_meta["trainable_params"])
    n_params_gate = n_stack + n_lora
    need_tokens = int(n_params_gate * args.token_ratio)
    print(
        f"params stack={n_stack} lora={n_lora} gate_params={n_params_gate} "
        f"need_loss_tokens>={need_tokens} (ratio={args.token_ratio})",
        flush=True,
    )
    print(f"backend={note}", flush=True)
    print(
        f"flags ada_temp={cfg['use_ada_temp']} gumbel={cfg['use_gumbel']} "
        f"topk={cfg['deslice_topk']} stiefel={cfg['use_stiefel']} "
        f"local={cfg.get('local_kind', 'dw3')} "
        f"dual_patch={cfg.get('dual_patch', False)} "
        f"unpatch={cfg.get('use_unpatch', True)}",
        flush=True,
    )

    model = SliceMoTMiniVLM(stack, llm).to(device)
    for p in model.llm.parameters():
        p.requires_grad_(False)
    enable_lora_grads(model.llm)

    vis, lang = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (lang if n.startswith("llm") else vis).append(p)
    base_lrs = [float(args.lr), float(args.llm_lr)]
    opt = torch.optim.AdamW(
        [{"params": vis, "lr": base_lrs[0]}, {"params": lang, "lr": base_lrs[1]}],
        weight_decay=0.1, betas=(0.9, 0.95),
    )
    # Schedule horizon = max_steps (smoke / safety cap). Cosine bottoms near end of run.
    sched_total = max(int(args.max_steps), 1)
    use_sched = not bool(args.no_schedule)
    apply_param_group_lrs(
        opt, base_lrs,
        warmup_cosine_factor(1, sched_total, args.warmup_ratio, args.min_lr_ratio)
        if use_sched else 1.0,
    )
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    rng = np.random.default_rng(args.seed)
    model.train()
    accum = max(1, args.grad_accum)
    opt.zero_grad(set_to_none=True)

    tokens_seen = 0
    step = 0
    hist = []
    t0 = time.time()
    last_loss = float("nan")
    last_lrs = [g["lr"] for g in opt.param_groups]

    print(
        f"lr schedule={'warmup+cosine' if use_sched else 'constant'} "
        f"peak_stack={args.lr} peak_lora={args.llm_lr} "
        f"warmup_ratio={args.warmup_ratio} min_lr_ratio={args.min_lr_ratio} "
        f"horizon_steps={sched_total}",
        flush=True,
    )
    print(
        f"training until loss_tokens>={need_tokens} or steps>{args.max_steps} …",
        flush=True,
    )
    while tokens_seen < need_tokens and step < args.max_steps:
        step += 1
        if use_sched:
            last_lrs = apply_param_group_lrs(
                opt, base_lrs,
                warmup_cosine_factor(
                    step, sched_total, args.warmup_ratio, args.min_lr_ratio,
                ),
            )
        idxs = rng.integers(0, len(store), size=args.batch)
        batch = sample_batch(store, idxs, res=cfg["res"])
        img = batch["image"].to(device)
        ids, mask, lab = answer_only_labels(
            tokenizer, batch["prompt"], batch["text"], max_length=args.max_len,
        )
        ids, mask, lab = ids.to(device), mask.to(device), lab.to(device)
        n_tok = count_loss_tokens(lab)

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
            loss, meta = model(img, ids, mask, text_labels=lab)
            loss = loss / accum
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if step % accum == 0 or tokens_seen + n_tok >= need_tokens:
            if use_amp:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(_trainable(model), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(_trainable(model), 1.0)
                opt.step()
            opt.zero_grad(set_to_none=True)

        tokens_seen += n_tok
        last_loss = float(loss.detach().float().item() * accum)
        if step % args.log_every == 0 or step == 1:
            vram = torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else 0
            hist.append({
                "step": step, "loss": last_loss, "tokens_seen": tokens_seen,
                "frac_budget": tokens_seen / need_tokens, "vram_mb": vram,
                "lr_stack": last_lrs[0], "lr_lora": last_lrs[1] if len(last_lrs) > 1 else None,
            })
            print(
                f"  step {step:5d} loss={last_loss:.4f} tokens={tokens_seen}/{need_tokens} "
                f"({100*tokens_seen/need_tokens:.1f}%) "
                f"lr={last_lrs[0]:.2e}/{last_lrs[1]:.2e} vram={vram:.0f}MB",
                flush=True,
            )
        if device.type == "cuda" and step % 50 == 0:
            torch.cuda.empty_cache()

    eligible = tokens_seen >= need_tokens
    print(f"stopped step={step} tokens={tokens_seen} eligible={eligible}", flush=True)

    probes = eval_matched_vs_shuffle(
        model, tokenizer, store_val, device, cfg["res"], args.max_len,
        n=args.probe_n, seed=args.seed + 9,
    )
    print("probes", probes, flush=True)

    row = {
        "experiment": cfg["name"],
        "arm": args.arm,
        "arm_note": arm["note"],
        "status": "ok",
        "eligible_1x": eligible,
        "n_params_stack": n_stack,
        "n_params_lora": n_lora,
        "n_params_gate": n_params_gate,
        "tokens_needed": need_tokens,
        "tokens_seen": tokens_seen,
        "token_ratio_achieved": tokens_seen / max(n_params_gate, 1),
        "steps": step,
        "final_loss": last_loss,
        "seconds": time.time() - t0,
        "peak_vram_mb": (
            torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else 0
        ),
        "config": cfg,
        "lr_schedule": {
            "kind": "warmup_cosine" if use_sched else "constant",
            "peak_lr_stack": args.lr,
            "peak_lr_lora": args.llm_lr,
            "warmup_ratio": args.warmup_ratio,
            "min_lr_ratio": args.min_lr_ratio,
            "horizon_steps": sched_total,
        },
        "flags": {
            "use_ada_temp": cfg["use_ada_temp"],
            "use_gumbel": cfg["use_gumbel"],
            "deslice_topk": cfg["deslice_topk"],
            "use_stiefel": cfg["use_stiefel"],
            "local_kind": cfg.get("local_kind", "dw3"),
            "dual_patch": cfg.get("dual_patch", False),
            "patch_size": cfg.get("patch_size", 4),
            "use_unpatch": cfg.get("use_unpatch", True),
        },
        "knob_attribution": {
            "ada_temp": "Transolver++ (not T3)",
            "gumbel": "Transolver++ (not T3)",
            "deslice_topk": "this repo",
            "stiefel": "this repo",
            "local_kind": "this repo (LocalVisual)",
            "dual_patch": "this repo (PatchEmbed(X)+Unpatch on point field)",
        },
        "backend": note,
        "dataset": f"jxie/flickr8k train n={len(store)}",
        "no_patch_control": True,
        "probes": probes,
        "history": hist,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(row, f, indent=2, default=str)

    lines = [
        f"# {cfg['name']} arm=`{args.arm}`",
        "",
        f"- note: {arm['note']}",
        f"- layers={cfg['n_layers']} M={cfg['n_slices']} d={cfg['d']} d_x={cfg['d_x']} (aligned)",
        f"- Gemma + LoRA r={cfg['lora_r']} last-{cfg['llm_last_n']}; layer Slice params **independent**",
        f"- flags: ada_temp={cfg['use_ada_temp']} gumbel={cfg['use_gumbel']} "
        f"topk={cfg['deslice_topk']} stiefel={cfg['use_stiefel']} "
        f"local={cfg.get('local_kind', 'dw3')} "
        f"dual_patch={cfg.get('dual_patch', False)}",
        "- Native MoT (optional dual PatchEmbed stream on point field)",
        "",
        "## 1× token gate",
        f"- gate_params (stack+LoRA) = **{n_params_gate}**",
        f"- loss tokens needed (≥{args.token_ratio}×) = **{need_tokens}**",
        f"- loss tokens seen = **{tokens_seen}** (ratio={row['token_ratio_achieved']:.3f})",
        f"- **eligible_1x = {eligible}**",
        f"- steps={step} final_loss={last_loss:.4f} VRAM={row['peak_vram_mb']:.0f}MB",
        "",
        "## Lite visual-causality probes",
        f"- matched NLL = {probes['matched_nll']:.4f}",
        f"- shuffled-image NLL = {probes['shuffled_nll']:.4f}",
        f"- Δ (shuffle−matched) = {probes['delta_nll_shuffle_minus_matched']:+.4f}",
        f"- frac matched lower NLL = {probes['frac_matched_lower_nll']:.3f}",
        f"- TF caption-span acc = {probes['tf_acc']:.3f}",
        "",
        "## Decision",
    ]
    if not eligible:
        lines.append("- **No efficacy claim**: did not reach 1× token budget.")
    elif probes["delta_nll_shuffle_minus_matched"] > 0.05 and probes["frac_matched_lower_nll"] > 0.55:
        lines.append(
            "- **Weak positive mechanism signal** on this arm "
            "(matched beats shuffle); not full Gate A."
        )
    else:
        lines.append(
            "- **1× met but no clear visual-causality signal** under lite probes."
        )
    lines.extend([
        "",
        "Knob attribution: Ada-Temp/Gumbel = Transolver++; topk/Stiefel = this repo; "
        "T3 scale stack not in scope.",
        f"Backend: `{note}`",
    ])
    Path(args.conclusion).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"→ {args.out}", flush=True)


if __name__ == "__main__":
    main()
