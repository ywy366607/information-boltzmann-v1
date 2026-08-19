#!/usr/bin/env python3
"""Train/compare A/B/C vision frontends: linear probe and/or frozen-LLM VQA.

Examples:
  set ML_CACHE_ROOT=D:\\ml_cache
  python scripts/train_vlm_frontends.py --protocol probe --mode sweep --T_list 64 32 16
  python scripts/train_vlm_frontends.py --protocol vqa --mode sweep --prefer gemma --steps 400 --seeds 0 1
  python scripts/train_vlm_frontends.py --protocol both --mode sweep --prefer gemma --steps 400 --seeds 0 1
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
import torch.nn as nn
import torch.nn.functional as F

# D: caches before any HF/modelscope import side effects
os.environ.setdefault("ML_CACHE_ROOT", r"D:\ml_cache")
os.environ["MODELSCOPE_CACHE"] = str(Path(os.environ["ML_CACHE_ROOT"]) / "modelscope")
os.environ["HF_HOME"] = str(Path(os.environ["ML_CACHE_ROOT"]) / "huggingface")
os.environ["TORCH_HOME"] = str(Path(os.environ["ML_CACHE_ROOT"]) / "torch")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fine_grain.frontends import (  # noqa: E402
    build_frontend,
    patch_token_count,
    suggest_T_grid,
)
from fine_grain.linear_probe import train_linear_probe  # noqa: E402
from fine_grain.llm_backend import cache_root, load_frozen_lm  # noqa: E402
from fine_grain.vlm_data import (  # noqa: E402
    answer_candidates,
    answer_only_labels,
    exact_match,
    make_vqa_batch,
    tokenize_captions,
)


class VLMBridge(nn.Module):
    """Frozen LLM + trainable vision frontend; visual tokens prepended to text."""

    def __init__(self, frontend: nn.Module, llm: nn.Module):
        super().__init__()
        self.frontend = frontend
        self.llm = llm
        for p in self.llm.parameters():
            p.requires_grad_(False)

    def forward(self, images, input_ids, attention_mask, text_labels=None):
        """Causal LM loss on text tokens (labels ignore visual prefix).

        If ``text_labels`` is provided ([B,L], with -100 ignore), those replace
        the default full-text labels — use for answer-only supervision.
        """
        vis = self.frontend(images)  # FrontendOut
        v = vis.tokens  # [B,T,d]
        if v.dtype != next(self.llm.parameters()).dtype:
            v = v.to(dtype=next(self.llm.parameters()).dtype)
        emb = self.llm.get_input_embeddings()(input_ids)  # [B,L,d]
        inputs_embeds = torch.cat([v, emb], dim=1)
        T = v.shape[1]
        B, L = input_ids.shape
        # attention: visual always kept
        vis_mask = torch.ones(B, T, device=input_ids.device, dtype=attention_mask.dtype)
        attn = torch.cat([vis_mask, attention_mask], dim=1)
        # labels: ignore visual prefix; default = full text ids (pad → -100)
        ignore = torch.full((B, T), -100, device=input_ids.device, dtype=input_ids.dtype)
        if text_labels is None:
            text_lab = input_ids.clone()
            text_lab = text_lab.masked_fill(attention_mask == 0, -100)
        else:
            text_lab = text_labels
        labels = torch.cat([ignore, text_lab], dim=1)
        out = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            labels=labels,
            use_cache=False,
        )
        meta = dict(vis.meta)
        meta["T"] = vis.T
        return out.loss, meta


def _trainable(m: nn.Module):
    return [p for p in m.parameters() if p.requires_grad]


def align_text_token_preds(logits: torch.Tensor, T: int, ids: torch.Tensor, mask: torch.Tensor):
    """Map causal LM logits → predictions for each text token (teacher-forced).

    Sequence: [vis_0 .. vis_{T-1}, txt_0 .. txt_{L-1}].
    ``logits[:, t]`` predicts the token after position t → text id j from T-1+j.
    """
    L = ids.shape[1]
    pred = logits[:, T - 1 : T + L - 1].argmax(dim=-1)
    assert pred.shape == ids.shape, (pred.shape, ids.shape)
    return pred, ids, mask > 0


def tf_answer_span_accuracy(
    logits: torch.Tensor,
    T: int,
    ids: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[float, int, int]:
    """Teacher-forced token accuracy on the answer span (labels != -100).

    Uses the same causal shift as ``align_text_token_preds``. Returns
    ``(acc, n_hit, n_ans)`` where ``n_ans`` is the number of supervised answer
    tokens across the batch. Empty span → ``(nan, 0, 0)``.
    """
    L = ids.shape[1]
    pred = logits[:, T - 1 : T + L - 1].argmax(dim=-1)
    assert pred.shape == ids.shape, (pred.shape, ids.shape)
    ans = labels != -100
    n_ans = int(ans.sum().item())
    if n_ans == 0:
        return float("nan"), 0, 0
    n_hit = int((pred[ans] == labels[ans]).sum().item())
    return n_hit / n_ans, n_hit, n_ans


@torch.no_grad()
def bridge_tf_logits(bridge: nn.Module, images: torch.Tensor, input_ids, attention_mask):
    """Run vision frontend + frozen LLM; return (logits, T_vis) for TF metrics.

    Works with ``VLMBridge`` / any module that exposes ``.frontend`` and ``.llm``.
    For VisualLatentCoT with two-look, use pass-1 tokens only (stable TF probe).
    """
    from fine_grain.visual_latent_cot import encode_frontend

    vis = encode_frontend(bridge.frontend, images, external_h=None)
    v = vis.tokens
    dtype = next(bridge.llm.parameters()).dtype
    if v.dtype != dtype:
        v = v.to(dtype=dtype)
    emb = bridge.llm.get_input_embeddings()(input_ids)
    inputs_embeds = torch.cat([v, emb], dim=1)
    T = v.shape[1]
    B, L = input_ids.shape
    vis_mask = torch.ones(B, T, device=input_ids.device, dtype=attention_mask.dtype)
    attn = torch.cat([vis_mask, attention_mask], dim=1)
    out = bridge.llm(
        inputs_embeds=inputs_embeds,
        attention_mask=attn,
        use_cache=False,
    )
    return out.logits, int(T)


@torch.no_grad()
def _greedy_answer(
    bridge,
    tokenizer,
    images: torch.Tensor,
    prompts: List[str],
    max_new: int = 6,
    early_stop: bool = True,
) -> List[str]:
    """Greedy decode after visual tokens + prompt (vectorized batch loop over steps)."""
    device = images.device
    dtype = next(bridge.llm.parameters()).dtype
    vis = bridge.frontend(images)
    v = vis.tokens.to(dtype=dtype)
    T = v.shape[1]
    B = images.shape[0]
    ids, mask = tokenize_captions(tokenizer, prompts, max_length=48)
    ids, mask = ids.to(device), mask.to(device)
    emb = bridge.llm.get_input_embeddings()(ids)
    inputs = torch.cat([v, emb], dim=1)
    attn = torch.cat([
        torch.ones(B, T, device=device, dtype=mask.dtype), mask,
    ], dim=1)
    # autoregressive append (small max_new; no Python per-pixel)
    gen = ids
    for _ in range(max_new):
        out = bridge.llm(inputs_embeds=inputs, attention_mask=attn, use_cache=False)
        next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [B,1]
        gen = torch.cat([gen, next_id], dim=1)
        next_emb = bridge.llm.get_input_embeddings()(next_id)
        inputs = torch.cat([inputs, next_emb], dim=1)
        attn = torch.cat([
            attn, torch.ones(B, 1, device=device, dtype=attn.dtype),
        ], dim=1)
    # decode only the newly generated suffix
    texts = []
    prompt_lens = mask.sum(dim=1).tolist()
    for i in range(B):
        pl = int(prompt_lens[i])
        # gen includes original prompt tokens then new; original ids length may pad
        # take tokens after non-pad prompt length
        new_ids = gen[i, pl:]
        t = tokenizer.decode(new_ids.tolist(), skip_special_tokens=True)
        if early_stop:
            # Cut continuation leaks only — do NOT strip on "answer:" (models often
            # emit "? Answer: red" as the whole span; cutting there kills true answers).
            low = t.lower()
            cut = len(t)
            for stop in ("\nquestion:", "\n\n", "question:"):
                idx = low.find(stop)
                if idx > 0:
                    cut = min(cut, idx)
            t = t[:cut]
        texts.append(t)
    return texts


@torch.no_grad()
def constrained_rank_answer(
    bridge,
    tokenizer,
    images: torch.Tensor,
    prompts: List[str],
    tasks: List[str],
    max_length: int = 48,
) -> List[str]:
    """Pick best short answer from task vocab by mean log-prob (no free-gen junk).

    For each sample, score candidate strings under teacher-forced LM given
    vision + prompt; return argmax candidate. This is the standard constrained
    VQA readout when the answer set is closed.
    """
    device = images.device
    dtype = next(bridge.llm.parameters()).dtype
    vis = bridge.frontend(images)
    v = vis.tokens.to(dtype=dtype)
    Tvis = v.shape[1]
    B = images.shape[0]
    preds: List[str] = []
    for i in range(B):
        cands = answer_candidates(tasks[i])
        prompt = prompts[i]
        p_ids = tokenizer(
            prompt, add_special_tokens=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        )["input_ids"].to(device)
        best_s, best_score = cands[0], -1e30
        plen = int(p_ids.shape[1])
        for ans in cands:
            # format_prompt ends with "Answer:"; append short gold with a space
            a_ids = tokenizer(
                " " + ans, add_special_tokens=False, return_tensors="pt",
            )["input_ids"].to(device)
            f_ids = torch.cat([p_ids, a_ids], dim=1)
            emb = bridge.llm.get_input_embeddings()(f_ids)
            vi = v[i : i + 1]
            inputs = torch.cat([vi, emb], dim=1)
            attn = torch.ones(1, Tvis + f_ids.shape[1], device=device, dtype=torch.long)
            out = bridge.llm(inputs_embeds=inputs, attention_mask=attn, use_cache=False)
            # sequence [vis | txt0..]; logit at Tvis+j-1 predicts token j
            logp = torch.log_softmax(out.logits[0], dim=-1)
            score = 0.0
            n_tok = 0
            for j in range(plen, f_ids.shape[1]):
                pos = Tvis + j - 1
                if pos < 0:
                    continue
                tid = int(f_ids[0, j].item())
                score += float(logp[pos, tid].item())
                n_tok += 1
            if n_tok == 0:
                continue
            score /= n_tok
            if score > best_score:
                best_score = score
                best_s = ans
        preds.append(best_s)
    return preds


@torch.no_grad()
def probe_task_accuracy(
    bridge,
    tokenizer,
    device,
    res: int,
    n: int = 64,
    batch: int = 8,
    val_seed: int = 90_001,
) -> dict:
    """Held-out synthetic VQA exact-match (color / kinks). No angles.

    Uses a fixed val_seed distinct from training seed.
    Primary metric: free-gen exact match on short answers.
    """
    bridge.eval()
    rng = np.random.default_rng(val_seed)
    by = {"color": {"hit": 0, "tot": 0}, "kinks": {"hit": 0, "tot": 0}}
    left = n
    while left > 0:
        b = min(batch, left)
        data = make_vqa_batch(rng, b, res=res, mix=("color", "kinks"))
        img = data["image"].to(device)
        preds = _greedy_answer(bridge, tokenizer, img, data["prompt"], max_new=6)
        for pred, gold, kind in zip(preds, data["answer"], data["probe"]):
            ok = exact_match(pred, gold)
            by[kind]["tot"] += 1
            by[kind]["hit"] += int(ok)
        left -= b
    bridge.train()
    out = {}
    hits = tots = 0
    for k, v in by.items():
        acc = v["hit"] / max(v["tot"], 1)
        out[f"acc_{k}"] = acc
        out[f"n_{k}"] = v["tot"]
        hits += v["hit"]
        tots += v["tot"]
    out["acc_overall"] = hits / max(tots, 1)
    out["n_overall"] = tots
    return out


def train_one(kind, T, args, llm, tokenizer, d_llm, device, seed: Optional[int] = None):
    seed = int(args.seed if seed is None else seed)
    torch.manual_seed(seed)
    fe = build_frontend(
        kind, d_llm, res=args.res, T=T, patch=args.patch,
        T_slice=args.T_slice, dim=args.dim, depth=args.depth,
        deslice_topk=args.topk,
        projector=getattr(args, "projector", "mlp"),
    ).to(device)
    bridge = VLMBridge(fe, llm).to(device)
    opt = torch.optim.AdamW(_trainable(bridge), lr=args.lr, weight_decay=0.01)
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    rng = np.random.default_rng(seed + 7)
    hist, slots = [], []
    t0 = time.time()
    bridge.train()
    meta = {}
    for step in range(1, args.steps + 1):
        # train seed stream (disjoint from val_seed in probe)
        data = make_vqa_batch(
            rng, args.batch, res=args.res, mix=("color", "kinks"),
        )
        img = data["image"].to(device, non_blocking=True)
        ids, mask = tokenize_captions(tokenizer, data["text"], max_length=args.max_len)
        ids, mask = ids.to(device), mask.to(device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            loss, meta = bridge(img, ids, mask)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        if step % args.log_every == 0 or step == 1 or step == args.steps:
            row = {
                "step": step,
                "loss": float(loss.detach().float().item()),
                "T": int(meta.get("T", T)),
                "kind": kind,
                "seconds": time.time() - t0,
            }
            if "slot" in meta:
                row["slot"] = meta["slot"]
                slots.append(meta["slot"])
            if "slice_slot" in meta:
                row["slot"] = meta["slice_slot"]
                slots.append(meta["slice_slot"])
            if "budget" in meta:
                row["budget"] = meta["budget"]
            hist.append(row)
            print(
                f"  [vqa {kind} T={row['T']} seed={seed}] step {step:4d} "
                f"loss={row['loss']:.4f} "
                f"meta={ {k: meta[k] for k in meta if k in ('budget','G_le_C','T_patch','T_slice')} }",
                flush=True,
            )
    vqa = probe_task_accuracy(
        bridge, tokenizer, device, args.res,
        n=args.probe_n, batch=min(8, args.batch), val_seed=args.val_seed,
    )
    # failure samples for kinks diagnosis when all wrong
    failures = []
    if float(vqa.get("acc_kinks", 0.0)) == 0.0:
        failures = _vqa_failure_samples(
            bridge, tokenizer, device, args.res, n=8, val_seed=args.val_seed + 1,
        )
    final_loss = hist[-1]["loss"] if hist else float("nan")
    return {
        "kind": kind,
        "T": int(hist[-1]["T"]) if hist else T,
        "protocol": "vqa",
        "seed": seed,
        "final_loss": final_loss,
        "probe": vqa,
        "probe_overall": float(vqa.get("acc_overall", 0.0)),
        "probe_color": float(vqa.get("acc_color", 0.0)),
        "probe_kinks": float(vqa.get("acc_kinks", 0.0)),
        "acc_overall": float(vqa.get("acc_overall", 0.0)),
        "acc_color": float(vqa.get("acc_color", 0.0)),
        "acc_kinks": float(vqa.get("acc_kinks", 0.0)),
        "failures": failures,
        "history": hist,
        "slot_last": slots[-1] if slots else {},
        "meta_example": meta if hist else {},
        "budget": meta.get("budget") if meta else None,
        "seconds": time.time() - t0,
        "status": "ok",
    }


@torch.no_grad()
def _vqa_failure_samples(bridge, tokenizer, device, res, n=8, val_seed=90_002):
    """Collect free-gen mismatches for documentation."""
    bridge.eval()
    rng = np.random.default_rng(val_seed)
    fails = []
    left = n * 4  # sample until n fails or budget
    while left > 0 and len(fails) < n:
        data = make_vqa_batch(rng, 1, res=res, mix=("color", "kinks"))
        img = data["image"].to(device)
        preds = _greedy_answer(bridge, tokenizer, img, data["prompt"], max_new=6)
        pred, gold, kind = preds[0], data["answer"][0], data["probe"][0]
        if not exact_match(pred, gold):
            fails.append({
                "probe": kind,
                "gold": gold,
                "pred": pred,
                "prompt": data["prompt"][0],
            })
        left -= 1
    bridge.train()
    return fails


def _run_abc_grid(train_fn, T_list, native, device):
    """Shared A/B/C × T loop; train_fn(kind, T) -> row dict."""
    rows = []
    for T in T_list:
        if T <= native:
            print(f"\n=== A T={T} ===", flush=True)
            try:
                rows.append(train_fn("A", T))
            except Exception as e:
                rows.append({"kind": "A", "T": T, "error": str(e), "status": "error"})
                print(f"  A T={T} FAIL {e}", flush=True)
        print(f"\n=== B T={T} ===", flush=True)
        try:
            rows.append(train_fn("B", T))
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "oom" in str(e).lower():
                rows.append({
                    "kind": "B", "T": T, "error": "OOM",
                    "detail": str(e)[:200], "status": "error",
                })
                print(f"  B T={T} OOM", flush=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            else:
                rows.append({"kind": "B", "T": T, "error": str(e), "status": "error"})
                print(f"  B T={T} FAIL {e}", flush=True)
        except Exception as e:
            rows.append({"kind": "B", "T": T, "error": str(e), "status": "error"})
            print(f"  B T={T} FAIL {e}", flush=True)
        print(f"\n=== C T≈{T} (split) ===", flush=True)
        try:
            rows.append(train_fn("C", T))
        except Exception as e:
            err = "OOM" if "out of memory" in str(e).lower() else str(e)
            rows.append({"kind": "C", "T": T, "error": err, "status": "error"})
            print(f"  C T={T} FAIL {e}", flush=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return rows


def run_sweep(args, llm, tokenizer, d_llm, device, seed: Optional[int] = None):
    T_list = args.T_list or suggest_T_grid(args.res, args.patch)
    print(
        f"VQA T grid (high→low): {T_list}  "
        f"patch_native={patch_token_count(args.res, args.patch)} seed={seed if seed is not None else args.seed}",
        flush=True,
    )
    native = patch_token_count(args.res, args.patch)

    def train_fn(kind, T):
        return train_one(kind, T, args, llm, tokenizer, d_llm, device, seed=seed)

    return _run_abc_grid(train_fn, T_list, native, device)


def run_probe_sweep(args, device):
    """Linear classification probe; no LLM load/generation."""
    T_list = args.T_list or suggest_T_grid(args.res, args.patch)
    d_feat = int(args.d_feat)
    print(
        f"PROBE T grid: {T_list} d_feat={d_feat} steps={args.probe_steps} "
        f"(no LLM)",
        flush=True,
    )
    native = patch_token_count(args.res, args.patch)
    seeds = args.seeds if args.seeds else [args.seed]
    rows = []
    for seed in seeds:
        def train_fn(kind, T, _seed=seed):
            return train_linear_probe(
                kind, T,
                d_feat=d_feat,
                res=args.res,
                patch=args.patch,
                dim=args.dim,
                depth=args.depth,
                topk=args.topk,
                T_slice=args.T_slice,
                projector=getattr(args, "projector", "mlp"),
                steps=args.probe_steps,
                batch=max(args.batch, 4),
                lr=args.lr,
                seed=_seed,
                val_seed=args.val_seed,
                probe_n=args.probe_n,
                device=device,
                log_every=args.log_every,
                collect_failures=8,
            )

        print(f"\n##### linear probe seed={seed} #####", flush=True)
        rows.extend(_run_abc_grid(train_fn, T_list, native, device))
    return rows


def _row_to_table_entry(r: dict) -> dict:
    if r.get("status") == "error" or "error" in r and r.get("status") != "ok":
        return {
            "kind": r.get("kind"),
            "T": r.get("T"),
            "protocol": r.get("protocol"),
            "seed": r.get("seed"),
            "status": "error",
            "error": r.get("error"),
            "detail": r.get("detail"),
        }
    return {
        "kind": r["kind"],
        "T": r["T"],
        "protocol": r.get("protocol", "vqa"),
        "seed": r.get("seed"),
        "final_loss": r.get("final_loss"),
        "acc_overall": r.get("acc_overall", r.get("probe_overall", 0.0)),
        "acc_color": r.get("acc_color", r.get("probe_color", 0.0)),
        "acc_kinks": r.get("acc_kinks", r.get("probe_kinks", 0.0)),
        "probe": r.get("probe", {}),
        "failures": r.get("failures", []),
        "slot_last": r.get("slot_last", {}),
        "budget": r.get("budget") or (r.get("meta_example") or {}).get("budget"),
        "seconds": r.get("seconds"),
        "status": "ok",
    }


def _mean_by_kind_T(rows, protocol: str):
    """Average acc across seeds for kind×T under protocol."""
    buckets = {}
    for r in rows:
        if r.get("status") != "ok" or r.get("protocol") != protocol:
            continue
        key = (r["kind"], r["T"])
        buckets.setdefault(key, []).append(r)
    out = []
    for (kind, T), vs in sorted(buckets.items(), key=lambda x: (x[0][0], -x[0][1])):
        out.append({
            "kind": kind,
            "T": T,
            "protocol": protocol,
            "n_seeds": len(vs),
            "acc_overall": float(np.mean([v["acc_overall"] for v in vs])),
            "acc_color": float(np.mean([v["acc_color"] for v in vs])),
            "acc_kinks": float(np.mean([v["acc_kinks"] for v in vs])),
            "acc_overall_std": float(np.std([v["acc_overall"] for v in vs])) if len(vs) > 1 else 0.0,
        })
    return out


def write_diagnosis_artifacts(payload: dict, out_path: str, conclusion_path: str, fail_log_path: Optional[Path] = None):
    """Write published JSON + conclusion with frontend/alignment/undertrained call."""
    table = payload["table"]
    probe_rows = [t for t in table if t.get("protocol") == "linear_probe" and t.get("status") == "ok"]
    vqa_rows = [t for t in table if t.get("protocol") == "vqa" and t.get("status") == "ok"]
    probe_mean = _mean_by_kind_T(table, "linear_probe")
    vqa_mean = _mean_by_kind_T(table, "vqa")

    def best_kind(means, kind):
        rs = [m for m in means if m["kind"] == kind]
        return max(rs, key=lambda x: x["acc_overall"]) if rs else None

    def best_overall(means):
        return max(means, key=lambda x: x["acc_overall"]) if means else None

    b_probe = best_kind(probe_mean, "B")
    b_vqa = best_kind(vqa_mean, "B")
    a_probe = best_kind(probe_mean, "A")
    a_vqa = best_kind(vqa_mean, "A")
    c_probe = best_kind(probe_mean, "C")
    c_vqa = best_kind(vqa_mean, "C")

    # Diagnosis heuristics
    probe_kinks_max = max((r["acc_kinks"] for r in probe_rows), default=0.0)
    vqa_kinks_max = max((r["acc_kinks"] for r in vqa_rows), default=0.0)
    probe_color_max = max((r["acc_color"] for r in probe_rows), default=0.0)
    vqa_color_max = max((r["acc_color"] for r in vqa_rows), default=0.0)

    diagnoses = []
    # frontend-weak: probe also fails structure/color
    if probe_color_max < 0.35 and probe_kinks_max < 0.30:
        diagnoses.append("frontend-weak")
    # alignment-weak: probe OK-ish, VQA much worse
    if probe_color_max >= 0.40 and vqa_color_max < probe_color_max - 0.15:
        diagnoses.append("alignment-weak")
    elif probe_kinks_max >= 0.30 and vqa_kinks_max < 0.05:
        diagnoses.append("alignment-weak")
    # undertrained: long VQA still near chance on color (~0.25) while probe also weak-mid
    if vqa_color_max < 0.35 and probe_color_max < 0.50:
        diagnoses.append("undertrained")
    if not diagnoses:
        # relative call
        if b_probe and b_vqa and b_probe["acc_overall"] > (b_vqa["acc_overall"] + 0.1):
            diagnoses.append("alignment-weak")
        elif b_probe and b_probe["acc_overall"] < 0.3:
            diagnoses.append("frontend-weak")
        else:
            diagnoses.append("mixed")
    # de-dupe preserve order
    seen_d = set()
    diagnoses = [d for d in diagnoses if not (d in seen_d or seen_d.add(d))]

    # B flip: does B rank better on probe than VQA relative to A?
    flip_note = "n/a"
    if a_probe and b_probe and a_vqa and b_vqa:
        probe_gap = b_probe["acc_overall"] - a_probe["acc_overall"]
        vqa_gap = b_vqa["acc_overall"] - a_vqa["acc_overall"]
        flipped = (probe_gap > 0 and vqa_gap < 0) or (probe_gap < 0 and vqa_gap > 0)
        flip_note = (
            f"B−A probe={probe_gap:+.3f} (B={b_probe['acc_overall']:.3f}@T{b_probe['T']}, "
            f"A={a_probe['acc_overall']:.3f}@T{a_probe['T']}); "
            f"B−A vqa={vqa_gap:+.3f} (B={b_vqa['acc_overall']:.3f}@T{b_vqa['T']}, "
            f"A={a_vqa['acc_overall']:.3f}@T{a_vqa['T']}); "
            f"flip={'YES' if flipped else 'NO'}"
        )

    # failure log
    all_failures = []
    for r in table:
        for f in r.get("failures") or []:
            all_failures.append({
                "kind": r.get("kind"), "T": r.get("T"),
                "protocol": r.get("protocol"), "seed": r.get("seed"), **f,
            })
    if fail_log_path is not None and all_failures:
        fail_log_path.parent.mkdir(parents=True, exist_ok=True)
        fail_log_path.write_text(
            json.dumps(all_failures[:80], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    kinks_all_zero = (probe_kinks_max == 0.0 and vqa_kinks_max == 0.0)

    payload = dict(payload)
    payload["diagnosis"] = {
        "classes": diagnoses,
        "b_probe_vs_vqa": flip_note,
        "probe_color_max": probe_color_max,
        "probe_kinks_max": probe_kinks_max,
        "vqa_color_max": vqa_color_max,
        "vqa_kinks_max": vqa_kinks_max,
        "kinks_all_zero": kinks_all_zero,
        "probe_mean": probe_mean,
        "vqa_mean": vqa_mean,
        "failure_log": str(fail_log_path) if fail_log_path and all_failures else None,
    }

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    lines = [
        "# Feature vs alignment: linear probe vs long VQA (A/B/C)",
        "",
        f"- Cache: `{payload.get('cache_root')}`",
        f"- Backend (VQA): `{payload.get('backend', 'n/a (probe-only)')}`",
        f"- res={payload.get('res')} patch={payload.get('patch')} topk={payload.get('topk')} "
        f"projector={payload.get('projector', 'mlp')} ST=on for B/C; **no angles**",
        f"- Vision→LLM connector: LLaVA-1.5 style "
        f"**{payload.get('projector', 'mlp')}** (mlp = Linear→GELU→Linear)",
        f"- Linear probe steps={payload.get('probe_steps')} d_feat={payload.get('d_feat')} "
        f"(mean-pool tokens → linear heads; **no LLM generation**)",
        f"- VQA steps={payload.get('vqa_steps')} seeds={payload.get('seeds')} "
        f"protocol=`Question:`/`Answer:` greedy exact-match",
        f"- val_seed={payload.get('val_seed')}",
        "",
        "## Linear probe (classification)",
        "",
        "| kind | T | seed | acc | color | kinks | notes |",
        "|------|---|------|-----|-------|-------|-------|",
    ]
    for t in table:
        if t.get("protocol") != "linear_probe":
            continue
        if t.get("status") != "ok":
            lines.append(
                f"| {t.get('kind')} | {t.get('T')} | {t.get('seed')} | — | — | — | {t.get('error')} |"
            )
        else:
            notes = t.get("budget") or ""
            slot = t.get("slot_last") or {}
            if slot:
                notes += (
                    f" PR={slot.get('PR_mass', float('nan')):.1f}"
                    f" sup={slot.get('support', float('nan')):.1f}"
                )
            lines.append(
                f"| {t['kind']} | {t['T']} | {t.get('seed')} | "
                f"{t['acc_overall']:.3f} | {t['acc_color']:.3f} | {t['acc_kinks']:.3f} | {notes} |"
            )

    lines.extend([
        "",
        "## Long VQA (exact-match free-gen)",
        "",
        "| kind | T | seed | acc | color | kinks | notes |",
        "|------|---|------|-----|-------|-------|-------|",
    ])
    for t in table:
        if t.get("protocol") != "vqa":
            continue
        if t.get("status") != "ok":
            lines.append(
                f"| {t.get('kind')} | {t.get('T')} | {t.get('seed')} | — | — | — | {t.get('error')} |"
            )
        else:
            notes = t.get("budget") or ""
            lines.append(
                f"| {t['kind']} | {t['T']} | {t.get('seed')} | "
                f"{t['acc_overall']:.3f} | {t['acc_color']:.3f} | {t['acc_kinks']:.3f} | {notes} |"
            )

    lines.extend(["", "## Diagnosis", ""])
    lines.append(f"- **Class**: {', '.join(diagnoses)}")
    lines.append(f"- **B probe vs VQA**: {flip_note}")
    lines.append(
        f"- Max probe color={probe_color_max:.3f} kinks={probe_kinks_max:.3f}; "
        f"max VQA color={vqa_color_max:.3f} kinks={vqa_kinks_max:.3f}"
    )
    if b_probe and b_vqa:
        lines.append(
            f"- B best probe acc={b_probe['acc_overall']:.3f} (T={b_probe['T']}) vs "
            f"best VQA acc={b_vqa['acc_overall']:.3f} (T={b_vqa['T']})"
        )
    if a_probe and a_vqa:
        lines.append(
            f"- A best probe acc={a_probe['acc_overall']:.3f} vs VQA {a_vqa['acc_overall']:.3f}"
        )
    if c_probe and c_vqa:
        lines.append(
            f"- C best probe acc={c_probe['acc_overall']:.3f} vs VQA {c_vqa['acc_overall']:.3f}"
        )

    if kinks_all_zero:
        lines.append(
            "- **Kinks remain 0** on both probe and VQA across reported cells. "
            "Likely hard at res=32 with 1px polylines + mixed batch + short heads; "
            "see failure samples."
        )
        if fail_log_path and all_failures:
            lines.append(f"- Failure sample log: `{fail_log_path}`")
            # inline a few
            for ex in all_failures[:5]:
                lines.append(
                    f"  - {ex.get('protocol')}/{ex.get('kind')}T{ex.get('T')}: "
                    f"gold={ex.get('gold', ex.get('gold_k'))} pred={ex.get('pred', ex.get('pred_k'))}"
                )
    else:
        lines.append(
            f"- Kinks learnable on at least one protocol "
            f"(probe_max={probe_kinks_max:.3f}, vqa_max={vqa_kinks_max:.3f})."
        )

    lines.extend([
        "",
        "## Method notes",
        "",
        "- Linear probe: freeze-free frontend + mean-pool over T vision tokens + CE heads; LLM unused.",
        "- VQA: frozen Gemma-3-270M @ D:\\ml_cache; train frontend only; greedy exact-match.",
        "- Same on-the-fly synthetic mix (color 4-way, kinks 5–8 corners); train/val seed split.",
    ])

    Path(conclusion_path).parent.mkdir(parents=True, exist_ok=True)
    Path(conclusion_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n=== DIAGNOSIS TABLE ===", flush=True)
    print("\n".join(lines), flush=True)
    print(f"json → {out_path}", flush=True)
    print(f"conclusion → {conclusion_path}", flush=True)
    return payload


def write_vqa_only_artifacts(rows, args, note, out_path, conclusion_path):
    """Back-compat writer for pure VQA sweeps (legacy default paths)."""
    table = [_row_to_table_entry(r) for r in rows]
    for t in table:
        if t.get("protocol") is None:
            t["protocol"] = "vqa"
    payload = {
        "backend": note,
        "cache_root": str(cache_root()),
        "res": args.res,
        "patch": args.patch,
        "steps": args.steps,
        "batch": args.batch,
        "topk": args.topk,
        "projector": getattr(args, "projector", "mlp"),
        "val_seed": args.val_seed,
        "probe": "held-out synthetic VQA exact-match (color, kinks); no angles",
        "T_list": args.T_list or suggest_T_grid(args.res, args.patch),
        "table": table,
        "rows": rows,
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    ok = [t for t in table if t.get("status") == "ok"]
    lines = [
        "# VLM frontend A/B/C comparison (task probe)",
        "",
        f"- Backend: `{note}`",
        f"- Cache: `{cache_root()}` (D: drive)",
        f"- res={args.res} patch={args.patch} steps={args.steps} topk={args.topk} "
        f"projector={getattr(args, 'projector', 'mlp')} ST=on for B/C",
        f"- **Probe**: held-out synthetic VQA exact-match (val_seed={args.val_seed}); "
        "tasks=color + kinks; **no angles**",
        f"- Train: `Question: … Answer: …` on-the-fly mix seeds={getattr(args, 'seeds', [args.seed])}",
        "",
        "| kind | T | seed | loss | acc | color | kinks | notes |",
        "|------|---|------|------|-----|-------|-------|-------|",
    ]
    for t in table:
        if t.get("status") != "ok":
            lines.append(
                f"| {t.get('kind')} | {t.get('T')} | {t.get('seed')} | — | — | — | — | {t.get('error')} |"
            )
        else:
            notes = t.get("budget") or ""
            slot = t.get("slot_last") or {}
            if slot:
                notes += (
                    f" PR={slot.get('PR_mass', float('nan')):.1f}"
                    f" r99={slot.get('r99', float('nan')):.1f}"
                    f" sup={slot.get('support', float('nan')):.1f}"
                )
            lines.append(
                f"| {t['kind']} | {t['T']} | {t.get('seed')} | {t['final_loss']:.3f} | "
                f"{t['acc_overall']:.3f} | {t['acc_color']:.3f} | {t['acc_kinks']:.3f} | "
                f"{notes} |"
            )
    lines.extend(["", "## Short conclusion", ""])
    if ok:
        by_kind = {}
        for t in ok:
            by_kind.setdefault(t["kind"], []).append(t)
        best = {k: max(vs, key=lambda x: x["acc_overall"]) for k, vs in by_kind.items()}
        order = sorted(best.items(), key=lambda kv: -kv[1]["acc_overall"])
        lines.append(
            "- Best **held-out VQA exact-match** by kind: "
            + ", ".join(
                f"{k}=T{v['T']} acc={v['acc_overall']:.3f} "
                f"(color={v['acc_color']:.2f}, kinks={v['acc_kinks']:.2f})"
                for k, v in order
            )
        )
    else:
        lines.append("- No successful runs; see JSON errors.")
    Path(conclusion_path).parent.mkdir(parents=True, exist_ok=True)
    Path(conclusion_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n=== TABLE ===", flush=True)
    print("\n".join(lines), flush=True)
    print(f"json → {out_path}", flush=True)
    print(f"conclusion → {conclusion_path}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["smoke", "sweep"], default="smoke")
    ap.add_argument(
        "--protocol",
        choices=["vqa", "probe", "both"],
        default="vqa",
        help="vqa=free-gen exact-match; probe=linear heads no LLM; both=diagnosis table",
    )
    ap.add_argument("--prefer", default="gemma", choices=["gemma", "pythia", "local"],
                    help="prefer gemma-3-270m on D:\\ml_cache; pythia/local fallbacks")
    ap.add_argument("--res", type=int, default=32)
    ap.add_argument("--patch", type=int, default=4)
    ap.add_argument("--T_list", type=int, nargs="*", default=None)
    ap.add_argument("--T_slice", type=int, default=None)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument(
        "--projector",
        default="mlp",
        choices=["mlp", "linear"],
        help="vision→LLM connector: LLaVA-1.5 2-layer MLP (default) or single Linear",
    )
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--steps", type=int, default=40, help="VQA train steps")
    ap.add_argument("--probe_steps", type=int, default=200, help="linear probe train steps")
    ap.add_argument("--d_feat", type=int, default=640,
                    help="frontend proj dim for probe (match Gemma d_llm=640 by default)")
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max_len", type=int, default=48)
    ap.add_argument("--probe_n", type=int, default=64)
    ap.add_argument("--val_seed", type=int, default=90_001,
                    help="RNG seed for held-out eval (disjoint from train)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="*", default=None,
                    help="multi train seeds for VQA/probe (default: single --seed)")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    ap.add_argument("--conclusion", default=None)
    ap.add_argument("--fail_log", default=None,
                    help="optional path for failure sample JSON")
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"cache_root={cache_root()} (must be on D:)", flush=True)
    assert str(cache_root()).upper().startswith("D:"), cache_root()

    if args.seeds is None:
        args.seeds = [args.seed]

    if args.mode == "smoke":
        args.steps = min(args.steps, 20)
        args.probe_steps = min(args.probe_steps, 20)
        args.T_list = args.T_list or [patch_token_count(args.res, args.patch), 32]
        seen = set()
        tl = []
        for t in args.T_list:
            if t not in seen:
                seen.add(t)
                tl.append(t)
        args.T_list = tl
        # smoke: single seed unless user passed many
        if len(args.seeds) > 1:
            args.seeds = args.seeds[:1]

    # default output paths
    if args.protocol == "both":
        args.out = args.out or "results/published/vlm_probe_vs_vqa_table.json"
        args.conclusion = args.conclusion or "results/published/vlm_probe_vs_vqa_conclusion.md"
    elif args.protocol == "probe":
        args.out = args.out or "results/published/vlm_probe_table.json"
        args.conclusion = args.conclusion or "results/published/vlm_probe_conclusion.md"
    else:
        args.out = args.out or "results/published/vlm_abc_table.json"
        args.conclusion = args.conclusion or "results/published/vlm_abc_conclusion.md"

    all_rows = []
    note = "n/a (probe-only; no LLM)"
    llm = tokenizer = None
    d_llm = args.d_feat

    if args.protocol in ("probe", "both"):
        print("\n######## PROTOCOL: linear_probe ########", flush=True)
        all_rows.extend(run_probe_sweep(args, device))

    if args.protocol in ("vqa", "both"):
        print(f"\n######## PROTOCOL: vqa (prefer={args.prefer}) ########", flush=True)
        print(f"loading LLM prefer={args.prefer} …", flush=True)
        llm, tokenizer, d_llm, note = load_frozen_lm(prefer=args.prefer, device=str(device))
        print(f"backend: {note}", flush=True)
        for seed in args.seeds:
            print(f"\n##### VQA seed={seed} steps={args.steps} #####", flush=True)
            all_rows.extend(
                run_sweep(args, llm, tokenizer, d_llm, device, seed=seed)
            )

    table = [_row_to_table_entry(r) for r in all_rows]

    if args.protocol == "both" or (
        args.protocol == "probe" and "probe_vs_vqa" in (args.out or "")
    ):
        payload = {
            "backend": note,
            "cache_root": str(cache_root()),
            "res": args.res,
            "patch": args.patch,
            "vqa_steps": args.steps if args.protocol in ("vqa", "both") else None,
            "probe_steps": args.probe_steps,
            "d_feat": args.d_feat,
            "batch": args.batch,
            "topk": args.topk,
            "projector": getattr(args, "projector", "mlp"),
            "val_seed": args.val_seed,
            "seeds": args.seeds,
            "T_list": args.T_list or suggest_T_grid(args.res, args.patch),
            "protocols": [args.protocol] if args.protocol != "both" else ["linear_probe", "vqa"],
            "table": table,
            "rows": all_rows,
        }
        fail_log = Path(args.fail_log) if args.fail_log else None
        write_diagnosis_artifacts(payload, args.out, args.conclusion, fail_log_path=fail_log)
    elif args.protocol == "probe":
        # probe-only but still use diagnosis writer with empty VQA
        payload = {
            "backend": note,
            "cache_root": str(cache_root()),
            "res": args.res,
            "patch": args.patch,
            "vqa_steps": None,
            "probe_steps": args.probe_steps,
            "d_feat": args.d_feat,
            "batch": args.batch,
            "topk": args.topk,
            "projector": getattr(args, "projector", "mlp"),
            "val_seed": args.val_seed,
            "seeds": args.seeds,
            "T_list": args.T_list or suggest_T_grid(args.res, args.patch),
            "protocols": ["linear_probe"],
            "table": table,
            "rows": all_rows,
        }
        fail_log = Path(args.fail_log) if args.fail_log else None
        write_diagnosis_artifacts(payload, args.out, args.conclusion, fail_log_path=fail_log)
    else:
        write_vqa_only_artifacts(all_rows, args, note, args.out, args.conclusion)


if __name__ == "__main__":
    main()

