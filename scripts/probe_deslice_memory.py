#!/usr/bin/env python3
"""Causal probe: is absolute Deslice working-memory broadcast?

GPT claim: Deslice(S) stamps current belief onto X so the next SliceRead
recovers S; Deslice(ΔS) only writes velocity and kills the kinks loop.

  python scripts/probe_deslice_memory.py --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fine_grain.vlm_data import make_vqa_batch
from scripts.run_v0_surprise_eval import DualStreamVQAModel, eval_model

CKPTS = {
    "champ_b": ROOT / "checkpoints" / "v1_bayes_2000step_upd_rms_run1_best.pt",
    "p0_id": ROOT / "checkpoints" / "v1_bayes_2000step_p0_id_run1_best.pt",
}
# Champion B trained: absolute write, H/Local ungated.
# P0 trained: increment write, H/Local gated.
CONFIGS = [
    ("abs_ungated", "absolute", False),   # original Champion B forward
    ("inc_ungated", "increment", False),  # only swap the write quantity
    ("abs_gated", "absolute", True),      # only add g_G on H/Local
    ("inc_gated", "increment", True),     # full P0 forward
]
FIG = ROOT / "present" / "figs"
OUT = ROOT / "results" / "published" / "deslice_memory_probe.json"


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    an = F.normalize(a.float(), dim=-1)
    bn = F.normalize(b.float(), dim=-1)
    return float((an * bn).sum(-1).mean())


def _retrofit_fulld_prior(model: DualStreamVQAModel) -> None:
    """Champion-B ckpt: multi-head QK, but prior/post MLP still over full d."""
    import types
    import torch.nn as nn

    def _head_full(self, x, mlp):
        return mlp(x)

    d = model.d_model
    for layer in model.mot_stack.layers:
        g = layer.surprise_gate
        g.prior_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2 * d))
        g.post_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2 * d))
        g._head_mlp = types.MethodType(_head_full, g)


def load_model(ckpt: Path, device: torch.device) -> DualStreamVQAModel:
    m = DualStreamVQAModel(
        d_model=128, n_slices=32, n_layers=4, res=32,
        surprise_mode="v1_bayes", surprise_beta=1.5,
        s_update="rms", use_stiefel=True, deslice_topk=2, n_heads=4,
    )
    raw = torch.load(ckpt, map_location="cpu")
    key = "mot_stack.layers.0.surprise_gate.prior_head.0.weight"
    if key in raw and tuple(raw[key].shape) == (128, 128):
        _retrofit_fulld_prior(m)
        print("  retrofit full-d prior/post (Champion B ckpt)", flush=True)
    m = m.to(device)
    miss = m.load_state_dict(raw, strict=False)
    print(
        f"  load {ckpt.name}  missing={len(miss.missing_keys)} unexpected={len(miss.unexpected_keys)}",
        flush=True,
    )
    m.eval()
    return m


def apply_knobs(model: DualStreamVQAModel, write: str, gated: bool) -> None:
    model.mot_stack.set_write_knobs(deslice_write=write, gate_h_local=gated)


@torch.no_grad()
def memory_trace(model: DualStreamVQAModel, img, prompts, device) -> dict:
    tok_ids, mask = model.tokenize(prompts, device)
    text_emb = model.embed(tok_ids)
    X = model.mot_stack.encode_X(img)
    H = model.mot_stack.text_in(text_emb)
    layers = []
    for i, layer in enumerate(model.mot_stack.layers):
        S_in, _ = layer.read(X)
        rms_x = float(X.pow(2).mean().sqrt())
        X2, H, tr = layer(X, H, text_mask=mask, prompt_mask=mask, layer_idx=i)
        S_reread, _ = layer.read(X2)
        dS = layer.last_S_write - layer.last_S
        layers.append({
            "layer": i,
            "U": tr.surprise_u,
            "g_G": tr.g_global,
            "x_delta": tr.x_delta,
            "h_delta": tr.h_delta,
            "rms_X": rms_x,
            "rms_X2": float(X2.pow(2).mean().sqrt()),
            "cos_reread_Swrite": _cos(S_reread, layer.last_S_write),
            "cos_reread_Sread": _cos(S_reread, layer.last_S),
            "cos_reread_dS": _cos(S_reread, dS) if dS.abs().mean() > 1e-6 else 0.0,
            "cos_reread_minus_in_vs_dS": _cos(S_reread - S_in, dS) if dS.abs().mean() > 1e-6 else 0.0,
        })
        X = X2
    return {"rms_X0": float(model.mot_stack.encode_X(img).pow(2).mean().sqrt()), "layers": layers}


@torch.no_grad()
def belief_identity(model: DualStreamVQAModel, img, prompts, device) -> dict:
    """force_gate=0: does X stay put, or only the Read-belief?"""
    tok_ids, mask = model.tokenize(prompts, device)
    text_emb = model.embed(tok_ids)
    X = model.mot_stack.encode_X(img)
    H = model.mot_stack.text_in(text_emb)
    z = torch.zeros(img.shape[0], model.mot_stack.n_slices, 1, device=device)
    rows = []
    for i, layer in enumerate(model.mot_stack.layers):
        S0, _ = layer.read(X)
        X2, H2, _ = layer(X, H, text_mask=mask, force_gate=z, layer_idx=i)
        S1, _ = layer.read(X2)
        rows.append({
            "layer": i,
            "dx": float((X2 - X).norm(dim=-1).mean()),
            "dh": float((H2 - H).norm(dim=-1).mean()),
            "d_read": float((S1 - S0).norm(dim=-1).mean()),
            "cos_read": _cos(S1, S0),
        })
        X, H = X2, H2
    return rows


def task_eval(model, device, n_batches=20, seed=9001) -> dict:
    rng = np.random.default_rng(seed)
    ev = eval_model(model, rng, val_batches=n_batches, batch_size=32, res=32, device=device)
    return {
        "acc": ev["acc"],
        "ocr": ev["task_accs"].get("ocr", 0.0),
        "kinks": ev["task_accs"].get("kinks", 0.0),
        "color": ev["task_accs"].get("color", 0.0),
        "loss": ev["loss"],
    }


def render(all_rows: dict) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    # 1) task bars: champ_b and p0_id under abs_ungated vs inc_gated vs inc_ungated
    keys = ["abs_ungated", "inc_ungated", "abs_gated", "inc_gated"]
    colors = {"abs_ungated": "#fbbf24", "inc_ungated": "#fb7185",
              "abs_gated": "#38bdf8", "inc_gated": "#2dd4bf"}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), dpi=140)
    fig.patch.set_facecolor("#0b1120")
    tasks = ["ocr", "kinks", "color"]
    for ax, ck in zip(axes, ("champ_b", "p0_id")):
        ax.set_facecolor("#0f172a")
        x = np.arange(len(tasks))
        w = 0.2
        for i, k in enumerate(keys):
            rec = all_rows[ck][k]["eval"]
            vals = [rec[t] * 100 for t in tasks]
            ax.bar(x + (i - 1.5) * w, vals, w, label=k, color=colors[k], alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(["OCR", "Kinks", "Color"], color="#e2e8f0")
        ax.set_ylim(0, 115)
        ax.set_title(ck, color="white", fontweight="bold")
        ax.tick_params(colors="#94a3b8")
        ax.grid(True, axis="y", linestyle=":", alpha=0.3)
        if ck == "champ_b":
            ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=7)
    fig.suptitle("Write quantity × gate   (same weights, eval-only swap)", color="white", fontweight="bold")
    fig.tight_layout()
    p1 = FIG / "deslice_memory_task_swap.png"
    fig.savefig(p1, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"  saved {p1}", flush=True)

    # 2) memory cosine + ||X|| + U for champ_b on kinks batch
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8), dpi=140)
    fig.patch.set_facecolor("#0b1120")
    titles = ["cos(Read(X'), S_write)", "||X|| RMS", "U"]
    getters = [
        lambda L: [r["cos_reread_Swrite"] for r in L],
        lambda L: [r["rms_X2"] for r in L],
        lambda L: [r["U"] for r in L],
    ]
    for ax, title, get in zip(axes, titles, getters):
        ax.set_facecolor("#0f172a")
        for k in keys:
            ys = get(all_rows["champ_b"][k]["mem_kinks"]["layers"])
            ax.plot(range(len(ys)), ys, "o-", label=k, color=colors[k], linewidth=2)
        ax.set_title(title, color="white", fontsize=10)
        ax.set_xlabel("layer", color="#cbd5e1")
        ax.set_xticks([0, 1, 2, 3])
        ax.tick_params(colors="#94a3b8")
        ax.grid(True, linestyle=":", alpha=0.3)
    axes[0].legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=7)
    fig.suptitle("Champion B weights  ·  kinks batch  ·  memory vs drift", color="white", fontweight="bold")
    fig.tight_layout()
    p2 = FIG / "deslice_memory_layer_trace.png"
    fig.savefig(p2, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"  saved {p2}", flush=True)

    # 3) belief identity under force_gate=0
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), dpi=140)
    fig.patch.set_facecolor("#0b1120")
    for ax, ck in zip(axes, ("champ_b", "p0_id")):
        ax.set_facecolor("#0f172a")
        for k, ls in (("abs_ungated", "-"), ("inc_gated", "--")):
            rows = all_rows[ck][k]["identity"]
            ax.plot([r["dx"] for r in rows], "o-", color=colors[k], label=f"{k} ||ΔX||")
            ax.plot([r["d_read"] for r in rows], "s--", color=colors[k], alpha=0.7, label=f"{k} ||ΔRead||")
        ax.set_title(f"{ck}  force g=0", color="white")
        ax.set_xlabel("layer", color="#cbd5e1")
        ax.tick_params(colors="#94a3b8")
        ax.grid(True, linestyle=":", alpha=0.3)
        ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=7)
    fig.suptitle("Identity: field ΔX vs belief ΔRead  (g=0)", color="white", fontweight="bold")
    fig.tight_layout()
    p3 = FIG / "deslice_memory_identity.png"
    fig.savefig(p3, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"  saved {p3}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batches", type=int, default=20)
    args = ap.parse_args()
    os.environ.setdefault("ML_CACHE_ROOT", r"D:\ml_cache")
    device = torch.device(args.device)

    rng_k = np.random.default_rng(7)
    rng_o = np.random.default_rng(8)
    bk = make_vqa_batch(rng_k, batch=16, res=32, mix=["kinks"])
    bo = make_vqa_batch(rng_o, batch=16, res=32, mix=["ocr"])

    all_rows = {}
    for ck_name, ck_path in CKPTS.items():
        if not ck_path.exists():
            print(f"SKIP {ck_name}: missing {ck_path}", flush=True)
            continue
        print(f"\n=== {ck_name} ===", flush=True)
        model = load_model(ck_path, device)
        all_rows[ck_name] = {}
        img_k, pr_k = bk["image"].to(device), bk["prompt"]
        img_o, pr_o = bo["image"].to(device), bo["prompt"]
        for tag, write, gated in CONFIGS:
            apply_knobs(model, write, gated)
            print(f"  [{tag}] eval {args.batches} batches...", flush=True)
            ev = task_eval(model, device, n_batches=args.batches)
            mem_k = memory_trace(model, img_k, pr_k, device)
            mem_o = memory_trace(model, img_o, pr_o, device)
            ident = belief_identity(model, img_k, pr_k, device)
            all_rows[ck_name][tag] = {
                "write": write, "gate_h_local": gated,
                "eval": ev, "mem_kinks": mem_k, "mem_ocr": mem_o, "identity": ident,
            }
            print(
                f"    acc={ev['acc']*100:.1f} ocr={ev['ocr']*100:.1f} "
                f"kinks={ev['kinks']*100:.1f} color={ev['color']*100:.1f}  "
                f"cos_SW L3={mem_k['layers'][-1]['cos_reread_Swrite']:.3f}  "
                f"U={ [round(x['U'],2) for x in mem_k['layers']] }",
                flush=True,
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    OUT.write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    print(f"\nsaved {OUT}", flush=True)
    render(all_rows)

    # verdict lines
    print("\n=== GPT claims vs numbers ===", flush=True)
    if "champ_b" in all_rows:
        b_abs = all_rows["champ_b"]["abs_ungated"]["eval"]
        b_inc = all_rows["champ_b"]["inc_ungated"]["eval"]
        print(
            f"B weights, only swap write:  kinks {b_abs['kinks']*100:.1f} → {b_inc['kinks']*100:.1f}  "
            f"(ocr {b_abs['ocr']*100:.1f} → {b_inc['ocr']*100:.1f})",
            flush=True,
        )
        c_abs = all_rows["champ_b"]["abs_ungated"]["mem_kinks"]["layers"][-1]["cos_reread_Swrite"]
        c_inc = all_rows["champ_b"]["inc_ungated"]["mem_kinks"]["layers"][-1]["cos_reread_Swrite"]
        print(f"B Read(X')~S_write L3:  abs {c_abs:.3f}  inc {c_inc:.3f}", flush=True)
        id_abs = all_rows["champ_b"]["abs_ungated"]["identity"][-1]
        id_inc = all_rows["champ_b"]["inc_gated"]["identity"][-1]
        print(
            f"g=0 last layer: abs  dX={id_abs['dx']:.3f} dRead={id_abs['d_read']:.3f}  "
            f"inc  dX={id_inc['dx']:.3f} dRead={id_inc['d_read']:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
