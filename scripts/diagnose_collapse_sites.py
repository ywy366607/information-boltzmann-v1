#!/usr/bin/env python3
"""Locate representation / routing collapse in Dual-Stream Native MoT.

Measures several independent sites on the same forward pass:

  param/Q_slice     learned language probes  Q ∈ R^{M×d}
  act/H_ctx         Q attending into language H
  act/S_hat         language prior over slices (S_hat or μ_p)
  act/S_stiefel     visual slices after NS, *before* SliceRead.out
  act/S_read        visual slices after SliceRead.out (the Linear can undo Stiefel)
  act/S_mot         visual slices after MoT
  act/H             language hidden states
  act/X             full-res point field
  route/w           assignment mass (PR_mass / r99 / H_point)

Usage:
  python scripts/diagnose_collapse_sites.py --device cuda --batches 4
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fine_grain.vlm_data import make_vqa_batch
from scripts.run_v0_surprise_eval import DualStreamVQAModel


def _svdvals(A: torch.Tensor) -> torch.Tensor:
    """A: [n, d] → singular values (float32, no grad)."""
    A = A.detach().float()
    if A.ndim != 2 or min(A.shape) == 0:
        return torch.zeros(0)
    return torch.linalg.svdvals(A)


def geom_of_tokens(tok: torch.Tensor) -> Dict[str, float]:
    """tok: [..., M, d]  — treat last two dims as M tokens in R^d.

    Metrics:
      erank     Roy–Vetterli effective rank of the M×d matrix (mean over batch)
      pr        participation ratio (∑σ²)² / ∑σ⁴
      top1      σ1² / ∑σ²   (1 = rank-1 collapse)
      cos       mean off-diagonal |cosine|
      nrm_cv    std/mean of token L2 norms (scale collapse)
    """
    t = tok.detach().float()
    if t.ndim == 2:
        t = t.unsqueeze(0)
    t = t.reshape(-1, t.shape[-2], t.shape[-1])  # [B, M, d]
    B, M, d = t.shape
    if M < 2:
        return dict(erank=1.0, pr=1.0, top1=1.0, cos=float("nan"), nrm_cv=0.0, M=M, d=d)

    eranks, prs, top1s, coses, cvs = [], [], [], [], []
    for b in range(B):
        A = t[b]  # [M, d]
        s = _svdvals(A)
        p = (s * s).clamp_min(1e-12)
        p = p / p.sum()
        eranks.append(float(torch.exp(-(p * p.log()).sum())))
        prs.append(float((p.sum() ** 2) / (p.square().sum())))
        top1s.append(float(p[0]))
        n = A.norm(dim=-1)
        cvs.append(float((n.std() / n.mean().clamp_min(1e-8)).item()) if M > 1 else 0.0)
        u = A / n.clamp_min(1e-8).unsqueeze(-1)
        sim = u @ u.T
        eye = torch.eye(M, dtype=torch.bool, device=sim.device)
        coses.append(float(sim.abs().masked_select(~eye).mean()))
    return dict(
        erank=float(np.mean(eranks)),
        pr=float(np.mean(prs)),
        top1=float(np.mean(top1s)),
        cos=float(np.mean(coses)),
        nrm_cv=float(np.mean(cvs)),
        M=M,
        d=d,
    )


def route_stats(w: torch.Tensor) -> Dict[str, float]:
    """w: [B, N, M] assignment."""
    w = w.detach().float()
    B, N, M = w.shape
    mass = w.sum(1)
    p = mass / mass.sum(-1, keepdim=True).clamp_min(1e-8)
    pr_mass = float((1.0 / p.pow(2).sum(-1)).mean())
    h_mass = float((-(p * (p + 1e-8).log()).sum(-1) / math.log(M)).mean())
    ps, _ = p.sort(dim=-1, descending=True)
    cume = ps.cumsum(-1)
    hit = cume >= 0.99
    idx = hit.float().argmax(dim=-1)
    none = ~hit.any(dim=-1)
    r99 = (idx + 1).float()
    r99[none] = float(M)
    h_point = float((-(w * (w + 1e-8).log()).sum(-1) / math.log(M)).mean())
    return dict(PR_mass=pr_mass, H_mass=h_mass, r99=float(r99.mean()), H_point=h_point, M=M)


class Capture:
    def __init__(self):
        self.by_layer: List[Dict[str, torch.Tensor]] = []
        self.hooks: List[torch.utils.hooks.RemovableHook] = []

    def clear_acts(self):
        self.by_layer = []

    def close(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def attach(self, model: DualStreamVQAModel):
        layers = model.mot_stack.layers
        self.by_layer = [{} for _ in layers]

        for i, layer in enumerate(layers):
            def make_out_hook(idx):
                def hook(mod, inp, out):
                    # SliceRead.out: inp = post-Stiefel S, out = after Linear
                    if idx >= len(self.by_layer):
                        return
                    self.by_layer[idx]["S_stiefel"] = inp[0].detach()
                    self.by_layer[idx]["S_read"] = out.detach()
                return hook

            def make_mot_hook(idx):
                def hook(mod, inp, out):
                    S2, H2, _ = out
                    self.by_layer[idx]["S_mot"] = S2.detach()
                    self.by_layer[idx]["H"] = H2.detach()
                return hook

            def make_gate_hook(idx):
                def hook(mod, inp, out):
                    S, H = inp[0], inp[1]
                    gate, meta = out
                    self.by_layer[idx]["S_obs"] = S.detach()
                    self.by_layer[idx]["H_in"] = H.detach()
                    if hasattr(mod, "slice_queries"):
                        self.by_layer[idx]["Q"] = mod.slice_queries.detach()[0]
                    sh = meta.get("s_hat", meta.get("mu_p", None))
                    if sh is not None:
                        self.by_layer[idx]["S_hat"] = sh.detach()
                    if hasattr(mod, "_predict_prior_from_h") and hasattr(mod, "slice_queries"):
                        with torch.no_grad():
                            self.by_layer[idx]["H_ctx"] = mod._predict_prior_from_h(H).detach()
                    if getattr(layer, "last_w", None) is not None:
                        self.by_layer[idx]["w"] = layer.last_w.detach()
                    self.by_layer[idx]["X"] = layer.last_X.detach() if getattr(layer, "last_X", None) is not None else None
                return hook

            self.hooks.append(layer.read.out.register_forward_hook(make_out_hook(i)))
            self.hooks.append(layer.mot.register_forward_hook(make_mot_hook(i)))
            self.hooks.append(layer.surprise_gate.register_forward_hook(make_gate_hook(i)))

        def make_readout_hook(store):
            def hook(mod, inp, out):
                store["S_readout"] = out.detach()
            return hook

        self.readout_box: Dict[str, torch.Tensor] = {}
        self.hooks.append(model.mot_stack.readout.out.register_forward_hook(make_readout_hook(self.readout_box)))


def param_Q_stats(model: DualStreamVQAModel) -> List[Dict[str, float]]:
    rows = []
    for i, layer in enumerate(model.mot_stack.layers):
        g = layer.surprise_gate
        if not hasattr(g, "slice_queries"):
            rows.append({"layer": i, "present": False})
            continue
        Q = g.slice_queries.detach()[0]  # [M, d]
        gstat = geom_of_tokens(Q)
        gstat["layer"] = i
        gstat["present"] = True
        rows.append(gstat)
    return rows


CKPTS = [
    ("init_jepa", None, "v0_jepa"),
    ("init_bayes", None, "v1_bayes"),
    ("baseline_600", "checkpoints/baseline_best.pt", "baseline"),
    ("v0_jepa_600", "checkpoints/v0_jepa_best.pt", "v0_jepa"),
    ("v1_bayes_600", "checkpoints/v1_bayes_best.pt", "v1_bayes"),
    ("v1_bayes_2k_r1", "checkpoints/v1_bayes_2000step_run1_best.pt", "v1_bayes"),
    ("v1_bayes_2k_r2", "checkpoints/v1_bayes_2000step_run2_best.pt", "v1_bayes"),
    ("v1_bayes_2k_r3", "checkpoints/v1_bayes_2000step_run3_best.pt", "v1_bayes"),
]


def load_model(ckpt: Optional[str], mode: str, device: torch.device) -> DualStreamVQAModel:
    model = DualStreamVQAModel(
        d_model=128, n_slices=32, n_layers=4, res=32,
        surprise_mode=mode, surprise_beta=1.5,
    ).to(device)
    if ckpt is not None:
        path = ROOT / ckpt
        sd = torch.load(path, map_location=device)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f"  [warn] {ckpt} missing {len(missing)} keys (first 4): {missing[:4]}")
        if unexpected:
            print(f"  [warn] {ckpt} unexpected {len(unexpected)} keys (first 4): {unexpected[:4]}")
    model.eval()
    return model


def run_arm(name: str, ckpt: Optional[str], mode: str, device: torch.device, batches: int, seed: int = 0) -> Dict:
    print(f"\n=== {name}  mode={mode}  ckpt={ckpt or 'FRESH INIT'} ===", flush=True)
    model = load_model(ckpt, mode, device)
    q_param = param_Q_stats(model)

    cap = Capture()
    cap.attach(model)
    rng = np.random.default_rng(seed + 12345)

    accums: Dict[Tuple[int, str], List[Dict[str, float]]] = {}
    route_acc: Dict[int, List[Dict[str, float]]] = {}
    readout_acc: List[Dict[str, float]] = []

    with torch.no_grad():
        for _ in range(batches):
            batch = make_vqa_batch(rng, batch=16, res=32, mix=["ocr", "kinks", "color"])
            imgs = batch["image"].to(device)
            cap.clear_acts()
            cap.by_layer = [{} for _ in model.mot_stack.layers]
            cap.readout_box.clear()
            _ = model(imgs, batch["prompt"])
            # last_w / last_X are written *after* surprise_gate, so refill w/X
            for i, layer in enumerate(model.mot_stack.layers):
                if getattr(layer, "last_w", None) is not None:
                    cap.by_layer[i]["w"] = layer.last_w.detach()
                if getattr(layer, "last_X", None) is not None:
                    cap.by_layer[i]["X"] = layer.last_X.detach()

            for i, box in enumerate(cap.by_layer):
                for key in ("Q", "H_ctx", "S_hat", "S_stiefel", "S_read", "S_mot", "H", "X"):
                    t = box.get(key)
                    if t is None:
                        continue
                    accums.setdefault((i, key), []).append(geom_of_tokens(t))
                if "w" in box and box["w"] is not None:
                    route_acc.setdefault(i, []).append(route_stats(box["w"]))
            if "S_readout" in cap.readout_box:
                readout_acc.append(geom_of_tokens(cap.readout_box["S_readout"]))

    cap.close()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    def mean_dict(xs: List[Dict[str, float]]) -> Dict[str, float]:
        if not xs:
            return {}
        keys = [k for k in xs[0] if isinstance(xs[0][k], (int, float))]
        return {k: float(np.mean([x[k] for x in xs])) for k in keys}

    layers_out = []
    n_layers = 4
    for i in range(n_layers):
        row = {"layer": i, "Q_param": q_param[i] if i < len(q_param) else {}}
        for key in ("Q", "H_ctx", "S_hat", "S_stiefel", "S_read", "S_mot", "H", "X"):
            row[key] = mean_dict(accums.get((i, key), []))
        row["route"] = mean_dict(route_acc.get(i, []))
        layers_out.append(row)

    return {
        "name": name,
        "mode": mode,
        "ckpt": ckpt,
        "layers": layers_out,
        "S_readout": mean_dict(readout_acc),
    }


def fmt(x, nd=2):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "  n/a"
    return f"{x:6.{nd}f}"


def print_report(results: List[Dict]):
    print("\n" + "=" * 108)
    print("COLLAPSE MAP  (erank → M is healthy; top1 → 1 and |cos| → 1 is collapse)")
    print("Sites: Q_slice (lang probes) | H_ctx (Q→H) | S_hat (lang prior)")
    print("       S_st (visual after Stiefel) | S_rd (after Linear out) | S_mot | H | X | route")
    print("=" * 108)

    header = (
        f"{'arm':<16} {'L':>1} "
        f"{'Q.er':>6} {'Q.t1':>5} {'Q.cos':>6}  "
        f"{'Hctx':>6} {'Shat':>6}  "
        f"{'Sst':>6} {'Srd':>6} {'Smot':>6}  "
        f"{'H':>6} {'X':>6}  "
        f"{'PRm':>5} {'r99':>5}"
    )
    print(header)
    print("-" * 108)

    for res in results:
        for row in res["layers"]:
            def er(k):
                return row.get(k, {}).get("erank")

            def t1(k):
                return row.get(k, {}).get("top1")

            def cs(k):
                return row.get(k, {}).get("cos")

            qp = row.get("Q_param") or {}
            q_er = qp.get("erank", er("Q"))
            q_t1 = qp.get("top1", t1("Q"))
            q_cs = qp.get("cos", cs("Q"))
            rt = row.get("route") or {}
            print(
                f"{res['name']:<16} {row['layer']:>1} "
                f"{fmt(q_er)} {fmt(q_t1, 2)} {fmt(q_cs, 2)}  "
                f"{fmt(er('H_ctx'))} {fmt(er('S_hat'))}  "
                f"{fmt(er('S_stiefel'))} {fmt(er('S_read'))} {fmt(er('S_mot'))}  "
                f"{fmt(er('H'))} {fmt(er('X'))}  "
                f"{fmt(rt.get('PR_mass'), 1)} {fmt(rt.get('r99'), 1)}"
            )
        rd = res.get("S_readout") or {}
        if rd:
            print(f"{res['name']:<16} R {fmt(rd.get('erank'))}  readout S  top1={fmt(rd.get('top1'),2)}  cos={fmt(rd.get('cos'),2)}")
        print()

    print("Legend: Q.er = language-probe effective rank (max 32).  Q.t1 = top singular-energy fraction.")
    print("        Sst/Srd/Smot = visual-slice erank after Stiefel / Linear / MoT.")
    print("        PRm / r99 = assignment participation-ratio / slices holding 99% mass.")
    print("        Collapse signature: erank ≈ 1, top1 ≈ 1, |cos| ≈ 1.  Healthy: erank ≫ 1, |cos| small.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batches", type=int, default=4)
    ap.add_argument("--out", default="results/published/collapse_sites.json")
    args = ap.parse_args()
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    print(f"device={device}  batches={args.batches}", flush=True)

    results = []
    for name, ckpt, mode in CKPTS:
        if ckpt is not None and not (ROOT / ckpt).exists():
            print(f"[skip] missing {ckpt}")
            continue
        results.append(run_arm(name, ckpt, mode, device, args.batches))

    print_report(results)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
