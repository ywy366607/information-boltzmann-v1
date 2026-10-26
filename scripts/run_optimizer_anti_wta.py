"""Is Muon's gain anti-winner-take-all on the competing axis? Long-tail linear associative memory.

Prereg: docs/information_boltzmann/CGB_OPTIMIZER_ANTI_WTA_PREREG_20261026.md
usage: python scripts/run_optimizer_anti_wta.py --out results/published/optimizer_anti_wta_20261026.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

K, D, STEPS, BATCH = 256, 128, 400, 256
LRS = (0.01, 0.03, 0.1, 0.3, 1.0)
NS = (3.4445, -4.7750, 2.0315)


def embeddings(kind, gen):
    g = torch.randn(K, D, generator=gen)
    if kind == "correlated":
        c = torch.randn(1, D, generator=gen)
        g = c / c.norm() * D ** 0.5 + 0.7 * g
    return F.normalize(g, dim=-1)


def newton_schulz(G, steps=5):
    a, b, c = NS
    X = G / (G.norm() + 1e-7)
    tall = X.shape[0] > X.shape[1]
    if tall:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X.T if tall else X


def polar(G):
    U, _, Vh = torch.linalg.svd(G, full_matrices=False)
    return U @ Vh


def step_fn(name, lr):
    state = {}

    def step(W, G):
        if name == "adam":
            m = state.setdefault("m", torch.zeros_like(W))
            v = state.setdefault("v", torch.zeros_like(W))
            state["t"] = state.get("t", 0) + 1
            m.mul_(0.9).add_(G, alpha=0.1)
            v.mul_(0.999).add_(G * G, alpha=0.001)
            mh, vh = m / (1 - 0.9 ** state["t"]), v / (1 - 0.999 ** state["t"])
            W -= lr * mh / (vh.sqrt() + 1e-8)
            return
        mom = state.setdefault("mom", torch.zeros_like(W))
        mom.mul_(0.9).add_(G)
        if name == "sgd":
            u = mom
        elif name == "muon_ns":
            u = newton_schulz(mom) * max(1.0, W.shape[0] / W.shape[1]) ** 0.5
        elif name == "muon_svd":
            u = polar(mom) * max(1.0, W.shape[0] / W.shape[1]) ** 0.5
        elif name == "rownorm":
            u = mom / (mom.norm(dim=1, keepdim=True) + 1e-8)
        elif name == "rowres":
            r = state.setdefault("r", torch.zeros(W.shape[0], 1))
            r.mul_(0.95).add_(mom.pow(2).sum(1, keepdim=True), alpha=0.05)
            u = mom / (r.sqrt() + 1e-8)
        W -= lr * u
    return step


def run(name, kind, lr, seed=0):
    gen = torch.Generator().manual_seed(seed)
    E = embeddings(kind, gen)
    p = torch.arange(1, K + 1, dtype=torch.float).pow(-1.2)
    p = p / p.sum()
    W = torch.zeros(K, D)
    step = step_fn(name, lr)
    for _ in range(STEPS):
        y = torch.multinomial(p, BATCH, replacement=True, generator=gen)
        Wg = W.clone().requires_grad_(True)
        loss = F.cross_entropy(E[y] @ Wg.T, y)
        loss.backward()
        with torch.no_grad():
            step(W, Wg.grad)
    with torch.no_grad():
        logits = E @ W.T
        acc = (logits.argmax(1) == torch.arange(K)).float()
        total = float(F.cross_entropy(logits, torch.arange(K), reduction="none") @ p)
        s = torch.linalg.svdvals(W)
        q = s / s.sum()
        erank = float(torch.exp(-(q * torch.log(q + 1e-12)).sum()))
    return dict(opt=name, emb=kind, lr=lr, loss=total, head_acc=float(acc[: K // 2].mean()),
                tail_acc=float(acc[K // 2:].mean()), erank=erank)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    best = []
    for kind in ("orthogonal", "correlated"):
        for name in ("sgd", "adam", "muon_ns", "muon_svd", "rownorm", "rowres"):
            runs = [run(name, kind, lr) for lr in LRS]
            b = min(runs, key=lambda r: r["loss"])
            b["lr_sweep"] = [dict(lr=r["lr"], loss=r["loss"], tail_acc=r["tail_acc"]) for r in runs]
            best.append(b)
            print(json.dumps({k: v for k, v in b.items() if k != "lr_sweep"}), flush=True)
    get = {(r["emb"], r["opt"]): r["tail_acc"] for r in best}
    pred = {}
    for kind in ("orthogonal", "correlated"):
        pred[f"O1_{kind}"] = abs(get[(kind, "muon_ns")] - get[(kind, "muon_svd")]) < 0.03
        pred[f"O4_{kind}"] = get[(kind, "muon_ns")] > get[(kind, "sgd")] and get[(kind, "rownorm")] > get[(kind, "sgd")]
        pred[f"O5_{kind}"] = get[(kind, "rowres")] >= get[(kind, "rownorm")] - 0.02
    pred["O2"] = get[("orthogonal", "rownorm")] >= get[("orthogonal", "muon_ns")] - 0.03
    pred["O3"] = get[("correlated", "rownorm")] < get[("correlated", "muon_ns")] - 0.05
    print(json.dumps(pred))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(dict(prereg="docs/information_boltzmann/CGB_OPTIMIZER_ANTI_WTA_PREREG_20261026.md",
                                           best=best, predictions=pred), indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
