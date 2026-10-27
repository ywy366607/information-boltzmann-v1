"""Shell-Kalman optimizer (KES) against SGD, Adam, AdaBelief, Muon, SOAP, RowRes.

Prereg: docs/information_boltzmann/CGB_OPTIMIZER_SHELL_KALMAN_PREREG_20261027.md
usage: python scripts/run_optimizer_shell_kalman.py --out results/published/optimizer_shell_kalman_20261027.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(4)
LRS = (1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3)
NS = (3.4445, -4.7750, 2.0315)
EPS = 1e-8
GAIN_FLOOR = 0.01      # registered in the prereg: an EMA horizon of 100 steps at most


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


class Basis:
    """SOAP / Shampoo eigenbasis of the left and right Kronecker factors (EMA 0.95, every 10 steps)."""

    def __init__(self, shape):
        K, D = shape
        self.L, self.R = torch.zeros(K, K), torch.zeros(D, D)
        self.QL, self.QR = torch.eye(K), torch.eye(D)
        self.t = 0

    def update(self, G):
        self.t += 1
        self.L.mul_(0.95).add_(G @ G.T, alpha=0.05)
        self.R.mul_(0.95).add_(G.T @ G, alpha=0.05)
        if self.t == 1 or self.t % 10 == 0:
            old = (self.QL, self.QR)
            self.QL = torch.linalg.eigh(self.L + 1e-12 * torch.eye(self.L.shape[0]))[1]
            self.QR = torch.linalg.eigh(self.R + 1e-12 * torch.eye(self.R.shape[0]))[1]
            return old
        return None

    def to(self, G):
        return self.QL.T @ G @ self.QR

    def back(self, U):
        return self.QL @ U @ self.QR.T

    def rebase(self, X, old):
        """Re-express a rotated-space quantity in the new basis."""
        QLo, QRo = old
        return self.QL.T @ (QLo @ X @ QRo.T) @ self.QR


class Optimizer:
    def __init__(self, name, lr, shape):
        self.name, self.lr, self.t = name, lr, 0
        self.s = {}
        self.basis = Basis(shape) if name in ("soap", "kes", "kes_ss") else None

    def z(self, key, like):
        if key not in self.s:
            self.s[key] = torch.zeros_like(like)
        return self.s[key]

    def step(self, W, G):
        self.t += 1
        t, n, lr = self.t, self.name, self.lr
        if n == "sgd":
            m = self.z("m", G).mul_(0.9).add_(G)
            W -= lr * m
        elif n == "adam":
            m = self.z("m", G).mul_(0.9).add_(G, alpha=0.1)
            v = self.z("v", G).mul_(0.999).add_(G * G, alpha=0.001)
            W -= lr * (m / (1 - 0.9 ** t)) / ((v / (1 - 0.999 ** t)).sqrt() + EPS)
        elif n == "adabelief":
            m = self.z("m", G).mul_(0.9).add_(G, alpha=0.1)
            s = self.z("s", G).mul_(0.999).add_((G - m) ** 2 + EPS, alpha=0.001)
            W -= lr * (m / (1 - 0.9 ** t)) / ((s / (1 - 0.999 ** t)).sqrt() + EPS)
        elif n == "muon":
            m = self.z("m", G).mul_(0.9).add_(G)
            W -= lr * newton_schulz(m) * max(1.0, W.shape[0] / W.shape[1]) ** 0.5
        elif n == "rowres":
            m = self.z("m", G).mul_(0.9).add_(G)
            if "r" not in self.s:
                self.s["r"] = torch.zeros(W.shape[0], 1)
            r = self.s["r"].mul_(0.95).add_(m.pow(2).sum(1, keepdim=True), alpha=0.05)
            W -= lr * m / (r.sqrt() + EPS)
        elif n == "soap":
            old = self.basis.update(G)
            g = self.basis.to(G)
            if old is not None and "m" in self.s:
                self.s["m"] = self.basis.rebase(self.s["m"], old)
            m = self.z("m", g).mul_(0.9).add_(g, alpha=0.1)
            v = self.z("v", g).mul_(0.95).add_(g * g, alpha=0.05)
            u = (m / (1 - 0.9 ** t)) / ((v / (1 - 0.95 ** t)).sqrt() + EPS)
            W -= lr * self.basis.back(u)
        elif n in ("kes", "kes_nr", "kes_ss"):
            if self.basis is not None:
                old = self.basis.update(G)
                g = self.basis.to(G)
                if old is not None and "m" in self.s:
                    for k in ("m", "gprev"):
                        self.s[k] = self.basis.rebase(self.s[k], old)
            else:
                g = G
            if t == 1:
                self.s.update(m=g.clone(), P=g * g + EPS, gprev=g.clone(), dprev=torch.zeros_like(g),
                              d2=torch.zeros_like(g), d11=torch.zeros_like(g), vm=torch.zeros_like(g))
            else:
                s = self.s
                d = g - s["gprev"]
                s["d2"].mul_(0.95).add_(d * d, alpha=0.05)
                s["d11"].mul_(0.95).add_(d * s["dprev"], alpha=0.05)
                c2 = 1 - 0.95 ** (t - 1)
                c11 = max(1 - 0.95 ** (t - 2), 0.05)
                r = (-s["d11"] / c11).clamp(min=0) + EPS
                q = (s["d2"] / c2 - 2 * r).clamp(min=0)
                if n == "kes_ss":          # steady state: P* = q/2 + sqrt(q^2/4 + q r), no shells
                    Pm = q / 2 + (q * q / 4 + q * r).sqrt()
                else:                      # time-varying posterior: evidence deepens the shell
                    Pm = s["P"] + q
                K = (Pm / (Pm + r)).clamp(min=GAIN_FLOOR, max=1.0)
                s["m"] += K * (g - s["m"])
                s["P"] = (1 - K) * Pm
                s["gprev"], s["dprev"] = g.clone(), d
            s = self.s
            s["vm"].mul_(0.95).add_(s["m"] * s["m"], alpha=0.05)
            u = s["m"] / ((s["vm"] / (1 - 0.95 ** t)) + s["P"] + EPS).sqrt()
            W -= lr * (self.basis.back(u) if self.basis is not None else u)
        else:
            raise ValueError(n)


def finite(x):
    return x if math.isfinite(x) else float("inf")


# ---------- task A: long-tail associative memory ----------
KA, DA, STEPS_A, BATCH_A = 1024, 64, 200, 256


def task_a(name, lr, seed, kind):
    gen = torch.Generator().manual_seed(seed)
    g = torch.randn(KA, DA, generator=gen)
    if kind == "correlated":
        c = torch.randn(1, DA, generator=gen)
        g = c / c.norm() * DA ** 0.5 + 0.7 * g
    E = F.normalize(g, dim=-1)
    p = torch.arange(1, KA + 1, dtype=torch.float).pow(-1.2)
    p = p / p.sum()
    W = torch.zeros(KA, DA)
    opt = Optimizer(name, lr, W.shape)
    for _ in range(STEPS_A):
        y = torch.multinomial(p, BATCH_A, replacement=True, generator=gen)
        Wg = W.clone().requires_grad_(True)
        F.cross_entropy(E[y] @ Wg.T, y).backward()
        with torch.no_grad():
            opt.step(W, Wg.grad)
    with torch.no_grad():
        logits = E @ W.T
        acc = (logits.argmax(1) == torch.arange(KA)).float()
        loss = float(F.cross_entropy(logits, torch.arange(KA), reduction="none") @ p)
    return dict(loss=finite(loss), head_acc=float(acc[: KA // 2].mean()), tail_acc=float(acc[KA // 2:].mean()))


# ---------- task B / B0: rotated regression with heterogeneous (or isotropic) SNR ----------
DB, STEPS_B, BATCH_B = 64, 500, 32


def rot(n, gen):
    q, r = torch.linalg.qr(torch.randn(n, n, generator=gen))
    return q * torch.sign(torch.diagonal(r))


def task_b(name, lr, seed, isotropic=False):
    gen = torch.Generator().manual_seed(1000 + seed)
    if isotropic:
        lam, sig = torch.ones(DB), torch.full((DB,), 0.3)
        U = V = torch.eye(DB)
    else:
        lam = torch.logspace(-2, 0, DB)
        sig = torch.logspace(math.log10(0.03), 0, DB)
        U, V = rot(DB, gen), rot(DB, gen)
    Sig = U @ torch.diag(lam) @ U.T
    Ws = torch.randn(DB, DB, generator=gen) / DB ** 0.5
    W = torch.zeros(DB, DB)
    opt = Optimizer(name, lr, W.shape)
    for _ in range(STEPS_B):
        X = U @ (lam.sqrt()[:, None] * torch.randn(DB, BATCH_B, generator=gen))
        Y = Ws @ X + V @ (sig[:, None] * torch.randn(DB, BATCH_B, generator=gen))
        G = (W @ X - Y) @ X.T / BATCH_B
        opt.step(W, G)
        if not torch.isfinite(W).all():
            return dict(loss=float("inf"))
    E = W - Ws
    return dict(loss=finite(float(0.5 * torch.trace(E @ Sig @ E.T))))


# ---------- task C: backbone rows + transient rows ----------
DC, STEPS_C, BATCH_C, NOISE_C = 32, 600, 32, 0.3


def task_c(name, lr, seed):
    gen = torch.Generator().manual_seed(2000 + seed)
    Ws = torch.randn(DC, DC, generator=gen) / DC ** 0.5
    W = torch.zeros(DC, DC)
    opt = Optimizer(name, lr, W.shape)
    back, trans = [], []
    for t in range(STEPS_C):
        Ws[:16] += 0.001 * torch.randn(16, DC, generator=gen)
        if t % 40 == 0:
            Ws[16:] = torch.randn(16, DC, generator=gen) / DC ** 0.5
        X = torch.randn(DC, BATCH_C, generator=gen)
        Y = Ws @ X + NOISE_C * torch.randn(DC, BATCH_C, generator=gen)
        opt.step(W, (W @ X - Y) @ X.T / BATCH_C)
        if not torch.isfinite(W).all():
            return dict(loss=float("inf"), backbone=float("inf"), transient=float("inf"))
        if t >= 200:
            E = W - Ws
            back.append(0.5 * float((E[:16] ** 2).sum()))
            trans.append(0.5 * float((E[16:] ** 2).sum()))
    return dict(loss=finite(float(np.mean(back) + np.mean(trans))), backbone=finite(float(np.mean(back))),
                transient=finite(float(np.mean(trans))))


TASKS = {
    "A_orthogonal": (lambda n, lr, s: task_a(n, lr, s, "orthogonal")),
    "A_correlated": (lambda n, lr, s: task_a(n, lr, s, "correlated")),
    "B_rotated_heterogeneous": (lambda n, lr, s: task_b(n, lr, s)),
    "B0_isotropic": (lambda n, lr, s: task_b(n, lr, s, isotropic=True)),
    "C_backbone_transient": task_c,
}
OPTS = ("sgd", "adam", "adabelief", "muon", "soap", "kes", "kes_nr", "kes_ss")


def summary(xs):
    xs = np.asarray(xs, dtype=float)
    return dict(min=float(xs.min()), median=float(np.median(xs)), mean=float(xs.mean()), max=float(xs.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default=",".join(TASKS))
    a = ap.parse_args()
    res = {}
    for task in a.tasks.split(","):
        fn = TASKS[task]
        opts = OPTS + (("rowres",) if task.startswith("A_") else ())
        res[task] = {}
        for n in opts:
            sweep = [(lr, fn(n, lr, 0)["loss"]) for lr in LRS]
            lr = min(sweep, key=lambda x: x[1])[0]
            runs = [fn(n, lr, s) for s in range(1, 6)]
            res[task][n] = dict(lr=lr, sweep=sweep, **{k: summary([r[k] for r in runs]) for k in runs[0]})
            print(task, n, json.dumps({k: (v["median"] if isinstance(v, dict) else v)
                                       for k, v in res[task][n].items() if k != "sweep"}), flush=True)
        Path(a.out).write_text(json.dumps(dict(prereg="docs/information_boltzmann/CGB_OPTIMIZER_SHELL_KALMAN_PREREG_20261027.md",
                                               results=res), indent=1))
    med = lambda t, n, k="loss": res[t][n][k]["median"]   # noqa: E731
    pred = {}
    if "B_rotated_heterogeneous" in res and "C_backbone_transient" in res:
        pred["P1"] = all(med(t, n) < med(t, "sgd") for t in ("B_rotated_heterogeneous", "C_backbone_transient")
                         for n in OPTS if n != "sgd")
    if "B_rotated_heterogeneous" in res:
        t = "B_rotated_heterogeneous"
        pred["P2"] = med(t, "soap") < med(t, "adam") and med(t, "kes") < med(t, "adam") and med(t, "kes") < med(t, "kes_nr")
        pred["P3"] = med(t, "kes") < med(t, "soap") and med(t, "kes") < med(t, "muon")
    if "C_backbone_transient" in res:
        t = "C_backbone_transient"
        pred["P4"] = med(t, "kes") < med(t, "kes_ss") and med(t, "kes", "backbone") < med(t, "kes_ss", "backbone")
    for kind in ("A_orthogonal", "A_correlated"):
        if kind in res:
            best = max(med(kind, n, "tail_acc") for n in ("adam", "muon", "soap"))
            pred[f"P5_{kind}"] = med(kind, "kes", "tail_acc") >= best - 0.03
    if "B0_isotropic" in res:
        t = "B0_isotropic"
        spread = res[t]["soap"]["loss"]["max"] - res[t]["soap"]["loss"]["min"]
        pred["P6"] = (med(t, "soap") - med(t, "kes")) <= spread
    print(json.dumps(pred), flush=True)
    Path(a.out).write_text(json.dumps(dict(prereg="docs/information_boltzmann/CGB_OPTIMIZER_SHELL_KALMAN_PREREG_20261027.md",
                                           gain_floor=GAIN_FLOOR, results=res, predictions=pred), indent=1))


if __name__ == "__main__":
    main()
