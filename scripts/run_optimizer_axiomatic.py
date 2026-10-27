"""AXO: the optimizer fixed by the axiomatic derivation, against SGD, Adam, Muon, SOAP, RowRes.

Prereg: docs/information_boltzmann/CGB_OPTIMIZER_AXIOMATIC_DERIVATION_20261027.md (第二版预注册)
usage: python scripts/run_optimizer_axiomatic.py --out results/published/optimizer_axiomatic_20261027.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_optimizer_shell_kalman as V1  # noqa: E402  (baselines, unchanged)

torch.set_num_threads(4)
LRS = (1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
EPS = 1e-12
OBS_FRACTION = 0.1      # registered: a row "occurs" when its gradient norm >= 0.1 x the layer's max
TRUST_ACCEPT = 0.5      # registered: success when actual >= 0.5 x predicted decrease
TRUST_GAMMA = 2.0       # registered
RHO0 = 1e-3             # registered starting bound


class AXO:
    """Per matrix (one clique): input-side eigenbasis, per-direction filter with an occurrence
    clock, least-squares secant curvature, step (m/|h|) * max(m^2 - P, 0) / m^2 clipped to rho,
    rho from realized / predicted decrease on the same batch. No momentum."""

    def __init__(self, shape, *, basis=True, steady=False, clock=True, fixed_rho=None):
        K, D = shape
        self.basis, self.steady, self.clock, self.fixed = basis, steady, clock, fixed_rho
        self.rho = RHO0 if fixed_rho is None else fixed_rho
        self.Rf = torch.zeros(D, D)
        self.Q = torch.eye(D)
        self.t = 0
        z = lambda: torch.zeros(K, D)  # noqa: E731
        self.m, self.P, self.gprev, self.dprev = z(), z(), z(), z()
        self.d2, self.d11, self.sxy, self.sxx = z(), z(), z(), z()
        self.th_prev, self.m_prev = z(), z()
        self.n = torch.zeros(K, 1)          # observations per row (occurrence clock)
        self.nd = torch.zeros(K, 1)         # observations since the last reset of difference stats
        self.evals = 0                      # extra forward passes (trust check)

    def _rebase(self, Qn):
        T = self.Q.T @ Qn
        for k in ("m", "gprev", "th_prev", "m_prev", "dprev"):
            setattr(self, k, getattr(self, k) @ T)
        self.d2.zero_()
        self.d11.zero_()
        self.nd.zero_()
        self.Q = Qn

    def step(self, W, G, closure):
        self.t += 1
        if self.basis:
            self.Rf.mul_(0.95).add_(G.T @ G, alpha=0.05)
            if self.t == 1 or self.t % 10 == 0:
                self._rebase(torch.linalg.eigh(self.Rf + EPS * torch.eye(self.Rf.shape[0]))[1])
        g = G @ self.Q
        th = W @ self.Q
        rn = G.norm(dim=1, keepdim=True)
        obs = (rn >= OBS_FRACTION * rn.max()) if self.clock else torch.ones_like(rn, dtype=torch.bool)
        obs = obs.expand_as(g)
        first = obs & (self.n == 0)
        later = obs & (self.n > 0)
        # first observation of a row: the estimate starts at the observation, fully uncertain
        self.m = torch.where(first, g, self.m)
        self.P = torch.where(first, g * g + EPS, self.P)
        # later observations: Kalman with noise and drift from difference statistics
        d = g - self.gprev
        has_prev = later & (self.nd > 0)
        self.d2 = torch.where(has_prev, 0.95 * self.d2 + 0.05 * d * d, self.d2)
        self.d11 = torch.where(has_prev & (self.nd > 1), 0.95 * self.d11 + 0.05 * d * self.dprev, self.d11)
        c2 = (1 - 0.95 ** self.nd).clamp(min=0.05)
        c11 = (1 - 0.95 ** (self.nd - 1)).clamp(min=0.05)
        r = (-self.d11 / c11).clamp(min=0) + EPS
        q = (self.d2 / c2 - 2 * r).clamp(min=0)
        Pm = (q / 2 + (q * q / 4 + q * r).sqrt()) if self.steady else (self.P + q)
        Kg = Pm / (Pm + r)
        self.m = torch.where(later, self.m + Kg * (g - self.m), self.m)
        self.P = torch.where(later, (1 - Kg) * Pm, self.P)
        # least-squares secant curvature between observations
        dm, dth = self.m - self.m_prev, th - self.th_prev
        upd = later
        self.sxy = torch.where(upd, 0.9 * self.sxy + 0.1 * dm * dth, self.sxy)
        self.sxx = torch.where(upd, 0.9 * self.sxx + 0.1 * dth * dth, self.sxx)
        self.dprev = torch.where(has_prev, d, self.dprev)
        self.gprev = torch.where(obs, g, self.gprev)
        self.m_prev = torch.where(obs, self.m, self.m_prev)
        self.th_prev = torch.where(obs, th, self.th_prev)
        rows = obs[:, :1].float()
        self.n += rows
        self.nd += rows
        # step: curvature scale x signal fraction, bounded by rho
        h = (self.sxy / (self.sxx + EPS)).abs().clamp(min=EPS)
        frac = (self.m * self.m - self.P).clamp(min=0) / (self.m * self.m + EPS)
        u = (self.m / h * frac).clamp(-self.rho, self.rho)
        if self.fixed is not None:
            W -= u @ self.Q.T
            return
        pred = float((self.m * u - 0.5 * h * u * u).sum())
        before = closure(W)
        W -= u @ self.Q.T
        after = closure(W)
        self.evals += 2
        actual = before - after
        ok = pred > 0 and actual >= TRUST_ACCEPT * pred
        self.rho = self.rho * TRUST_GAMMA if ok else self.rho / TRUST_GAMMA


def make(name, lr, shape):
    if name.startswith("axo"):
        kw = dict(axo={}, axo_diag=dict(basis=False), axo_steady=dict(steady=True),
                  axo_noclock=dict(clock=False), axo_fixedrho=dict(fixed_rho=lr))[name]
        return AXO(shape, **kw)
    opt = V1.Optimizer(name, lr, shape)
    return type("Wrap", (), dict(step=lambda self, W, G, closure: opt.step(W, G), evals=0))()


# ---------- tasks with a same-batch loss closure ----------
def task_a(name, lr, seed, kind):
    gen = torch.Generator().manual_seed(seed)
    g = torch.randn(V1.KA, V1.DA, generator=gen)
    if kind == "correlated":
        c = torch.randn(1, V1.DA, generator=gen)
        g = c / c.norm() * V1.DA ** 0.5 + 0.7 * g
    E = F.normalize(g, dim=-1)
    p = torch.arange(1, V1.KA + 1, dtype=torch.float).pow(-1.2)
    p = p / p.sum()
    W = torch.zeros(V1.KA, V1.DA)
    opt = make(name, lr, W.shape)
    for _ in range(V1.STEPS_A):
        y = torch.multinomial(p, V1.BATCH_A, replacement=True, generator=gen)
        Wg = W.clone().requires_grad_(True)
        F.cross_entropy(E[y] @ Wg.T, y).backward()
        with torch.no_grad():
            opt.step(W, Wg.grad, lambda Wc: float(F.cross_entropy(E[y] @ Wc.T, y)))
        if not torch.isfinite(W).all():
            return dict(loss=float("inf"), head_acc=0.0, tail_acc=0.0, evals=opt.evals)
    with torch.no_grad():
        logits = E @ W.T
        acc = (logits.argmax(1) == torch.arange(V1.KA)).float()
        loss = float(F.cross_entropy(logits, torch.arange(V1.KA), reduction="none") @ p)
    return dict(loss=V1.finite(loss), head_acc=float(acc[: V1.KA // 2].mean()),
                tail_acc=float(acc[V1.KA // 2:].mean()), evals=opt.evals)


def task_b(name, lr, seed, isotropic=False):
    gen = torch.Generator().manual_seed(1000 + seed)
    DB = V1.DB
    if isotropic:
        lam, sig = torch.ones(DB), torch.full((DB,), 0.3)
        U = Vr = torch.eye(DB)
    else:
        lam = torch.logspace(-2, 0, DB)
        sig = torch.logspace(math.log10(0.03), 0, DB)
        U, Vr = V1.rot(DB, gen), V1.rot(DB, gen)
    Sig = U @ torch.diag(lam) @ U.T
    Ws = torch.randn(DB, DB, generator=gen) / DB ** 0.5
    W = torch.zeros(DB, DB)
    opt = make(name, lr, W.shape)
    for _ in range(V1.STEPS_B):
        X = U @ (lam.sqrt()[:, None] * torch.randn(DB, V1.BATCH_B, generator=gen))
        Y = Ws @ X + Vr @ (sig[:, None] * torch.randn(DB, V1.BATCH_B, generator=gen))
        G = (W @ X - Y) @ X.T / V1.BATCH_B
        opt.step(W, G, lambda Wc: float(0.5 * ((Wc @ X - Y) ** 2).sum(0).mean()))
        if not torch.isfinite(W).all():
            return dict(loss=float("inf"), evals=opt.evals)
    E = W - Ws
    return dict(loss=V1.finite(float(0.5 * torch.trace(E @ Sig @ E.T))), evals=opt.evals)


def task_c(name, lr, seed):
    gen = torch.Generator().manual_seed(2000 + seed)
    DC = V1.DC
    Ws = torch.randn(DC, DC, generator=gen) / DC ** 0.5
    W = torch.zeros(DC, DC)
    opt = make(name, lr, W.shape)
    back, trans = [], []
    for t in range(V1.STEPS_C):
        Ws[:16] += 0.001 * torch.randn(16, DC, generator=gen)
        if t % 40 == 0:
            Ws[16:] = torch.randn(16, DC, generator=gen) / DC ** 0.5
        X = torch.randn(DC, V1.BATCH_C, generator=gen)
        Y = Ws @ X + V1.NOISE_C * torch.randn(DC, V1.BATCH_C, generator=gen)
        opt.step(W, (W @ X - Y) @ X.T / V1.BATCH_C, lambda Wc: float(0.5 * ((Wc @ X - Y) ** 2).sum(0).mean()))
        if not torch.isfinite(W).all():
            return dict(loss=float("inf"), backbone=float("inf"), transient=float("inf"), evals=opt.evals)
        if t >= 200:
            E = W - Ws
            back.append(0.5 * float((E[:16] ** 2).sum()))
            trans.append(0.5 * float((E[16:] ** 2).sum()))
    return dict(loss=V1.finite(float(np.mean(back) + np.mean(trans))), backbone=V1.finite(float(np.mean(back))),
                transient=V1.finite(float(np.mean(trans))), evals=opt.evals)


TASKS = {
    "A_orthogonal": (lambda n, lr, s: task_a(n, lr, s, "orthogonal")),
    "A_correlated": (lambda n, lr, s: task_a(n, lr, s, "correlated")),
    "B_rotated_heterogeneous": (lambda n, lr, s: task_b(n, lr, s)),
    "B0_isotropic": (lambda n, lr, s: task_b(n, lr, s, isotropic=True)),
    "C_backbone_transient": task_c,
}
BASE = ("sgd", "adam", "muon", "soap")
UNTUNED = ("axo", "axo_diag", "axo_steady", "axo_noclock")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default=",".join(TASKS))
    a = ap.parse_args()
    res = {}
    for task in a.tasks.split(","):
        fn = TASKS[task]
        res[task] = {}
        tuned = BASE + (("rowres",) if task.startswith("A_") else ()) + ("axo_fixedrho",)
        for n in tuned + UNTUNED:
            if n in UNTUNED:
                lr, sweep = None, None
            else:
                sweep = [(lr_, fn(n, lr_, 0)["loss"]) for lr_ in LRS]
                lr = min(sweep, key=lambda x: x[1])[0]
            runs = [fn(n, lr, s) for s in range(1, 6)]
            res[task][n] = dict(lr=lr, sweep=sweep, **{k: V1.summary([r[k] for r in runs]) for k in runs[0]})
            print(task, n, json.dumps({k: (v["median"] if isinstance(v, dict) else v)
                                       for k, v in res[task][n].items() if k != "sweep"}), flush=True)
        Path(a.out).write_text(json.dumps(dict(results=res), indent=1))
    med = lambda t, n, k="loss": res[t][n][k]["median"]   # noqa: E731
    pred = {}
    t = "B_rotated_heterogeneous"
    pred["Q1"] = med(t, "axo") < med(t, "adam") and med(t, "axo") < med(t, "axo_diag")
    t = "C_backbone_transient"
    pred["Q2"] = med(t, "axo") < med(t, "axo_steady") and med(t, "axo", "backbone") <= med(t, "adam", "backbone")
    for kind in ("A_orthogonal", "A_correlated"):
        best = max(med(kind, n, "tail_acc") for n in BASE + ("rowres",))
        pred[f"Q3_{kind}"] = (med(kind, "axo", "tail_acc") >= med(kind, "axo_noclock", "tail_acc")
                              and med(kind, "axo", "tail_acc") >= best - 0.03)
    for t in ("A_orthogonal", "A_correlated", "B_rotated_heterogeneous", "C_backbone_transient"):
        best = min(med(t, n) for n in BASE + (("rowres",) if t.startswith("A_") else ()))
        pred[f"Q4_{t}"] = med(t, "axo") <= 1.5 * best
        pred[f"Q5_{t}"] = med(t, "axo") <= 1.2 * med(t, "axo_fixedrho")
    t = "B0_isotropic"
    pred["Q6"] = (med(t, "soap") - med(t, "axo")) <= (res[t]["soap"]["loss"]["max"] - res[t]["soap"]["loss"]["min"])
    print(json.dumps(pred), flush=True)
    Path(a.out).write_text(json.dumps(dict(prereg="docs/information_boltzmann/CGB_OPTIMIZER_AXIOMATIC_DERIVATION_20261027.md",
                                           registered=dict(obs_fraction=OBS_FRACTION, trust_accept=TRUST_ACCEPT,
                                                           trust_gamma=TRUST_GAMMA, rho0=RHO0),
                                           results=res, predictions=pred), indent=1))


if __name__ == "__main__":
    main()
