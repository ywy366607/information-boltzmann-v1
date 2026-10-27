"""CLO: the continual optimizer derived in GenerationGradientFilter + GenerationContinualOptimizer.

Prereg: docs/information_boltzmann/CGB_OPTIMIZER_AXIOMATIC_DERIVATION_20261027.md (第三版预注册)
usage: python scripts/run_optimizer_continual.py --out results/published/optimizer_continual_20261027.json
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
import run_optimizer_shell_kalman as V1  # noqa: E402  (baselines and v1 tasks, unchanged)

torch.set_num_threads(4)
LRS = (1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
EPS = 1e-12
# registered choices (no theorem fixes them)
OBS_FRACTION = 0.1      # a row occurs when its gradient norm >= 0.1 x the layer's max row norm
TRUST_ACCEPT = 0.5      # success when the fresh-batch decrease >= 0.5 x predicted
TRUST_GAMMA = 2.0
RHO0 = 1e-3
RESOLUTION = 1e-5       # float floor: a predicted decrease below 1e-5 x |loss| carries no evidence
EVIDENCE_Z = 2.0        # and a prediction within 2 sd of the fresh-batch measurement noise carries none
WINDOW = 8              # latch window c (consecutive observations)
Z2 = 4.0                # signal detected when m^2 > 4 P (two posterior standard deviations)
RES_RATE = 0.05         # row reservoir: recovery and use rates (G22a)
SINKHORN_ITERS = 10


class CLO:
    """Per matrix (one clique), in the eigenbasis of the pre-activity covariance:
    per-element filter with occurrence clock; every consequence of the own step is measured on
    the next batch, which the step did not see: the gradient at the previous parameters on the
    current batch gives the efference copy g_t(th_t) - g_t(th_{t-1}), the secant, the noise
    residual g_t(th_{t-1}) - g_{t-1}(th_{t-1}) and the fresh score of the step. Then step
    (m/|h|) max(m^2-P,0)/m^2, consolidation latch (hold c, release c), allocation under a
    conserved per-direction budget and per-row reservoirs (Sinkhorn), bound rho scored on the
    next (fresh) batch with a resolution dead band. No momentum."""

    def __init__(self, shape, *, latch=True, exit=True, alloc="both", scoring="fresh"):
        K, D = shape
        self.latch, self.exit, self.alloc, self.scoring = latch, exit, alloc, scoring
        self.rho = RHO0
        self.Rx = torch.zeros(D, D)
        self.Q = torch.eye(D)
        self.t = 0
        z = lambda: torch.zeros(K, D)  # noqa: E731
        self.m, self.P = z(), z()
        self.gprev, self.thprev, self.eprev = z(), z(), z()
        self.e2, self.e11, self.sxy, self.sxx = z(), z(), z(), z()
        self.cons = torch.zeros(K, D, dtype=torch.bool)
        self.run = torch.zeros(K, D)
        self.n = torch.zeros(K, 1)
        self.nd = torch.zeros(K, 1)
        self.avail = torch.ones(K, 1)
        self.Wprev, self.pred, self.pred_sd, self.binding, self.umax = None, 0.0, 0.0, False, 0.0
        self.evals = 0
        self.stats = dict(consolidations=0, releases=0, rho_up=0, rho_down=0, rho_none=0)

    # ----- basis -----
    def _rebase(self, Qn):
        T = self.Q.T @ Qn
        # keep the identity of directions: match each old direction to its closest new one
        perm = torch.full((T.shape[0],), -1, dtype=torch.long)
        used = torch.zeros(T.shape[0], dtype=torch.bool)
        for j in torch.argsort(-T.abs().max(1).values):
            cand = T[j].abs().masked_fill(used, -1)
            k = int(cand.argmax())
            perm[j], used[k] = k, True
        Qn = Qn[:, perm]
        Qn = Qn * torch.sign(torch.diagonal(self.Q.T @ Qn)).clamp(min=0).mul(2).sub(1)
        T = self.Q.T @ Qn
        T2 = T * T
        for k in ("m", "gprev", "thprev", "eprev"):
            setattr(self, k, getattr(self, k) @ T)
        for k in ("P", "sxy", "sxx", "e2", "e11"):
            setattr(self, k, getattr(self, k) @ T2)
        self.Q = Qn

    def _trust(self, actual, pred, loss):
        # the fresh decrease is the prediction plus noise sum(u xi') with sd sqrt(sum u^2 r)
        if abs(pred) <= max(RESOLUTION * abs(loss), EVIDENCE_Z * self.pred_sd, EPS):
            self.stats["rho_none"] += 1
            return
        if pred > 0 and actual >= TRUST_ACCEPT * pred:
            if self.binding:                    # only a binding bound learns from success
                self.rho *= TRUST_GAMMA
                self.stats["rho_up"] += 1
        else:                                   # below the step that failed
            self.rho = min(self.rho, self.umax) / TRUST_GAMMA
            self.stats["rho_down"] += 1

    def step(self, W, G, X, fgrad):
        self.t += 1
        Gp = None
        if self.Wprev is not None:
            now = fgrad(W, False)[0]
            lp, Gp = fgrad(self.Wprev, True)
            self.evals += 2
            # global modulator: the previous step scored on this batch, which it did not see
            if self.scoring == "fresh":
                self._trust(lp - now, self.pred, now)
        # relation basis from the pre-activity covariance
        self.Rx.mul_(0.95).add_(X @ X.T / X.shape[1], alpha=0.05)
        if self.t == 1 or self.t % 10 == 0:
            self._rebase(torch.linalg.eigh(self.Rx + EPS * torch.eye(self.Rx.shape[0]))[1])
        g, th = G @ self.Q, W @ self.Q
        gp = Gp @ self.Q if Gp is not None else g
        thp = self.Wprev @ self.Q if self.Wprev is not None else th
        rn = G.norm(dim=1, keepdim=True)
        obs = (rn >= OBS_FRACTION * rn.max()).expand_as(g) if rn.max() > 0 else torch.zeros_like(g, dtype=torch.bool)
        first = obs & (self.n == 0)
        later = obs & (self.n > 0)
        # own step measured on the current batch: efference copy and secant (same noise at both points)
        own = g - gp
        dth = th - thp
        self.sxy = torch.where(later, 0.9 * self.sxy + 0.1 * own * dth, self.sxy)
        self.sxx = torch.where(later, 0.9 * self.sxx + 0.1 * dth * dth, self.sxx)
        h = self.sxy / (self.sxx + EPS)
        # noise and drift: new batch minus old batch at the same (previous) parameters
        e = gp - self.gprev
        has1 = later & (self.nd > 0)
        has2 = later & (self.nd > 1)
        self.e2 = torch.where(has1, 0.95 * self.e2 + 0.05 * e * e, self.e2)
        self.e11 = torch.where(has2, 0.95 * self.e11 + 0.05 * e * self.eprev, self.e11)
        c2 = (1 - 0.95 ** self.nd).clamp(min=0.05)
        c11 = (1 - 0.95 ** (self.nd - 1)).clamp(min=0.05)
        r = (-self.e11 / c11).clamp(min=0) + EPS
        q = (self.e2 / c2 - 2 * r).clamp(min=0)
        # Kalman with the own step predicted exactly: m- = m + (g_t(th_t) - g_t(th_{t-1}))
        mp = self.m + own
        Pp = self.P + q
        Kg = Pp / (Pp + r)
        self.m = torch.where(later, mp + Kg * (g - mp), torch.where(first, g, self.m))
        self.P = torch.where(later, (1 - Kg) * Pp, torch.where(first, g * g + EPS, self.P))
        self.eprev = torch.where(has1, e, self.eprev)
        self.gprev = torch.where(obs, g, self.gprev)
        self.thprev = torch.where(obs, th, self.thprev)
        rows = obs[:, :1].float()
        self.n += rows
        self.nd += rows
        # consolidation latch (hold for c contrary observations, release after c)
        signal = self.m * self.m > Z2 * self.P
        if self.latch:
            rest_run = torch.where(obs & ~self.cons, torch.where(signal, 0.0, self.run + 1), self.run)
            hold_run = torch.where(obs & self.cons, torch.where(signal, self.run + 1, 0.0), self.run)
            self.run = torch.where(self.cons, hold_run, rest_run)
            new_cons = ~self.cons & (self.run >= WINDOW)
            release = self.cons & (self.run >= WINDOW) & self.exit
            self.stats["consolidations"] += int(new_cons.sum())
            self.stats["releases"] += int(release.sum())
            self.cons = (self.cons | new_cons) & ~release
            self.run = torch.where(new_cons | release, 0.0, self.run)
        # step: curvature scale x signal fraction, bounded by rho
        ha = h.abs().clamp(min=EPS)
        frac = (self.m * self.m - self.P).clamp(min=0) / (self.m * self.m + EPS)
        u = self.m / ha * frac
        live = obs & ~self.cons
        binding = bool(((u.abs() >= self.rho) & live).any())
        u = torch.where(live, u.clamp(-self.rho, self.rho), torch.zeros_like(u))
        # allocation: conserved mass per direction (competition), per-row reservoir (no WTA)
        a = u * u
        rowm = a.sum(1, keepdim=True)
        act = rowm > 0
        if self.alloc != "none" and act.sum() > 1:
            colT = a.sum(0, keepdim=True)
            tgt = torch.where(act, self.avail, torch.zeros_like(self.avail))
            tgt = tgt / tgt.sum() * a.sum()
            iters = SINKHORN_ITERS if self.alloc == "both" else 1
            for _ in range(iters):
                a = a * torch.where(act, tgt / (a.sum(1, keepdim=True) + EPS), torch.zeros_like(tgt))
                if self.alloc == "both":
                    a = a * torch.where(colT > 0, colT / (a.sum(0, keepdim=True) + EPS), torch.zeros_like(colT))
            u = (torch.sign(u) * a.sqrt()).clamp(-self.rho, self.rho)
        use = (u * u).sum(1, keepdim=True)
        if act.any():
            use = use / (use[act].mean() + EPS)
            self.avail += RES_RATE * (1 - self.avail) - RES_RATE * use * self.avail
            self.avail.clamp_(min=1e-3, max=1.0)
        pred = float((self.m * u - 0.5 * ha * u * u).sum())
        pred_sd = float(((u * u) * r).sum().sqrt())
        step = u @ self.Q.T
        if self.scoring == "same":
            before = fgrad(W, False)[0]
            after = fgrad(W - step, False)[0]
            self.evals += 2
            self.pred_sd, self.binding, self.umax = pred_sd, binding, float(u.abs().max())
            self._trust(before - after, pred, before)
        self.Wprev = W.clone()
        self.pred, self.pred_sd, self.binding, self.umax = pred, pred_sd, binding, float(u.abs().max())
        W -= step


class Decay:
    """Baseline with the conventional linear learning-rate decay to zero over the run."""

    def __init__(self, name, lr, shape, T):
        self.opt, self.lr0, self.T, self.t, self.evals = V1.Optimizer(name, lr, shape), lr, T, 0, 0

    def step(self, W, G, X, fgrad):
        self.t += 1
        self.opt.lr = self.lr0 * max(0.0, 1 - (self.t - 1) / self.T)
        self.opt.step(W, G)


class Plain:
    def __init__(self, name, lr, shape):
        self.opt, self.evals = V1.Optimizer(name, lr, shape), 0

    def step(self, W, G, X, fgrad):
        self.opt.step(W, G)


CLO_ARMS = dict(clo={}, clo_nolatch=dict(latch=False), clo_noexit=dict(exit=False),
                clo_noalloc=dict(alloc="none"), clo_rowonly=dict(alloc="row"), clo_samebatch=dict(scoring="same"))


def make(name, lr, shape, T):
    if name in CLO_ARMS:
        return CLO(shape, **CLO_ARMS[name])
    if name.endswith("_decay"):
        return Decay(name[: -len("_decay")], lr, shape, T)
    return Plain(name, lr, shape)


def lin_fg(X, Y):
    def fg(Wc, need):
        R = Wc @ X - Y
        return float(0.5 * (R ** 2).sum(0).mean()), (R @ X.T / X.shape[1] if need else None)
    return fg


def fin(x):
    return V1.finite(float(x))


# ---------- tasks (v1 tasks with the pre-activity and a loss closure exposed) ----------
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
    opt = make(name, lr, W.shape, V1.STEPS_A)
    for _ in range(V1.STEPS_A):
        y = torch.multinomial(p, V1.BATCH_A, replacement=True, generator=gen)
        def fg(Wc, need, y=y):
            Wg = Wc.clone().requires_grad_(need)
            with torch.enable_grad():
                L = F.cross_entropy(E[y] @ Wg.T, y)
                if need:
                    L.backward()
            return float(L), (Wg.grad if need else None)
        Gc = fg(W, True)[1]
        with torch.no_grad():
            opt.step(W, Gc, E[y].T, fg)
        if not torch.isfinite(W).all():
            return dict(loss=float("inf"), head_acc=0.0, tail_acc=0.0)
    with torch.no_grad():
        logits = E @ W.T
        acc = (logits.argmax(1) == torch.arange(V1.KA)).float()
        loss = float(F.cross_entropy(logits, torch.arange(V1.KA), reduction="none") @ p)
    return dict(loss=fin(loss), head_acc=float(acc[: V1.KA // 2].mean()), tail_acc=float(acc[V1.KA // 2:].mean()))


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
    opt = make(name, lr, W.shape, V1.STEPS_B)
    for _ in range(V1.STEPS_B):
        X = U @ (lam.sqrt()[:, None] * torch.randn(DB, V1.BATCH_B, generator=gen))
        Y = Ws @ X + Vr @ (sig[:, None] * torch.randn(DB, V1.BATCH_B, generator=gen))
        opt.step(W, (W @ X - Y) @ X.T / V1.BATCH_B, X, lin_fg(X, Y))
        if not torch.isfinite(W).all():
            return dict(loss=float("inf"))
    E = W - Ws
    return dict(loss=fin(0.5 * torch.trace(E @ Sig @ E.T)))


def task_c(name, lr, seed):
    gen = torch.Generator().manual_seed(2000 + seed)
    DC = V1.DC
    Ws = torch.randn(DC, DC, generator=gen) / DC ** 0.5
    W = torch.zeros(DC, DC)
    opt = make(name, lr, W.shape, V1.STEPS_C)
    back, trans = [], []
    for t in range(V1.STEPS_C):
        Ws[:16] += 0.001 * torch.randn(16, DC, generator=gen)
        if t % 40 == 0:
            Ws[16:] = torch.randn(16, DC, generator=gen) / DC ** 0.5
        X = torch.randn(DC, V1.BATCH_C, generator=gen)
        Y = Ws @ X + V1.NOISE_C * torch.randn(DC, V1.BATCH_C, generator=gen)
        opt.step(W, (W @ X - Y) @ X.T / V1.BATCH_C, X, lin_fg(X, Y))
        if not torch.isfinite(W).all():
            return dict(loss=float("inf"), backbone=float("inf"), transient=float("inf"))
        if t >= 200:
            E = W - Ws
            back.append(0.5 * float((E[:16] ** 2).sum()))
            trans.append(0.5 * float((E[16:] ** 2).sum()))
    return dict(loss=fin(np.mean(back) + np.mean(trans)), backbone=fin(np.mean(back)), transient=fin(np.mean(trans)))


# D: revertible topic bursts on the backbone, then one lasting change
DD, STEPS_D, BATCH_D, NOISE_D = 32, 1200, 64, 0.1
BURSTS = (2, 4, 8, 16)


def low_rank(gen, scale):
    a = torch.randn(DD, 2, generator=gen)
    b = torch.randn(2, DD, generator=gen)
    M = a @ b
    return M / M.norm() * scale


def task_d(name, lr, seed):
    gen = torch.Generator().manual_seed(3000 + seed)
    Ws = torch.randn(DD, DD, generator=gen) / DD ** 0.5
    size = 0.5 * float(Ws.norm())
    starts = [200 + 50 * i for i in range(8)]
    lengths = [BURSTS[i % 4] for i in torch.randperm(8, generator=gen).tolist()]
    deltas = [low_rank(gen, size) for _ in starts]
    change_at, change = 700, low_rank(gen, size)
    W = torch.zeros(DD, DD)
    opt = make(name, lr, W.shape, STEPS_D)
    exc, post = [], {L: [] for L in BURSTS}
    adopt, lag = [], []
    for t in range(STEPS_D):
        if t == change_at:
            Ws = Ws + change
        target = Ws
        active = [k for k, s in enumerate(starts) if s <= t < s + lengths[k]]
        if active:
            target = Ws + deltas[active[0]]
        X = torch.randn(DD, BATCH_D, generator=gen)
        Y = target @ X + NOISE_D * torch.randn(DD, BATCH_D, generator=gen)
        opt.step(W, (W @ X - Y) @ X.T / BATCH_D, X, lin_fg(X, Y))
        if not torch.isfinite(W).all():
            return dict(loss=float("inf"), final=float("inf"), adopt100=float("inf"),
                        **{f"post{L}": float("inf") for L in BURSTS})
        e = 0.5 * float(((W - Ws) ** 2).sum())
        if t >= 200 and not active:
            exc.append(e)
        for k, s in enumerate(starts):
            if s + lengths[k] <= t < s + lengths[k] + 10:
                post[lengths[k]].append(e)
        if t >= STEPS_D - 100:
            adopt.append(e)
        if change_at <= t < change_at + 100:
            lag.append(e)
    return dict(loss=fin(np.mean(exc)), final=fin(np.mean(adopt)), adopt100=fin(np.mean(lag)),
                **{f"post{L}": fin(np.mean(v)) for L, v in post.items()})


# E: two-layer student-teacher; hidden units must differentiate
DE, KE, STEPS_E, BATCH_E = 32, 8, 1500, 64


def task_e(name, lr, seed):
    gen = torch.Generator().manual_seed(4000 + seed)
    Fq = torch.linalg.qr(torch.randn(DE, KE, generator=gen))[0].T          # teacher features, orthonormal
    amp = torch.arange(1, KE + 1, dtype=torch.float).pow(-0.7)
    teach = lambda X: amp @ F.relu(Fq @ X)                                   # noqa: E731
    W1 = torch.randn(KE, DE, generator=gen) / DE ** 0.5
    W2 = torch.randn(1, KE, generator=gen) / KE ** 0.5
    o1, o2 = make(name, lr, W1.shape, STEPS_E), make(name, lr, W2.shape, STEPS_E)
    lossf = lambda A, B, X, y: float(0.5 * ((B @ F.relu(A @ X) - y) ** 2).mean())  # noqa: E731

    def grads(A, B, X, y):
        A, B = A.clone().requires_grad_(True), B.clone().requires_grad_(True)
        with torch.enable_grad():
            L = 0.5 * ((B @ F.relu(A @ X) - y) ** 2).mean()
            L.backward()
        return float(L), A.grad, B.grad
    for _ in range(STEPS_E):
        X = torch.randn(DE, BATCH_E, generator=gen)
        y = teach(X) + 0.01 * torch.randn(BATCH_E, generator=gen)
        _, G1, G2 = grads(W1, W2, X, y)
        with torch.no_grad():
            H0 = F.relu(W1 @ X)
            W1o, W2o = W1.clone(), W2.clone()
            o1.step(W1, G1, X, lambda Wc, need: (lambda r: (r[0], r[1] if need else None))(grads(Wc, W2o, X, y)))
            o2.step(W2, G2, H0, lambda Wc, need: (lambda r: (r[0], r[2] if need else None))(grads(W1o, Wc, X, y)))
        if not (torch.isfinite(W1).all() and torch.isfinite(W2).all()):
            return dict(loss=float("inf"), coverage=0.0, copies=float(KE))
    with torch.no_grad():
        Xt = torch.randn(DE, 4096, generator=gen)
        loss = lossf(W1, W2, Xt, teach(Xt))
        cos = F.normalize(W1, dim=1) @ Fq.T                                  # student x teacher
        best = cos.max(1)
        matched = best.indices[best.values > 0.9]
        coverage = len(set(matched.tolist()))
        copies = int(len(matched) - coverage)
    return dict(loss=fin(loss), coverage=float(coverage), copies=float(copies))


TASKS = {
    "A_orthogonal": (lambda n, lr, s: task_a(n, lr, s, "orthogonal")),
    "A_correlated": (lambda n, lr, s: task_a(n, lr, s, "correlated")),
    "B_rotated_heterogeneous": (lambda n, lr, s: task_b(n, lr, s)),
    "B0_isotropic": (lambda n, lr, s: task_b(n, lr, s, isotropic=True)),
    "C_backbone_transient": task_c,
    "D_topic_bursts": task_d,
    "E_two_layer": task_e,
}
BASE = ("sgd", "adam", "muon", "soap", "adam_decay", "muon_decay")
UNTUNED = tuple(CLO_ARMS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default=",".join(TASKS))
    a = ap.parse_args()
    res = {}
    for task in a.tasks.split(","):
        fn = TASKS[task]
        res[task] = {}
        for n in BASE + (("rowres",) if task.startswith("A_") else ()) + UNTUNED:
            sweep, lr = None, None
            if n not in UNTUNED:
                sweep = [(lr_, fn(n, lr_, 0)["loss"]) for lr_ in LRS]
                lr = min(sweep, key=lambda x: x[1])[0]
            runs = [fn(n, lr, s) for s in range(1, 6)]
            res[task][n] = dict(lr=lr, sweep=sweep, **{k: V1.summary([r[k] for r in runs]) for k in runs[0]})
            print(task, n, json.dumps({k: (v["median"] if isinstance(v, dict) else v)
                                       for k, v in res[task][n].items() if k != "sweep"}), flush=True)
            Path(a.out).write_text(json.dumps(dict(results=res), indent=1))
    Path(a.out).write_text(json.dumps(dict(
        prereg="docs/information_boltzmann/CGB_OPTIMIZER_AXIOMATIC_DERIVATION_20261027.md",
        registered=dict(obs_fraction=OBS_FRACTION, trust_accept=TRUST_ACCEPT, trust_gamma=TRUST_GAMMA, rho0=RHO0,
                        resolution=RESOLUTION, window=WINDOW, z2=Z2, res_rate=RES_RATE, sinkhorn_iters=SINKHORN_ITERS),
        results=res), indent=1))


if __name__ == "__main__":
    main()
