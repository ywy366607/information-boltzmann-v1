"""The nested world (Lean `CGB.NestedWorld`, T-E2) as a simulator generated from its table.

Every event is a rule of the Lean table, with the Lean rate `c * exp(-dE / 2)`:
  * supply (single cell, `.inr (.inr (.inr (food, dir)))`): at a source cell (kind 1) a pellet of
    food enters or leaves, exchanging with the outer reservoir at the cell's stimulus potential;
    at a sink cell (kind 2) one quantum of waste does the same;
  * diffusion (neighbours, `.inl (food, dir)`): one quantum of food or waste moves between
    adjacent cells. Both ordered pairs (u, v) and (v, u) are events of the medium, so a move from
    v to u has two events with the same rate (factor 2 below);
  * cell energy `cellF V n = -log C(V, n)` (G26, `cell_step`).
Rain, drying, waste removal and the user's clicks are stimuli (the reservoir potential of boundary
cells), not code paths. Organism events plug in later from the organism's own table.

Two integrators on the same rates: `ssa_world` (exact Gillespie, numba, small grids, reference)
and `World` (tau-leaping on torch, batched, CPU or CUDA).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WORLDS = ROOT / "information_boltzmann" / "worlds"


def load(name):
    return json.loads((WORLDS / f"{name}.json").read_text(encoding="utf-8"))


# ---------- geometry (torusPage, torusAhead) ----------

def kinds(G):
    """Kind map of the rain torus: sink (2) at x % 3 == 1 and y % 3 == 1, source (1) elsewhere."""
    x, y = np.meshgrid(np.arange(G), np.arange(G), indexing="ij")
    return np.where((x % 3 == 1) & (y % 3 == 1), 2, 1).astype(np.int64)


def pages(G, kind):
    x, y = np.meshgrid(np.arange(G), np.arange(G), indexing="ij")
    return 9 * kind + 3 * (x % 3) + y % 3


def ahead(i, j, h):
    """`torusAhead`: is the cell with page j the neighbour ahead of page i for heading h
    (0: +x, 1: +y, 2: -x, 3: -y)."""
    ix, iy, jx, jy = i % 9 // 3, i % 3, j % 9 // 3, j % 3
    return {0: jx == (ix + 1) % 3 and jy == iy, 1: jy == (iy + 1) % 3 and jx == ix,
            2: jx == (ix + 2) % 3 and jy == iy, 3: jy == (iy + 2) % 3 and jx == ix}[h]


# ---------- cell energy ----------

def cellF(V, n):
    return -(math.lgamma(V + 1) - math.lgamma(n + 1) - math.lgamma(V - n + 1))


# ---------- exact reference (numba Gillespie) ----------

def _ssa_kernel():
    from numba import njit

    @njit(cache=True)
    def lf(V, n):
        return -(math.lgamma(V + 1.0) - math.lgamma(n + 1.0) - math.lgamma(V - n + 1.0))

    @njit(cache=True)
    def rates(f, w, mu, kind, V, P, kin, kdiff, fE, out):
        """All effective events of the world. out[c, e]: c = cell (x * G + y);
        e 0/1: food pellet in/out (source), 2/3: waste in/out (sink),
        4..7: food quantum to neighbour d, 8..11: waste quantum to neighbour d
        (d: 0 +x, 1 +y, 2 -x, 3 -y)."""
        G = f.shape[0]
        tot = 0.0
        for x in range(G):
            for y in range(G):
                c = x * G + y
                for e in range(12):
                    out[c, e] = 0.0
                m = mu[x, y]
                if kind[x, y] == 1:
                    if f[x, y] + P <= V:
                        out[c, 0] = kin * math.exp(-(lf(V, f[x, y] + P) - lf(V, f[x, y]) - (m - fE) * P) / 2.0)
                    if f[x, y] >= P:
                        out[c, 1] = kin * math.exp(-(lf(V, f[x, y] - P) - lf(V, f[x, y]) + (m - fE) * P) / 2.0)
                elif kind[x, y] == 2:
                    if w[x, y] + 1 <= V:
                        out[c, 2] = kin * math.exp(-(lf(V, w[x, y] + 1) - lf(V, w[x, y]) - m) / 2.0)
                    if w[x, y] >= 1:
                        out[c, 3] = kin * math.exp(-(lf(V, w[x, y] - 1) - lf(V, w[x, y]) + m) / 2.0)
                for d in range(4):
                    tx = (x + (1 if d == 0 else (G - 1 if d == 2 else 0))) % G
                    ty = (y + (1 if d == 1 else (G - 1 if d == 3 else 0))) % G
                    for s in range(2):
                        a = f if s == 0 else w
                        if a[x, y] >= 1 and a[tx, ty] + 1 <= V:
                            dE = (lf(V, a[tx, ty] + 1) - lf(V, a[tx, ty]) +
                                  lf(V, a[x, y] - 1) - lf(V, a[x, y]))
                            out[c, 4 + 4 * s + d] = 2.0 * kdiff * math.exp(-dE / 2.0)
                for e in range(12):
                    tot += out[c, e]
        return tot

    @njit(cache=True)
    def run(f, w, mu, kind, V, P, kin, kdiff, fE, T, seed):
        """Exact Gillespie for time T (stimulus held fixed). Recomputes all rates per event:
        a reference for small grids, not a fast simulator."""
        np.random.seed(seed)
        G = f.shape[0]
        out = np.zeros((G * G, 12))
        t = 0.0
        nev = 0
        while True:
            tot = rates(f, w, mu, kind, V, P, kin, kdiff, fE, out)
            if tot <= 0.0:
                break
            t += -math.log(1.0 - np.random.random()) / tot
            if t > T:
                break
            u = np.random.random() * tot
            acc = 0.0
            cc, ee = 0, 0
            done = False
            for c in range(G * G):
                for e in range(12):
                    acc += out[c, e]
                    if acc >= u and out[c, e] > 0.0:
                        cc, ee = c, e
                        done = True
                        break
                if done:
                    break
            x, y = cc // G, cc % G
            if ee == 0:
                f[x, y] += P
            elif ee == 1:
                f[x, y] -= P
            elif ee == 2:
                w[x, y] += 1
            elif ee == 3:
                w[x, y] -= 1
            else:
                s = (ee - 4) // 4
                d = (ee - 4) % 4
                tx = (x + (1 if d == 0 else (G - 1 if d == 2 else 0))) % G
                ty = (y + (1 if d == 1 else (G - 1 if d == 3 else 0))) % G
                if s == 0:
                    f[x, y] -= 1
                    f[tx, ty] += 1
                else:
                    w[x, y] -= 1
                    w[tx, ty] += 1
            nev += 1
        return nev

    return rates, run


_SSA = None


def ssa_kernels():
    global _SSA
    if _SSA is None:
        _SSA = _ssa_kernel()
    return _SSA


def ssa_world(f, w, mu, kind, tbl, T, seed):
    _, run = ssa_kernels()
    t = tbl["table"]
    return run(f, w, mu, kind, t["V"], t["pellet"], t["kappa_in"], t["kappa_diff"], t.get("foodE", 0.0), float(T), seed)


def rates_np(f, w, mu, kind, tbl):
    """Rate array (G*G, 12) from the exact kernel (for the gate)."""
    r, _ = ssa_kernels()
    t = tbl["table"]
    out = np.zeros((f.shape[0] * f.shape[1], 12))
    r(f, w, mu, kind, t["V"], t["pellet"], t["kappa_in"], t["kappa_diff"], t.get("foodE", 0.0), out)
    return out


# ---------- batched tau-leaping on torch ----------

class World:
    """B worlds on one device. State f, w: (B, G, G) int64; stimulus mu: (B, G, G).
    `rates()` returns the same 12 event classes as the exact kernel, as (B, 12, G, G)."""

    SHIFT = ((1, 0), (0, 1), (-1, 0), (0, -1))   # neighbour d at (x + dx, y + dy)

    def __init__(self, tbl, B=1, device="cpu", seed=0, dt=0.05, dtype=None):
        """dtype: of the rates (float64 for the gate; float32 is the fast default on CUDA)."""
        import torch
        self.torch, self.tbl, self.B, self.dev, self.dt = torch, tbl, B, device, dt
        self.dtype = dtype or (torch.float32 if str(device).startswith("cuda") else torch.float64)
        t = tbl["table"]
        self.V, self.P, self.kin, self.kdiff = t["V"], t["pellet"], t["kappa_in"], t["kappa_diff"]
        self.foodE = t.get("foodE", 0.0)
        self.G = G = tbl["geometry"]["G"]
        k = torch.tensor(kinds(G), device=device)
        self.src, self.snk = (k == 1), (k == 2)
        self.gen = torch.Generator(device=device).manual_seed(seed)
        self.f = torch.zeros(B, G, G, dtype=torch.int64, device=device)
        self.w = torch.zeros(B, G, G, dtype=torch.int64, device=device)
        self.lgV = math.lgamma(self.V + 1)

    def lf(self, n):
        n = n.to(self.torch.float64)
        return -(self.lgV - self.torch.lgamma(n + 1) - self.torch.lgamma(self.V - n + 1))

    def _nb(self, a, d):
        dx, dy = self.SHIFT[d]
        return self.torch.roll(a, shifts=(-dx, -dy), dims=(1, 2))

    def _tables(self):
        """exp(-(cellF(n+1) - cellF(n)) / 2) = sqrt((V - n) / (n + 1)) (`cell_step`) for adding a
        quantum, and its inverse at n - 1 for removing one; 0 where the shift leaves 0..V."""
        if not hasattr(self, "_up"):
            th, V = self.torch, self.V
            n = th.arange(V + 1, dtype=th.float64, device=self.dev)
            self._up = th.where(n < V, th.sqrt((V - n) / (n + 1)), 0.0).to(self.dtype)
            self._dn = th.where(n > 0, th.sqrt(n / (V - n + 1)), 0.0).to(self.dtype)
        return self._up, self._dn

    def rates(self, mu):
        th, V, P = self.torch, self.V, self.P
        f, w = self.f, self.w
        out = th.empty(self.B, 12, self.G, self.G, dtype=self.dtype, device=self.dev)
        if P == 1:
            mu = mu.to(self.dtype)
            up, dn = self._tables()
            ufa, dfa, uwa, dwa = up[f], dn[f], up[w], dn[w]
            em, ep = th.exp(mu / 2), th.exp(-mu / 2)
            emf, epf = th.exp((mu - self.foodE) / 2), th.exp(-(mu - self.foodE) / 2)   # food carries foodE
            out[:, 0] = self.kin * ufa * emf * self.src
            out[:, 1] = self.kin * dfa * epf * self.src
            out[:, 2] = self.kin * uwa * em * self.snk
            out[:, 3] = self.kin * dwa * ep * self.snk
            for s, (ua, da) in enumerate(((ufa, dfa), (uwa, dwa))):
                for d in range(4):
                    out[:, 4 + 4 * s + d] = (2.0 * self.kdiff) * da * self._nb(ua, d)
            return out
        lff, lfw = self.lf(f), self.lf(w)
        r = self.kin * th.exp(-(self.lf(f + P) - lff - (mu - self.foodE) * P) / 2)
        out[:, 0] = th.where(self.src & (f + P <= V), r, 0.0)
        r = self.kin * th.exp(-(self.lf(f - P) - lff + (mu - self.foodE) * P) / 2)
        out[:, 1] = th.where(self.src & (f >= P), r, 0.0)
        r = self.kin * th.exp(-(self.lf(w + 1) - lfw - mu) / 2)
        out[:, 2] = th.where(self.snk & (w + 1 <= V), r, 0.0)
        r = self.kin * th.exp(-(self.lf(w - 1) - lfw + mu) / 2)
        out[:, 3] = th.where(self.snk & (w >= 1), r, 0.0)
        for s, a in enumerate((f, w)):
            la, lam = self.lf(a), self.lf(a - 1)
            for d in range(4):
                b = self._nb(a, d)
                dE = self.lf(b + 1) - self.lf(b) + lam - la
                ok = (a >= 1) & (b + 1 <= V)
                out[:, 4 + 4 * s + d] = th.where(ok, 2.0 * self.kdiff * th.exp(-dE / 2), 0.0)
        return out

    def step(self, mu, T=1.0):
        """Advance time T by tau-leaping with step dt; counts clipped to the pools' bounds."""
        th = self.torch
        for _ in range(max(1, int(round(T / self.dt)))):
            k = th.poisson(self.rates(mu) * self.dt, generator=self.gen).to(th.int64)
            P, V = self.P, self.V
            self.f += (th.minimum(k[:, 0] * P, V - self.f) // P - th.minimum(k[:, 1] * P, self.f) // P) * P
            self.w += th.minimum(k[:, 2], V - self.w) - th.minimum(k[:, 3], self.w)
            # diffusion: one direction at a time (each cell sends to one and receives from one)
            for s in range(2):
                a = self.f if s == 0 else self.w
                for d in range(4):
                    n = th.minimum(th.minimum(k[:, 4 + 4 * s + d], a), V - self._nb(a, d))
                    dx, dy = self.SHIFT[d]
                    a += th.roll(n, shifts=(dx, dy), dims=(1, 2)) - n

    def captured(self, T=1.0):
        """A CUDA Graph of `step(mu, T)` on a static stimulus buffer. Returns (mu_buffer, replay):
        copy the stimulus into the buffer, then replay. Random numbers advance on every replay."""
        th = self.torch
        mu = th.zeros(self.B, self.G, self.G, dtype=th.float64, device=self.dev)
        self.gen = None   # the default CUDA generator is graph-safe
        s = th.cuda.Stream()
        s.wait_stream(th.cuda.current_stream())
        f0, w0 = self.f.clone(), self.w.clone()
        with th.cuda.stream(s):
            self.step(mu, T)
        th.cuda.current_stream().wait_stream(s)
        g = th.cuda.CUDAGraph()
        with th.cuda.graph(g):
            self.step(mu, T)
        self.f.copy_(f0)
        self.w.copy_(w0)
        return mu, g.replay


# ---------- the stimulus: rain, richness drift, clicks ----------

class Weather:
    """The world's outer environment, as the stimulus field mu (B, G, G): sinks at mu_sink;
    sources at mu_dry, or mu_rain during a rain at that cell. Rains start at random source cells
    at rate rain_rate * exp(X), X an Ornstein-Uhlenbeck process (rich and lean times)."""

    def __init__(self, tbl, B=1, device="cpu", seed=0):
        import torch
        self.torch, self.B, self.dev = torch, B, device
        s = tbl["stimulus"]
        self.s = s
        self.G = G = tbl["geometry"]["G"]
        k = kinds(G)
        self.src_idx = np.argwhere(k == 1)
        self.src_flat = torch.tensor(self.src_idx[:, 0] * G + self.src_idx[:, 1], device=device)
        self.rng = np.random.default_rng(seed)
        self.X = self.rng.normal(0.0, s["rate_sigma"], size=B)
        self.left = torch.zeros(B, G * G, dtype=torch.int64, device=device)   # rain ticks left
        self.snk = torch.tensor(k == 2, device=device)

    def rate(self):
        return self.s["rain_rate"] * np.exp(self.X)

    def click(self, b, x, y):
        """A rain at the source cell nearest to (x, y)."""
        d = np.abs(self.src_idx - np.array([x, y]))
        d = np.minimum(d, self.G - d).sum(1)
        cx, cy = self.src_idx[int(np.argmin(d))]
        self.left[b, int(cx) * self.G + int(cy)] = self.s["rain_dur"]

    def tick(self):
        s = self.s
        a = math.exp(-1.0 / s["rate_tau"])
        self.X = a * self.X + s["rate_sigma"] * math.sqrt(1 - a * a) * self.rng.normal(size=self.B)
        th = self.torch
        nr = self.rng.poisson(self.rate())
        if nr.sum():
            bb = np.repeat(np.arange(self.B), nr)
            cc = self.src_flat[th.tensor(self.rng.integers(len(self.src_idx), size=len(bb)), device=self.dev)]
            self.left[th.tensor(bb, device=self.dev), cc] = s["rain_dur"]
        raining = (self.left > 0).view(self.B, self.G, self.G)
        mu = th.where(raining, s["mu_rain"], s["mu_dry"]).to(th.float64)
        mu = th.where(self.snk, s["mu_sink"], mu)
        self.left.sub_(1).clamp_(min=0)
        return mu
