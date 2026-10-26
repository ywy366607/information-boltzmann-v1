"""Organisms nested in the rain torus (Lean `CGB.NestedWorld`, prereg
CGB_NESTED_ORGANISM_PREREG_20261027): the organism's part of the same Lean table, batched over B
worlds (one organism per world) on torch, one leap after each world leap.

Events and Lean rates (`c * exp(-dE / 2)`; the organism's cell may be a boundary cell, but every
organism event conserves the cell's content, so the reservoir term vanishes):
  * inner pair rule k on units (u, v), u != v: kappa_k * exp(-dE_phi / 2), summed over ordered
    unit pairs in the source states (multiplicity n[i, s] * n[j, t]);
  * exchange j of unit u with the cell's food or waste pool: kappa_j * gate * exp(-dE / 2) with
    dE = dphi + cellF(pool + d) - cellF(pool), i.e. the factor sqrt((V - n) / (n + 1)) when the
    pool gains one quantum and sqrt(n / (V - n + 1)) when it loses one; gate = intact membrane
    units for gated exchanges, 1 for wear;
  * turn from heading a to each b != a while the motor is not running: kappa_flip each;
  * move to the cell ahead (forward, fuel rule a -> b) or behind (fuel b -> a) while the motor
    runs before and after: kappa_move * exp(-dE_fuel / 2) * fuel multiplicity.
"""

from __future__ import annotations

import math

import numpy as np
import torch

F, Y, M, E, D, S = range(6)
PAGES = ("F", "Y", "M", "E", "D", "S")
STATES = (3, 2, 2, 2, 2, 2)
SMAX = 3
UNITS = (8, 8, 1, 20, 10, 10)
RUN = 1
# inner rules (i, j, a1, a2, b1, b2)
RULES = ((Y, F, 0, 1, 1, 0),    # memory follows the present (slow)
         (Y, M, 1, 0, 0, 1),    # past high -> run
         (M, F, 1, 1, 0, 2),    # present high -> stop
         (S, E, 0, 1, 1, 0))    # repair the membrane
# exchanges (page, a, b, food pool?, gated?)
EXCH = ((F, 0, 1, True, True),    # bind
        (F, 2, 1, True, True),    # unload
        (E, 0, 1, True, True),    # uptake
        (D, 1, 0, False, True),   # excrete
        (S, 1, 0, False, False))  # wear (physics)
FUEL = (E, D, 1, 0, 0, 1)
GATE = (S, 1)
PHI_S1 = -5.0                      # physics
LOGK_WEAR = math.log(6e-4)         # physics
SELECTED_PHI = ((F, 1), (F, 2), (Y, 1), (M, 1), (E, 1), (D, 1))
SHIFT = ((1, 0), (0, 1), (-1, 0), (0, -1))


def random_genomes(B, rng, lo=-9.0, hi=0.0):
    """Selected numbers, uniform in the prereg ranges, with the time-scale premise
    logk(Y0 + F1) <= logk(bind) - ln 20."""
    g = dict(phi=rng.uniform(-8.0, 0.0, size=(B, len(SELECTED_PHI))),
             logk_rule=rng.uniform(lo, hi, size=(B, len(RULES))),
             logk_x=rng.uniform(lo, hi, size=(B, 4)),
             log_flip=rng.uniform(lo, hi, size=B),
             log_move=rng.uniform(lo, hi, size=B))
    return premise(g)


def premise(g):
    g["logk_rule"][:, 0] = np.minimum(g["logk_rule"][:, 0], g["logk_x"][:, 0] - math.log(20.0))
    return g


class Organisms:
    """B organisms, organism b lives in world b of a `FusedWorld` (int32 pools)."""

    def __init__(self, world, genomes, seed=0, blind=False, still=False, freeze_y=False):
        self.W = world
        B, dev = world.B, world.f.device
        self.B, self.dev, self.G, self.V, self.dt = B, dev, world.G, world.V, world.dt
        g = {k: torch.tensor(np.asarray(v), dtype=torch.float32, device=dev) for k, v in genomes.items()}
        phi = torch.zeros(B, 6, SMAX, device=dev)
        for c, (p, s) in enumerate(SELECTED_PHI):
            phi[:, p, s] = g["phi"][:, c]
        phi[:, S, 1] = PHI_S1
        self.phi = phi
        self.kr = torch.exp(g["logk_rule"])
        kx = torch.cat([torch.exp(g["logk_x"]), torch.full((B, 1), math.exp(LOGK_WEAR), device=dev)], 1)
        if blind:
            kx[:, 0] = 0.0
            kx[:, 1] = 0.0
        if freeze_y:
            self.kr[:, 0] = 0.0
            self.kr[:, 1] = 0.0
        self.kx = kx
        self.kflip = torch.exp(g["log_flip"])
        self.kmove = torch.exp(g["log_move"]) * (0.0 if still else 1.0)
        rng = np.random.default_rng(seed)
        self.n = torch.zeros(B, 6, SMAX, dtype=torch.int32, device=dev)
        for p, u in enumerate(UNITS):
            self.n[:, p, 0] = u
        self.n[:, S, 0], self.n[:, S, 1] = 0, UNITS[S]
        self.n[:, E, 0], self.n[:, E, 1] = UNITS[E] - UNITS[E] // 2, UNITS[E] // 2
        self.x = torch.tensor(rng.integers(self.G, size=B), dtype=torch.int64, device=dev)
        self.y = torch.tensor(rng.integers(self.G, size=B), dtype=torch.int64, device=dev)
        self.h = torch.tensor(rng.integers(4, size=B), dtype=torch.int64, device=dev)
        self.death = torch.full((B,), -1, dtype=torch.int64, device=dev)
        self.tick = torch.zeros((), dtype=torch.int64, device=dev)
        self.intake = torch.zeros(B, dtype=torch.int64, device=dev)
        self.moves = torch.zeros(B, dtype=torch.int64, device=dev)
        self.run_time = torch.zeros(B, dtype=torch.int64, device=dev)
        self.ar = torch.arange(B, device=dev)
        self.sx = torch.tensor([s[0] for s in SHIFT], device=dev)
        self.sy = torch.tensor([s[1] for s in SHIFT], device=dev)

    # ---------- rates ----------

    def _up(self, n):
        n = n.to(torch.float32)
        return torch.where(n < self.V, torch.sqrt((self.V - n) / (n + 1.0)), torch.zeros_like(n))

    def _dn(self, n):
        n = n.to(torch.float32)
        return torch.where(n > 0, torch.sqrt(n / (self.V - n + 1.0)), torch.zeros_like(n))

    def pools(self):
        cell = self.x * self.G + self.y
        return cell, self.W.f.view(self.B, -1)[self.ar, cell], self.W.w.view(self.B, -1)[self.ar, cell]

    def rates(self, fc=None, wc=None):
        """(B, C) rates: rules (2 per rule), exchanges (2 each), turn, forward, back."""
        if fc is None:
            _, fc, wc = self.pools()
        n, phi = self.n.to(torch.float32), self.phi
        cols = []
        for r, (i, j, a1, a2, b1, b2) in enumerate(RULES):
            dE = phi[:, i, b1] + phi[:, j, b2] - phi[:, i, a1] - phi[:, j, a2]
            cols.append(self.kr[:, r] * torch.exp(-dE / 2) * n[:, i, a1] * (n[:, j, a2] - float(i == j and a1 == a2)))
            cols.append(self.kr[:, r] * torch.exp(dE / 2) * n[:, i, b1] * (n[:, j, b2] - float(i == j and b1 == b2)))
        gate = n[:, GATE[0], GATE[1]]
        for x, (p, a, b, food, gated) in enumerate(EXCH):
            pool = fc if food else wc
            d = a - b                         # quanta the pool gains in direction a -> b
            fwd_pool = self._up(pool) if d > 0 else self._dn(pool)
            bwd_pool = self._dn(pool) if d > 0 else self._up(pool)
            k = self.kx[:, x] * (gate if gated else 1.0)
            dphi = phi[:, p, b] - phi[:, p, a]
            cols.append(k * torch.exp(-dphi / 2) * fwd_pool * n[:, p, a])
            cols.append(k * torch.exp(dphi / 2) * bwd_pool * n[:, p, b])
        running = (self.n[:, M, RUN] > 0)
        cols.append(torch.where(running, 0.0, 3.0 * self.kflip))
        i, j, a1, a2, b1, b2 = FUEL
        dE = phi[:, i, b1] + phi[:, j, b2] - phi[:, i, a1] - phi[:, j, a2]
        cols.append(torch.where(running, self.kmove * torch.exp(-dE / 2) * n[:, i, a1] * n[:, j, a2], 0.0))
        cols.append(torch.where(running, self.kmove * torch.exp(dE / 2) * n[:, i, b1] * n[:, j, b2], 0.0))
        return torch.stack(cols, 1)

    # ---------- one leap ----------

    def _pair(self, i, s0, j, s1, i2, d0, j2, d1, k):
        k = torch.minimum(k, torch.minimum(self.n[:, i, s0], self.n[:, j, s1]))
        self.n[:, i, s0] -= k
        self.n[:, j, s1] -= k
        self.n[:, i2, d0] += k
        self.n[:, j2, d1] += k
        return k

    def leap(self):
        """One leap of length dt. State tensors are updated in place (CUDA Graph replays)."""
        cell, fc, wc = self.pools()
        k = torch.poisson(self.rates(fc, wc) * self.dt).to(torch.int32)
        c = 0
        for (i, j, a1, a2, b1, b2) in RULES:
            self._pair(i, a1, j, a2, i, b1, j, b2, k[:, c])
            self._pair(i, b1, j, b2, i, a1, j, a2, k[:, c + 1])
            c += 2
        fc, wc = fc.clone(), wc.clone()
        f0 = fc.clone()
        for (p, a, b, food, gated) in EXCH:
            for (s0, s1, kk) in ((a, b, k[:, c]), (b, a, k[:, c + 1])):
                pool = fc if food else wc
                d = s0 - s1
                lim = pool if d < 0 else self.V - pool
                kk = torch.minimum(torch.minimum(kk, self.n[:, p, s0]), lim)
                self.n[:, p, s0] -= kk
                self.n[:, p, s1] += kk
                pool += d * kk
            c += 2
        self.intake += (f0 - fc).to(torch.int64)
        fl = self.W.f.view(self.B, -1)
        wl = self.W.w.view(self.B, -1)
        fl[self.ar, cell] = fc
        wl[self.ar, cell] = wc
        turn = k[:, c] > 0
        self.h.copy_(torch.where(turn, (self.h + 1 + torch.randint(0, 3, (self.B,), device=self.dev)) % 4, self.h))
        c += 1
        i, j, a1, a2, b1, b2 = FUEL
        fwd = torch.minimum(k[:, c], torch.ones_like(k[:, c]))
        back = torch.minimum(k[:, c + 1], torch.ones_like(k[:, c + 1]))
        fwd = self._pair(i, a1, j, a2, i, b1, j, b2, fwd)
        back = self._pair(i, b1, j, b2, i, a1, j, a2, back)
        step = (fwd - back).to(torch.int64)
        self.x.copy_((self.x + step * self.sx[self.h]) % self.G)
        self.y.copy_((self.y + step * self.sy[self.h]) % self.G)
        self.moves += (fwd + back).to(torch.int64)

    def end_tick(self):
        self.tick += 1
        dead = (self.n[:, S, 1] == 0) & (self.death < 0)
        self.death.copy_(torch.where(dead, self.tick, self.death))
        self.run_time += (self.n[:, M, RUN] > 0).to(torch.int64)


def run(tbl, genomes, world_seeds, ticks=5000, device="cuda", graph=True, **org_kw):
    """Lives of B organisms (genomes rows) in B fresh worlds (world_seeds). Returns numpy dict."""
    import cgb_nested_world as NW
    import cgb_nested_world_triton as NT
    B = len(world_seeds)
    W = NT.FusedWorld(tbl, B=B, device=device, seed=int(world_seeds[0]))
    wea = NW.Weather(tbl, B=B, device=device, seed=int(world_seeds[0]) * 7 + 3)
    wea.rng = np.random.default_rng(list(map(int, world_seeds)))
    # warm the world to its own steady state before birth
    buf, replay = W.captured()
    for _ in range(1500):
        buf.copy_(wea.tick())
        replay()
    O = Organisms(W, genomes, seed=int(world_seeds[0]) + 1, **org_kw)
    nleap = max(1, int(round(1.0 / W.dt)))

    def tick_fn():
        for _ in range(nleap):
            W.leap()
            O.leap()
        O.end_tick()

    if graph:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        snap = (W.f.clone(), W.w.clone(), O.n.clone(), O.x.clone(), O.y.clone(), O.h.clone())
        with torch.cuda.stream(s):
            tick_fn()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            tick_fn()
        for t, v in zip((W.f, W.w, O.n, O.x, O.y, O.h), snap):
            t.copy_(v)
        for t in (O.death, O.tick, O.intake, O.moves, O.run_time):
            t.zero_() if t is not O.death else t.fill_(-1)
        step = g.replay
    else:
        step = tick_fn
    for _ in range(ticks):
        W.mu32.copy_(wea.tick())
        step()
    death = O.death.cpu().numpy()
    surv = np.where(death < 0, ticks, death)
    return dict(survival=surv, intake=O.intake.cpu().numpy() / np.maximum(1, surv),
                moves=O.moves.cpu().numpy(), run_frac=O.run_time.cpu().numpy() / ticks)
