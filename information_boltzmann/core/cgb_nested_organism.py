"""Organisms nested in the rain torus (Lean `CGB.NestedWorld`, prereg
CGB_NESTED_ORGANISM_PREREG_20261027): the organism's part of the same Lean table, batched over B
worlds (one organism per world) on torch, one leap after each world leap.

Events and Lean rates (`c * exp(-dE / 2)`; the organism's cell may be a boundary cell, but every
organism event conserves the cell's content, so the reservoir term vanishes):
  * pair rule k on units (u, v), u != v: kappa_k * gate * poolCond * exp(-dE / 2), summed over
    ordered unit pairs in the source states (multiplicity n[i, s] * n[j, t]); a rule may couple to
    the cell's pools (`sdf`, `sdw`): dE then includes foodE * df and the cell energies, and the
    pool conductance makes it mass action (factor V - n when a pool gains a quantum, n when it
    loses one, `mass_action_up/down`); gate = intact membrane units for gated rules, else 1;
  * exchange j of unit u with the cell's food or waste pool: the same with one unit;
  * turn from heading a to each b != a while the motor is not running: kappa_flip each;
  * move to the cell ahead (forward, fuel rule a -> b) or behind (fuel b -> a) while the motor
    runs before and after: kappa_move * exp(-dE_fuel / 2) * fuel multiplicity.

The organism's structure is a `Table`. `LEGACY` is the hand-written first table (four inner rules,
four gated exchanges); `GRAMMAR` enumerates every rule the Lean grammar allows under one stated
principle (`grammar_table`), and selection decides which of them carry weight.
"""

from __future__ import annotations

from dataclasses import dataclass
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
LEAKS = (  # physics (`leak_bound`): every content-holding state of every page loses one quantum to the
           # cell's waste pool through no membrane, at the same prefactor
    (F, 1, 0, False, False),
    (F, 2, 1, False, False),
    (Y, 1, 0, False, False),
    (M, 1, 0, False, False),
    (E, 1, 0, False, False),
    (D, 1, 0, False, False),
    (S, 1, 0, False, False))  # membrane wear
EXCH = ((F, 0, 1, True, True),    # bind
        (F, 2, 1, True, True),    # unload
        (E, 0, 1, True, True),    # uptake
        (D, 1, 0, False, True),   # excrete
        ) + LEAKS
FUEL = (E, D, 1, 0, 0, 1)
GATE = (S, 1)
PHI_S1 = 2.0                       # physics: intact membrane, between waste (< 0) and food (foodE + ln c)
LOGK_LEAK = math.log(2.6e-7)       # physics: leak prefactor of every page; under mass action a leak into an
                                   # empty waste pool fires at kappa * V, the same per-quantum rate as the
                                   # first table's 1e-5 * sqrt(V) (V = 1500)
PHI_PER_QUANTUM = (0.0, 8.0)       # physics: energy per content quantum of a held state; the floor 0 keeps
                                   # every body state above waste up to a half-full pool (no condensation
                                   # of bodies out of waste); the scale bounds holds (`leak_bound`)
PHI_BOUND = 16.0                   # |phi| <= PHI_BOUND for every state (content <= 2)
N_SELECTED_X = 4                   # bind, unload, uptake, excrete
SELECTED_PHI = ((F, 1), (F, 2), (Y, 1), (M, 1), (E, 1), (D, 1))
SHIFT = ((1, 0), (0, 1), (-1, 0), (0, -1))
LOGK_RANGE = (-9.0, 0.0)           # selection range of rule, turn and move prefactors
LOGK_X_RANGE = (-16.0, -4.0)       # exchanges: under mass action the pool factor is n or V - n (up to 1500),
                                   # so the range moves down by ~ln sqrt(8 V) ~ 4.7 and widens


@dataclass(frozen=True)
class Table:
    """An organism's structure. RULES: (i, j, a1, a2, b1, b2, df, dw, gated); in the forward
    direction the food and waste pools change by df, dw (Lean `sdf`, `sdw`, `sgated`). EXCH:
    (page, a, b, food, gated); the first N_SELECTED_X are selected, the rest are physics leaks.
    RULE_RANGE: per rule, the selection range of its log prefactor."""
    name: str
    RULES: tuple
    EXCH: tuple
    N_SELECTED_X: int
    RULE_RANGE: tuple
    PREMISE: bool
    UNITS: tuple = UNITS
    STATES: tuple = STATES
    SMAX: int = SMAX
    FUEL: tuple = FUEL
    GATE: tuple = GATE
    RUN: int = RUN
    SELECTED_PHI: tuple = SELECTED_PHI
    PHI_S1: float = PHI_S1
    LOGK_LEAK: float = LOGK_LEAK
    PHI_PER_QUANTUM: tuple = PHI_PER_QUANTUM
    X_RANGE: tuple = LOGK_X_RANGE      # selection range of the selected exchanges' log prefactors

    def structure(self):
        return dict(table=self.name, pages=list(PAGES), units=list(self.UNITS), states=list(self.STATES),
                    rules=[list(r) for r in self.RULES], exchanges=[list(x) for x in self.EXCH],
                    n_selected_x=self.N_SELECTED_X, fuel=list(self.FUEL), gate=list(self.GATE), run=self.RUN,
                    physics=dict(phi_S1=self.PHI_S1, logk_leak=self.LOGK_LEAK,
                                 phi_per_quantum=list(self.PHI_PER_QUANTUM)))


LEGACY = Table(name="legacy", RULES=tuple(r + (0, 0, False) for r in RULES), EXCH=EXCH,
               N_SELECTED_X=N_SELECTED_X, RULE_RANGE=tuple(LOGK_RANGE for _ in RULES), PREMISE=True)


# Ranges of the grammar table, by one principle: the bottom of every range is negligible, at most
# ~1e-4 events per tick at the largest multiplicity (crossing: membrane 10 x pool factor 1500 x units
# ~260 ~ 4e6, so e^-24; inside the body: units ~260, so e^-14); a coordinate of 0 is an absent event.
GRAMMAR_X_RANGE = (-24.0, -4.0)
GRAMMAR_INNER_RANGE = (-14.0, 0.0)


def grammar_table():
    """Every pair rule and exchange the Lean grammar (`NestedWorld`: pair rules coupled to the
    pools, exchanges) allows under one principle, stated before any selection:
      * a pair rule moves two units on different pages one state step each (content 0 <-> 1, or
        1 <-> 2 on the food-binding page); the pools absorb the net content change with at most
        one quantum each (no change for a rule inside the body);
      * a rule that crosses the boundary is the organism's own channel and goes through the
        membrane (gated); ungated crossings are physics only (the leaks);
      * the membrane page takes part in no crossing rule (Lean `hsg`: gated rules leave the
        membrane alone), only in rules inside the body;
      * an exchange moves one unit one step, with the food or the waste pool, gated; the membrane
        page has none (Lean `hgate`).
    Movement (fuel), the membrane, the motor and the leaks are the fixed body (`LEGACY`)."""
    trans = {p: [(s, s + 1) for s in range(STATES[p] - 1)] for p in range(6)}
    rules, rng_ = [], []
    for i in range(6):
        for j in range(i + 1, 6):
            for ti in trans[i]:
                for tj in trans[j]:
                    for sigma in (1, -1):
                        a1, b1 = ti
                        a2, b2 = tj if sigma == 1 else (tj[1], tj[0])
                        delta = 1 + sigma               # content the units gain going a -> b
                        if delta == 0:
                            opts = [(0, 0), (-1, 1), (1, -1)]
                        else:
                            opts = [(-1, -1)]
                        for df, dw in opts:
                            crossing = (df, dw) != (0, 0)
                            if crossing and S in (i, j):
                                continue
                            rules.append((i, j, a1, a2, b1, b2, df, dw, crossing))
                            rng_.append(GRAMMAR_X_RANGE if crossing else GRAMMAR_INNER_RANGE)
    exch = []
    for p in range(6):
        if p == S:
            continue
        for (a, b) in trans[p]:
            for food in (True, False):
                exch.append((p, a, b, food, True))     # unit a -> b takes one quantum from the pool
    return Table(name="grammar_v1", RULES=tuple(rules), EXCH=tuple(exch) + LEAKS, N_SELECTED_X=len(exch),
                 RULE_RANGE=tuple(rng_), PREMISE=False, X_RANGE=GRAMMAR_X_RANGE)


GRAMMAR = grammar_table()
TABLES = {"legacy": LEGACY, "grammar_v1": GRAMMAR}


def table_of(name_or_table):
    if name_or_table is None:
        return LEGACY
    if isinstance(name_or_table, Table):
        return name_or_table
    return TABLES[name_or_table]


def random_genomes(B, rng, lo=LOGK_RANGE[0], hi=LOGK_RANGE[1], table=None):
    """Selected numbers, uniform in the prereg ranges (legacy: with the time-scale premise
    logk(Y0 + F1) <= logk(bind) - ln 20)."""
    T = table_of(table)
    plo, phi_hi = phi_range(T)
    rl = np.array([r[0] for r in T.RULE_RANGE])
    rh = np.array([r[1] for r in T.RULE_RANGE])
    g = dict(phi=rng.uniform(plo, phi_hi, size=(B, len(T.SELECTED_PHI))),
             logk_rule=rl + (rh - rl) * rng.uniform(size=(B, len(T.RULES))),
             logk_x=rng.uniform(*T.X_RANGE, size=(B, T.N_SELECTED_X)),
             log_flip=rng.uniform(lo, hi, size=B),
             log_move=rng.uniform(lo, hi, size=B))
    return premise(g, T)


def phi_range(table=None):
    """Per selected energy: content s of the state times the per-quantum physics range."""
    T = table_of(table)
    lo = np.array([T.PHI_PER_QUANTUM[0] * s for (_, s) in T.SELECTED_PHI])
    hi = np.array([T.PHI_PER_QUANTUM[1] * s for (_, s) in T.SELECTED_PHI])
    return lo, hi


def premise(g, table=None):
    if table_of(table).PREMISE:
        g["logk_rule"][..., 0] = np.minimum(g["logk_rule"][..., 0], g["logk_x"][..., 0] - math.log(20.0))
    return g


class Organisms:
    """B organisms, organism b lives in world b of a `FusedWorld` (int32 pools)."""

    def __init__(self, world, genomes, seed=0, blind=False, still=False, freeze_y=False, table=None):
        T = self.T = table_of(table)
        self.W = world
        self.foodE = float(getattr(world, "foodE", 0.0))
        B, dev = world.B, world.f.device
        self.B, self.dev, self.G, self.V, self.dt = B, dev, world.G, world.V, world.dt
        g = {k: torch.tensor(np.asarray(v), dtype=torch.float32, device=dev) for k, v in genomes.items()}
        phi = torch.zeros(B, 6, T.SMAX, device=dev)
        for c, (p, s) in enumerate(T.SELECTED_PHI):
            phi[:, p, s] = g["phi"][:, c]
        phi[:, S, 1] = T.PHI_S1
        self.phi = phi
        self.kr = torch.exp(g["logk_rule"]).reshape(B, len(T.RULES))
        nleak = len(T.EXCH) - T.N_SELECTED_X
        kx = torch.cat([torch.exp(g["logk_x"]).reshape(B, T.N_SELECTED_X),
                        torch.full((B, nleak), math.exp(T.LOGK_LEAK), device=dev)], 1)
        if blind:   # receptors: every selected exchange or crossing rule that binds food on the sensing page
            for x, (p, a, b, food, gated) in enumerate(T.EXCH[:T.N_SELECTED_X]):
                if p == F and food:
                    kx[:, x] = 0.0
            for r, rule in enumerate(T.RULES):
                if F in (rule[0], rule[1]) and rule[6] != 0:
                    self.kr[:, r] = 0.0
        if freeze_y:
            for r, rule in enumerate(T.RULES):
                if Y in (rule[0], rule[1]):
                    self.kr[:, r] = 0.0
        self.kx = kx
        self.kflip = torch.exp(g["log_flip"]).reshape(B)
        self.kmove = torch.exp(g["log_move"]).reshape(B) * (0.0 if still else 1.0)
        rng = np.random.default_rng(seed)
        self.n = torch.zeros(B, 6, T.SMAX, dtype=torch.int32, device=dev)
        for p, u in enumerate(T.UNITS):
            self.n[:, p, 0] = u
        self.n[:, S, 0], self.n[:, S, 1] = 0, T.UNITS[S]
        self.n[:, E, 0], self.n[:, E, 1] = T.UNITS[E] - T.UNITS[E] // 2, T.UNITS[E] // 2
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
        return torch.where(n < self.V, self.V - n, torch.zeros_like(n))

    def _dn(self, n):
        n = n.to(torch.float32)
        return torch.where(n > 0, n, torch.zeros_like(n))

    def _pf(self, pool, d):
        """Mass-action pool factor of a change by d in {-1, 0, 1}."""
        if d > 0:
            return self._up(pool)
        if d < 0:
            return self._dn(pool)
        return 1.0

    def pools(self):
        cell = self.x * self.G + self.y
        return cell, self.W.f.view(self.B, -1)[self.ar, cell], self.W.w.view(self.B, -1)[self.ar, cell]

    def rates(self, fc=None, wc=None):
        """(B, C) rates: rules (2 per rule), exchanges (2 each), turn, forward, back."""
        T = self.T
        if fc is None:
            _, fc, wc = self.pools()
        n, phi = self.n.to(torch.float32), self.phi
        gate = n[:, T.GATE[0], T.GATE[1]]
        cols = []
        for r, (i, j, a1, a2, b1, b2, df, dw, gated) in enumerate(T.RULES):
            dE = phi[:, i, b1] + phi[:, j, b2] - phi[:, i, a1] - phi[:, j, a2] + self.foodE * df
            k = self.kr[:, r] * (gate if gated else 1.0)
            cols.append(k * torch.exp(-dE / 2) * n[:, i, a1] * (n[:, j, a2] - float(i == j and a1 == a2)) *
                        self._pf(fc, df) * self._pf(wc, dw))
            cols.append(k * torch.exp(dE / 2) * n[:, i, b1] * (n[:, j, b2] - float(i == j and b1 == b2)) *
                        self._pf(fc, -df) * self._pf(wc, -dw))
        for x, (p, a, b, food, gated) in enumerate(T.EXCH):
            pool = fc if food else wc
            d = a - b                         # quanta the pool gains in direction a -> b
            k = self.kx[:, x] * (gate if gated else 1.0)
            dphi = phi[:, p, b] - phi[:, p, a] + (self.foodE * d if food else 0.0)   # food carries foodE
            cols.append(k * torch.exp(-dphi / 2) * self._pf(pool, d) * n[:, p, a])
            cols.append(k * torch.exp(dphi / 2) * self._pf(pool, -d) * n[:, p, b])
        running = (self.n[:, M, T.RUN] > 0)
        cols.append(torch.where(running, 0.0, 3.0 * self.kflip))
        i, j, a1, a2, b1, b2 = T.FUEL
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
        """One tau leap of length dt (the reference engine; runs use the exact GPU kernel).
        State tensors are updated in place (CUDA Graph replays)."""
        T = self.T
        cell, fc, wc = self.pools()
        k = torch.poisson(self.rates(fc, wc) * self.dt).to(torch.int32)
        fc, wc = fc.clone(), wc.clone()
        f0 = fc.clone()
        c = 0
        for (i, j, a1, a2, b1, b2, df, dw, gated) in T.RULES:
            for (s, kk) in ((1, k[:, c]), (-1, k[:, c + 1])):
                dfs, dws = s * df, s * dw
                for pool, d in ((fc, dfs), (wc, dws)):
                    if d < 0:
                        kk = torch.minimum(kk, pool)
                    elif d > 0:
                        kk = torch.minimum(kk, self.V - pool)
                if s == 1:
                    kk = self._pair(i, a1, j, a2, i, b1, j, b2, kk)
                else:
                    kk = self._pair(i, b1, j, b2, i, a1, j, a2, kk)
                fc += dfs * kk
                wc += dws * kk
            c += 2
        for (p, a, b, food, gated) in T.EXCH:
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
        i, j, a1, a2, b1, b2 = T.FUEL
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
        self.run_time += (self.n[:, M, self.T.RUN] > 0).to(torch.int64)


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


def _rule_key(rule):
    """A pair rule up to the order of its units and its direction."""
    i, j, a1, a2, b1, b2, df, dw, _g = rule
    fwd = tuple(sorted([(i, a1, b1), (j, a2, b2)])) + ((df, dw),)
    rev = tuple(sorted([(i, b1, a1), (j, b2, a2)])) + ((-df, -dw),)
    return min(fwd, rev)


def _exch_key(x):
    p, a, b, food, _g = x
    return (p, min(a, b), max(a, b), food)


def map_genome(g, src, dst):
    """A genome of table `src` written in table `dst`: rules and exchanges that are the same event
    keep their prefactor; every other rule of `dst` is absent (prefactor 0, log -inf)."""
    S_, D_ = table_of(src), table_of(dst)
    rk = {_rule_key(r): k for k, r in enumerate(S_.RULES)}
    xk = {_exch_key(x): k for k, x in enumerate(S_.EXCH[:S_.N_SELECTED_X])}
    lr = np.asarray(g["logk_rule"], dtype=float)
    lx = np.asarray(g["logk_x"], dtype=float)
    out = {k: np.asarray(v, dtype=float).copy() for k, v in g.items() if k not in ("logk_rule", "logk_x")}
    out["logk_rule"] = np.array([lr[..., rk[_rule_key(r)]] if _rule_key(r) in rk else -np.inf
                                 for k, r in enumerate(D_.RULES)]).T
    out["logk_x"] = np.array([lx[..., xk[_exch_key(x)]] if _exch_key(x) in xk else -np.inf
                              for x in D_.EXCH[:D_.N_SELECTED_X]]).T
    return out
