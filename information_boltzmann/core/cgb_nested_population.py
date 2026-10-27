"""Populations in the rain torus: many organisms per world, division with inheritance and mutation,
death and dissolution, all on the GPU (Lean `CGB.NestedWorld`: division by recipes, dissolution,
moves that exchange with the occupant; prereg docs/information_boltzmann/CGB_GENERATED_ECOLI_GOAL_20261027.md).

A world has SL organism slots; slot b of world b // SL is one lane of the exact event kernel
(`cgb_nested_organism_triton._cols`, the organism's own events, checked against Lean), with:
  * moves exchange the mover with the occupant of the target (Lean `NestedWorld.move_exchanges`,
    2026-10-27; runs before that date moved only into empty cells). Into an empty cell: claimed by
    an atomic compare-and-swap on the occupancy grid, undone (fuel returned) if another lane took
    the cell in the same leap. Into a body's cell: the mover pays its fuel, files an exchange request
    and ends its leap there (a lane caches its cell's pools, so two lanes cannot share a cell inside
    a leap); after the leap `_exchange` swaps the pair if the occupant is still in place, filed no
    request itself and no other mover won it (lowest lane wins), and otherwise returns the fuel, as
    for a lost claim. Engine conventions (not Lean): the mover loses the rest of that leap
    (dt = 0.05 tick), and a cell claimed by a birth in this tick (the newborn joins at the end of the
    tick, as all births here) cannot be entered: its move column is zero;
  * division (Lean recipe events `(S, rb, h2, dir = false)`): a body with an intact membrane whose
    division page is full puts a newborn into an empty neighbour; its rate is
    4 headings x kdiv x empty neighbours x min(1, e^{-dE}) (Metropolis conductance),
    dE = bugE(newborn) - BQ (phi_E1 - phi_E0); the parent pays BQ = B + content(newborn) quanta,
    one per unit of the division page (`bornQ`, `drain`). The newborn's genotype is drawn from the
    mutation kernel (recipe conductances kdiv mu(g, g'); dE does not depend on g', so the total
    rate is kdiv and the genotype is drawn afterwards);
  * dissolution of a body without intact membrane into the cell's waste pool with everything it
    holds: kappa_dis min(1, e^{-dE}), dE = cellF(w + q) - cellF(w) - bugE (Lean c1 dissolution).
Births are claimed in the kernel (a free slot of the world, then the empty cell) and become lanes
at the end of the tick.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import triton

import cgb_nested_organism as NO
import cgb_nested_organism_triton as NX

GEN = Path(__file__).resolve().parent / "generated"


# ---------- the dividing body ----------

def division_table(base=None, B_body=16):
    """`base` (default GRAMMAR) with a division page that holds exactly one newborn: E has
    B + |S| units (the body plus an intact membrane), so a full reserve pays for one newborn and
    the drain has multiplicity one (more units would multiply the event by C(n, BQ))."""
    T = NO.table_of(base or NO.GRAMMAR)
    units = list(T.UNITS)
    units[NO.E] = B_body + T.UNITS[NO.S]
    return NO.Table(**{**T.__dict__, "name": T.name + "_div", "UNITS": tuple(units)})


DIV_GRAMMAR = division_table()
NO.TABLES[DIV_GRAMMAR.name] = DIV_GRAMMAR

PMUT = 0.1          # physics (replication fidelity): a division changes one genome coordinate with
                    # probability PMUT, by one grid step (DELTA) up or down; the kernel is symmetric
DELTA = 0.05        # grid of the genome in [0, 1] coordinates (21 levels: a finite genotype set)
LOGK_DIV_RANGE = NO.LOGK_RANGE


# ---------- genome coordinates ----------

@dataclass
class Coder:
    """[0, 1] coordinates of a genome of table T (as `run_cgb_es_nested`, plus log_div)."""
    T: object

    def __post_init__(self):
        T = self.T
        plo, phi_hi = NO.phi_range(T)
        lo = [plo, np.array([r[0] for r in T.RULE_RANGE]), np.full(T.N_SELECTED_X, T.X_RANGE[0]),
              np.full(3, NO.LOGK_RANGE[0])]
        hi = [phi_hi, np.array([r[1] for r in T.RULE_RANGE]), np.full(T.N_SELECTED_X, T.X_RANGE[1]),
              np.full(3, NO.LOGK_RANGE[1])]
        self.lo, self.hi = np.concatenate(lo), np.concatenate(hi)
        self.sizes = [("phi", len(T.SELECTED_PHI)), ("logk_rule", len(T.RULES)), ("logk_x", T.N_SELECTED_X),
                      ("log_flip", 0), ("log_move", 0), ("log_div", 0)]
        self.D = len(self.lo)

    def encode(self, g):
        v = np.concatenate([np.atleast_1d(np.asarray(g[k], dtype=float)) for k, _ in self.sizes])
        return np.clip((v - self.lo) / (self.hi - self.lo), 0.0, 1.0)

    def decode(self, U):
        U = np.atleast_2d(U)
        V = self.lo + np.clip(U, 0.0, 1.0) * (self.hi - self.lo)
        nphi = len(self.T.SELECTED_PHI)    # a prefactor at coordinate 0 is an absent event (conductance 0)
        V[:, nphi:] = np.where(U[:, nphi:] <= 0.0, -np.inf, V[:, nphi:])
        g, i = {}, 0
        for k, n in self.sizes:
            if n == 0:
                g[k] = V[:, i].copy()
                i += 1
            else:
                g[k] = V[:, i:i + n].copy()
                i += n
        return g


# ---------- the kernel ----------

def kernel_source(K, T):
    T = NO.table_of(T)
    cols = NX._cols(T)
    P, SM = len(T.STATES), T.SMAX
    nvars = [(p, s) for p in range(P) for s in range(T.STATES[p])]
    Fp, Ep, Sp = NO.F, NO.E, NO.S
    # the move columns are zero only into a cell claimed by a birth in this tick (`ea`, `eb`)
    (efwd, ffwd), (eback, fback) = cols[-2], cols[-1]
    cols[-2] = (f"({efwd}) * ea", ffwd)
    cols[-1] = (f"({eback}) * eb", fback)
    i_, j_, a1, a2, b1, b2 = T.FUEL
    L = []
    w = L.append
    w('"""Generated by cgb_nested_population.kernel_source from the organism table. Do not edit."""')
    w("import triton")
    w("import triton.language as tl")
    w("")
    w("@triton.jit")
    w("def _up(n, V):")
    w("    nf = n.to(tl.float32)")
    w("    return tl.where(n < V, V - nf, 0.0)")
    w("")
    w("@triton.jit")
    w("def _dn(n, V):")
    w("    nf = n.to(tl.float32)")
    w("    return tl.where(n > 0, nf, 0.0)")
    w("")
    w("@triton.jit")
    w("def population_leap(n_ptr, phi_ptr, kr_ptr, kx_ptr, kflip_ptr, kmove_ptr, kdiv_ptr, x_ptr, y_ptr, h_ptr,")
    w("                    alive_ptr, occ_ptr, free_ptr, birth_ptr, bh_ptr, scratch_ptr, cellF_ptr,")
    w("                    f_ptr, w_ptr, seed_ptr, intake_ptr, moves_ptr, over_ptr, gone_ptr, rates_ptr,")
    w("                    xreq_ptr, xcell_ptr, xdir_ptr,")
    w("                    B, SL, G, V, dt, FE, BODY, BQ, NS, PHIS1, KDIS, KMAX, RATES_ONLY: tl.constexpr,")
    w("                    BLOCK: tl.constexpr):")
    w("    pid = tl.program_id(0)")
    w("    b = pid * BLOCK + tl.arange(0, BLOCK)")
    w("    inb = b < B")
    w("    alive = tl.load(alive_ptr + b, mask=inb, other=0)")
    w("    m = inb & (alive == 1)")
    w("    pend = tl.load(birth_ptr + b, mask=m, other=0)")
    w("    wi = b // SL")
    w("    base = wi * G * G")
    w("    seed = tl.load(seed_ptr)")
    for p, s in nvars:
        w(f"    n{p}_{s} = tl.load(n_ptr + b * {P * SM} + {p * SM + s}, mask=m, other=0)")
        w(f"    phi{p}_{s} = tl.load(phi_ptr + b * {P * SM} + {p * SM + s}, mask=m, other=0.0)")
    for r in range(len(T.RULES)):
        w(f"    kr{r} = tl.load(kr_ptr + b * {len(T.RULES)} + {r}, mask=m, other=0.0)")
    for x in range(len(T.EXCH)):
        w(f"    kx{x} = tl.load(kx_ptr + b * {len(T.EXCH)} + {x}, mask=m, other=0.0)")
    w("    kflip = tl.load(kflip_ptr + b, mask=m, other=0.0)")
    w("    kmove = tl.load(kmove_ptr + b, mask=m, other=0.0)")
    w("    kdiv = tl.load(kdiv_ptr + b, mask=m, other=0.0)")
    w("    x = tl.load(x_ptr + b, mask=m, other=0)")
    w("    y = tl.load(y_ptr + b, mask=m, other=0)")
    w("    h = tl.load(h_ptr + b, mask=m, other=0)")
    w("    cell = base + x * G + y")
    w("    fc = tl.load(f_ptr + cell, mask=m, other=0)")
    w("    wc = tl.load(w_ptr + cell, mask=m, other=0)")
    w("    intake = tl.zeros_like(fc)")
    w("    moves = tl.zeros_like(fc)")
    w("    born = tl.zeros_like(fc)")
    w("    bhd = tl.zeros_like(fc)")
    w("    gone = tl.zeros_like(fc)")
    w("    t = tl.zeros([BLOCK], dtype=tl.float32)")
    w("    go = m")
    w("    it = pid * 0")
    w("    while (tl.max(go.to(tl.int32), axis=0) > 0) & (it < KMAX):")
    w("        xp = (x + 1) % G")
    w("        xm = (x + G - 1) % G")
    w("        yp = (y + 1) % G")
    w("        ym = (y + G - 1) % G")
    w("        c0 = base + xp * G + y")
    w("        c1 = base + x * G + yp")
    w("        c2 = base + xm * G + y")
    w("        c3 = base + x * G + ym")
    w("        o0 = tl.load(occ_ptr + c0, mask=go, other=1)")
    w("        o1 = tl.load(occ_ptr + c1, mask=go, other=1)")
    w("        o2 = tl.load(occ_ptr + c2, mask=go, other=1)")
    w("        o3 = tl.load(occ_ptr + c3, mask=go, other=1)")
    w("        e0 = tl.where(o0 == 0, 1, 0)")
    w("        e1 = tl.where(o1 == 0, 1, 0)")
    w("        e2 = tl.where(o2 == 0, 1, 0)")
    w("        e3 = tl.where(o3 == 0, 1, 0)")
    w("        nempty = e0 + e1 + e2 + e3")
    w("        ca = tl.where(h == 0, c0, tl.where(h == 1, c1, tl.where(h == 2, c2, c3)))")
    w("        cb = tl.where(h == 0, c2, tl.where(h == 1, c3, tl.where(h == 2, c0, c1)))")
    w("        oa = tl.where(h == 0, o0, tl.where(h == 1, o1, tl.where(h == 2, o2, o3)))")
    w("        ob = tl.where(h == 0, o2, tl.where(h == 1, o3, tl.where(h == 2, o0, o1)))")
    w("        ea = tl.where(oa != -1, 1.0, 0.0)")
    w("        eb = tl.where(ob != -1, 1.0, 0.0)")
    for c, (expr, _) in enumerate(cols):
        w(f"        r{c} = {expr}")
    # division
    gate = f"n{T.GATE[0]}_{T.GATE[1]}"
    w(f"        dEdiv = NS * PHIS1 - BQ * (phi{Ep}_1 - phi{Ep}_0)")
    w(f"        rdiv = tl.where(({gate} > 0) & (n{Ep}_1 == BQ) & (pend == 0), 4.0 * kdiv * nempty.to(tl.float32) * "
      f"tl.minimum(1.0, tl.exp(-dEdiv)), 0.0)")
    # dissolution
    cont = " + ".join(f"{s} * n{p}_{s}" for p, s in nvars if s > 0)
    bug = " + ".join(f"phi{p}_{s} * n{p}_{s}.to(tl.float32)" for p, s in nvars)
    w(f"        q = BODY + {cont}")
    w(f"        bug = {bug}")
    w("        wq = tl.minimum(wc + q, V)")
    w("        dCF = tl.load(cellF_ptr + wq, mask=go, other=0.0) - tl.load(cellF_ptr + wc, mask=go, other=0.0)")
    w("        dEdis = dCF.to(tl.float32) - bug")
    w(f"        rdis = tl.where(({gate} == 0) & (wc + q <= V), KDIS * tl.minimum(1.0, tl.exp(-dEdis)), 0.0)")
    nc = len(cols)
    w("        tot = " + " + ".join(f"r{c}" for c in range(nc)) + " + rdiv + rdis")
    w("        if RATES_ONLY:")
    for c in range(nc):
        w(f"            tl.store(rates_ptr + b * {nc + 2} + {c}, r{c}, mask=m)")
    w(f"            tl.store(rates_ptr + b * {nc + 2} + {nc}, rdiv, mask=m)")
    w(f"            tl.store(rates_ptr + b * {nc + 2} + {nc + 1}, rdis, mask=m)")
    w("            go = go & (t < 0.0)")
    w("        u1 = tl.rand(seed, b * 4 * KMAX + 4 * it)")
    w("        u2 = tl.rand(seed, b * 4 * KMAX + 4 * it + 1)")
    w("        u3 = tl.rand(seed, b * 4 * KMAX + 4 * it + 2)")
    w("        t += tl.where(tot > 0.0, -tl.log(1.0 - u1) / tl.maximum(tot, 1e-30), 1e30)")
    w("        go = go & (t < dt)")
    w("        target = u2 * tot")
    w("        acc = tl.zeros_like(tot)")
    w("        step = tl.zeros_like(x)")
    for c, (_, eff) in enumerate(cols):
        w(f"        sel = go & (target >= acc) & (target < acc + r{c}) & (r{c} > 0.0)")
        w(f"        frac = (target - acc) / tl.maximum(r{c}, 1e-30)")
        for e in eff:
            var, op, val = e.split(" ", 2)
            if op == "=":
                w(f"        {var} = tl.where(sel, {val}, {var})")
            else:
                sign = "+" if op == "+=" else "-"
                w(f"        {var} = tl.where(sel, {var} {sign} ({val}), {var})")
        w(f"        acc += r{c}")
    w("        sdiv = go & (target >= acc) & (target < acc + rdiv) & (rdiv > 0.0)")
    w("        fdiv = (target - acc) / tl.maximum(rdiv, 1e-30)")
    w("        acc += rdiv")
    w("        sdis = go & (target >= acc) & (target < acc + rdis) & (rdis > 0.0)")
    # moves: into an empty cell claim it (undo the fuel step if another lane took it); into a body's
    # cell file an exchange request and end the leap (resolved by Population._exchange)
    w("        moved = step != 0")
    w("        tcell = tl.where(step == 1, ca, cb)")
    w("        tocc = tl.load(occ_ptr + tcell, mask=moved, other=0)")
    w("        xch = moved & (tocc > 0)")
    w("        plain = moved & (tocc == 0)")
    w("        old = tl.atomic_cas(tl.where(plain, occ_ptr + tcell, scratch_ptr + b), tl.zeros_like(b), b + 1)")
    w("        ok = plain & (old == 0)")
    w("        bad = (plain & (old != 0)) | (moved & (tocc < 0))")
    w("        tl.store(xreq_ptr + b, tocc, mask=xch)")
    w("        tl.store(xcell_ptr + b, tcell, mask=xch)")
    w("        tl.store(xdir_ptr + b, step, mask=xch)")
    w(f"        badf = bad & (step == 1)")
    w(f"        badb = bad & (step == -1)")
    for (msk, src, dst) in (("badf", (a1, a2), (b1, b2)), ("badb", (b1, b2), (a1, a2))):
        w(f"        n{i_}_{src[0]} = tl.where({msk}, n{i_}_{src[0]} + 1, n{i_}_{src[0]})")
        w(f"        n{j_}_{src[1]} = tl.where({msk}, n{j_}_{src[1]} + 1, n{j_}_{src[1]})")
        w(f"        n{i_}_{dst[0]} = tl.where({msk}, n{i_}_{dst[0]} - 1, n{i_}_{dst[0]})")
        w(f"        n{j_}_{dst[1]} = tl.where({msk}, n{j_}_{dst[1]} - 1, n{j_}_{dst[1]})")
    w("        tl.store(f_ptr + cell, fc, mask=ok)")
    w("        tl.store(w_ptr + cell, wc, mask=ok)")
    w("        tl.store(occ_ptr + cell, tl.zeros_like(cell), mask=ok)")
    w("        cell = tl.where(ok, tcell, cell)")
    w("        x = tl.where(ok, (cell - base) // G, x)")
    w("        y = tl.where(ok, (cell - base) % G, y)")
    w("        fc = tl.where(ok, tl.load(f_ptr + cell, mask=ok, other=0), fc)")
    w("        wc = tl.where(ok, tl.load(w_ptr + cell, mask=ok, other=0), wc)")
    w("        moves += tl.where(ok, 1, 0)")
    # division: a free slot of the world, then the k-th empty neighbour
    w("        kk = tl.minimum((fdiv * nempty.to(tl.float32)).to(tl.int32), nempty - 1)")
    w("        td = tl.where(kk < e0, c0, tl.where(kk < e0 + e1, c1, tl.where(kk < e0 + e1 + e2, c2, c3)))")
    w("        prev = tl.atomic_add(free_ptr + wi, -1, mask=sdiv)")
    w("        slot = sdiv & (prev > 0)")
    w("        tl.atomic_add(free_ptr + wi, 1, mask=sdiv & (prev <= 0))")
    w("        oldd = tl.atomic_cas(tl.where(slot, occ_ptr + td, scratch_ptr + b), tl.zeros_like(b), tl.zeros_like(b) - 1)")
    w("        okd = slot & (oldd == 0)")
    w("        tl.atomic_add(free_ptr + wi, 1, mask=slot & (oldd != 0))")
    w(f"        n{Ep}_1 = tl.where(okd, n{Ep}_1 - BQ, n{Ep}_1)")
    w(f"        n{Ep}_0 = tl.where(okd, n{Ep}_0 + BQ, n{Ep}_0)")
    w("        born = tl.where(okd, td - base + 1, born)")
    w("        bhd = tl.where(okd, tl.minimum((u3 * 4.0).to(tl.int32), 3), bhd)")
    w("        pend = tl.where(okd, 1, pend)")
    # dissolution: everything the body holds goes to the waste pool, the place is free
    w("        wc = tl.where(sdis, wc + q, wc)")
    w("        gone = tl.where(sdis, 1, gone)")
    w("        go = go & (gone == 0) & (tl.where(xch, 1, 0) == 0)")
    w("        it += 1")
    w("    if not RATES_ONLY:")
    w("        over = tl.where(go, 1, 0)")
    w("        live = m & (gone == 0)")
    for p, s in nvars:
        w(f"        tl.store(n_ptr + b * {P * SM} + {p * SM + s}, tl.where(gone == 1, 0, n{p}_{s}), mask=m)")
    w("        tl.store(x_ptr + b, x, mask=m)")
    w("        tl.store(y_ptr + b, y, mask=m)")
    w("        tl.store(h_ptr + b, h, mask=m)")
    w("        tl.store(f_ptr + cell, fc, mask=m)")
    w("        tl.store(w_ptr + cell, wc, mask=m)")
    w("        tl.store(occ_ptr + cell, tl.zeros_like(cell), mask=m & (gone == 1))")
    w("        tl.store(alive_ptr + b, tl.where(gone == 1, 0, alive), mask=m)")
    w("        tl.store(gone_ptr + b, tl.load(gone_ptr + b, mask=m, other=0) + gone, mask=m)")
    w("        tl.store(birth_ptr + b, tl.where(born > 0, born, pend * 0 + tl.load(birth_ptr + b, mask=m, other=0)), mask=m)")
    w("        tl.store(bh_ptr + b, tl.where(born > 0, bhd, tl.load(bh_ptr + b, mask=m, other=0)), mask=m)")
    w("        tl.store(intake_ptr + b, tl.load(intake_ptr + b, mask=m, other=0) + intake, mask=m)")
    w("        tl.store(moves_ptr + b, tl.load(moves_ptr + b, mask=m, other=0) + moves, mask=m)")
    w("        tl.store(over_ptr + b, tl.load(over_ptr + b, mask=m, other=0) + over, mask=m)")
    w("        if pid == 0:")
    w("            tl.store(seed_ptr, tl.load(seed_ptr) + 1)")
    return "\n".join(L) + "\n", nc + 2


_MODS = {}


def kernel(T, K=64):
    T = NO.table_of(T)
    name = f"population_leap_{T.name}"
    if name not in _MODS:
        src, ncol = kernel_source(K, T)
        GEN.mkdir(exist_ok=True)
        path = GEN / f"{name}.py"
        if not path.exists() or path.read_text(encoding="utf-8") != src:
            path.write_text(src, encoding="utf-8")
        import importlib.util
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _MODS[name] = (mod.population_leap, ncol)
    return _MODS[name]


def cellF_table(V):
    k = np.arange(V + 1)
    from scipy.special import gammaln
    return torch.tensor(-(gammaln(V + 1) - gammaln(k + 1) - gammaln(V - k + 1)), dtype=torch.float64)


# ---------- the population ----------

class Population:
    """Worlds (a `FusedWorld`) with SL organism slots each."""

    BLOCK = 64

    def __init__(self, world, tbl, SL=128, table=None, seed=0, K=64, pmut=PMUT, delta=DELTA):
        T = self.T = NO.table_of(table or DIV_GRAMMAR)
        self.W, self.tblw = world, tbl
        self.NW, self.SL, self.G, self.V = world.B, SL, world.G, world.V
        self.B = self.NW * SL
        self.dev = dev = world.f.device
        self.coder = Coder(T)
        self.pmut, self.delta = pmut, delta
        self.rng = np.random.default_rng(seed)
        self.foodE = float(tbl["table"].get("foodE", 0.0))
        self.BODY = int(tbl["table"]["B"])
        self.BQ = self.BODY + T.UNITS[NO.S]
        assert T.UNITS[NO.E] == self.BQ, "the division page must hold exactly one newborn"
        self.KDIS = float(tbl["table"]["kappa_dis"])
        P, SM = 6, T.SMAX
        B = self.B
        self.n = torch.zeros(B, P, SM, dtype=torch.int32, device=dev)
        self.phi = torch.zeros(B, P, SM, dtype=torch.float32, device=dev)
        self.kr = torch.zeros(B, len(T.RULES), dtype=torch.float32, device=dev)
        self.kx = torch.zeros(B, len(T.EXCH), dtype=torch.float32, device=dev)
        self.kflip = torch.zeros(B, dtype=torch.float32, device=dev)
        self.kmove = torch.zeros(B, dtype=torch.float32, device=dev)
        self.kdiv = torch.zeros(B, dtype=torch.float32, device=dev)
        self.theta = np.zeros((B, self.coder.D))
        for name in ("x", "y", "h", "alive", "birth", "bh", "intake", "moves", "over", "gone"):
            setattr(self, name, torch.zeros(B, dtype=torch.int32, device=dev))
        self.scratch = torch.ones(B + self.BLOCK, dtype=torch.int32, device=dev)   # CAS target of lanes that claim nothing
        # occupancy (0 empty, -1 claimed by a birth, lane + 1); one spare element at the end receives
        # the scatters of lanes that change nothing in `_exchange`
        self._occ_buf = torch.zeros(self.NW * self.G * self.G + 1, dtype=torch.int32, device=dev)
        self.occ = self._occ_buf[:-1].view(self.NW, self.G * self.G)
        for name in ("xreq", "xcell", "xdir"):
            setattr(self, name, torch.zeros(B, dtype=torch.int32, device=dev))
        self.exchanges_total = torch.zeros(self.NW, dtype=torch.int64, device=dev)
        self.exchanges_lost = torch.zeros(self.NW, dtype=torch.int64, device=dev)
        self.free = torch.full((self.NW,), SL, dtype=torch.int32, device=dev)
        self.cellF = cellF_table(self.V).to(dev)
        self.seed = torch.tensor([(seed * 7919 + 17) % 2000000000], dtype=torch.int32, device=dev)
        self.fn, self.ncol = kernel(T, K)
        self.K = K
        self.rates_buf = torch.zeros(B, self.ncol, dtype=torch.float32, device=dev)
        # lineage bookkeeping (host)
        self.gen = np.zeros(B, dtype=np.int64)
        self.born_at = np.zeros(B, dtype=np.int64)
        self.lineage = np.full(B, -1, dtype=np.int64)
        self.tick = 0
        self.births_total = np.zeros(self.NW, dtype=np.int64)
        self.deaths_total = np.zeros(self.NW, dtype=np.int64)

    # genome -> lane parameters
    def _set_params(self, lanes, theta):
        T, dev = self.T, self.dev
        g = self.coder.decode(theta)
        L = torch.as_tensor(lanes, device=dev, dtype=torch.long)
        phi = torch.zeros(len(lanes), 6, T.SMAX, dtype=torch.float32, device=dev)
        gp = torch.tensor(g["phi"], dtype=torch.float32, device=dev)
        for c, (p, s) in enumerate(T.SELECTED_PHI):
            phi[:, p, s] = gp[:, c]
        phi[:, NO.S, 1] = T.PHI_S1
        self.phi[L] = phi
        self.kr[L] = torch.exp(torch.tensor(g["logk_rule"], dtype=torch.float32, device=dev))
        nleak = len(T.EXCH) - T.N_SELECTED_X
        self.kx[L] = torch.cat([torch.exp(torch.tensor(g["logk_x"], dtype=torch.float32, device=dev)),
                                torch.full((len(lanes), nleak), math.exp(T.LOGK_LEAK), device=dev)], 1)
        self.kflip[L] = torch.exp(torch.tensor(g["log_flip"], dtype=torch.float32, device=dev))
        self.kmove[L] = torch.exp(torch.tensor(g["log_move"], dtype=torch.float32, device=dev))
        self.kdiv[L] = torch.exp(torch.tensor(g["log_div"], dtype=torch.float32, device=dev))
        self.theta[lanes] = theta

    def _state(self, kind):
        """Inner state counts: 'adult' (intact membrane, half-full reserve) or 'newborn' (intact
        membrane, everything else empty: Lean `yb`)."""
        T = self.T
        n = torch.zeros(6, T.SMAX, dtype=torch.int32)
        for p, u in enumerate(T.UNITS):
            n[p, 0] = u
        n[NO.S, 0], n[NO.S, 1] = 0, T.UNITS[NO.S]
        if kind == "adult":
            n[NO.E, 0], n[NO.E, 1] = T.UNITS[NO.E] - T.UNITS[NO.E] // 2, T.UNITS[NO.E] // 2
        return n.to(self.dev)

    def place(self, world, cell, theta, kind="adult", h=None, parent=None):
        """Put a body into an empty cell of `world` (the user's hand, or a birth). Returns lane."""
        occ = self.occ[world]
        if int(occ[cell]) not in (0, -1):
            return -1
        alive = self.alive[world * self.SL:(world + 1) * self.SL]
        free = torch.nonzero(alive == 0).flatten()
        if len(free) == 0:
            return -1
        lane = world * self.SL + int(free[0])
        self.n[lane] = self._state(kind)
        self.x[lane], self.y[lane] = cell // self.G, cell % self.G
        self.h[lane] = int(self.rng.integers(4)) if h is None else int(h)
        self.alive[lane] = 1
        for name in ("birth", "bh", "intake", "moves", "over", "gone"):
            getattr(self, name)[lane] = 0
        self._set_params([lane], np.asarray(theta, dtype=float)[None])
        occ[cell] = lane + 1
        self.gen[lane] = 0 if parent is None else self.gen[parent] + 1
        self.lineage[lane] = lane if parent is None else self.lineage[parent]
        self.born_at[lane] = self.tick
        return lane

    def seed_population(self, theta, per_world, kind="adult"):
        G2 = self.G * self.G
        for wld in range(self.NW):
            cells = self.rng.choice(G2, size=per_world, replace=False)
            for c in cells:
                self.place(wld, int(c), theta, kind)
        self._sync_free()

    def _sync_free(self):
        self.free.copy_(self.SL - self.alive.view(self.NW, self.SL).sum(1).to(torch.int32))

    def mutate(self, theta):
        t = np.array(theta, dtype=float)
        if self.rng.random() < self.pmut:
            k = int(self.rng.integers(len(t)))
            t[k] = np.clip(np.round((t[k] + self.delta * self.rng.choice((-1, 1))) / self.delta) * self.delta, 0, 1)
        return t

    def leap(self, rates_only=False):
        grid = (triton.cdiv(self.B, self.BLOCK),)
        self.xreq.zero_()
        self.fn[grid](self.n, self.phi, self.kr, self.kx, self.kflip, self.kmove, self.kdiv, self.x, self.y, self.h,
                      self.alive, self.occ, self.free, self.birth, self.bh, self.scratch, self.cellF,
                      self.W.f, self.W.w, self.seed, self.intake, self.moves, self.over, self.gone, self.rates_buf,
                      self.xreq, self.xcell, self.xdir,
                      self.B, self.SL, self.G, self.V, self.W.dt, self.foodE, self.BODY, self.BQ,
                      self.T.UNITS[NO.S], float(self.T.PHI_S1), self.KDIS, self.K, RATES_ONLY=rates_only,
                      BLOCK=self.BLOCK)
        if not rates_only:
            self._exchange()

    def _exchange(self):
        """Resolve the leap's exchange requests on the device (no host sync). A request of lane `a`
        to swap with lane `t` holds if `t` is alive and still in the requested cell, filed no request
        itself, and `a` is the lowest lane asking for `t`; then the two bodies swap cells. Otherwise
        the mover's fuel step is returned (as for a lost claim of an empty cell)."""
        B, G, dev = self.B, self.G, self.dev
        lane = torch.arange(B, device=dev)
        has = self.xreq > 0
        t = (self.xreq - 1).clamp(min=0).long()
        base = (lane // self.SL) * G * G
        cell_a = base + self.x.long() * G + self.y.long()
        cell_t = base + self.x[t].long() * G + self.y[t].long()
        occf = self._occ_buf
        xc = self.xcell.long().clamp(min=0, max=occf.numel() - 1)
        valid = (has & (self.alive[t] == 1) & (self.alive == 1) & (cell_t == xc)
                 & (occf[xc] == self.xreq) & (self.xreq[t] == 0))
        win = torch.full((B,), B, dtype=torch.long, device=dev)
        win.scatter_reduce_(0, t, torch.where(valid, lane, torch.full_like(lane, B)), reduce="amin")
        acc = valid & (win[t] == lane)
        # positions: the mover takes the target's cell, the target the mover's
        for name in ("x", "y"):
            v = getattr(self, name)
            tmp = torch.full((B + 1,), -1, dtype=v.dtype, device=dev)
            tmp.scatter_(0, torch.where(acc, t, torch.full_like(t, B)), v)
            new = torch.where(tmp[:B] >= 0, tmp[:B], v)
            new = torch.where(acc, v[t], new)
            v.copy_(new)
        spare = torch.full_like(cell_a, occf.numel() - 1)
        occf.scatter_(0, torch.where(acc, cell_a, spare), (t + 1).to(torch.int32))
        occf.scatter_(0, torch.where(acc, cell_t, spare), (lane + 1).to(torch.int32))
        occf[-1] = 0
        self.moves += acc.to(torch.int32)
        # lost requests: return the fuel step (forward: (a1, a2) -> (b1, b2), backward the reverse)
        lost = has & ~acc
        i_, j_, a1, a2, b1, b2 = self.T.FUEL
        d = (lost & (self.xdir == 1)).to(torch.int32) - (lost & (self.xdir == -1)).to(torch.int32)
        n = self.n
        n[:, i_, a1] += d
        n[:, j_, a2] += d
        n[:, i_, b1] -= d
        n[:, j_, b2] -= d
        w = lane // self.SL
        self.exchanges_total.index_add_(0, w, acc.long())
        self.exchanges_lost.index_add_(0, w, lost.long())

    def end_tick(self):
        """Births claimed during the tick become lanes; dissolved lanes are free."""
        self.tick += 1
        gone = self.gone.cpu().numpy()
        if gone.any():
            idx = np.nonzero(gone)[0]
            np.add.at(self.deaths_total, idx // self.SL, 1)
            self.gone.zero_()
        req = self.birth.cpu().numpy()
        if req.any():
            for parent in np.nonzero(req)[0]:
                wld = parent // self.SL
                cell = int(req[parent]) - 1
                child = self.mutate(self.theta[parent])
                lane = self.place(wld, cell, child, kind="newborn", h=int(self.bh[parent]), parent=parent)
                assert lane >= 0, "a claimed birth found no slot"
                self.births_total[wld] += 1
            self.birth.zero_()
        self._sync_free()

    def counts(self):
        a = self.alive.view(self.NW, self.SL)
        gate = self.n[:, NO.S, 1].view(self.NW, self.SL)
        return (a.sum(1).cpu().numpy(), ((a == 1) & (gate > 0)).sum(1).cpu().numpy())
