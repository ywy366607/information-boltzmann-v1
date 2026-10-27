"""Event-driven execution of the open local medium's generator (contract v2:
CGB_UNIFIED_REFERENCE_20261026, "事件驱动执行与按事件计费", 2026-10-27).

Rates are those of `OpenMedium.prepare`: r_i = kappa_family * 2**exponent_i, zero when a
capacity or an empty stock blocks the move. Every family is closed under reverse and kappa
is symmetric within a reverse pair, so any kappa_family > 0 keeps pi ~ 2**(-2E) (uniform
mu); kappa sets the transients (relative family clocks).

Two executions of the same rates:
- `evolve`: physical time tau, next-event method, one fee per event that happens.
- `evolve_attempts`: the attempt-counted chain of `OpenMedium.tick_window` without rejected
  attempts (n-fold way): with matched kappa = 1 / (F * N_family * 2**L_family) the
  probability of event i per attempt is exactly the old one, gaps between accepted moves
  are geometric, and every attempt is still paid.

- `evolve_first_reaction`: every enabled event keeps its own exponential clock and the earliest
  fires (Gillespie's first-reaction method). It reads no global sum; drawing the time from the
  total rate and the event in proportion to its rate (`evolve`) gives the same law (the minimum
  of independent exponentials), which the validation checks against `expm`.

The rate vector, its sum and the p/q/w arrays are derived execution storage, not body state;
`check_sync` compares the arrays with the packed body. This reference implementation rebuilds
all rates after every event (O(events) work per event); a clique-local update is not done.
"""

import math
import random

import numpy as np

FAMILIES = 5  # cross-end, plastic q<->w, P transport, Q transport, exchange


def matched_conductance(opened):
    """kappa per family that reproduces the attempt chain of `sample_proposal` + dyadic tickets."""
    f = opened.body.law
    sizes = (4 * f.edges, 2 * f.edges, f.edges * (f.labels - 1), f.edges * (f.rows - 1), 2 * f.edges)
    bounds = tuple(f.bounds) + (opened.exchange_bound,)
    F = sum(1 for n in sizes if n)
    return tuple(1.0 / (F * n * 2.0 ** L) if n else 0.0 for n, L in zip(sizes, bounds))


class ContinuousOpenMedium:
    def __init__(self, opened, *, tau, max_events, family_conductance, seed):
        if (type(max_events) is not int or max_events < 0 or not tau >= 0
                or len(family_conductance) != FAMILIES or any(k < 0 for k in family_conductance)):
            raise ValueError("invalid event-driven window")
        self.opened, self.tau, self.max_events = opened, float(tau), max_events
        self.kappa = tuple(float(k) for k in family_conductance)
        self.rng = random.Random(seed)
        self.family_events = [0] * FAMILIES        # observer counters, snapshotted with the rng
        self.intake_on_target = 0
        self.draws = 0                             # uniform draws taken from the rng
        f = opened.body.law
        self.H, self.B, self.Q, self.W, self.E = f.rows, f.labels, f.charge_cap, f.slot_cap, f.edges
        self.U = np.array(f.potential, dtype=np.int64)
        H, B = self.H, self.B
        self._tcol = np.array([[j + (j >= c) for j in range(B - 1)] for c in range(B)], dtype=np.int64).reshape(1, B, B - 1)
        self._trow = np.array([[j + (j >= r) for j in range(H - 1)] for r in range(H)], dtype=np.int64).reshape(H, 1, H - 1)
        self._sync()

    # ---------- state (derived arrays mirror the packed body) ----------
    def _sync(self):
        body = self.opened.body
        s = np.array([body.get(e) for e in range(self.E)], dtype=np.int64).reshape(self.H, self.B, 3)
        self.p, self.q, self.w = s[..., 0].copy(), s[..., 1].copy(), s[..., 2].copy()

    def _unit(self, p, q, w):
        f = self.opened.body.law
        return f.epsilon * w - f.coupling * w * p * q

    def _Uat(self, n):
        return self.U[np.clip(np.asarray(n), 0, len(self.U) - 1)]

    @staticmethod
    def _rate(kappa, ok, ex):
        return np.where(ok, kappa * np.exp2(np.where(ok, ex, 0).astype(np.float64)), 0.0)

    # ---------- rates ----------
    def rates(self, label):
        """All open events in `OpenMedium` index order (blocked events have rate 0)."""
        H, B, Q, W, E = self.H, self.B, self.Q, self.W, self.E
        p, q, w = self.p, self.q, self.w
        NP = p.sum(1)[:, None]
        NQ = q.sum(0)[None, :]
        u0 = self._unit(p, q, w)
        k0, k1, k2, k3, kx = self.kappa
        out = np.zeros(self.opened.events, dtype=np.float64)
        r0 = np.zeros((H, B, 4))
        for choice in range(4):                       # cross-end: index 4 * edge + choice
            s = 1 if choice < 2 else -1
            pn, qn = p - s, q + s
            ok = (pn >= 0) & (pn <= Q) & (qn >= 0) & (qn <= Q)
            du = self._unit(pn, qn, w) - u0
            dU = (self._Uat(NP - s) - self._Uat(NP)) if choice % 2 == 0 else (self._Uat(NQ + s) - self._Uat(NQ))
            r0[..., choice] = self._rate(k0, ok, -(du + 2 * dU))
        out[:4 * E] = r0.reshape(-1)
        r1 = np.zeros((H, B, 2))
        for choice in range(2):                       # plastic: 4E + 2 * edge + choice; 0 = q -> w
            s = 1 if choice == 0 else -1
            qn, wn = q - s, w + s
            ok = (qn >= 0) & (qn <= Q) & (wn >= 0) & (wn <= W)
            du = self._unit(p, qn, wn) - u0
            r1[..., choice] = self._rate(k1, ok, -(du + self._Uat(NQ - s) - self._Uat(NQ)))
        out[4 * E:6 * E] = r1.reshape(-1)
        base = 6 * E
        if B > 1:                                     # p moves to another column of the same row
            t = np.broadcast_to(self._tcol, (H, B, B - 1))
            rows = np.arange(H)[:, None, None]
            pt, qt, wt = p[rows, t], q[rows, t], w[rows, t]
            ok = (p[..., None] >= 1) & (pt + 1 <= Q)
            d = (self._unit(p - 1, q, w) - u0)[..., None] + self._unit(pt + 1, qt, wt) - self._unit(pt, qt, wt)
            out[base:base + E * (B - 1)] = self._rate(k2, ok, -d).reshape(-1)
        base += E * (B - 1)
        if H > 1:                                     # q moves to another row of the same column
            t = np.broadcast_to(self._trow, (H, B, H - 1))
            cols = np.arange(B)[None, :, None]
            pt, qt, wt = p[t, cols], q[t, cols], w[t, cols]
            ok = (q[..., None] >= 1) & (qt + 1 <= Q)
            d = (self._unit(p, q - 1, w) - u0)[..., None] + self._unit(pt, qt + 1, wt) - self._unit(pt, qt, wt)
            out[base:base + E * (H - 1)] = self._rate(k3, ok, -d).reshape(-1)
        base += E * (H - 1)
        res = self.opened.reservoir
        mu = res.chemical_base + res.stimulus * (np.arange(B) == label)[None, :]
        rx = np.zeros((H, B, 2))
        for d_ in range(2):                           # exchange: events + 2 * edge + direction; 0 = intake
            s = 1 if d_ == 0 else -1
            qn = q + s
            ok = (qn >= 0) & (qn <= Q)
            if s == 1 and self.opened.stock == 0:
                ok = np.zeros_like(ok)
            du = self._unit(p, qn, w) - u0
            ex = -(du + self._Uat(NQ + s) - self._Uat(NQ)) - (res.content_cost - mu) * s
            rx[..., d_] = self._rate(kx, ok, ex)
        out[base:base + 2 * E] = rx.reshape(-1)
        return out

    def rates_reference(self, label):
        """Oracle: one `OpenMedium.prepare` per event, times its family conductance."""
        o, f = self.opened, self.opened.body.law
        out = np.zeros(o.events)
        for i in range(o.events):
            _edits, exponent, _bound, _sign = o.prepare(i, label)
            if exponent is not None:
                out[i] = self.kappa[o.body.decode(i)[0] if i < f.events else 4] * 2.0 ** exponent
        return out

    # ---------- execution ----------
    def _apply(self, index, label):
        edits, exponent, _bound, sign = self.opened.prepare(index, label)
        assert exponent is not None, "a positive-rate event must be enabled"
        self.opened._commit(edits, sign)
        f = self.opened.body.law
        fam = 4 if index >= f.events else self.opened.body.decode(index)[0]
        self.family_events[fam] += 1                    # observer counters, not body state
        if fam == 4 and sign == 1:
            self.intake_on_target += int(edits[0][0] % self.B == label)
        for edge, (p, q, w) in edits:
            r, c = divmod(edge, self.B)
            self.p[r, c], self.q[r, c], self.w[r, c] = p, q, w
        return sign

    def _choose(self, r):
        c = np.cumsum(r)
        self.draws += 1
        i = min(int(np.searchsorted(c, self.rng.random() * c[-1], side="right")), len(r) - 1)
        while r[i] <= 0.0:          # float edge of the cumulative sum
            i -= 1
        return i

    def evolve(self, label, tau, pay=lambda: True, max_events=None):
        """Physical time tau at a fixed label. Returns (events, material_delta, capped); a refused
        payment ends the window (see `self.refused`)."""
        cap = self.max_events if max_events is None else max_events
        t, n, material = 0.0, 0, 0
        while True:
            r = self.rates(label)
            R = float(r.sum())
            if R <= 0.0:
                return n, material, False
            t += self.rng.expovariate(R)
            self.draws += 1
            if t > tau:
                return n, material, False
            if n >= cap:
                return n, material, True
            if not pay():
                self.refused = True
                return n, material, False
            material += self._apply(self._choose(r), label)
            n += 1

    def evolve_first_reaction(self, label, tau):
        """Local clocks: each enabled event draws its own exponential time; the earliest fires.
        Same law as `evolve` (no cap, no payment); used only to check the equivalence."""
        t, n = 0.0, 0
        while True:
            r = self.rates(label)
            idx = np.nonzero(r)[0]
            if len(idx) == 0:
                return n
            times = np.array([self.rng.expovariate(float(r[i])) for i in idx])
            self.draws += len(idx)
            k = int(np.argmin(times))
            t += float(times[k])
            if t > tau:
                return n
            self._apply(int(idx[k]), label)
            n += 1

    def check_sync(self):
        body = self.opened.body
        s = np.array([body.get(e) for e in range(self.E)], dtype=np.int64).reshape(self.H, self.B, 3)
        return bool((s[..., 0] == self.p).all() and (s[..., 1] == self.q).all() and (s[..., 2] == self.w).all())

    def evolve_attempts(self, label, attempts):
        """The attempt-counted chain without rejected attempts (needs matched kappa: sum of
        rates <= 1 is the acceptance probability of one attempt). Returns (accepted, material)."""
        used, n, material = 0, 0, 0
        while True:
            r = self.rates(label)
            R = float(r.sum())
            if R <= 0.0:
                return n, material
            assert R <= 1.0 + 1e-12, "attempt chain needs matched conductances"
            u = self.rng.random()
            self.draws += 1
            gap = 1 if R >= 1.0 else int(math.floor(math.log1p(-u) / math.log1p(-R))) + 1
            used += gap
            if used > attempts:
                return n, material
            material += self._apply(self._choose(r), label)
            n += 1

    def window(self, world):
        """One publication, then physical time tau with the new byte; each event is paid."""
        o = self.opened
        if o.body.law.labels != 256:
            raise ValueError("window requires the native 256-byte boundary")
        masses, total = o.masses()          # observer readout BEFORE the target
        boundary = world.begin()
        observed = world.publish()
        fee = o.body.law.attempt_fee
        paid = [0]
        self.refused = False
        draws0 = self.draws

        def pay():
            ok = world.pay_write(fee)
            paid[0] += int(ok)
            return ok
        if boundary.active:
            events, material, capped = self.evolve(observed.payload, self.tau, pay)
        else:                               # an inactive body does not react (as tick_window's refusal)
            events, material, capped = 0, 0, False
        return dict(tick=observed.tick, byte=observed.payload, active=boundary.active,
                    target_mass=masses[observed.payload], total_mass=total, events=events,
                    material_delta=material, capped=capped, refused=self.refused,
                    draws=self.draws - draws0, spent=paid[0] * fee)

    def state_dict(self):
        v, internal, gauss = self.rng.getstate()
        return dict(version=1, tau=self.tau, max_events=self.max_events, kappa=list(self.kappa),
                    opened=self.opened.state_dict(), rng=[v, list(internal), gauss],
                    counters=dict(family_events=list(self.family_events),
                                  intake_on_target=self.intake_on_target, draws=self.draws))

    def load_state_dict(self, snap):
        if (snap.get("version") != 1 or snap["tau"] != self.tau or snap["max_events"] != self.max_events
                or list(snap["kappa"]) != list(self.kappa)):
            raise ValueError("event-driven checkpoint mismatch")
        self.opened.load_state_dict(snap["opened"])
        v, internal, gauss = snap["rng"]
        self.rng.setstate((v, tuple(internal), gauss))
        c = snap["counters"]
        self.family_events, self.intake_on_target, self.draws = list(c["family_events"]), c["intake_on_target"], c["draws"]
        self._sync()
