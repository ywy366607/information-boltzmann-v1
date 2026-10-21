"""Conserved E9 capture -> retrograde transport -> re-release.

This is an opt-in effective kinetic candidate, not a new utility score. Each
internal source debits its donor pool. F and T remain the medium's existing
arrays; only the cell cargo H is a new physical pool. Frozen-rate Lie splitting
is first order in time. Transport is exponential residence, not cable delay.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class SupportTransportParams:
    transport_rate: float = 1.0 / 1000.0
    release_rate: float = 1.0 / 1000.0
    cargo_clearance: float = 1.0 / 20000.0

    def __post_init__(self):
        for value in asdict(self).values():
            if not np.isfinite(value) or value < 0:
                raise ValueError('Transport, release and clearance rates must be finite and nonnegative')


def _fraction(rate, dt):
    """Integral of a unit pool with competing first-order exit rates."""
    result = np.full_like(np.asarray(rate, dtype=float), dt)
    np.divide(-np.expm1(-dt * rate), rate, out=result, where=rate > 0)
    return result


class SupportTransport:
    """O(N+E) state/workspace; no stored history or dense graph solver.

    Ownership: captured T_e belongs to pre(e), while free F_i belongs to i.
    Consequently capture is the cross-owner transfer; cargo transit and release
    are internal transfers. This convention makes any module's balance exact.
    Material/energy supporting synthesis of external q remains outside this
    support ledger; conservation here does not establish energetic closure.
    """

    _fields = ('source', 'cargo', 'initial', 'imported', 'cleared', 'spent',
               'unpaid', 'captured', 'transported', 'rereleased')

    def __init__(self, n, pre, post, free, bound, source, params=None):
        self.params = params or SupportTransportParams()
        self.n = int(n)
        self.pre = np.asarray(pre, dtype=np.int64).copy()
        self.post = np.asarray(post, dtype=np.int64).copy()
        if (self.n < 0 or self.pre.ndim != 1 or self.pre.shape != self.post.shape
                or np.any(self.pre < 0) or np.any(self.pre >= n)
                or np.any(self.post < 0) or np.any(self.post >= n)):
            raise ValueError('Valid directed endpoints required')
        self.source = np.broadcast_to(np.asarray(source, float), (n,)).copy()
        free, bound = np.asarray(free), np.asarray(bound)
        if free.shape != (n,) or bound.shape != self.pre.shape:
            raise ValueError('One free pool per cell and captured pool per edge required')
        for value in (self.source, free, bound):
            if not np.isfinite(value).all() or np.any(value < 0):
                raise ValueError('Finite nonnegative support pools and external sources required')
        self.cargo = np.zeros(n)
        self.initial = free + self._sum(self.pre, bound)
        self.imported = np.zeros(n)
        self.cleared = np.zeros(n)
        self.spent = np.zeros(n)
        self.unpaid = np.zeros(n)
        self.captured = np.zeros(len(self.pre))
        self.transported = np.zeros(n)
        self.rereleased = np.zeros(n)

    def _sum(self, indices, values):
        return np.bincount(indices, weights=values, minlength=self.n)

    def advance(self, free, bound, carrier, structure, enabled, dt, trophic):
        """Advance existing F/T in place; return actual per-channel amounts.

        `carrier` is inherited positive tag or activity, not a global reward.
        Maintenance is an availability-limited sink, recorded without inventing
        a debt-triggered death rule. No q is minted on internal re-release.
        """
        p = self.params
        carrier = np.asarray(carrier, float)
        structure = np.asarray(structure, float)
        enabled = np.asarray(enabled, bool)
        if (free.shape != (self.n,) or bound.shape != self.pre.shape
                or carrier.shape != bound.shape or structure.shape != bound.shape
                or enabled.shape != bound.shape):
            raise ValueError('Support update shape mismatch')
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('Positive finite elapsed time required')
        if (not np.isfinite(trophic.kappa) or trophic.kappa < 0
                or not np.isfinite(trophic.phi) or trophic.phi < 0
                or np.isnan(trophic.tau_N) or trophic.tau_N <= 0
                or np.isnan(trophic.tau_T) or trophic.tau_T <= 0):
            raise ValueError('Nonnegative capture/cost and positive clearance lifetimes required')
        for value in (free, bound, carrier, structure, self.source, self.cargo):
            if not np.isfinite(value).all() or np.any(value < 0):
                raise ValueError('Finite nonnegative pools, sources, tags and structure required')
        rates = trophic.kappa * carrier * enabled
        total_capture = self._sum(self.post, rates)
        requests = dt * trophic.phi * structure * enabled
        if not np.isfinite(total_capture).all() or not np.isfinite(requests).all():
            raise ValueError('Nonfinite aggregate capture or maintenance rate')

        injected = dt * self.source
        if not np.isfinite(injected).all():
            raise ValueError('Nonfinite external injection')
        free += injected
        self.imported += injected

        # Old soma cargo: release and clearance compete for the same molecules.
        amount = self.cargo * _fraction(np.asarray(p.release_rate + p.cargo_clearance), dt)
        released = amount * p.release_rate
        cargo_loss = amount * p.cargo_clearance
        self.cargo *= np.exp(-dt * (p.release_rate + p.cargo_clearance))
        free += released
        self.rereleased += released

        # Captured terminal cargo moves to its owning presynaptic cell.
        terminal_decay = 1.0 / trophic.tau_T
        amount = bound * _fraction(np.asarray(p.transport_rate + terminal_decay), dt)
        transported = amount * p.transport_rate
        terminal_loss = amount * terminal_decay
        bound *= np.exp(-dt * (p.transport_rate + terminal_decay))
        received = self._sum(self.pre, transported)
        self.cargo += received
        self.transported += received

        # All competing terminals see the same donor; no edge-order priority.
        free_decay = 1.0 / trophic.tau_N
        integral = free * _fraction(total_capture + free_decay, dt)
        captured = rates * integral[self.post]
        free_loss = free_decay * integral
        free *= np.exp(-dt * (total_capture + free_decay))
        bound += captured
        self.captured += captured

        paid = np.minimum(requests, bound)
        bound -= paid
        self.spent += self._sum(self.pre, paid)
        self.unpaid += self._sum(self.pre, requests - paid)
        cleared = cargo_loss + free_loss + self._sum(self.pre, terminal_loss)
        self.cleared += cleared
        return {'captured': captured, 'transported': transported,
                'released': released, 'cleared': cleared, 'maintenance': paid}

    def balance(self, free, bound):
        """Per-owner residual, including all cumulative boundary transfers."""
        inventory = free + self.cargo + self._sum(self.pre, bound)
        return (inventory + self.cleared + self.spent - self.initial - self.imported
                - self._sum(self.pre, self.captured)
                + self._sum(self.post, self.captured))

    def state_dict(self):
        return {'schema': 1, 'params': asdict(self.params), 'n': self.n,
                'pre': self.pre.copy(), 'post': self.post.copy(),
                'arrays': {name: getattr(self, name).copy() for name in self._fields}}

    @classmethod
    def from_state_dict(cls, state, free, bound):
        if state['schema'] != 1:
            raise ValueError('Unsupported support transport state schema')
        arrays = state['arrays']
        obj = cls(state['n'], state['pre'], state['post'], free, bound,
                  arrays['source'], SupportTransportParams(**state['params']))
        for name in cls._fields:
            value = np.asarray(arrays[name], dtype=float)
            if value.shape != getattr(obj, name).shape or not np.isfinite(value).all() or np.any(value < 0):
                raise ValueError('Invalid support transport archive: ' + name)
            setattr(obj, name, value.copy())
        # A persistent lifetime ledger can greatly exceed present inventory;
        # allow summation roundoff scaled to actual traffic, not a fixed age cap.
        traffic = (1. + obj.initial + obj.imported + obj.cleared + obj.spent
                   + free + obj.cargo + obj._sum(obj.pre, bound)
                   + obj._sum(obj.pre, obj.captured) + obj._sum(obj.post, obj.captured))
        # Pool integration and ledger summation both accumulate roundoff; retain
        # the interface's absolute tolerance as well as traffic-relative error.
        if np.any(np.abs(obj.balance(free, bound)) > 1e-9 + 128 * np.finfo(float).eps * traffic):
            raise ValueError('Support transport archive does not balance')
        return obj
