"""Finite reversible exchange along Claude's flow/pool/turnover framework.

No learned objective or biological component is required. Contact occupancy,
support composition and finite driving composition are explicit. All energies
are in kBT units at one fixed bath temperature and reference volume one.
This subsystem does not yet evolve the brain's structural conductance.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.special import rel_entr, xlogy

from .material_reactions import material_slot_step


@dataclass(frozen=True)
class ExchangeParams:
    binding_rate: float = .01
    transport_rate: float = .001
    release_rate: float = .001
    bound_energy: float = -2.
    carried_energy: float = 0.
    charged_energy: float = 4.

    def __post_init__(self):
        if not all(np.isfinite(x) for x in asdict(self).values()):
            raise ValueError('Finite rates and dimensionless energies required')
        if min(self.binding_rate, self.transport_rate, self.release_rate, self.charged_energy) < 0:
            raise ValueError('Nonnegative rates and charging energy required')
        for exponent in (self.bound_energy,
                         self.carried_energy-self.bound_energy-self.charged_energy,
                         -self.carried_energy):
            with np.errstate(over='ignore', under='ignore'):
                value = np.exp(exponent)
            if not np.isfinite(value) or value <= 0:
                raise ValueError('Energy differences exceed reversible float64 rate range')


class FiniteExchange:
    """F+vacancy <-> C; C+P <-> H+vacancy+D; H <-> F.

    F/C/H conserve one support composition. P/D conserve one drive composition
    per node. The second event draws on finite P rather than a free one-way
    transport rate. Rates encode a declared ideal-mixture/finite-site effective
    material, not a universal law or a calibrated biological implementation.
    State and work arrays scale as O(N+E); no trajectory or Jacobian is kept.
    """

    _arrays = ('free', 'bound', 'carried', 'charged', 'depleted',
               'initial_owned', 'initial_drive', 'binding_net')

    def __init__(self, pre, post, capacity, free, bound, carried, charged,
                 depleted, params=None):
        self.params = params or ExchangeParams()
        pre, post = np.asarray(pre), np.asarray(post)
        if (pre.ndim != 1 or pre.shape != post.shape
                or not np.issubdtype(pre.dtype, np.integer)
                or not np.issubdtype(post.dtype, np.integer)):
            raise ValueError('Integer edge endpoints required')
        self.pre, self.post = pre.astype(np.int64).copy(), post.astype(np.int64).copy()
        self.free = np.asarray(free, float).copy()
        if self.free.ndim != 1:
            raise ValueError('One scalar free pool per node required')
        self.n = len(self.free)
        if (np.any(pre < 0) or np.any(pre >= self.n)
                or np.any(post < 0) or np.any(post >= self.n)):
            raise ValueError('Endpoint outside node panel')
        self.capacity = np.broadcast_to(np.asarray(capacity, float), pre.shape).copy()
        self.bound = np.asarray(bound, float).copy()
        self.carried, self.charged, self.depleted = (
            np.asarray(a, float).copy() for a in (carried, charged, depleted))
        self._validate()
        self.initial_owned = self.owned()
        self.initial_drive = self.charged + self.depleted
        self.binding_net = np.zeros(len(pre))
        self.initial_energy = self.energy()
        self.initial_free_energy = self.free_energy()
        self.heat_out = 0.
        self.entropy_balance = 0.
        self.t = 0.

    def _validate(self):
        if (self.bound.shape != self.pre.shape
                or any(a.shape != self.free.shape for a in
                       (self.carried, self.charged, self.depleted))):
            raise ValueError('Pool shapes differ from node/edge layout')
        for value in (self.free, self.bound, self.capacity, self.carried,
                      self.charged, self.depleted):
            if not np.isfinite(value).all() or np.any(value < 0):
                raise ValueError('Finite nonnegative occupancies required')
        if np.any(self.bound > self.capacity):
            raise ValueError('Bound occupancy exceeds finite contact capacity')

    def _sum(self, indices, values):
        return np.bincount(indices, weights=values, minlength=self.n)

    def owned(self):
        return self.free + self.carried + self._sum(self.pre, self.bound)

    def balances(self):
        return dict(support=self.owned()-self.initial_owned
                    - self._sum(self.pre, self.binding_net)
                    + self._sum(self.post, self.binding_net),
                    drive=self.charged+self.depleted-self.initial_drive,
                    energy=self.energy()+self.heat_out-self.initial_energy,
                    entropy=self.initial_free_energy-self.free_energy()-self.entropy_balance)

    def energy(self):
        p = self.params
        return float(p.bound_energy*self.bound.sum()
                     + p.carried_energy*self.carried.sum()
                     + p.charged_energy*self.charged.sum())

    def free_energy(self):
        dilute = sum(float(np.sum(xlogy(a, a)-a)) for a in
                     (self.free, self.carried, self.charged, self.depleted))
        empty = self.capacity-self.bound
        sites = np.sum(xlogy(self.bound, self.bound)+xlogy(empty, empty)
                       -xlogy(self.capacity, self.capacity))
        return float(dilute+sites+self.energy())

    def fluxes(self):
        """Forward/backward event rates; zero-boundary affinities may diverge."""
        p = self.params
        empty = self.capacity-self.bound
        return dict(binding=(p.binding_rate*self.free[self.post]*empty,
                             p.binding_rate*np.exp(p.bound_energy)*self.bound),
                    transport=(p.transport_rate*self.bound*self.charged[self.pre],
                               p.transport_rate*np.exp(p.carried_energy-p.bound_energy-p.charged_energy)
                               *self.carried[self.pre]*self.depleted[self.pre]*empty),
                    release=(p.release_rate*self.carried,
                             p.release_rate*np.exp(-p.carried_energy)*self.free))

    def entropy_rate(self):
        return float(sum(np.sum(rel_entr(a, b)+rel_entr(b, a))
                         for a, b in self.fluxes().values()))

    def _transport(self, dt):
        p = self.params
        if p.transport_rate == 0 or not len(self.pre):
            return
        old = self.bound
        total = self._sum(self.pre, old)
        slots = self._sum(self.pre, self.capacity)
        low = np.maximum.reduce((-self.carried, -self.depleted, total-slots))
        high = np.minimum(self.charged, total)
        backward = p.transport_rate*np.exp(p.carried_energy-p.bound_energy-p.charged_energy)

        def occupancy(x):
            plus = dt*p.transport_rate*(self.charged-x)[self.pre]
            minus = dt*backward*((self.carried+x)*(self.depleted+x))[self.pre]
            return (old + minus*self.capacity)/(1+plus+minus)

        # F'(X)>=1: all donor groups have a unique feasible implicit extent.
        for _ in range(64):
            mid = (low+high)*.5
            residual = mid+self._sum(self.pre, occupancy(mid))-total
            below = residual < 0
            low = np.where(below, mid, low)
            high = np.where(below, high, mid)
        x = (low+high)*.5
        self.bound = occupancy(x)
        self.carried += x
        self.charged -= x
        self.depleted += x

    def step(self, dt):
        """First-order splitting, two implicit groups and one exact pair.

        The free-energy balance includes discrete numerical dissipation. Step
        refinement is needed to identify the continuous-model dissipation.
        No externally replenished energy or material enters this closed kernel.
        """
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('Positive finite elapsed time required')
        p = self.params
        before_g, before_u = self.free_energy(), self.energy()
        if p.binding_rate and len(self.pre):
            bound_before = self.bound
            result = material_slot_step(self.free, self.bound, self.post, self.capacity,
                p.binding_rate, p.binding_rate*np.exp(p.bound_energy), dt)
            self.free, self.bound = result.free, result.structure
            self.binding_net += self.bound-bound_before
        self._transport(dt)
        if p.release_rate:
            forward = p.release_rate
            backward = forward*np.exp(-p.carried_energy)
            rate = forward+backward
            decay, exchange = np.exp(-dt*rate), -np.expm1(-dt*rate)
            total = self.carried+self.free
            carried = self.carried*decay+total*(backward/rate)*exchange
            self.free = self.free*decay+total*(forward/rate)*exchange
            self.carried = carried
        self._validate()
        after_g, after_u = self.free_energy(), self.energy()
        self.heat_out += before_u-after_u
        produced = before_g-after_g
        self.entropy_balance += produced
        self.t += dt
        return dict(free_energy_drop=produced, heat_out=before_u-after_u)

    def state_dict(self):
        return dict(schema=1, params=asdict(self.params), pre=self.pre.copy(),
                    post=self.post.copy(), capacity=self.capacity.copy(),
                    arrays={name:getattr(self, name).copy() for name in self._arrays},
                    initial_energy=self.initial_energy, initial_free_energy=self.initial_free_energy,
                    heat_out=self.heat_out, entropy_balance=self.entropy_balance, t=self.t)

    @classmethod
    def from_state_dict(cls, state):
        if state['schema'] != 1:
            raise ValueError('Unsupported exchange archive schema')
        a = state['arrays']
        obj = cls(state['pre'], state['post'], state['capacity'], a['free'], a['bound'],
                  a['carried'], a['charged'], a['depleted'], ExchangeParams(**state['params']))
        for name in cls._arrays:
            value = np.asarray(a[name], float)
            if value.shape != getattr(obj, name).shape or not np.isfinite(value).all():
                raise ValueError('Invalid exchange state: '+name)
            setattr(obj, name, value.copy())
        for name in ('initial_energy', 'initial_free_energy', 'heat_out', 'entropy_balance', 't'):
            if not np.isfinite(state[name]):
                raise ValueError('Nonfinite exchange ledger')
            setattr(obj, name, float(state[name]))
        obj._validate()
        if any(np.max(np.abs(value), initial=0.) > 1e-8 for value in obj.balances().values()):
            raise ValueError('Exchange archive fails composition/energy accounting')
        return obj
