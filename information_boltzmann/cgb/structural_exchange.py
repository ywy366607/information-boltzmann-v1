"""Finite U/S/C positions couple support occupancy and structural turnover.

The minimal transition graph disassembles only unbound installed positions;
neglecting direct bound disassembly is an explicit barrier/time-scale model
assumption. It is not imposed by conservation alone. No utility score occurs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import xlogy

from .finite_exchange import ExchangeParams, FiniteExchange


@dataclass(frozen=True)
class StructuralExchangeParams(ExchangeParams):
    assembly_rate: float = .001
    structure_energy: float = 2.

    def __post_init__(self):
        super().__post_init__()
        if self.assembly_rate < 0:
            raise ValueError('Nonnegative assembly rate required')
        with np.errstate(over='ignore', under='ignore'):
            rate = np.exp(self.structure_energy-self.charged_energy)
        if not np.isfinite(rate) or rate <= 0:
            raise ValueError('Structural energy difference exceeds reversible float64 range')


class StructuralExchange(FiniteExchange):
    _arrays = FiniteExchange._arrays + ('positions', 'material', 'initial_material')

    def __init__(self, pre, post, capacity, free, bound, carried, charged,
                 depleted, positions, material, params=None):
        self.positions = np.broadcast_to(np.asarray(positions, float), np.shape(bound)).copy()
        self.material = np.broadcast_to(np.asarray(material, float), np.shape(free)).copy()
        super().__init__(pre, post, capacity, free, bound, carried, charged,
                         depleted, params or StructuralExchangeParams())
        self.initial_material = self.material+self._sum(self.post, self.capacity)

    def _validate(self):
        super()._validate()
        if (self.positions.shape != self.capacity.shape or self.material.shape != self.free.shape
                or not np.isfinite(self.positions).all() or np.any(self.positions < self.capacity)
                or not np.isfinite(self.material).all() or np.any(self.material < 0)):
            raise ValueError('Finite nonnegative material and installed positions <= maximum required')

    def energy(self):
        return super().energy()+float(self.params.structure_energy*self.capacity.sum())

    def free_energy(self):
        uninstalled = self.positions-self.capacity
        correction = (np.sum(xlogy(self.capacity, self.capacity)+xlogy(uninstalled, uninstalled)
                             -xlogy(self.positions, self.positions))
                      +np.sum(xlogy(self.material, self.material)-self.material))
        # super().free_energy calls the extended energy() exactly once.
        return float(super().free_energy()+correction)

    def balances(self):
        values = super().balances()
        values['material'] = self.material+self._sum(self.post, self.capacity)-self.initial_material
        return values

    def fluxes(self):
        values = super().fluxes()
        p = self.params
        values['assembly'] = (
            p.assembly_rate*self.material[self.post]*(self.positions-self.capacity)*self.charged[self.post],
            p.assembly_rate*np.exp(p.structure_energy-p.charged_energy)
            *(self.capacity-self.bound)*self.depleted[self.post])
        return values

    def _assembly(self, dt):
        p = self.params
        if p.assembly_rate == 0 or not len(self.pre):
            return
        old = self.capacity
        removable = self._sum(self.post, old-self.bound)
        empty = self._sum(self.post, self.positions-old)
        low = np.maximum(-self.depleted, -removable)
        high = np.minimum.reduce((self.material, self.charged, empty))
        backward = p.assembly_rate*np.exp(p.structure_energy-p.charged_energy)

        def installed(x):
            plus = dt*p.assembly_rate*((self.material-x)*(self.charged-x))[self.post]
            minus = dt*backward*(self.depleted+x)[self.post]
            return (old+plus*self.positions+minus*self.bound)/(1+plus+minus)

        for _ in range(64):
            mid = (low+high)*.5
            residual = mid-self._sum(self.post, installed(mid)-old)
            below = residual < 0
            low = np.where(below, mid, low)
            high = np.where(below, high, mid)
        x = (low+high)*.5
        self.capacity = installed(x)
        self.material -= x
        self.charged -= x
        self.depleted += x

    def step(self, dt):
        before_g, before_u = self.free_energy(), self.energy()
        super().step(dt)
        assembly_g, assembly_u = self.free_energy(), self.energy()
        self._assembly(dt)
        self._validate()
        after_g, after_u = self.free_energy(), self.energy()
        self.heat_out += assembly_u-after_u
        self.entropy_balance += assembly_g-after_g
        return dict(free_energy_drop=before_g-after_g, heat_out=before_u-after_u)

    def state_dict(self):
        state = super().state_dict()
        state['kind'] = 'structural_exchange'
        return state

    @classmethod
    def from_state_dict(cls, state):
        if state['schema'] != 1 or state.get('kind') != 'structural_exchange':
            raise ValueError('Unsupported structural exchange archive')
        a = state['arrays']
        obj = cls(state['pre'], state['post'], state['capacity'], a['free'], a['bound'],
                  a['carried'], a['charged'], a['depleted'], a['positions'], a['material'],
                  StructuralExchangeParams(**state['params']))
        for name in cls._arrays:
            value = np.asarray(a[name], float)
            if value.shape != getattr(obj, name).shape or not np.isfinite(value).all():
                raise ValueError('Invalid structural exchange state: '+name)
            setattr(obj, name, value.copy())
        for name in ('initial_energy', 'initial_free_energy', 'heat_out', 'entropy_balance', 't'):
            if not np.isfinite(state[name]):
                raise ValueError('Nonfinite structural exchange ledger')
            setattr(obj, name, float(state[name]))
        obj._validate()
        if any(np.max(np.abs(value), initial=0.) > 1e-8 for value in obj.balances().values()):
            raise ValueError('Structural exchange archive fails composition/energy accounting')
        return obj
