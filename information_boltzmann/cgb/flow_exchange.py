"""Capacitive work <-> finite drive <-> installed parallel conductance.

One explicit isothermal material realization of the CGB physical core. This
is a reciprocal passive fast network with reversible transduction, not a
replacement for the existing directed spiking brain or its world interface.
No objective gradient or activity-dependent free supply appears here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg
from scipy.special import expit

from .structural_exchange import StructuralExchange, StructuralExchangeParams


@dataclass(frozen=True)
class FlowExchangeParams(StructuralExchangeParams):
    conversion_rate: float = .01
    charge_per_event: float = 1.

    def __post_init__(self):
        super().__post_init__()
        if self.conversion_rate < 0 or self.charge_per_event <= 0:
            raise ValueError('Nonnegative conversion rate and positive transferred charge required')


class FlowExchange(StructuralExchange):
    """Fixed capacitors, identical parallel units, and a two-state transducer.

    All arrays and matrix-free solver work scale as O(N+E). CG iterations can
    depend on graph conditioning; no constant per-event runtime is claimed.
    The reference electrode is an explicit zero-potential boundary. Charge
    returned there by D->P is tracked, rather than silently disappearing.
    """

    _arrays = StructuralExchange._arrays + (
        'charge', 'capacitance', 'unit_conductance', 'initial_charge',
        'injected_charge', 'returned_charge', 'transported_charge')
    _ledgers = ('initial_energy', 'initial_free_energy', 'heat_out',
                'entropy_balance', 't', 'work_in', 'conduction_heat',
                'conduction_numerical_heat', 'conduction_energy_drop', 'last_cg_iterations')

    def __init__(self, pre, post, capacity, free, bound, carried, charged,
                 depleted, positions, material, charge, capacitance=1.,
                 unit_conductance=.01, params=None):
        shape = np.shape(free)
        self.charge = np.broadcast_to(np.asarray(charge, float), shape).copy()
        self.capacitance = np.broadcast_to(np.asarray(capacitance, float), shape).copy()
        self.unit_conductance = np.broadcast_to(
            np.asarray(unit_conductance, float), np.shape(bound)).copy()
        self.initial_charge = self.charge.copy()
        self.injected_charge = np.zeros(shape)
        self.returned_charge = np.zeros(shape)
        self.transported_charge = np.zeros(shape)
        self.work_in = 0.
        self.conduction_heat = 0.
        self.conduction_numerical_heat = 0.
        self.conduction_energy_drop = 0.
        self.last_cg_iterations = 0
        super().__init__(pre, post, capacity, free, bound, carried, charged,
                         depleted, positions, material, params or FlowExchangeParams())

    def _validate(self):
        super()._validate()
        for name in ('charge', 'capacitance', 'initial_charge', 'injected_charge',
                     'returned_charge', 'transported_charge'):
            a = getattr(self, name)
            if a.shape != self.free.shape or not np.isfinite(a).all():
                raise ValueError('Finite nodal electrical state required: '+name)
        if (np.any(self.capacitance <= 0)
                or self.unit_conductance.shape != self.capacity.shape
                or not np.isfinite(self.unit_conductance).all()
                or np.any(self.unit_conductance < 0)):
            raise ValueError('Positive capacitance and nonnegative unit conductance required')
        # Reject rather than clip a field outside the float64 reversible-rate
        # range. expit is stable, but exactly zero rates lose finite affinities.
        if np.any(np.abs(self.params.charge_per_event*self.voltage
                         -self.params.charged_energy) > 600.):
            raise ValueError('Electrical affinity exceeds supported reversible-rate range')

    @property
    def voltage(self):
        return self.charge/self.capacitance

    @property
    def conductance(self):
        return self.unit_conductance*self.capacity

    def field_energy(self):
        return float(.5*np.dot(self.charge, self.voltage))

    def energy(self):
        return super().energy()+self.field_energy()

    def balances(self):
        result = super().balances()
        result['energy'] -= self.work_in
        result['entropy'] += self.work_in
        result['charge_local'] = (self.charge-self.initial_charge-self.injected_charge
                                  +self.returned_charge-self.transported_charge)
        result['charge_transport_global'] = self.transported_charge.sum()
        return result

    def fluxes(self):
        result = super().fluxes()
        p = self.params
        affinity = p.charge_per_event*self.voltage-p.charged_energy
        result['conversion'] = (p.conversion_rate*self.depleted*expit(affinity),
                                p.conversion_rate*self.charged*expit(-affinity))
        return result

    def entropy_rate(self):
        dv = self.voltage[self.pre]-self.voltage[self.post]
        return super().entropy_rate()+float(np.dot(self.conductance, dv*dv))

    def _conduction(self, dt):
        g = self.conductance
        self.last_cg_iterations = 0
        if not len(g) or not np.any(g):
            return
        old_q, old_v = self.charge.copy(), self.voltage

        def laplacian(v):
            j = g*(v[self.pre]-v[self.post])
            return self._sum(self.pre, j)-self._sum(self.post, j)

        diagonal = self.capacitance+dt*(self._sum(self.pre, g)+self._sum(self.post, g))
        operator = LinearOperator((self.n, self.n),
            matvec=lambda v: self.capacitance*v+dt*laplacian(v), dtype=float)
        preconditioner = LinearOperator((self.n, self.n),
            matvec=lambda v: v/diagonal, dtype=float)

        def count(_):
            self.last_cg_iterations += 1

        v, info = cg(operator, old_q, x0=old_v, M=preconditioner,
                     rtol=2e-13, atol=1e-14, maxiter=max(100, 4*self.n), callback=count)
        residual = operator@v-old_q
        if info or np.max(np.abs(residual), initial=0.) > 1e-10*max(1., np.linalg.norm(old_q)):
            raise RuntimeError('Passive field solve did not converge')
        before_g, before_u = self.free_energy(), self.energy()
        # Advance by antisymmetric edge currents, rather than accumulating
        # the iterative solver residual as a fictitious boundary charge.
        self.charge = old_q-dt*laplacian(v)
        self.transported_charge += self.charge-old_q
        dv = v[self.pre]-v[self.post]
        self.conduction_heat += dt*float(np.dot(g, dv*dv))
        self.conduction_numerical_heat += .5*float(np.dot(self.capacitance, (self.voltage-old_v)**2))
        self.conduction_energy_drop += before_u-self.energy()
        self.heat_out += before_u-self.energy()
        self.entropy_balance += before_g-self.free_energy()

    def _conversion(self, dt):
        p = self.params
        if p.conversion_rate == 0:
            return
        old_p, old_d, old_q = self.charged.copy(), self.depleted.copy(), self.charge.copy()
        low, high = -old_p, old_d
        for _ in range(64):
            x = (low+high)*.5
            vbar = (old_q-.5*p.charge_per_event*x)/self.capacitance
            a = p.charge_per_event*vbar-p.charged_energy
            net = p.conversion_rate*((old_d-x)*expit(a)-(old_p+x)*expit(-a))
            below = x-dt*net < 0
            low = np.where(below, x, low)
            high = np.where(below, high, x)
        x = (low+high)*.5
        before_g, before_u = self.free_energy(), self.energy()
        self.charged, self.depleted = old_p+x, old_d-x
        self.charge = old_q-p.charge_per_event*x
        self.returned_charge += p.charge_per_event*x
        self.heat_out += before_u-self.energy()
        self.entropy_balance += before_g-self.free_energy()

    def step(self, dt, current=None):
        """Boundary work, passive flow, reversible conversion, slow exchange.

        Current is held for this integration interval, not an instantaneous
        biological observation. Lie splitting is first order and must be
        refined before attributing physical significance to a trajectory.
        """
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('Positive finite elapsed time required')
        current = np.zeros(self.n) if current is None else np.asarray(current, float)
        if current.shape != self.charge.shape or not np.isfinite(current).all():
            raise ValueError('Finite current with one value per node required')
        dq = dt*current
        future_q = self.charge+dq
        if np.any(np.abs(self.params.charge_per_event*future_q/self.capacitance
                         -self.params.charged_energy) > 600.):
            raise ValueError('Boundary current exceeds supported reversible-rate range')
        before_g, before_heat, before_work = self.free_energy(), self.heat_out, self.work_in
        self.work_in += float(np.dot((self.charge+future_q)/(2*self.capacitance), dq))
        self.charge = future_q
        self.injected_charge += dq
        self._conduction(dt)
        self._conversion(dt)
        super().step(dt)
        return dict(work_in=self.work_in-before_work, heat_out=self.heat_out-before_heat,
                    free_energy_drop=before_g+self.work_in-before_work-self.free_energy())

    def state_dict(self):
        state = super().state_dict()
        state['kind'] = 'flow_exchange'
        for name in self._ledgers:
            state[name] = getattr(self, name)
        return state

    @classmethod
    def from_state_dict(cls, state):
        if state['schema'] != 1 or state.get('kind') != 'flow_exchange':
            raise ValueError('Unsupported flow exchange archive')
        a = state['arrays']
        obj = cls(state['pre'], state['post'], state['capacity'], a['free'], a['bound'],
                  a['carried'], a['charged'], a['depleted'], a['positions'], a['material'],
                  a['charge'], a['capacitance'], a['unit_conductance'],
                  FlowExchangeParams(**state['params']))
        for name in cls._arrays:
            value = np.asarray(a[name], float)
            if value.shape != getattr(obj, name).shape or not np.isfinite(value).all():
                raise ValueError('Invalid flow exchange state: '+name)
            setattr(obj, name, value.copy())
        for name in cls._ledgers:
            if not np.isfinite(state[name]):
                raise ValueError('Nonfinite flow exchange ledger: '+name)
            setattr(obj, name, state[name])
        obj._validate()
        if any(np.max(np.abs(value), initial=0.) > 1e-8 for value in obj.balances().values()):
            raise ValueError('Flow exchange archive fails composition/charge/energy accounting')
        return obj
