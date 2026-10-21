"""Finite activity-written configuration, reciprocal binding and rate memory.

This realizes Claude's trace/capture/retention route using declared storage
and interaction energies. It does not label high activity as useful learning.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit, xlogy

from .driven_exchange import DrivenExchange, DrivenExchangeParams


@dataclass(frozen=True)
class TraceExchangeParams(DrivenExchangeParams):
    write_rate: float = .003
    recovery_rate: float = .001
    trace_energy: float = 2.
    capture_coupling: float = 2.
    gate_low: float = 1.
    gate_high: float = 2.

    def __post_init__(self):
        super().__post_init__()
        if min(self.write_rate, self.recovery_rate, self.capture_coupling,
               self.gate_low, self.gate_high) < 0:
            raise ValueError('Nonnegative rates, binding interaction and gate populations required')


class TraceExchange(DrivenExchange):
    _arrays = DrivenExchange._arrays + ('trace', 'trace_capacity', 'trace_written')
    _ledgers = DrivenExchange._ledgers + ('trace_heat',)

    def __init__(self, pre, post, capacity, free, bound, carried, charged,
                 depleted, positions, material, charge, capacitance=1.,
                 unit_conductance=.01, params=None, *, trace=0., trace_capacity=1., **ports):
        self.trace = np.broadcast_to(np.asarray(trace, float), np.shape(bound)).copy()
        self.trace_capacity = np.broadcast_to(np.asarray(trace_capacity, float), np.shape(bound)).copy()
        self.trace_written = np.zeros(np.shape(bound))
        self.trace_heat = 0.
        super().__init__(pre, post, capacity, free, bound, carried, charged, depleted,
            positions, material, charge, capacitance, unit_conductance,
            params or TraceExchangeParams(), **ports)

    def _validate(self):
        super()._validate()
        if (self.trace.shape != self.capacity.shape or self.trace_capacity.shape != self.capacity.shape
                or not np.isfinite(self.trace).all() or not np.isfinite(self.trace_capacity).all()
                or np.any(self.trace_capacity <= 0) or np.any(self.trace < 0)
                or np.any(self.trace > self.trace_capacity)
                or not np.isfinite(self.trace_written).all()):
            raise ValueError('Finite trace inside positive carrier capacity required')
        if np.any(np.abs(self.binding_energy()) > 500.):
            raise ValueError('Binding interaction exceeds supported reversible-rate range')

    def binding_energy(self):
        return self.params.bound_energy-self.params.capture_coupling*self.trace

    def trace_local_energy(self):
        return self.params.trace_energy-self.params.capture_coupling*self.bound

    def gate_factor(self):
        p = self.params
        return p.gate_low+(p.gate_high-p.gate_low)*self.trace/self.trace_capacity

    @property
    def conductance(self):
        return super().conductance*self.gate_factor()

    def active_attempt_rate(self):
        return super().active_attempt_rate()*self.gate_factor()

    def energy(self):
        return super().energy()+float(np.dot(self.trace, self.trace_local_energy()))

    def free_energy(self):
        empty = self.trace_capacity-self.trace
        return float(super().free_energy()+np.sum(xlogy(self.trace, self.trace)
            +xlogy(empty, empty)-xlogy(self.trace_capacity, self.trace_capacity)))

    def fluxes(self):
        values = super().fluxes()
        p = self.params
        energy = p.charged_energy-self.trace_local_energy()+p.charge_per_event*(
            self.voltage[self.pre]-self.voltage[self.post])
        rate = p.write_rate*self.capacity*(self.pre != self.post)
        values['trace_write'] = (
            rate*self.charged[self.pre]*(self.trace_capacity-self.trace)*expit(energy),
            rate*self.depleted[self.pre]*self.trace*expit(-energy))
        local = self.trace_local_energy()
        values['trace_thermal'] = (p.recovery_rate*(self.trace_capacity-self.trace)*expit(-local),
                                   p.recovery_rate*self.trace*expit(local))
        return values

    def _write_trace(self, dt):
        p = self.params
        if p.write_rate == 0:
            return
        for edges in self._edge_groups:
            pre, post = self.pre[edges], self.post[edges]
            old_r = self.trace[edges].copy()
            old_p, old_d = self.charged[pre].copy(), self.depleted[pre].copy()
            qa, qb = self.charge[pre].copy(), self.charge[post].copy()
            lo = np.maximum(-old_d, -old_r)
            hi = np.minimum(old_p, self.trace_capacity[edges]-old_r)
            rate = p.write_rate*self.capacity[edges]
            local = self.trace_local_energy()[edges]
            for _ in range(64):
                x = (lo+hi)*.5
                va = (qa-.5*p.charge_per_event*x)/self.capacitance[pre]
                vb = (qb+.5*p.charge_per_event*x)/self.capacitance[post]
                a = p.charged_energy-local+p.charge_per_event*(va-vb)
                j = rate*((old_p-x)*(self.trace_capacity[edges]-old_r-x)*expit(a)
                          -(old_d+x)*(old_r+x)*expit(-a))
                below = x-dt*j < 0
                lo, hi = np.where(below, x, lo), np.where(below, hi, x)
            x = (lo+hi)*.5
            dq = p.charge_per_event*x
            self.trace[edges] += x
            self.trace_written[edges] += x
            self.active_net[edges] += x
            self.charged[pre] -= x
            self.depleted[pre] += x
            self.charge[pre] -= dq
            self.charge[post] += dq
            self.transported_charge[pre] -= dq
            self.transported_charge[post] += dq
            field_change = .5*np.sum((self.charge[pre]**2-qa**2)/self.capacitance[pre]
                                     +(self.charge[post]**2-qb**2)/self.capacitance[post])
            energy_change = float(field_change-p.charged_energy*x.sum()+np.dot(local, x))
            def dilute(v):
                return xlogy(v, v)-v
            def sites(v):
                return xlogy(v, v)+xlogy(self.trace_capacity[edges]-v,
                                        self.trace_capacity[edges]-v)
            mix_change = float(np.sum(dilute(self.charged[pre])+dilute(self.depleted[pre])
                -dilute(old_p)-dilute(old_d)+sites(self.trace[edges])-sites(old_r)))
            heat = -energy_change
            self.trace_heat += heat
            self.heat_out += heat
            self.entropy_balance -= energy_change+mix_change

    def _recover_trace(self, dt):
        p = self.params
        if p.recovery_rate == 0:
            return
        before_g, before_u = self.free_energy(), self.energy()
        decay = np.exp(-p.recovery_rate*dt)
        self.trace = self.trace*decay+self.trace_capacity*expit(-self.trace_local_energy())*(
            -np.expm1(-p.recovery_rate*dt))
        heat = before_u-self.energy()
        self.trace_heat += heat
        self.heat_out += heat
        self.entropy_balance += before_g-self.free_energy()

    def step(self, dt, current=None, world_voltage=None):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('Positive finite elapsed time required')
        # Validate external inputs before mutating the new local storage.
        if current is not None and (np.shape(current) != self.charge.shape
                                   or not np.isfinite(current).all()):
            raise ValueError('Finite current with one value per node required')
        if world_voltage is not None and (np.shape(world_voltage) != self.port_nodes.shape
                or not np.isfinite(world_voltage).all() or np.any(np.abs(world_voltage) > 100.)):
            raise ValueError('Finite source voltage per world port required')
        before_g, before_heat, before_work = self.free_energy(), self.heat_out, self.work_in
        self._write_trace(dt)
        self._recover_trace(dt)
        super().step(dt, current, world_voltage)
        return dict(work_in=self.work_in-before_work, heat_out=self.heat_out-before_heat,
                    free_energy_drop=before_g+self.work_in-before_work-self.free_energy())

    def state_dict(self):
        state = super().state_dict()
        state['kind'] = 'trace_exchange'
        return state

    @classmethod
    def from_state_dict(cls, state):
        if state['schema'] != 1 or state.get('kind') != 'trace_exchange':
            raise ValueError('Unsupported trace exchange archive')
        a = state['arrays']
        obj = cls(state['pre'], state['post'], state['capacity'], a['free'], a['bound'],
            a['carried'], a['charged'], a['depleted'], a['positions'], a['material'],
            a['charge'], a['capacitance'], a['unit_conductance'], TraceExchangeParams(**state['params']),
            trace=a['trace'], trace_capacity=a['trace_capacity'], port_nodes=state['port_nodes'],
            world_charge=a['world_charge'], world_capacitance=a['world_capacitance'],
            port_conductance=a['port_conductance'], source_conductance=a['source_conductance'])
        for name in cls._arrays:
            value = np.asarray(a[name], float)
            if value.shape != getattr(obj, name).shape or not np.isfinite(value).all():
                raise ValueError('Invalid trace state: '+name)
            setattr(obj, name, value.copy())
        for name in cls._ledgers:
            if not np.isfinite(state[name]):
                raise ValueError('Nonfinite trace ledger')
            setattr(obj, name, state[name])
        obj._validate()
        if any(np.max(np.abs(v), initial=0.) > 1e-8 for v in obj.balances().values()):
            raise ValueError('Trace archive fails composition/charge/energy accounting')
        return obj
