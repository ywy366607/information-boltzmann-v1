"""Finite fueled directional channels and reciprocal stateful world ports.

The same local free-energy accounting as FlowExchange, without a learning
objective or a free active current. This is a material/port realization, not
yet a spiking body or a language-world agent.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit, xlogy

from .flow_exchange import FlowExchange, FlowExchangeParams


@dataclass(frozen=True)
class DrivenExchangeParams(FlowExchangeParams):
    active_rate: float = .001

    def __post_init__(self):
        super().__post_init__()
        if self.active_rate < 0:
            raise ValueError('Nonnegative active attempt rate required')


class DrivenExchange(FlowExchange):
    _arrays = FlowExchange._arrays + (
        'active_net', 'world_charge', 'world_capacitance', 'port_conductance',
        'source_conductance', 'initial_world_charge', 'world_injected', 'port_transfer')
    _ledgers = FlowExchange._ledgers + ('active_heat', 'port_heat', 'source_heat', 'source_work')

    def __init__(self, pre, post, capacity, free, bound, carried, charged,
                 depleted, positions, material, charge, capacitance=1.,
                 unit_conductance=.01, params=None, *, port_nodes=(),
                 world_charge=0., world_capacitance=1., port_conductance=.01,
                 source_conductance=.01):
        nodes = np.asarray(port_nodes)
        if nodes.ndim != 1 or (nodes.size and not np.issubdtype(nodes.dtype, np.integer)):
            raise ValueError('Integer one-dimensional port nodes required')
        self.port_nodes = nodes.astype(np.int64)
        self.world_charge = np.broadcast_to(np.asarray(world_charge, float), nodes.shape).copy()
        self.world_capacitance = np.broadcast_to(np.asarray(world_capacitance, float), nodes.shape).copy()
        self.port_conductance = np.broadcast_to(np.asarray(port_conductance, float), nodes.shape).copy()
        self.source_conductance = np.broadcast_to(np.asarray(source_conductance, float), nodes.shape).copy()
        self.initial_world_charge = self.world_charge.copy()
        self.world_injected = np.zeros(nodes.shape)
        self.port_transfer = np.zeros(nodes.shape)
        self.active_net = np.zeros(np.shape(bound))
        self.active_heat = self.port_heat = self.source_heat = self.source_work = 0.
        super().__init__(pre, post, capacity, free, bound, carried, charged,
            depleted, positions, material, charge, capacitance, unit_conductance,
            params or DrivenExchangeParams())
        # Each vector group has no shared endpoint. The grouping is numerical
        # execution, not a module partition or an alteration of anatomy.
        occupied, groups = [], []
        for e, (a, b) in enumerate(zip(self.pre, self.post)):
            if a == b:
                continue  # No distinct electrical terminals in one compartment.
            for used, group in zip(occupied, groups):
                if a not in used and b not in used:
                    used.update((a, b))
                    group.append(e)
                    break
            else:
                occupied.append({a, b})
                groups.append([e])
        self._edge_groups = [np.asarray(group, np.int64) for group in groups]

    def _validate(self):
        super()._validate()
        if np.any(self.port_nodes < 0) or np.any(self.port_nodes >= self.n):
            raise ValueError('Port node outside brain panel')
        for name in ('world_charge', 'world_capacitance', 'port_conductance',
                     'source_conductance', 'initial_world_charge', 'world_injected', 'port_transfer'):
            a = getattr(self, name)
            if a.shape != self.port_nodes.shape or not np.isfinite(a).all():
                raise ValueError('Finite world-port state required: '+name)
        if (np.any(self.world_capacitance <= 0) or np.any(self.port_conductance < 0)
                or np.any(self.source_conductance < 0)
                or self.active_net.shape != self.capacity.shape
                or not np.isfinite(self.active_net).all()):
            raise ValueError('Invalid world capacity, transport or active ledger')

    @property
    def world_voltage(self):
        return self.world_charge/self.world_capacitance

    def world_energy(self):
        return .5*float(np.dot(self.world_charge, self.world_voltage))

    def energy(self):
        return super().energy()+self.world_energy()

    def balances(self):
        result = super().balances()
        result['charge_world'] = (self.world_charge-self.initial_world_charge
                                  -self.world_injected+self.port_transfer)
        result['charge_transport_global'] += float(
            np.sum(self.world_charge-self.initial_world_charge-self.world_injected))
        return result

    def fluxes(self):
        result = super().fluxes()
        p = self.params
        a = p.charged_energy+p.charge_per_event*(self.voltage[self.pre]-self.voltage[self.post])
        rate = self.active_attempt_rate()*(self.pre != self.post)
        result['active'] = (rate*self.charged[self.pre]*expit(a),
                            rate*self.depleted[self.pre]*expit(-a))
        return result

    def active_attempt_rate(self):
        """Common forward/reverse attempt factor, distinct from affinity."""
        return self.params.active_rate*self.capacity

    def entropy_rate(self):
        dv = self.voltage[self.port_nodes]-self.world_voltage
        return super().entropy_rate()+float(np.dot(self.port_conductance, dv*dv))

    def _active_flow(self, dt):
        p = self.params
        if p.active_rate == 0:
            return
        for edges in self._edge_groups:
            pre, post = self.pre[edges], self.post[edges]
            old_p, old_d = self.charged[pre].copy(), self.depleted[pre].copy()
            old_pre, old_post = self.charge[pre].copy(), self.charge[post].copy()
            lo, hi = -old_d, old_p
            rate = self.active_attempt_rate()[edges]
            for _ in range(64):
                x = (lo+hi)*.5
                va = (old_pre-.5*p.charge_per_event*x)/self.capacitance[pre]
                vb = (old_post+.5*p.charge_per_event*x)/self.capacitance[post]
                a = p.charged_energy+p.charge_per_event*(va-vb)
                net = rate*((old_p-x)*expit(a)-(old_d+x)*expit(-a))
                below = x-dt*net < 0
                lo, hi = np.where(below, x, lo), np.where(below, hi, x)
            x = (lo+hi)*.5
            dq = p.charge_per_event*x
            self.charge[pre] -= dq
            self.charge[post] += dq
            self.transported_charge[pre] -= dq
            self.transported_charge[post] += dq
            self.charged[pre] -= x
            self.depleted[pre] += x
            self.active_net[edges] += x
            field_change = .5*np.sum((self.charge[pre]**2-old_pre**2)/self.capacitance[pre]
                                     +(self.charge[post]**2-old_post**2)/self.capacitance[post])
            energy_change = float(field_change-p.charged_energy*x.sum())
            mix_change = float(np.sum(xlogy(self.charged[pre], self.charged[pre])-self.charged[pre]
                +xlogy(self.depleted[pre], self.depleted[pre])-self.depleted[pre]
                -xlogy(old_p, old_p)+old_p-xlogy(old_d, old_d)+old_d))
            heat = -energy_change
            self.active_heat += heat
            self.heat_out += heat
            self.entropy_balance -= energy_change+mix_change

    def _world_source(self, dt, voltage):
        if voltage is None:
            return
        fraction = -np.expm1(-dt*self.source_conductance/self.world_capacitance)
        dq = (self.world_capacitance*voltage-self.world_charge)*fraction
        before_u = self.world_energy()
        work = float(np.dot(voltage, dq))
        self.world_charge += dq
        self.world_injected += dq
        self.work_in += work
        self.source_work += work
        heat = work+before_u-self.world_energy()
        self.source_heat += heat
        self.heat_out += heat
        self.entropy_balance += heat

    def _ports(self, dt):
        # Sequential exact capacitor pairs also permit repeated brain ports;
        # their simultaneous continuous limit is checked by step refinement.
        for a, node in enumerate(self.port_nodes):
            old_brain, old_world = self.charge[node], self.world_charge[a]
            inverse_c = 1/self.capacitance[node]+1/self.world_capacitance[a]
            delta = self.voltage[node]-self.world_voltage[a]
            sent = delta*(-np.expm1(-dt*self.port_conductance[a]*inverse_c))/inverse_c
            self.charge[node] -= sent
            self.world_charge[a] += sent
            self.transported_charge[node] -= sent
            self.port_transfer[a] -= sent
            heat = .5*((old_brain**2-self.charge[node]**2)/self.capacitance[node]
                       +(old_world**2-self.world_charge[a]**2)/self.world_capacitance[a])
            self.port_heat += heat
            self.heat_out += heat
            self.entropy_balance += heat

    def step(self, dt, current=None, world_voltage=None):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('Positive finite elapsed time required')
        if current is not None:
            current = np.asarray(current, float)
            if current.shape != self.charge.shape or not np.isfinite(current).all():
                raise ValueError('Finite current with one value per node required')
            if np.any(np.abs(self.params.charge_per_event*(self.charge+dt*current)/self.capacitance
                             -self.params.charged_energy) > 600.):
                raise ValueError('Boundary current exceeds supported reversible-rate range')
        if world_voltage is not None:
            world_voltage = np.asarray(world_voltage, float)
            if (world_voltage.shape != self.port_nodes.shape or not np.isfinite(world_voltage).all()
                    or np.any(np.abs(world_voltage) > 100.)):
                raise ValueError('Finite source voltage per world port within reference range required')
        before_g, before_work, before_heat = self.free_energy(), self.work_in, self.heat_out
        self._world_source(dt, world_voltage)
        self._ports(dt)
        self._active_flow(dt)
        super().step(dt, current)
        return dict(work_in=self.work_in-before_work, heat_out=self.heat_out-before_heat,
                    free_energy_drop=before_g+self.work_in-before_work-self.free_energy())

    def state_dict(self):
        state = super().state_dict()
        state['kind'] = 'driven_exchange'
        state['port_nodes'] = self.port_nodes.tolist()
        return state

    @classmethod
    def from_state_dict(cls, state):
        if state['schema'] != 1 or state.get('kind') != 'driven_exchange':
            raise ValueError('Unsupported driven exchange archive')
        a = state['arrays']
        obj = cls(state['pre'], state['post'], state['capacity'], a['free'], a['bound'],
            a['carried'], a['charged'], a['depleted'], a['positions'], a['material'],
            a['charge'], a['capacitance'], a['unit_conductance'], DrivenExchangeParams(**state['params']),
            port_nodes=state['port_nodes'], world_charge=a['world_charge'],
            world_capacitance=a['world_capacitance'], port_conductance=a['port_conductance'],
            source_conductance=a['source_conductance'])
        for name in cls._arrays:
            value = np.asarray(a[name], float)
            if value.shape != getattr(obj, name).shape or not np.isfinite(value).all():
                raise ValueError('Invalid driven exchange state: '+name)
            setattr(obj, name, value.copy())
        for name in cls._ledgers:
            if not np.isfinite(state[name]):
                raise ValueError('Nonfinite driven exchange ledger: '+name)
            setattr(obj, name, state[name])
        obj._validate()
        if any(np.max(np.abs(v), initial=0.) > 1e-8 for v in obj.balances().values()):
            raise ValueError('Driven exchange archive fails composition/charge/energy accounting')
        return obj
