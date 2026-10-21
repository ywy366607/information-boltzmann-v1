"""Finite material ends preserve crossing-versus-joined port responses.

This extends the same trace/support/fuel reactions, not a branch scheduler.
Ports at the same coordinate retain independent charge. A reversible bridge
consumes one end from each port, one material unit and one charged driver.
Candidate encounters and port inventories are explicit constitutive geometry;
they are not yet a continuously generated cell arbor or a lineage program.
"""
from __future__ import annotations

import numpy as np
from scipy.special import rel_entr, xlogy

from .spatial_exchange import SpatialExchange, SpatialExchangeParams
from .trace_exchange import TraceExchange


class EndExchange(SpatialExchange):
    _arrays = SpatialExchange._arrays + (
        'end_capacity', 'initial_end_inventory', 'reaction_scale')

    def __init__(self, pre, post, capacity, free, bound, carried, charged,
                 depleted, positions, material, charge, capacitance=1.,
                 unit_conductance=.01, params=None, *, coordinates,
                 end_capacity, reaction_scale=1., **kwargs):
        self.end_capacity = np.broadcast_to(
            np.asarray(end_capacity, float), np.shape(free)).copy()
        self.initial_end_inventory = self.end_capacity.copy()
        self.reaction_scale = np.broadcast_to(
            np.asarray(reaction_scale, float), np.shape(bound)).copy()
        super().__init__(pre, post, capacity, free, bound, carried, charged,
            depleted, positions, material, charge, capacitance, unit_conductance,
            params or SpatialExchangeParams(), coordinates=coordinates, **kwargs)

    def remaining_ends(self):
        return (self.end_capacity-self._sum(self.pre, self.capacity)
                - self._sum(self.post, self.capacity))

    def _spatial_edges(self):
        lengths = np.linalg.norm(
            self.coordinates[self.pre]-self.coordinates[self.post], axis=1)
        return lengths > 0.

    def _validate(self):
        # SpatialExchange assumes a single domain per voxel; its geometry
        # check intentionally does not apply to distinct colocated terminals.
        TraceExchange._validate(self)
        if (self.coordinates.shape != (self.n, 3)
                or not np.isfinite(self.coordinates).all()
                or np.any(self.pre == self.post)):
            raise ValueError('Finite coordinates and distinct port endpoints required')
        lengths = np.linalg.norm(
            self.coordinates[self.pre]-self.coordinates[self.post], axis=1)
        if not np.all((lengths == 0.) | np.isclose(
                lengths, self.params.spacing, rtol=1e-10, atol=1e-12)):
            raise ValueError('Encounters must be colocated or neighboring spatial ports')
        for name in ('end_capacity', 'initial_end_inventory'):
            a = getattr(self, name)
            if a.shape != self.free.shape or not np.isfinite(a).all() or np.any(a < 0):
                raise ValueError('Finite nonnegative end inventories required')
        if (self.reaction_scale.shape != self.capacity.shape
                or not np.isfinite(self.reaction_scale).all()
                or np.any(self.reaction_scale < 0)):
            raise ValueError('Nonnegative common forward/backward attempt scales required')
        if np.any(self.remaining_ends() < 0):
            raise ValueError('Installed bridges exceed an endpoint inventory')
        if np.any(np.abs(self.structure_energy_local()-self.params.charged_energy) > 500.):
            raise ValueError('Bridge energy exceeds reversible rate range')

    def free_energy(self):
        ends = self.remaining_ends()
        # End counts are finite ideal-mixture components. Their chemical
        # potential is log(e), contributing -log(e_a e_b) on bridge assembly.
        return float(super().free_energy()+np.sum(xlogy(ends, ends)-ends))

    def balances(self):
        values = super().balances()
        values['ends'] = (self.remaining_ends()+self._sum(self.pre, self.capacity)
            +self._sum(self.post, self.capacity)-self.initial_end_inventory)
        return values

    def fluxes(self):
        values = super().fluxes()
        plus, minus = values['assembly']
        ends = self.remaining_ends()
        values['assembly'] = (
            plus*ends[self.pre]*ends[self.post]*self.reaction_scale,
            minus*self.reaction_scale)
        return values

    def entropy_rate(self):
        rate = TraceExchange.entropy_rate(self)
        keep = self._spatial_edges()
        for name, d in self._diffusion_rates():
            if d:
                a = getattr(self, name)
                pre, post = self.pre[keep], self.post[keep]
                rate += d/self.params.spacing**2*float(np.sum(
                    rel_entr(a[pre], a[post])+rel_entr(a[post], a[pre])))
        return rate

    def _diffuse(self, dt):
        # Reservoir diffusion follows the declared positive-distance mesh.
        # A zero-distance electrical contact is not infinite diffusivity or
        # a silent merger of the two finite material/charge compartments.
        keep = self._spatial_edges()
        for name, d in self._diffusion_rates():
            if not d:
                continue
            pool = getattr(self, name)
            ledger = getattr(self, name+'_diffusion_net')
            fraction = -.5*np.expm1(-2*dt*d/self.params.spacing**2)
            for group in self._edge_groups:
                edges = group[keep[group]]
                pre, post = self.pre[edges], self.post[edges]
                a, b = pool[pre].copy(), pool[post].copy()
                transfer = fraction*(b-a)
                pool[pre] += transfer
                pool[post] -= transfer
                ledger[pre] += transfer
                ledger[post] -= transfer
                self.entropy_balance += float(np.sum(xlogy(a, a)-a+xlogy(b, b)-b
                    -xlogy(pool[pre], pool[pre])+pool[pre]
                    -xlogy(pool[post], pool[post])+pool[post]))

    def _assembly(self, dt):
        p = self.params
        if not p.assembly_rate:
            return
        incident = self._sum(self.pre, self.capacity)+self._sum(self.post, self.capacity)
        for edges in self._edge_groups:
            pre, post = self.pre[edges], self.post[edges]
            old = self.capacity[edges].copy()
            c, k = self.bound[edges], self.positions[edges]
            b = self.material[post].copy()
            fuel, waste = self.charged[post].copy(), self.depleted[post].copy()
            ea = self.end_capacity[pre]-incident[pre]
            eb = self.end_capacity[post]-incident[post]
            neighbors = incident[pre]+incident[post]-2*old
            local = p.structure_energy-p.cohesion_energy*neighbors
            rate = (p.assembly_rate*(p.bulk_attempt+p.surface_attempt*neighbors)
                    *self.reaction_scale[edges])
            lo = np.maximum(c-old, -waste)
            hi = np.minimum.reduce((k-old, b, fuel, ea, eb))
            # Stay on the feasible side of a saturated inventory at roundoff.
            hi = np.where(hi > 0., np.nextafter(hi, 0.), hi)
            for _ in range(64):
                x = (lo+hi)*.5
                net = rate*((b-x)*(fuel-x)*(k-old-x)*(ea-x)*(eb-x)
                    -np.exp(local-p.charged_energy)*(old+x-c)*(waste+x))
                below = x-dt*net < 0
                lo, hi = np.where(below, x, lo), np.where(below, hi, x)
            x = (lo+hi)*.5
            # A zero-rate event must remain bitwise stationary, including
            # inventory-saturated states where the implicit bracket is skewed.
            x = np.where(rate == 0., 0., x)
            self.capacity[edges] += x
            self.material[post] -= x
            self.charged[post] -= x
            self.depleted[post] += x
            incident[pre] += x
            incident[post] += x
        # StructuralExchange.step debits the complete coupled U and G once.

    def state_dict(self):
        state = super().state_dict()
        state['kind'] = 'end_exchange'
        return state

    @classmethod
    def from_state_dict(cls, state):
        if state['schema'] != 1 or state.get('kind') != 'end_exchange':
            raise ValueError('Unsupported finite end exchange archive')
        a = state['arrays']
        obj = cls(state['pre'], state['post'], state['capacity'], a['free'], a['bound'],
            a['carried'], a['charged'], a['depleted'], a['positions'], a['material'],
            a['charge'], a['capacitance'], a['unit_conductance'],
            SpatialExchangeParams(**state['params']), coordinates=a['coordinates'],
            end_capacity=a['end_capacity'], reaction_scale=a['reaction_scale'],
            trace=a['trace'], trace_capacity=a['trace_capacity'], port_nodes=state['port_nodes'],
            world_charge=a['world_charge'], world_capacitance=a['world_capacitance'],
            port_conductance=a['port_conductance'], source_conductance=a['source_conductance'])
        for name in cls._arrays:
            value = np.asarray(a[name], float)
            if value.shape != getattr(obj, name).shape or not np.isfinite(value).all():
                raise ValueError('Invalid finite end state: '+name)
            setattr(obj, name, value.copy())
        for name in cls._ledgers:
            if not np.isfinite(state[name]):
                raise ValueError('Nonfinite finite-end ledger')
            setattr(obj, name, state[name])
        obj._validate()
        if any(np.max(np.abs(v), initial=0.) > 1e-8 for v in obj.balances().values()):
            raise ValueError('End archive fails composition/charge/energy accounting')
        return obj
