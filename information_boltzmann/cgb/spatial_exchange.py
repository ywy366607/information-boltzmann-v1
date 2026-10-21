"""Spatial deposition from the same finite trace/support/drive reactions.

Coordinates are a finite-volume material mesh, not a list of neurons. Short
range cohesion, diffusion and reversible surface catalysis give a moving
material field. No target shape, steering vector, or branch count is used.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import rel_entr, xlogy

from .trace_exchange import TraceExchange, TraceExchangeParams


@dataclass(frozen=True)
class SpatialExchangeParams(TraceExchangeParams):
    cohesion_energy: float = .6
    bulk_attempt: float = .001
    surface_attempt: float = 2.
    material_diffusion: float = .005
    support_diffusion: float = .0025
    drive_diffusion: float = .0025
    spacing: float = 1.

    def __post_init__(self):
        super().__post_init__()
        if min(self.cohesion_energy, self.bulk_attempt, self.surface_attempt,
               self.material_diffusion, self.support_diffusion, self.drive_diffusion) < 0 or self.spacing <= 0:
            raise ValueError('Nonnegative material interactions/rates and positive spacing required')


def cubic_mesh(side, spacing=1.):
    """Equal-volume cubic reference discretization; no anatomical target."""
    if not isinstance(side, int) or side < 2 or not np.isfinite(spacing) or spacing <= 0:
        raise ValueError('Integer side>=2 and positive finite spacing required')
    xyz = np.stack(np.meshgrid(*([np.arange(side)]*3), indexing='ij'), axis=-1).reshape(-1, 3)*spacing
    ids = np.arange(side**3).reshape((side,)*3)
    pre, post = [], []
    for axis in range(3):
        low, high = [slice(None)]*3, [slice(None)]*3
        low[axis], high[axis] = slice(None,-1), slice(1,None)
        pre.append(ids[tuple(low)].ravel())
        post.append(ids[tuple(high)].ravel())
    return xyz, np.concatenate(pre), np.concatenate(post)


class SpatialExchange(TraceExchange):
    _arrays = TraceExchange._arrays + ('coordinates', 'material_diffusion_net',
        'free_diffusion_net', 'charged_diffusion_net', 'depleted_diffusion_net')

    def __init__(self, pre, post, capacity, free, bound, carried, charged,
                 depleted, positions, material, charge, capacitance=1.,
                 unit_conductance=.01, params=None, *, coordinates, **kwargs):
        self.coordinates = np.asarray(coordinates, float).copy()
        for field in ('material','free','charged','depleted'):
            setattr(self, field+'_diffusion_net', np.zeros(np.shape(free)))
        super().__init__(pre,post,capacity,free,bound,carried,charged,depleted,
            positions,material,charge,capacitance,unit_conductance,
            params or SpatialExchangeParams(), **kwargs)

    def _validate(self):
        super()._validate()
        if self.coordinates.shape != (self.n,3) or not np.isfinite(self.coordinates).all():
            raise ValueError('Three-dimensional finite-volume coordinates required')
        lengths = np.linalg.norm(self.coordinates[self.pre]-self.coordinates[self.post],axis=1)
        if not np.allclose(lengths,self.params.spacing,rtol=1e-10,atol=1e-12):
            raise ValueError('Spatial edges must join equal-spacing neighboring volumes')
        if np.any(np.abs(self.structure_energy_local()-self.params.charged_energy) > 500.):
            raise ValueError('Cohesive energy exceeds supported reversible-rate range')

    def neighbor_structure(self):
        incident = self._sum(self.pre,self.capacity)+self._sum(self.post,self.capacity)
        return incident[self.pre]+incident[self.post]-2*self.capacity

    def structure_energy_local(self):
        return self.params.structure_energy-self.params.cohesion_energy*self.neighbor_structure()

    def energy(self):
        return super().energy()-.5*self.params.cohesion_energy*float(
            np.dot(self.capacity,self.neighbor_structure()))

    def balances(self):
        values = super().balances()
        values['support'] -= self.free_diffusion_net
        values['material'] -= self.material_diffusion_net
        values['drive'] -= self.charged_diffusion_net+self.depleted_diffusion_net
        values['diffusion_global'] = np.array([getattr(self,f+'_diffusion_net').sum()
                                               for f in ('material','free','charged','depleted')])
        return values

    def fluxes(self):
        values = super().fluxes()
        p = self.params
        attempt = p.assembly_rate*(p.bulk_attempt+p.surface_attempt*self.neighbor_structure())
        values['assembly'] = (attempt*self.material[self.post]*(self.positions-self.capacity)*self.charged[self.post],
            attempt*np.exp(self.structure_energy_local()-p.charged_energy)
            *(self.capacity-self.bound)*self.depleted[self.post])
        return values

    def _diffusion_rates(self):
        p = self.params
        return (('material',p.material_diffusion),('free',p.support_diffusion),
                ('charged',p.drive_diffusion),('depleted',p.drive_diffusion))

    def entropy_rate(self):
        rate = super().entropy_rate()
        for name,d in self._diffusion_rates():
            if d:
                a = getattr(self,name)
                rate += d/self.params.spacing**2*float(np.sum(
                    rel_entr(a[self.pre],a[self.post])+rel_entr(a[self.post],a[self.pre])))
        return rate

    def _diffuse(self,dt):
        for name,d in self._diffusion_rates():
            if not d:
                continue
            pool, ledger = getattr(self,name), getattr(self,name+'_diffusion_net')
            fraction = -.5*np.expm1(-2*dt*d/self.params.spacing**2)
            for edges in self._edge_groups:
                pre,post = self.pre[edges],self.post[edges]
                a,b = pool[pre].copy(),pool[post].copy()
                transfer = fraction*(b-a)
                pool[pre] += transfer
                pool[post] -= transfer
                ledger[pre] += transfer
                ledger[post] -= transfer
                self.entropy_balance += float(np.sum(xlogy(a,a)-a+xlogy(b,b)-b
                    -xlogy(pool[pre],pool[pre])+pool[pre]-xlogy(pool[post],pool[post])+pool[post]))

    def _assembly(self,dt):
        p = self.params
        if not p.assembly_rate:
            return
        if p.cohesion_energy==0 and p.surface_attempt==0 and p.bulk_attempt==1:
            return super()._assembly(dt)
        incident = self._sum(self.pre,self.capacity)+self._sum(self.post,self.capacity)
        for edges in self._edge_groups:
            pre,post = self.pre[edges],self.post[edges]
            old = self.capacity[edges].copy()
            c,k = self.bound[edges],self.positions[edges]
            b,fuel,waste = self.material[post].copy(),self.charged[post].copy(),self.depleted[post].copy()
            neighbors = incident[pre]+incident[post]-2*old
            local = p.structure_energy-p.cohesion_energy*neighbors
            rate = p.assembly_rate*(p.bulk_attempt+p.surface_attempt*neighbors)
            lo = np.maximum(c-old,-waste)
            hi = np.minimum.reduce((k-old,b,fuel))
            for _ in range(64):
                x = (lo+hi)*.5
                net = rate*((b-x)*(k-old-x)*(fuel-x)
                            -np.exp(local-p.charged_energy)*(old+x-c)*(waste+x))
                below = x-dt*net < 0
                lo,hi = np.where(below,x,lo),np.where(below,hi,x)
            x = (lo+hi)*.5
            self.capacity[edges] += x
            self.material[post] -= x
            self.charged[post] -= x
            self.depleted[post] += x
            incident[pre] += x
            incident[post] += x
        # StructuralExchange.step owns the complete assembly energy/free-
        # energy debit, including cohesion. Do not debit this substep twice.

    def step(self,dt,current=None,world_voltage=None):
        if not np.isfinite(dt) or dt<=0:
            raise ValueError('Positive finite elapsed time required')
        if current is not None and (np.shape(current)!=self.charge.shape or not np.isfinite(current).all()):
            raise ValueError('Finite nodal current required')
        if world_voltage is not None and (np.shape(world_voltage)!=self.port_nodes.shape
                or not np.isfinite(world_voltage).all() or np.any(np.abs(world_voltage)>100.)):
            raise ValueError('Finite world-port voltage required')
        before_g,before_heat,before_work = self.free_energy(),self.heat_out,self.work_in
        self._diffuse(dt)
        super().step(dt,current,world_voltage)
        return dict(work_in=self.work_in-before_work,heat_out=self.heat_out-before_heat,
            free_energy_drop=before_g+self.work_in-before_work-self.free_energy())

    def state_dict(self):
        state = super().state_dict()
        state['kind'] = 'spatial_exchange'
        return state

    @classmethod
    def from_state_dict(cls,state):
        if state['schema']!=1 or state.get('kind')!='spatial_exchange':
            raise ValueError('Unsupported spatial exchange archive')
        a = state['arrays']
        obj = cls(state['pre'],state['post'],state['capacity'],a['free'],a['bound'],a['carried'],
            a['charged'],a['depleted'],a['positions'],a['material'],a['charge'],a['capacitance'],
            a['unit_conductance'],SpatialExchangeParams(**state['params']),coordinates=a['coordinates'],
            trace=a['trace'],trace_capacity=a['trace_capacity'],port_nodes=state['port_nodes'],
            world_charge=a['world_charge'],world_capacitance=a['world_capacitance'],
            port_conductance=a['port_conductance'],source_conductance=a['source_conductance'])
        for name in cls._arrays:
            value = np.asarray(a[name],float)
            if value.shape!=getattr(obj,name).shape or not np.isfinite(value).all():
                raise ValueError('Invalid spatial state: '+name)
            setattr(obj,name,value.copy())
        for name in cls._ledgers:
            if not np.isfinite(state[name]):
                raise ValueError('Nonfinite spatial ledger')
            setattr(obj,name,state[name])
        obj._validate()
        if any(np.max(np.abs(v),initial=0.)>1e-8 for v in obj.balances().values()):
            raise ValueError('Spatial archive fails material/charge/energy accounting')
        return obj
