"""Mobile finite-valence material, with the same trace/support/fuel reactions.

Integer local compositions and binary bridge configurations give a resolved
Markov realization of the earlier occupancy kernel. Geometry is material
state, not a prescribed tree. Candidate encounters are transient spatial
queries; only actual bridges and their paid storage carriers are retained.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import math

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import expit, gammaln

from .spatial_exchange import SpatialExchangeParams


@dataclass(frozen=True)
class MobileExchangeParams(SpatialExchangeParams):
    mobility: float = .002
    core_diameter: float = .3
    contact_radius: float = 1.5
    rest_length: float = .8
    elastic_stiffness: float = 4.
    box_size: float = 5.

    def __post_init__(self):
        super().__post_init__()
        if (self.mobility < 0 or self.elastic_stiffness < 0
                or not 0 < self.core_diameter < self.rest_length < self.contact_radius
                or self.box_size <= self.contact_radius):
            raise ValueError('Ordered finite material lengths and nonnegative mobility/stiffness required')


@dataclass(frozen=True)
class MaterialEvent:
    kind: str
    i: int
    j: int = -1
    edge: int = -1
    sign: int = 1
    pool: str = ''
    interface: int = -1


class MobileExchange:
    _arrays = ('coordinates', 'free', 'bound', 'carried', 'charged', 'depleted',
        'material', 'trace', 'end_capacity', 'charge', 'capacitance',
        'world_charge', 'world_capacitance', 'port_conductance', 'source_conductance',
        'initial_material', 'initial_drive', 'initial_support', 'initial_total_charge')
    _ledgers = ('initial_energy', 'initial_free_energy', 'heat_out', 'work_in',
        'entropy_balance', 'source_charge', 'electrode_charge', 't',
        'motion_heat', 'reaction_heat', 'source_heat', 'conduction_heat')

    def __init__(self, coordinates, material, free, charged, depleted, *,
                 end_capacity=3, carried=0, charge=0., capacitance=1.,
                 pre=(), post=(), bound=0, trace=0, unit_conductance=.01,
                 params=None, port_nodes=(), world_charge=0., world_capacitance=2.,
                 port_conductance=.05, source_conductance=.1, seed=0):
        self.params = params or MobileExchangeParams()
        self.coordinates = np.asarray(coordinates, float).copy()
        self.n = len(self.coordinates)
        def counts(value, shape):
            a = np.broadcast_to(np.asarray(value), shape)
            if not np.isfinite(a).all() or not np.equal(a, np.floor(a)).all():
                raise ValueError('Integer finite compositions required')
            return a.astype(np.int64).copy()
        self.pre, self.post = counts(pre, np.shape(pre)), counts(post, np.shape(post))
        self.capacity = np.ones(len(self.pre))
        self.bound, self.trace = counts(bound, self.pre.shape), counts(trace, self.pre.shape)
        for name, value in (('material', material), ('free', free), ('carried', carried),
                            ('charged', charged), ('depleted', depleted), ('end_capacity', end_capacity)):
            setattr(self, name, counts(value, (self.n,)))
        self.charge = np.broadcast_to(np.asarray(charge, float), (self.n,)).copy()
        self.capacitance = np.broadcast_to(np.asarray(capacitance, float), (self.n,)).copy()
        self.port_nodes = counts(port_nodes, np.shape(port_nodes))
        for name, value in (('world_charge', world_charge), ('world_capacitance', world_capacitance),
                            ('port_conductance', port_conductance), ('source_conductance', source_conductance)):
            setattr(self, name, np.broadcast_to(np.asarray(value, float), self.port_nodes.shape).copy())
        self.unit_conductance = float(unit_conductance)
        self.rng_reaction = np.random.default_rng(seed)
        self.rng_motion = np.random.default_rng(np.random.SeedSequence([seed, 1]))
        self.rng_field = np.random.default_rng(np.random.SeedSequence([seed, 2]))
        self.reaction_counts = {}
        self.motion_attempts = self.motion_accepted = 0
        self.event_observer = None
        self.source_observer = None  # disposable instrument; never physical state
        self.field_scheme = 'sequential_ou'
        for name in self._ledgers:
            setattr(self, name, 0.)
        self.initial_material = np.array(self.material.sum()+len(self.pre))
        self.initial_drive = np.array(self.charged.sum()+self.depleted.sum())
        self.initial_support = np.array(self.free.sum()+self.carried.sum()+self.bound.sum())
        self.initial_total_charge = np.array(self.charge.sum()+self.world_charge.sum())
        self._validate()
        self.initial_energy, self.initial_free_energy = self.energy(), self.free_energy()

    def degrees(self):
        return np.bincount(np.r_[self.pre, self.post], minlength=self.n)

    def remaining_ends(self):
        return self.end_capacity-self.degrees()

    def bridge_key(self, i, j):
        """Default material permits one bridge, of either polarity, per pair."""
        return tuple(sorted((int(i), int(j))))

    def encounters(self):
        return cKDTree(self.coordinates).query_pairs(self.params.contact_radius, output_type='ndarray')

    def _validate(self):
        if self.field_scheme not in ('sequential_ou', 'symmetric_ou'):
            raise ValueError('Unknown field integration scheme')
        p = self.params
        if (self.coordinates.shape != (self.n, 3) or not np.isfinite(self.coordinates).all()
                or np.any(self.coordinates < 0) or np.any(self.coordinates > p.box_size)):
            raise ValueError('Finite material coordinates inside the declared container required')
        for name in ('material', 'free', 'carried', 'charged', 'depleted', 'end_capacity'):
            a = getattr(self, name)
            if a.shape != (self.n,) or a.dtype.kind not in 'iu' or np.any(a < 0):
                raise ValueError('Nonnegative integer local composition required: '+name)
        if (self.pre.shape != self.post.shape or self.pre.ndim != 1
                or np.any(self.pre < 0) or np.any(self.post < 0)
                or np.any(self.pre >= self.n) or np.any(self.post >= self.n)
                or np.any(self.pre == self.post)):
            raise ValueError('Invalid actual bridge endpoints')
        if self.capacity.shape != self.pre.shape or not np.array_equal(self.capacity, np.ones(len(self.pre))):
            raise ValueError('Actual bridges have one material unit each')
        pairs = np.asarray([self.bridge_key(a, b) for a, b in zip(self.pre, self.post)])
        if len(np.unique(pairs, axis=0)) != len(pairs):
            raise ValueError('One actual bridge per material pair required')
        for name in ('bound', 'trace'):
            a = getattr(self, name)
            if a.shape != self.pre.shape or a.dtype.kind not in 'iu' or np.any((a < 0)|(a > 1)):
                raise ValueError('Binary internal bridge configuration required')
        if np.any(self.remaining_ends() < 0):
            raise ValueError('Finite material ends overcommitted')
        if len(self.pre) and np.any(self.lengths() > p.contact_radius+1e-12):
            raise ValueError('Bridge exceeds the declared extensibility domain')
        if len(self.coordinates) > 1:
            if cKDTree(self.coordinates).query_pairs(p.core_diameter-1e-12):
                raise ValueError('Material cores overlap')
        for name in ('charge', 'capacitance'):
            if getattr(self, name).shape != (self.n,) or not np.isfinite(getattr(self, name)).all():
                raise ValueError('Finite material electric state required')
        if np.any(self.capacitance <= 0) or self.unit_conductance < 0 or not np.isfinite(self.unit_conductance):
            raise ValueError('Positive capacitance and nonnegative conductance required')
        if self.port_nodes.ndim != 1 or np.any(self.port_nodes < 0) or np.any(self.port_nodes >= self.n):
            raise ValueError('Invalid world port nodes')
        for name in ('world_charge', 'world_capacitance', 'port_conductance', 'source_conductance'):
            a = getattr(self, name)
            if a.shape != self.port_nodes.shape or not np.isfinite(a).all():
                raise ValueError('Finite world port state required')
        if (np.any(self.world_capacitance <= 0) or np.any(self.port_conductance < 0)
                or np.any(self.source_conductance < 0)):
            raise ValueError('Invalid world coupling')

    def lengths(self):
        return np.linalg.norm(self.coordinates[self.pre]-self.coordinates[self.post], axis=1)

    def energy(self):
        p = self.params
        d = self.degrees()
        return float(.5*np.sum(self.charge**2/self.capacitance)
            +.5*np.sum(self.world_charge**2/self.world_capacitance)
            +p.charged_energy*self.charged.sum()+p.carried_energy*self.carried.sum()
            +p.structure_energy*len(self.pre)
            +.5*p.elastic_stiffness*np.sum((self.lengths()-p.rest_length)**2)
            +p.bound_energy*self.bound.sum()+p.trace_energy*self.trace.sum()
            -p.capture_coupling*np.dot(self.trace, self.bound)
            -.5*p.cohesion_energy*np.dot(d, d-1))

    def free_energy(self):
        mixture = sum(float(gammaln(getattr(self, name)+1).sum())
            for name in ('material', 'free', 'carried', 'charged', 'depleted'))
        end = np.sum(gammaln(self.remaining_ends()+1)-gammaln(self.end_capacity+1))
        return self.energy()+mixture+float(end)

    def forces(self):
        force = np.zeros_like(self.coordinates)
        if len(self.pre):
            delta = self.coordinates[self.pre]-self.coordinates[self.post]
            length = self.lengths()
            pair = -self.params.elastic_stiffness*(length-self.params.rest_length)[:, None]*delta/length[:, None]
            np.add.at(force, self.pre, pair)
            np.add.at(force, self.post, -pair)
        return force

    def _pools(self, event):
        i, j, s, e = event.i, event.j, event.sign, event.edge
        k = event.kind
        if k == 'assembly': return [('material', j, -s), ('charged', j, -s), ('depleted', j, s)]
        if k == 'binding': return [('free', j, -s)]
        if k == 'transport': return [('carried', i, s), ('charged', i, -s), ('depleted', i, s)]
        if k == 'release': return [('carried', i, -s), ('free', i, s)]
        if k in ('write', 'active'): return [('charged', i, -s), ('depleted', i, s)]
        if k == 'conversion': return [('charged', i, s), ('depleted', i, -s)]
        if k == 'exchange': return [(event.pool, i, -s), (event.pool, j, s)]
        if k == 'thermal': return []
        raise ValueError('Unknown material event: '+k)

    def _charge_changes(self, event):
        z = event.sign*self.params.charge_per_event
        if event.kind in ('write', 'active'): return [(event.i, -z), (event.j, z)]
        if event.kind == 'conversion': return [(event.i, -z)]
        return []

    def _charge_energy_increment(self, node, dq):
        if node < self.n:
            return (self.charge[node]*dq+.5*dq*dq)/self.capacitance[node]
        port = node-self.n
        return (self.world_charge[port]*dq+.5*dq*dq)/self.world_capacitance[port]

    def _add_charge(self, node, dq):
        if node < self.n:
            self.charge[node] += dq
        else:
            self.world_charge[node-self.n] += dq

    def _contact_arrays(self, event):
        if event.interface >= 0:
            raise ValueError('This material has no registered boundary storage')
        return self.bound, self.trace, event.edge

    def _assembly_delta(self, event, degree):
        p, s = self.params, event.sign
        neighbors = degree[event.i]+degree[event.j]-(2 if s < 0 else 0)
        length = np.linalg.norm(self.coordinates[event.i]-self.coordinates[event.j])
        du = s*(p.structure_energy+.5*p.elastic_stiffness*(length-p.rest_length)**2
                -p.cohesion_energy*neighbors)
        ends = self.end_capacity-degree
        mix = (-math.log(ends[event.i])-math.log(ends[event.j]) if s > 0 else
                math.log(ends[event.i]+1)+math.log(ends[event.j]+1))
        return du, mix

    def _assembly_attempt(self, event, degree):
        neighbors = degree[event.i]+degree[event.j]-(2 if event.sign < 0 else 0)
        return self.params.assembly_rate*(self.params.bulk_attempt+self.params.surface_attempt*neighbors)

    def _contact_transport_factor(self, event):
        return 1.

    def contact_conductance_increment(self, event):
        dn = event.sign if event.kind == 'assembly' else 0
        dr = event.sign if event.kind in ('write','thermal') else 0
        p = self.params
        return self.unit_conductance*self._contact_transport_factor(event)*(p.gate_low*dn+(p.gate_high-p.gate_low)*dr)

    def event_delta(self, event, degree=None):
        """Exact U/G increments, including finite-composition multiplicities."""
        p, e, s = self.params, event.edge, event.sign
        du, mix = 0., 0.
        for name, node, change in self._pools(event):
            count = getattr(self, name)[node]
            if count+change < 0: raise ValueError('Unavailable local composition')
            mix += math.log(count+1) if change > 0 else -math.log(count)
            if name == 'charged': du += p.charged_energy*change
            if name == 'carried': du += p.carried_energy*change
        for node, dq in self._charge_changes(event):
            du += self._charge_energy_increment(node, dq)
        if event.kind == 'assembly':
            degree = self.degrees() if degree is None else degree
            assembly_u, assembly_mix = self._assembly_delta(event, degree)
            du += assembly_u
            mix += assembly_mix
        elif event.kind in ('binding', 'transport'):
            _, trace, index = self._contact_arrays(event)
            dc = s if event.kind == 'binding' else -s
            du += dc*(p.bound_energy-p.capture_coupling*trace[index])
        elif event.kind in ('write', 'thermal'):
            bound, _, index = self._contact_arrays(event)
            du += s*(p.trace_energy-p.capture_coupling*bound[index])
        return float(du), float(du+mix)

    def attempt_rate(self, event, degree=None):
        p, k = self.params, event.kind
        if k == 'assembly':
            d = self.degrees() if degree is None else degree
            return self._assembly_attempt(event, d)
        if k == 'binding': return p.binding_rate
        if k == 'transport': return p.transport_rate
        if k == 'release': return p.release_rate
        if k == 'write': return p.write_rate
        if k == 'thermal': return p.recovery_rate
        if k == 'conversion': return p.conversion_rate
        if k == 'active':
            _, trace, index = self._contact_arrays(event)
            return p.active_rate*(p.gate_low+(p.gate_high-p.gate_low)*trace[index])
        if k == 'exchange':
            d = {'material': p.material_diffusion, 'free': p.support_diffusion,
                 'charged': p.drive_diffusion, 'depleted': p.drive_diffusion}[event.pool]
            return d/p.contact_radius**2
        raise ValueError('Unknown event')

    def event_rate(self, event, degree=None):
        return self.attempt_rate(event, degree)*float(expit(-self.event_delta(event, degree)[1]))

    def _installed_contact_events(self, i, j, edge, interface=-1):
        """One admission graph, with resource owners distinct from terminals."""
        def event(kind, sign=1):
            return MaterialEvent(kind,i,j,edge,sign,interface=interface)
        bound,trace,index = self._contact_arrays(event('active'))
        c,r = bound[index],trace[index]
        events = []
        if not c and not r and self.depleted[j]: events.append(event('assembly',-1))
        if not c and self.free[j]: events.append(event('binding'))
        if c: events.append(event('binding',-1))
        if c and self.charged[i]: events.append(event('transport'))
        if not c and self.carried[i] and self.depleted[i]: events.append(event('transport',-1))
        if not r and self.charged[i]: events.append(event('write'))
        if r and self.depleted[i]: events.append(event('write',-1))
        events.append(event('thermal',-1 if r else 1))
        if self.charged[i]: events.append(event('active'))
        if self.depleted[i]: events.append(event('active',-1))
        return events

    def events(self, pairs=None, degree=None):
        events = []
        pairs = self.encounters() if pairs is None else pairs
        degree = self.degrees() if degree is None else degree
        present = {self.bridge_key(a, b) for a, b in zip(self.pre, self.post)}
        ends = self.end_capacity-degree
        for a, b in pairs:
            a, b = int(a), int(b)
            for i, j in ((a, b), (b, a)):
                if (self.bridge_key(i, j) not in present and ends[a] and ends[b]
                        and self.material[j] and self.charged[j]):
                    events.append(MaterialEvent('assembly', i, j))
            for pool in ('material', 'free', 'charged', 'depleted'):
                if getattr(self, pool)[a]: events.append(MaterialEvent('exchange', a, b, pool=pool))
                if getattr(self, pool)[b]: events.append(MaterialEvent('exchange', a, b, sign=-1, pool=pool))
        for e, (i, j) in enumerate(zip(self.pre, self.post)):
            i, j = int(i), int(j)
            events.extend(self._installed_contact_events(i,j,e))
        for i in range(self.n):
            if self.carried[i]: events.append(MaterialEvent('release', i))
            if self.free[i]: events.append(MaterialEvent('release', i, sign=-1))
            if self.depleted[i]: events.append(MaterialEvent('conversion', i))
            if self.charged[i]: events.append(MaterialEvent('conversion', i, sign=-1))
        return [e for e in events if self.attempt_rate(e, degree) > 0.]

    def _apply_assembly(self, event):
        if event.sign > 0:
            self.pre, self.post = np.r_[self.pre, event.i], np.r_[self.post, event.j]
            self.bound, self.trace = np.r_[self.bound, 0], np.r_[self.trace, 0]
            self.capacity = np.ones(len(self.pre))
        else:
            keep = np.arange(len(self.pre)) != event.edge
            for name in ('pre', 'post', 'bound', 'trace', 'capacity'):
                setattr(self, name, getattr(self, name)[keep])

    def apply_event(self, event, *, event_time=None, check=True, degree=None):
        if check and event not in self.events():
            raise ValueError('Unavailable material event')
        degree = self.degrees() if degree is None else degree
        du, dg = self.event_delta(event, degree)
        before_degree = degree
        for name, node, change in self._pools(event):
            getattr(self, name)[node] += change
        for node, dq in self._charge_changes(event):
            self._add_charge(node, dq)
        e, s = event.edge, event.sign
        if event.kind == 'assembly':
            self._apply_assembly(event)
        elif event.kind in ('binding', 'transport'):
            bound, _, index = self._contact_arrays(event)
            bound[index] += s if event.kind == 'binding' else -s
        elif event.kind in ('write', 'thermal'):
            _, trace, index = self._contact_arrays(event)
            trace[index] += s
        if event.kind == 'conversion':
            self.electrode_charge += s*self.params.charge_per_event
        self.heat_out -= du
        self.reaction_heat -= du
        self.entropy_balance -= dg
        key = ('port_' if event.interface >= 0 else '')+event.kind+('+' if s > 0 else '-')
        self.reaction_counts[key] = self.reaction_counts.get(key, 0)+1
        if self.event_observer:
            record = dict(kind=event.kind, sign=s, i=event.i, j=event.j,
                time_ms=self.t if event_time is None else event_time,
                degrees_before=before_degree[[event.i, event.j]].tolist() if event.j >= 0 else [],
                distance=float(np.linalg.norm(self.coordinates[event.i]-self.coordinates[event.j]))
                if event.j >= 0 else None, delta_u=du, delta_g=dg)
            if event.interface >= 0:
                record['interface'] = event.interface
            self.event_observer(record)

    def _reactions(self, dt):
        elapsed = 0.
        # Geometry is fixed inside this operator substep; search once. Each
        # hazard refresh shares one degree/end panel instead of recomputing
        # an O(N+E) topology reduction for every candidate event.
        pairs = self.encounters()
        while True:
            degree = self.degrees()
            events = self.events(pairs, degree)
            rates = np.array([self.event_rate(e, degree) for e in events])
            total = rates.sum()
            if not np.isfinite(total): raise FloatingPointError('Nonfinite event hazard')
            if not total: break
            elapsed += self.rng_reaction.exponential(1/total)
            if elapsed > dt: break
            target = self.rng_reaction.random()*total
            index = min(int(np.searchsorted(np.cumsum(rates), target, side='right')), len(events)-1)
            self.apply_event(events[index], event_time=self.t+elapsed, check=False, degree=degree)

    def move_trial(self, i, proposal, acceptance_uniform):
        proposal = np.asarray(proposal, float)
        if not 0 <= i < self.n or not np.isfinite(acceptance_uniform) or not 0 <= acceptance_uniform < 1:
            raise ValueError('Valid material index and uniform variate required')
        self.motion_attempts += 1
        p = self.params
        if proposal.shape != (3,) or not np.isfinite(proposal).all():
            raise ValueError('Finite proposed coordinate required')
        if np.any(proposal < 0) or np.any(proposal > p.box_size): return False
        bins = getattr(self, '_motion_bins', None)
        if bins is None:
            candidates = cKDTree(self.coordinates).query_ball_point(proposal, p.core_diameter)
        else:
            key = np.floor(proposal/p.core_diameter).astype(int)
            candidates = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        candidates.extend(bins.get(tuple(key+[dx, dy, dz]), ()))
        if any(j != i and np.linalg.norm(self.coordinates[j]-proposal) < p.core_diameter
               for j in candidates): return False
        adjacency = getattr(self, '_motion_neighbors', None)
        neighbor = (np.r_[self.post[self.pre == i], self.pre[self.post == i]]
                    if adjacency is None else adjacency[i])
        new_lengths = np.linalg.norm(self.coordinates[neighbor]-proposal, axis=1)
        if np.any(new_lengths > p.contact_radius): return False
        old = self.coordinates[i].copy()
        old_lengths = np.linalg.norm(self.coordinates[neighbor]-old, axis=1)
        du = .5*p.elastic_stiffness*float(np.sum(
            (new_lengths-p.rest_length)**2-(old_lengths-p.rest_length)**2))
        if acceptance_uniform >= min(1., math.exp(min(0., -du))):
            return False
        self.coordinates[i] = proposal
        if bins is not None:
            old_key = tuple(np.floor(old/p.core_diameter).astype(int))
            new_key = tuple(np.floor(proposal/p.core_diameter).astype(int))
            bins[old_key].remove(i)
            if not bins[old_key]: del bins[old_key]
            bins.setdefault(new_key, set()).add(i)
        self.motion_accepted += 1
        self.motion_heat -= du
        self.heat_out -= du
        self.entropy_balance -= du
        return True

    def _motion(self, dt):
        scale = math.sqrt(2*self.params.mobility*dt)
        self._motion_bins = {}
        self._motion_neighbors = [[] for _ in range(self.n)]
        for i, xyz in enumerate(self.coordinates):
            self._motion_bins.setdefault(tuple(np.floor(xyz/self.params.core_diameter).astype(int)), set()).add(i)
        for a, b in zip(self.pre, self.post):
            self._motion_neighbors[a].append(b)
            self._motion_neighbors[b].append(a)
        self._motion_neighbors = [np.asarray(v, np.int64) for v in self._motion_neighbors]
        try:
            for i in self.rng_motion.permutation(self.n):
                displacement = self.rng_motion.normal(size=3)*scale
                u = self.rng_motion.random()
                if scale:
                    self.move_trial(int(i), self.coordinates[i]+displacement, u)
        finally:
            del self._motion_bins, self._motion_neighbors

    def _field(self, dt, world_voltage):
        if self.field_scheme == 'symmetric_ou':
            return self._field_symmetric(dt, world_voltage)
        if world_voltage is not None:
            old = self.world_charge.copy()
            rate = dt*self.source_conductance/self.world_capacitance
            delta = ((world_voltage-old/self.world_capacitance)*(-np.expm1(-rate))*self.world_capacitance
                +self.rng_field.normal(size=len(old))*np.sqrt(
                    -self.world_capacitance*np.expm1(-2*rate)))
            self.world_charge += delta
            work = float(np.dot(world_voltage, delta))
            heat = work-.5*float(np.sum((self.world_charge**2-old**2)/self.world_capacitance))
            self.work_in += work
            self.source_charge += float(delta.sum())
            self.source_heat += heat
            self.heat_out += heat
            self.entropy_balance += heat
            if self.source_observer:
                self.source_observer(dict(time_ms=self.t, dt_ms=dt,
                    before_charge=old.copy(), after_charge=self.world_charge.copy(),
                    source_work=work, source_heat=heat))
        port_g = self.effective_port_conductance()
        for a, i in enumerate(self.port_nodes):
            inverse_c = 1/self.capacitance[i]+1/self.world_capacitance[a]
            dv = self.charge[i]/self.capacitance[i]-self.world_charge[a]/self.world_capacitance[a]
            rate = dt*port_g[a]*inverse_c
            sent = (dv*(-np.expm1(-rate))/inverse_c
                +self.rng_field.normal()*math.sqrt(-np.expm1(-2*rate)/inverse_c))
            heat = (dv*sent-.5*inverse_c*sent*sent)
            self.charge[i] -= sent
            self.world_charge[a] += sent
            self.heat_out += heat
            self.conduction_heat += heat
            self.entropy_balance += heat
        if not len(self.pre): return
        g = self.bridge_conductance()
        # Exact conditional OU charge exchange at the same kBT=1 as motion
        # and reactions. Passive dissipation without this variance would
        # silently place the fast field at a different bath temperature.
        for e, (i, j) in enumerate(zip(self.pre, self.post)):
            inverse_c = 1/self.capacitance[i]+1/self.capacitance[j]
            dv = self.charge[i]/self.capacitance[i]-self.charge[j]/self.capacitance[j]
            rate = dt*g[e]*inverse_c
            sent = (dv*(-np.expm1(-rate))/inverse_c
                +self.rng_field.normal()*math.sqrt(-np.expm1(-2*rate)/inverse_c))
            heat = dv*sent-.5*inverse_c*sent*sent
            self.charge[i] -= sent
            self.charge[j] += sent
            self.heat_out += heat
            self.conduction_heat += heat
            self.entropy_balance += heat

    def _field_symmetric(self, dt, world_voltage):
        """Palindromic local OU composition, with O(N+E) workspace.

        Every channel advances dt in total; the global clock advances only
        once in step(). This changes execution accuracy, not constitutive
        laws. Source work and heat are measured at each actual half update.
        """
        n = self.n
        q = np.r_[self.charge, self.world_charge]
        capacitance = np.r_[self.capacitance, self.world_capacitance]
        conductance = self.bridge_conductance()
        port_g = self.effective_port_conductance()
        half = dt/2

        def source():
            if world_voltage is None:
                return
            old = q[n:].copy()
            rate = half*self.source_conductance/self.world_capacitance
            delta = ((world_voltage-old/self.world_capacitance)*(-np.expm1(-rate))*self.world_capacitance
                +self.rng_field.normal(size=len(old))*np.sqrt(-self.world_capacitance*np.expm1(-2*rate)))
            q[n:] += delta
            work = float(np.dot(world_voltage, delta))
            heat = work-.5*float(np.sum((q[n:]**2-old**2)/self.world_capacitance))
            self.work_in += work
            self.source_charge += float(delta.sum())
            self.source_heat += heat
            self.heat_out += heat
            self.entropy_balance += heat
            if self.source_observer:
                self.charge[:], self.world_charge[:] = q[:n], q[n:]
                self.source_observer(dict(time_ms=self.t, dt_ms=half,
                    before_charge=old.copy(), after_charge=q[n:].copy(),
                    source_work=work, source_heat=heat))

        def pair(i, j, g):
            inverse_c = 1/capacitance[i]+1/capacitance[j]
            dv = q[i]/capacitance[i]-q[j]/capacitance[j]
            rate = half*g*inverse_c
            sent = (dv*(-np.expm1(-rate))/inverse_c
                +self.rng_field.normal()*math.sqrt(-np.expm1(-2*rate)/inverse_c))
            heat = dv*sent-.5*inverse_c*sent*sent
            q[i] -= sent
            q[j] += sent
            self.heat_out += heat
            self.conduction_heat += heat
            self.entropy_balance += heat

        source()
        for a, i in enumerate(self.port_nodes):
            pair(int(i), n+a, port_g[a])
        for e, (i, j) in enumerate(zip(self.pre, self.post)):
            pair(int(i), int(j), conductance[e])
        for e in range(len(self.pre)-1, -1, -1):
            pair(int(self.pre[e]), int(self.post[e]), conductance[e])
        for a in range(len(self.port_nodes)-1, -1, -1):
            pair(int(self.port_nodes[a]), n+a, port_g[a])
        source()
        self.charge[:], self.world_charge[:] = q[:n], q[n:]

    def bridge_conductance(self):
        p = self.params
        return self.unit_conductance*(p.gate_low+(p.gate_high-p.gate_low)*self.trace)

    def effective_port_conductance(self):
        return self.port_conductance

    def step(self, dt, world_voltage=None):
        if not np.isfinite(dt) or dt <= 0: raise ValueError('Positive finite elapsed time required')
        if world_voltage is not None:
            world_voltage = np.asarray(world_voltage, float)
            if (world_voltage.shape != self.port_nodes.shape or not np.isfinite(world_voltage).all()
                    or np.any(np.abs(world_voltage) > 100.)):
                raise ValueError('Finite bounded world source voltage required')
        self._motion(dt)
        self._reactions(dt)
        self._field(dt, world_voltage)
        self.t += dt
        self._validate()

    def balances(self):
        return dict(material=self.material.sum()+len(self.pre)-self.initial_material,
            drive=self.charged.sum()+self.depleted.sum()-self.initial_drive,
            support=self.free.sum()+self.carried.sum()+self.bound.sum()-self.initial_support,
            ends=self.remaining_ends().sum()+2*len(self.pre)-self.end_capacity.sum(),
            charge=self.charge.sum()+self.world_charge.sum()+self.electrode_charge
                   -self.source_charge-self.initial_total_charge,
            energy=self.energy()+self.heat_out-self.work_in-self.initial_energy,
            free_energy=self.initial_free_energy+self.work_in-self.free_energy()-self.entropy_balance)

    def state_dict(self):
        if self.field_scheme not in ('sequential_ou', 'symmetric_ou'):
            raise ValueError('Unknown field integration scheme')
        state = dict(schema=2, kind='mobile_exchange', params=asdict(self.params),
            pre=self.pre.copy(), post=self.post.copy(), capacity=self.capacity.copy(),
            arrays={name: getattr(self, name).copy() for name in self._arrays},
            port_nodes=self.port_nodes.tolist(), unit_conductance=self.unit_conductance,
            rng_reaction=copy.deepcopy(self.rng_reaction.bit_generator.state),
            rng_motion=copy.deepcopy(self.rng_motion.bit_generator.state),
            rng_field=copy.deepcopy(self.rng_field.bit_generator.state),
            reaction_counts=dict(self.reaction_counts), motion_attempts=self.motion_attempts,
            motion_accepted=self.motion_accepted,
            **{name: float(getattr(self, name)) for name in self._ledgers})
        if self.field_scheme == 'symmetric_ou':
            state.update(schema=3, field_scheme=self.field_scheme)
        return state

    @classmethod
    def from_state_dict(cls, state):
        if state.get('schema') not in (2,3) or state.get('kind') != 'mobile_exchange':
            raise ValueError('Unsupported mobile material archive')
        if state.get('field_scheme', 'sequential_ou') != ('symmetric_ou' if state['schema']==3 else 'sequential_ou'):
            raise ValueError('Field scheme and archive schema disagree')
        a = state['arrays']
        obj = cls(a['coordinates'], a['material'], a['free'], a['charged'], a['depleted'],
            end_capacity=a['end_capacity'], carried=a['carried'], charge=a['charge'],
            capacitance=a['capacitance'], pre=state['pre'], post=state['post'],
            bound=a['bound'], trace=a['trace'], unit_conductance=state['unit_conductance'],
            params=MobileExchangeParams(**state['params']), port_nodes=state['port_nodes'],
            world_charge=a['world_charge'], world_capacitance=a['world_capacitance'],
            port_conductance=a['port_conductance'], source_conductance=a['source_conductance'])
        return cls._restore_archive(obj, state)

    @classmethod
    def _restore_archive(cls, obj, state):
        """Restore all arrays/ledgers/RNGs after a subtype builds its carrier."""
        a = state['arrays']
        obj.field_scheme = state.get('field_scheme', 'sequential_ou')
        if not np.array_equal(state['capacity'], obj.capacity):
            raise ValueError('Corrupt actual bridge inventory')
        for name in cls._arrays:
            value = np.asarray(a[name])
            if value.shape != getattr(obj, name).shape or not np.isfinite(value).all():
                raise ValueError('Invalid mobile state array: '+name)
            setattr(obj, name, value.copy())
        for name in cls._ledgers:
            if not np.isfinite(state[name]): raise ValueError('Nonfinite mobile ledger')
            setattr(obj, name, float(state[name]))
        obj.rng_reaction.bit_generator.state = state['rng_reaction']
        obj.rng_motion.bit_generator.state = state['rng_motion']
        obj.rng_field.bit_generator.state = state['rng_field']
        obj.reaction_counts = dict(state['reaction_counts'])
        obj.motion_attempts, obj.motion_accepted = state['motion_attempts'], state['motion_accepted']
        obj._validate()
        if any(abs(float(v)) > 1e-8 for v in obj.balances().values()):
            raise ValueError('Mobile archive fails full-state accounting')
        return obj
