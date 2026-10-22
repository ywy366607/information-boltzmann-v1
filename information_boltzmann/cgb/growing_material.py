"""Paid birth/recovery of material domains in the existing local generator.

Domains and bridges each carry one material unit. A reversible, marked
Poisson transition pays 2 B + P -> domain + idle bridge + D. Coordinates,
charge and finite compositions have explicit proposal densities. Compact
array labels are not physical states: log(N!) and the uniform insertion
proposal cancel. No target shape or predictive score enters these rates.
"""
from __future__ import annotations

import copy
import math

import numpy as np
from scipy.special import gammaln

from .boundary_material import BoundaryIdentityMobileExchange
from .identity_material import IdentityMaterialProgram
from .mobile_exchange import MobileExchangeParams


class GrowingIdentityMobileExchange(BoundaryIdentityMobileExchange):
    _ledgers = BoundaryIdentityMobileExchange._ledgers + ('domain_heat',)
    _pools_names = ('material', 'free', 'carried', 'charged', 'depleted')
    _node_arrays = ('coordinates', 'material', 'free', 'carried', 'charged',
                    'depleted', 'charge', 'capacitance', 'end_capacity', 'cell_type')

    def __init__(self, *args, domain_formation_energy, reference_volume,
                 reference_charge, domain_reactions_enabled=True,
                 domain_sampling='resolved', material_scheme='sequential', **kwargs):
        if material_scheme not in ('sequential', 'symmetric'):
            raise ValueError('Explicit supported whole-material integration scheme required')
        self.material_scheme = 'sequential'  # the parent initializes its legacy field first
        self.domain_formation_energy = float(domain_formation_energy)
        self.reference_volume = float(reference_volume)
        self.reference_charge = float(reference_charge)
        self.domain_reactions_enabled = domain_reactions_enabled
        self.domain_sampling = domain_sampling
        self.domain_attempts = self.domain_births = self.domain_deaths = 0
        self.domain_observer = None  # disposable receipts, never a rate input
        super().__init__(*args, **kwargs)
        self.initial_material += self.n
        self.material_scheme = material_scheme
        if material_scheme == 'symmetric':
            self.field_scheme = 'symmetric_ou'
        self._validate()

    def _validate(self):
        super()._validate()
        if (not np.isfinite(self.domain_formation_energy)
                or not np.isfinite(self.reference_volume) or self.reference_volume <= 0
                or not np.isfinite(self.reference_charge) or self.reference_charge <= 0
                or not isinstance(self.domain_reactions_enabled, bool)
                or self.domain_sampling not in ('resolved', 'marginal')):
            raise ValueError('Explicit finite domain energy, positive standard measures and barrier required')
        if (self.material_scheme not in ('sequential', 'symmetric')
                or self.material_scheme == 'symmetric' and self.field_scheme != 'symmetric_ou'):
            raise ValueError('Whole-material symmetric integration requires symmetric OU fields')

    def energy(self):
        return super().energy() + self.n * self.domain_formation_energy

    def free_energy(self):
        # Quotient the N! redundant array permutations; ports are remapped too.
        return super().free_energy() + float(gammaln(self.n + 1))

    def balances(self):
        values = super().balances()
        values['material'] += self.n
        return values

    def _domain_attempt_rate(self, old_parent_degree):
        p = self.params
        return p.assembly_rate * (p.bulk_attempt + p.surface_attempt * old_parent_degree)

    def birth_log_ratio(self, parent, position, *, old_counts=None,
                        old_charge=None, old_degree=None):
        """Exact marginal ratio: Gaussian/partition proposal terms cancel.

        All five pools and both charge degrees remain actual state. This
        only removes their auxiliary draws from the acceptance decision.
        No predictive utility or new material parameter enters this ratio.
        """
        counts = (np.array([getattr(self, name)[parent] for name in self._pools_names], np.int64)
                  if old_counts is None else np.asarray(old_counts, np.int64))
        b, f, h, fuel, depleted = map(int, counts)
        degree = int(self.degrees()[parent] if old_degree is None else old_degree)
        ends = int(self.end_capacity[parent])
        if min(counts) < 0 or b < 2 or fuel < 1 or degree >= ends:
            raise ValueError('Unavailable parent composition or ends')
        q = float(self.charge[parent] if old_charge is None else old_charge)
        c, p = float(self.capacitance[parent]), self.params
        length = float(np.linalg.norm(np.asarray(position)-self.coordinates[parent]))
        volume = 4*math.pi*(p.contact_radius**3-p.core_diameter**3)/3
        composition = (math.log(b)+math.log(b-1)+math.log(fuel)-math.log(depleted+1)
                       + (sum(counts)-2)*math.log(2))
        return float(-self.domain_formation_energy-p.structure_energy+p.charged_energy
                     + p.cohesion_energy*degree-float(self.recognition_free_energy(parent, parent))
                     + q*q/(4*c)+.5*math.log(math.pi*c)-math.log(self.reference_charge)
                     + math.log(ends-degree)+math.log(ends)+math.log(volume/self.reference_volume)
                     + composition-.5*p.elastic_stiffness*(length-p.rest_length)**2)

    def death_log_ratio(self, parent, child, degree=None):
        """Birth/recovery ratio evaluated on this inverse channel's preimage."""
        merged = np.array([getattr(self, name)[parent]+getattr(self, name)[child]
                           for name in self._pools_names], np.int64)
        merged += np.array([2, 0, 0, 1, -1], np.int64)
        degree = self.degrees() if degree is None else degree
        return self.birth_log_ratio(parent, self.coordinates[child], old_counts=merged,
                                    old_charge=self.charge[parent]+self.charge[child],
                                    old_degree=degree[parent]-1)

    def domain_generator_rates(self):
        """Exact recovery hazards and rigorous birth bounds, not a utility."""
        degree = self.degrees()
        parents = self.birth_parents(degree) if self.domain_reactions_enabled else []
        deaths = self.death_channels(degree) if self.domain_reactions_enabled else []
        birth_bound = float(sum(self._domain_attempt_rate(degree[i]) for i in parents))
        rows = []
        for i, j, edge in deaths:
            log_ratio = self.death_log_ratio(i, j, degree)
            attempt = self._domain_attempt_rate(degree[i]-1)
            rows.append(dict(parent=i, child=j, edge=edge, attempt_rate=attempt,
                             log_birth_ratio=log_ratio,
                             accepted_rate=attempt*math.exp(min(0., -log_ratio))))
        death_rate = sum(row['accepted_rate'] for row in rows)
        return dict(birth_attempt_rate_upper_bound=birth_bound, recovery_accepted_rate=death_rate,
                    birth_parent_count=len(parents), inverse_channels=rows,
                    domain_count_drift_bounds=[-death_rate, birth_bound-death_rate],
                    charged_count_drift_bounds=[death_rate-birth_bound, death_rate],
                    scope='Original physical generator, rates per reference ms; actual birth acceptance requires the allowed spatial-mark integral, not computed here')

    def birth_parents(self, degree=None):
        degree = self.degrees() if degree is None else degree
        return np.flatnonzero((self.material >= 2) & (self.charged > 0)
                              & (self.end_capacity-degree > 0))

    def death_channels(self, degree=None):
        """Only inverses of an idle directed domain-birth transition."""
        degree = self.degrees() if degree is None else degree
        ports = set(self.port_nodes.tolist())
        channels = []
        for edge, (i, j) in enumerate(zip(self.pre, self.post)):
            i, j = int(i), int(j)
            if (degree[j] == 1 and j not in ports and not self.bound[edge]
                    and not self.trace[edge] and self.cell_type[i] == self.cell_type[j]
                    and self.capacitance[i] == self.capacitance[j]
                    and self.end_capacity[i] == self.end_capacity[j]
                    and self.depleted[i] + self.depleted[j] > 0):
                channels.append((i, j, edge))
        return channels

    def _birth_metrics(self, parent, position, child_charge, child_counts, *,
                       old_counts=None, old_charge=None, old_degree=None, old_n=None):
        """Local exact increment; no model/genome copies or dense workspace."""
        p = self.params
        old = (np.array([getattr(self, name)[parent] for name in self._pools_names], np.int64)
               if old_counts is None else np.asarray(old_counts, np.int64))
        available = old + np.array([-2, 0, 0, -1, 1], np.int64)
        child = np.asarray(child_counts)
        if (available.shape != (5,) or child.shape != (5,)
                or not np.isfinite(child).all() or not np.equal(child, np.floor(child)).all()
                or np.any(available < 0) or np.any(child < 0) or np.any(child > available)):
            raise ValueError('Unavailable domain composition partition')
        child = child.astype(np.int64)
        remaining = available - child
        q = self.charge[parent] if old_charge is None else float(old_charge)
        degree = int(self.degrees()[parent] if old_degree is None else old_degree)
        n = self.n if old_n is None else int(old_n)
        c = float(self.capacitance[parent])  # identical local material phase
        ends = int(self.end_capacity[parent])
        if degree >= ends or not np.isfinite(child_charge):
            raise ValueError('Finite charge and a free parent end required')
        length = float(np.linalg.norm(np.asarray(position) - self.coordinates[parent]))
        recognition = float(self.recognition_free_energy(parent, parent))
        electrical = ((q - child_charge)**2 + child_charge**2 - q*q) / (2*c)
        du = (electrical + self.domain_formation_energy - p.charged_energy
              + p.structure_energy + .5*p.elastic_stiffness*(length-p.rest_length)**2
              - p.cohesion_energy*degree + self.recognition_energy_fraction*recognition)
        mixture = float(np.sum(gammaln(remaining+1) + gammaln(child+1) - gammaln(old+1)))
        physical_g = (du + mixture - math.log(ends-degree) - math.log(ends)
                      + (1-self.recognition_energy_fraction)*recognition)
        volume = 4*math.pi*(p.contact_radius**3 - p.core_diameter**3)/3
        log_position = math.log(self.reference_volume/volume)
        effective_c = c/2  # kBT=1, fixed total charge on two equal capacities
        log_charge = (math.log(self.reference_charge)
                      - .5*(math.log(2*math.pi*effective_c)
                            + (child_charge-q/2)**2/effective_c))
        log_partition = float(np.sum(gammaln(available+1) - gammaln(child+1)
                                     - gammaln(remaining+1) - available*math.log(2)))
        return dict(delta_u=float(du), delta_g=float(physical_g+math.log(n+1)),
                    delta_g_physical=float(physical_g), electrical_energy_delta=float(electrical),
                    drive_energy_delta=-float(p.charged_energy),
                    log_position_density=log_position, log_charge_density=log_charge,
                    log_partition_probability=log_partition,
                    log_birth_acceptance_ratio=float(-physical_g-log_position-log_charge-log_partition),
                    parent_counts=remaining, child_counts=child)

    @staticmethod
    def _accept(log_ratio, uniform):
        if not np.isfinite(uniform) or not 0 <= uniform < 1:
            raise ValueError('A uniform variate in [0,1) is required')
        return uniform == 0 or math.log(uniform) < min(0., log_ratio)

    def _record_domain(self, receipt, metrics=None, event_time=None):
        if self.domain_sampling == 'marginal':
            receipt['domain_sampling'] = 'marginal'
        if metrics is not None:
            self.domain_heat -= receipt['delta_u']
            self.heat_out -= receipt['delta_u']
            self.reaction_heat -= receipt['delta_u']
            self.entropy_balance -= receipt['delta_g']
            key = 'domain+' if receipt['kind'] == 'domain_birth' else 'domain-'
            self.reaction_counts[key] = self.reaction_counts.get(key, 0)+1
        receipt['time_ms'] = self.t if event_time is None else float(event_time)
        if self.domain_observer is not None:
            self.domain_observer(copy.deepcopy(receipt))
        return receipt

    def _valid_birth_geometry(self, parent, xyz):
        length = float(np.linalg.norm(xyz-self.coordinates[parent]))
        return (not np.any(xyz < 0) and not np.any(xyz > self.params.box_size)
                and self.params.core_diameter <= length <= self.params.contact_radius
                and not np.any(np.linalg.norm(self.coordinates-xyz, axis=1) < self.params.core_diameter))

    def birth_trial(self, parent, position, child_charge, child_counts, uniform, *,
                    insertion_index=None, event_time=None):
        """One marked Poisson proposal. Forced variates are numerical controls."""
        if parent not in self.birth_parents():
            raise ValueError('Unavailable domain birth parent')
        xyz = np.asarray(position, float)
        if xyz.shape != (3,) or not np.isfinite(xyz).all():
            raise ValueError('A finite 3D birth position is required')
        if not np.isfinite(uniform) or not 0 <= uniform < 1:
            raise ValueError('A uniform variate in [0,1) is required')
        metrics = self._birth_metrics(parent, xyz, child_charge, child_counts)
        self.domain_attempts += 1
        receipt = dict(kind='domain_birth', parent_before=int(parent), nodes_before=self.n,
                       accepted=False, position=xyz.tolist(), child_charge=float(child_charge),
                       child_compositions=metrics['child_counts'].tolist(),
                       **{k:v for k,v in metrics.items() if not k.endswith('_counts')})
        if not self._valid_birth_geometry(parent, xyz):
            receipt['rejection'] = 'outside allowed physical geometry'
            return self._record_domain(receipt, event_time=event_time)
        if not self._accept(metrics['log_birth_acceptance_ratio'], uniform):
            receipt['rejection'] = 'reversible acceptance'
            return self._record_domain(receipt, event_time=event_time)
        slot = (int(self.rng_reaction.integers(self.n+1))
                if insertion_index is None else int(insertion_index))
        if not 0 <= slot <= self.n:
            raise ValueError('Insertion index is outside the compact array')
        values = dict(coordinates=xyz, charge=float(child_charge),
                      capacitance=self.capacitance[parent], end_capacity=self.end_capacity[parent],
                      cell_type=self.cell_type[parent])
        for name, a, b in zip(self._pools_names, metrics['parent_counts'], metrics['child_counts']):
            getattr(self, name)[parent] = a
            values[name] = b
        self.charge[parent] -= child_charge
        for name in self._node_arrays:
            setattr(self, name, np.insert(getattr(self, name), slot, values[name], axis=0))
        self.cell_type.setflags(write=False)
        self.pre += self.pre >= slot
        self.post += self.post >= slot
        self.port_nodes += self.port_nodes >= slot
        parent_after = int(parent+(parent >= slot))
        self.pre = np.r_[self.pre, parent_after]
        self.post = np.r_[self.post, slot]
        self.bound, self.trace = np.r_[self.bound, 0], np.r_[self.trace, 0]
        self.capacity = np.ones(len(self.pre))
        self.n += 1
        self.domain_births += 1
        receipt.update(accepted=True, child_after=slot, parent_after=parent_after, nodes_after=self.n)
        return self._record_domain(receipt, metrics, event_time)

    def death_trial(self, parent, child, edge, uniform, *, event_time=None):
        if (parent, child, edge) not in self.death_channels():
            raise ValueError('Unavailable inverse domain transition')
        merged = np.array([getattr(self, name)[parent]+getattr(self, name)[child]
                           for name in self._pools_names], np.int64)
        merged += np.array([2, 0, 0, 1, -1], np.int64)
        child_counts = [getattr(self, name)[child] for name in self._pools_names]
        q = float(self.charge[parent]+self.charge[child])
        metrics = self._birth_metrics(parent, self.coordinates[child], self.charge[child],
                                      child_counts, old_counts=merged, old_charge=q,
                                      old_degree=self.degrees()[parent]-1, old_n=self.n-1)
        self.domain_attempts += 1
        receipt = dict(kind='domain_death', parent_before=int(parent), child_before=int(child),
                       nodes_before=self.n, accepted=False,
                       **{k:(-v if k in ('delta_u', 'delta_g', 'delta_g_physical',
                                         'electrical_energy_delta', 'drive_energy_delta') else v)
                          for k,v in metrics.items() if not k.endswith('_counts')})
        if not self._accept(-metrics['log_birth_acceptance_ratio'], uniform):
            receipt['rejection'] = 'reversible acceptance'
            return self._record_domain(receipt, event_time=event_time)
        for name, value in zip(self._pools_names, merged):
            getattr(self, name)[parent] = value
        self.charge[parent] = q
        for name in self._node_arrays:
            setattr(self, name, np.delete(getattr(self, name), child, axis=0))
        self.cell_type.setflags(write=False)
        keep = np.arange(len(self.pre)) != edge
        for name in ('pre', 'post', 'bound', 'trace', 'capacity'):
            setattr(self, name, getattr(self, name)[keep])
        self.pre -= self.pre > child
        self.post -= self.post > child
        self.port_nodes -= self.port_nodes > child
        self.n -= 1
        self.domain_deaths += 1
        receipt.update(accepted=True, parent_after=int(parent-(parent > child)), nodes_after=self.n)
        return self._record_domain(receipt, metrics, event_time)

    def _propose_birth(self, parent, event_time):
        p = self.params
        direction = self.rng_reaction.normal(size=3)
        direction /= np.linalg.norm(direction)
        radius = (p.core_diameter**3+self.rng_reaction.random()
                  *(p.contact_radius**3-p.core_diameter**3))**(1/3)
        xyz = self.coordinates[parent]+radius*direction
        if self.domain_sampling == 'marginal':
            log_ratio = self.birth_log_ratio(parent, xyz)
            uniform = self.rng_reaction.random()
            valid = self._valid_birth_geometry(parent, xyz)
            accepted = valid and self._accept(log_ratio, uniform)
            if not accepted:
                self.domain_attempts += 1
                receipt = dict(kind='domain_birth', parent_before=int(parent), nodes_before=self.n,
                               accepted=False, position=xyz.tolist(),
                               log_birth_acceptance_ratio=log_ratio,
                               rejection='reversible acceptance' if valid else 'outside allowed physical geometry')
                return self._record_domain(receipt, event_time=event_time)
        charge = self.charge[parent]/2+self.rng_reaction.normal()*math.sqrt(self.capacitance[parent]/2)
        available = np.array([getattr(self, name)[parent] for name in self._pools_names])
        available += np.array([-2, 0, 0, -1, 1])
        partition = self.rng_reaction.binomial(available, .5)
        # In marginal mode the position has already been accepted with its
        # exact marginal rate. Draw the remaining unchanged physical marks
        # conditionally; do not perform a second acceptance/rejection.
        uniform = 0. if self.domain_sampling == 'marginal' else self.rng_reaction.random()
        return self.birth_trial(parent, xyz, charge, partition, uniform,
                                event_time=event_time)

    def _reactions(self, dt):
        if not self.domain_reactions_enabled:
            return super()._reactions(dt)
        elapsed = 0.
        origin = self.t+getattr(self, '_reaction_clock_offset', 0.)
        pairs = self.encounters()
        while True:
            degree = self.degrees()
            events = self.events(pairs, degree)
            parents, deaths = self.birth_parents(degree), self.death_channels(degree)
            rates = [self.event_rate(e, degree) for e in events]
            rates += [self._domain_attempt_rate(degree[i]) for i in parents]
            rates += [self._domain_attempt_rate(degree[i]-1)
                      * (math.exp(min(0., -self.death_log_ratio(i, j, degree)))
                         if self.domain_sampling == 'marginal' else 1.) for i,j,e in deaths]
            rates = np.asarray(rates, float)
            total = float(rates.sum())
            if not np.isfinite(total):
                raise FloatingPointError('Nonfinite joint reaction/domain hazard')
            if not total:
                break
            elapsed += self.rng_reaction.exponential(1/total)
            if elapsed > dt:
                break
            index = min(int(np.searchsorted(np.cumsum(rates),
                                            self.rng_reaction.random()*total, side='right')), len(rates)-1)
            if index < len(events):
                self.apply_event(events[index], event_time=origin+elapsed, check=False, degree=degree)
            else:
                index -= len(events)
                if index < len(parents):
                    receipt = self._propose_birth(int(parents[index]), origin+elapsed)
                else:
                    receipt = self.death_trial(*deaths[index-len(parents)],
                                               0. if self.domain_sampling == 'marginal'
                                               else self.rng_reaction.random(), event_time=origin+elapsed)
                if receipt['accepted']:
                    pairs = self.encounters()  # geometry/index space actually changed

    def step(self, dt, world_voltage=None):
        if self.material_scheme == 'sequential':
            return super().step(dt, world_voltage)
        # Reject invalid calls before mutating any physical/RNG state.
        self._validate()
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('Positive finite elapsed time required')
        if world_voltage is not None:
            world_voltage = np.asarray(world_voltage, float)
            if (world_voltage.shape != self.port_nodes.shape or not np.isfinite(world_voltage).all()
                    or np.any(np.abs(world_voltage) > 100.)):
                raise ValueError('Finite bounded world source voltage required')
        if getattr(self, '_material_step_active', False):
            raise ValueError('A partial or failed material step cannot be restarted')
        self._material_step_active = True
        half = dt/2
        self._motion(half)
        self._reaction_clock_offset = 0.
        self._reactions(half)
        self._field(dt, world_voltage)
        self._reaction_clock_offset = half
        self._reactions(half)
        self._motion(half)
        self.t += dt
        self._validate()
        # On exceptions leave the guard set: intermediate operator cursors
        # are not archived. Resume the last complete checkpoint instead.
        del self._reaction_clock_offset, self._material_step_active

    def state_dict(self):
        self._validate()
        state = super().state_dict()
        state.update(kind='growing_identity_mobile_exchange', schema=1,
                     domain_formation_energy=self.domain_formation_energy,
                     reference_volume=self.reference_volume, reference_charge=self.reference_charge,
                     domain_reactions_enabled=self.domain_reactions_enabled,
                     domain_attempts=self.domain_attempts, domain_births=self.domain_births,
                     domain_deaths=self.domain_deaths)
        if self.domain_sampling == 'marginal':
            state.update(schema=2, domain_sampling='marginal')
        if self.material_scheme == 'symmetric':
            state.update(schema=3, material_scheme='symmetric', domain_sampling=self.domain_sampling)
        return state

    @classmethod
    def from_boundary(cls, model, *, domain_reactions_enabled=True):
        if type(model) is not BoundaryIdentityMobileExchange:
            raise TypeError('A complete boundary-material individual is required')
        state = copy.deepcopy(model.state_dict())
        energy = model.params.structure_energy
        state['arrays']['initial_material'] += model.n
        state['initial_energy'] += energy*model.n
        state['initial_free_energy'] += energy*model.n+float(gammaln(model.n+1))
        state.update(kind='growing_identity_mobile_exchange', schema=1,
                     domain_formation_energy=energy, reference_volume=model.params.spacing**3,
                     reference_charge=model.params.charge_per_event,
                     domain_reactions_enabled=domain_reactions_enabled,
                     domain_heat=0., domain_attempts=0, domain_births=0, domain_deaths=0)
        return cls.from_state_dict(state)

    @classmethod
    def from_state_dict(cls, state):
        if state.get('kind') != 'growing_identity_mobile_exchange' or state.get('schema') not in (1, 2, 3):
            raise ValueError('Unsupported growing material archive')
        schema = state['schema']
        if schema == 3:
            if (state.get('material_scheme') != 'symmetric' or state.get('field_scheme') != 'symmetric_ou'
                    or state.get('domain_sampling') not in ('resolved', 'marginal')):
                raise ValueError('Symmetric whole-material archive requires explicit field and phase modes')
        elif state.get('material_scheme', 'sequential') != 'sequential':
            raise ValueError('Whole-material mode disagrees with archive schema')
        if schema != 3 and state.get('domain_sampling', 'resolved') != ('resolved' if schema == 1 else 'marginal'):
            raise ValueError('Domain sampling mode disagrees with archive schema')
        if state.get('field_scheme') not in ('sequential_ou', 'symmetric_ou'):
            raise ValueError('Explicit field mode required')
        a, genome = state['arrays'], state['genome']
        program = IdentityMaterialProgram({name:a['gene_'+name] for name in IdentityMaterialProgram.array_names},
                                         genome['types'], genome['provenance'])
        if program.fingerprint() != genome['sha256']:
            raise ValueError('Material genome fingerprint changed')
        obj = cls(a['coordinates'], a['material'], a['free'], a['charged'], a['depleted'],
                  end_capacity=a['end_capacity'], carried=a['carried'], charge=a['charge'],
                  capacitance=a['capacitance'], pre=state['pre'], post=state['post'],
                  bound=a['bound'], trace=a['trace'], unit_conductance=state['unit_conductance'],
                  params=MobileExchangeParams(**state['params']), port_nodes=state['port_nodes'],
                  world_charge=a['world_charge'], world_capacitance=a['world_capacitance'],
                  port_conductance=a['port_conductance'], source_conductance=a['source_conductance'],
                  program=program, cell_type=a['cell_type'],
                  recognition_energy_fraction=state['recognition_energy_fraction'],
                  recognition_enabled=state['recognition_enabled'], transport_enabled=state['transport_enabled'],
                  port_installed=a['port_installed'], port_bound=a['port_bound'], port_trace=a['port_trace'],
                  boundary_reactions_enabled=state['boundary_reactions_enabled'],
                  boundary_conduction_enabled=state['boundary_conduction_enabled'],
                  domain_formation_energy=state['domain_formation_energy'],
                  reference_volume=state['reference_volume'], reference_charge=state['reference_charge'],
                  domain_reactions_enabled=state['domain_reactions_enabled'],
                  domain_sampling=state.get('domain_sampling', 'resolved'),
                  material_scheme=state.get('material_scheme', 'sequential'))
        for name in ('domain_attempts', 'domain_births', 'domain_deaths'):
            value = state[name]
            if not isinstance(value, int) or value < 0:
                raise ValueError('Nonnegative domain event counters required')
            setattr(obj, name, value)
        if obj.domain_births+obj.domain_deaths > obj.domain_attempts:
            raise ValueError('Domain acceptances exceed actual attempts')
        obj = cls._restore_archive(obj, state)
        obj.cell_type.setflags(write=False)
        for name in program.array_names:
            setattr(obj, 'gene_'+name, program.arrays[name])
        return obj
