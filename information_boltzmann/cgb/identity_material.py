"""Existing directed identities as relative recognition and transport genes.

Geometry is supplied by actual material encounters. The centered identity
components retain the original programs; their geometric terms and intercepts
are not re-applied to a local contact. The reference-ensemble interpretation
and unresolved energy/entropy split are explicit in the associated derivation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from collections import OrderedDict

import numpy as np

from .occurrence import OccurrenceProgram
from .type_generator import TypeStrengthGenerator
from .mobile_exchange import MobileExchange, MobileExchangeParams


class IdentityMaterialProgram:
    array_names = ('code', 'occurrence_affinity', 'occurrence_out', 'occurrence_in',
                   'strength_affinity', 'strength_out', 'strength_in')

    def __init__(self, arrays, types, provenance=None):
        self.arrays = {name: np.asarray(arrays[name]).copy() for name in self.array_names}
        self.types = np.asarray(types, dtype=str).copy()
        self.provenance = dict(provenance or {})
        code = self.arrays['code']
        if code.ndim != 2 or len(self.types) != len(code) or len(set(self.types)) != len(code):
            raise ValueError('One unique known type per code row required')
        n, k = code.shape
        shapes = dict(code=(n, k), occurrence_affinity=(k, k), occurrence_out=(k,), occurrence_in=(k,),
                      strength_affinity=(k, k), strength_out=(n,), strength_in=(n,))
        for name, shape in shapes.items():
            if self.arrays[name].shape != shape or not np.isfinite(self.arrays[name]).all():
                raise ValueError('Invalid identity gene: '+name)
        raw = code.astype(float)
        self._normalized = raw/np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1e-30)
        self._occ_left = self._normalized@self.arrays['occurrence_affinity']
        self._strength_left = raw@self.arrays['strength_affinity']
        co, cs = self._normalized.mean(0), raw.mean(0)
        self.occurrence_center = float(co@self.arrays['occurrence_out']+co@self.arrays['occurrence_in']
            +co@self.arrays['occurrence_affinity']@co)
        self.strength_center = float(np.mean(self.arrays['strength_out'])+np.mean(self.arrays['strength_in'])
            +cs@self.arrays['strength_affinity']@cs)
        for value in self.arrays.values():
            value.setflags(write=False)
        self.types.setflags(write=False)

    @classmethod
    def from_sources(cls, strength_path, occurrence_path):
        strength = TypeStrengthGenerator.load(strength_path)
        occurrence = OccurrenceProgram.load(occurrence_path)
        if (not np.array_equal(strength.types, occurrence.values['types'].astype(str))
                or not np.array_equal(strength.values['code'], occurrence.values['code'])):
            raise ValueError('Occurrence and strength must share the actual type/code population')
        arrays = dict(code=strength.values['code'], occurrence_affinity=occurrence.values['affinity'],
            occurrence_out=occurrence.values['out'], occurrence_in=occurrence.values['in'],
            strength_affinity=strength.affinity, strength_out=strength.values['out_bias'],
            strength_in=strength.values['in_bias'])
        sources = {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                   for p in (strength_path, occurrence_path)}
        return cls(arrays, strength.types, dict(source_sha256=sources,
            scope='centered directed identity components, not full geometric occurrence or strength predictions'))

    def _indices(self, pre, post):
        a, b = np.asarray(pre), np.asarray(post)
        if (a.shape != b.shape or a.dtype.kind not in 'iu' or b.dtype.kind not in 'iu'
                or np.any(a < 0) or np.any(b < 0) or np.any(a >= len(self.types)) or np.any(b >= len(self.types))):
            raise ValueError('Known integer identities with matching shapes required')
        return a, b

    def recognition_field(self, pre, post):
        a, b = self._indices(pre, post)
        return (self._normalized[a]@self.arrays['occurrence_out']
            +self._normalized[b]@self.arrays['occurrence_in']
            +np.sum(self._occ_left[a]*self._normalized[b], axis=-1)-self.occurrence_center)

    def transport_log_scale(self, pre, post):
        a, b = self._indices(pre, post)
        return (self.arrays['strength_out'][a]+self.arrays['strength_in'][b]
            +np.sum(self._strength_left[a]*self.arrays['code'][b], axis=-1)-self.strength_center)

    def fingerprint(self):
        h = hashlib.sha256(json.dumps(self.types.tolist(), ensure_ascii=False).encode())
        for name in self.array_names:
            value = self.arrays[name]
            h.update(name.encode()); h.update(str(value.dtype).encode())
            h.update(str(value.shape).encode()); h.update(value.tobytes())
        return h.hexdigest()


class IdentityMobileExchange(MobileExchange):
    _arrays = MobileExchange._arrays+('cell_type',)+tuple('gene_'+n for n in IdentityMaterialProgram.array_names)

    def __init__(self, *args, program, cell_type, recognition_energy_fraction,
                 recognition_enabled=True, transport_enabled=True, **kwargs):
        if not isinstance(program, IdentityMaterialProgram):
            raise TypeError('A complete inherited identity program is required')
        self.program = program
        self.cell_type = np.asarray(cell_type).copy()
        self.cell_type.setflags(write=False)
        self._pair_cache = OrderedDict()
        self.recognition_energy_fraction = float(recognition_energy_fraction)
        self.recognition_enabled, self.transport_enabled = bool(recognition_enabled), bool(transport_enabled)
        for name in program.array_names:
            setattr(self, 'gene_'+name, program.arrays[name])
        super().__init__(*args, **kwargs)

    def bridge_key(self, i, j):
        # Reciprocal polarized bridges are distinct finite material objects.
        return int(i), int(j)

    def _validate(self):
        super()._validate()
        if (self.cell_type.shape != (self.n,) or self.cell_type.dtype.kind not in 'iu'
                or np.any(self.cell_type < 0) or np.any(self.cell_type >= len(self.program.types))
                or not 0 <= self.recognition_energy_fraction <= 1):
            raise ValueError('Known material identities and explicit energy fraction in [0,1] required')
        values = self.program.transport_log_scale(self.cell_type, self.cell_type)
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite identity mobility')

    def recognition_free_energy(self, pre, post):
        if not self.recognition_enabled:
            return np.zeros(np.shape(pre))
        if np.ndim(pre) == 0 and np.ndim(post) == 0:
            return self._pair_properties(int(pre), int(post))[0]
        return -self.program.recognition_field(self.cell_type[pre], self.cell_type[post])

    def _pair_properties(self, pre, post):
        """Bounded, disposable memoization of immutable genotype responses.

        This is execution state, not learned memory. Eviction only recomputes
        the same full-rank rule; it never removes a physical contact or trace.
        """
        a, b = self.cell_type[pre], self.cell_type[post]
        key = int(a), int(b)
        if key in self._pair_cache:
            self._pair_cache.move_to_end(key)
            return self._pair_cache[key]
        f = -self.program.recognition_field(a, b)
        log_scale = self.program.transport_log_scale(a, b)
        result = f, log_scale
        if len(self._pair_cache) >= 4096:
            self._pair_cache.popitem(last=False)
        self._pair_cache[key] = result
        return result

    def _recognition_total(self):
        return float(np.sum(self.recognition_free_energy(self.pre, self.post)))

    def energy(self):
        return super().energy()+self.recognition_energy_fraction*self._recognition_total()

    def free_energy(self):
        return (super().free_energy()
            +(1-self.recognition_energy_fraction)*self._recognition_total())

    def event_delta(self, event, degree=None):
        du, dg = super().event_delta(event, degree)
        if event.kind == 'assembly':
            df = event.sign*float(self.recognition_free_energy(event.i, event.j))
            du += self.recognition_energy_fraction*df
            dg += df
        return du, dg

    def _transport_factor(self, pre, post):
        if not self.transport_enabled:
            return np.ones(np.shape(pre))
        if np.ndim(pre) == 0 and np.ndim(post) == 0:
            log_scale = self._pair_properties(int(pre), int(post))[1]
            if abs(log_scale) > 600:
                raise ValueError('Identity mobility exceeds reversible numerical range')
            return np.exp(log_scale)
        logs = self.program.transport_log_scale(self.cell_type[pre], self.cell_type[post])
        if np.any(np.abs(logs) > 600):
            raise ValueError('Identity mobility exceeds reversible numerical range')
        return np.exp(logs)

    def bridge_conductance(self):
        return super().bridge_conductance()*self._transport_factor(self.pre, self.post)

    def attempt_rate(self, event, degree=None):
        rate = super().attempt_rate(event, degree)
        if event.kind in ('active', 'write'):
            rate *= float(self._transport_factor(event.i, event.j))
        return rate

    def state_dict(self):
        state = super().state_dict()
        state.update(schema=1, kind='identity_mobile_exchange',
            recognition_energy_fraction=self.recognition_energy_fraction,
            recognition_enabled=self.recognition_enabled, transport_enabled=self.transport_enabled,
            genome=dict(types=self.program.types.tolist(), provenance=self.program.provenance,
                        sha256=self.program.fingerprint()))
        return state

    @classmethod
    def from_state_dict(cls, state):
        if state.get('schema') != 1 or state.get('kind') != 'identity_mobile_exchange':
            raise ValueError('Unsupported inherited identity material archive')
        a, genome = state['arrays'], state['genome']
        program = IdentityMaterialProgram({k: a['gene_'+k] for k in IdentityMaterialProgram.array_names},
            genome['types'], genome['provenance'])
        if program.fingerprint() != genome['sha256']:
            raise ValueError('Identity program differs from its retained fingerprint')
        obj = cls(a['coordinates'], a['material'], a['free'], a['charged'], a['depleted'],
            end_capacity=a['end_capacity'], carried=a['carried'], charge=a['charge'],
            capacitance=a['capacitance'], pre=state['pre'], post=state['post'],
            bound=a['bound'], trace=a['trace'], unit_conductance=state['unit_conductance'],
            params=MobileExchangeParams(**state['params']), port_nodes=state['port_nodes'],
            world_charge=a['world_charge'], world_capacitance=a['world_capacitance'],
            port_conductance=a['port_conductance'], source_conductance=a['source_conductance'],
            program=program, cell_type=a['cell_type'],
            recognition_energy_fraction=state['recognition_energy_fraction'],
            recognition_enabled=state['recognition_enabled'], transport_enabled=state['transport_enabled'])
        obj = cls._restore_archive(obj, state)
        for name in program.array_names:
            setattr(obj, 'gene_'+name, program.arrays[name])
        obj.cell_type.setflags(write=False)
        return obj
