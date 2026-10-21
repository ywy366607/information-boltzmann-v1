"""Local, finite substrate for structural synthesis, maintenance and recycling.

This is an abstract material account, not a claim that trophic signalling
proteins are the substance of a synapse. Trophic Nf/T still control selection;
this separate pool pays for the structure they help retain. No past graph or
global loss is stored. State is O(nodes), transfer workspace O(edges).

The proportional transfer allocator enforces feasibility; conservation alone
does not derive it as finite-rate chemistry. Linear costs and donor ownership
are declared effective-model assumptions. Unpaid maintenance has no feedback
yet, so a balanced ledger is not a completed self-production mechanism.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class MaterialParams:
    supply: float = 0.0           # material / target cell / ms
    clearance: float = 0.0        # free-material clearance rate, 1/ms
    cost: float = 1.0             # material / structural conductance unit
    maintenance: float = 0.0      # material / conductance / ms
    recycle: float = 1.0          # recoverable share of dismantled material

    def __post_init__(self):
        if not all(np.isfinite(v) for v in asdict(self).values()):
            raise ValueError('Material parameters must be finite')
        if min(self.supply, self.clearance, self.maintenance) < 0 or self.cost <= 0:
            raise ValueError('Nonnegative rates and positive structural cost required')
        if not 0 <= self.recycle <= 1:
            raise ValueError('Recycling fraction must lie in [0, 1]')


class StructuralBudget:
    """One material donor pool at each postsynaptic cell, with an exact ledger.

    B + reserved_arbors + cost * sum(g) + clearance + maintenance + discarded
      = initial_material + imported_material.

    All cumulative quantities are per target cell, so a globally balanced
    account cannot conceal an unbalanced compartment. Maintenance is an actual
    debit, not an unlimited negative pool. Unpaid demand is recorded; it does
    not silently add a new starvation rule to E6/E9.
    """
    _fields = ('free', 'initial', 'imported', 'cleared', 'spent', 'discarded',
               'unpaid', 'synthesized', 'dismantled', 'reserved')

    def __init__(self, n_cells, post, g, params=None, initial_free=0.0):
        self.n = int(n_cells)
        self.post = np.asarray(post, dtype=np.int64).copy()
        self.params = params or MaterialParams()
        self.free = np.broadcast_to(np.asarray(initial_free, dtype=float), (self.n,)).copy()
        if not np.isfinite(self.free).all() or (self.free < 0).any():
            raise ValueError('Initial free material must be finite and nonnegative')
        if self.post.shape != np.asarray(g).shape or (self.post < 0).any() or (self.post >= self.n).any():
            raise ValueError('One valid postsynaptic donor required per edge')
        if not np.isfinite(g).all() or (np.asarray(g) < 0).any():
            raise ValueError('Initial structure must be finite and nonnegative')
        for name in self._fields[2:]:
            setattr(self, name, np.zeros(self.n))
        self.initial = self.free + self.params.cost * self.total(g)

    def total(self, edge_values):
        return np.bincount(self.post, weights=edge_values, minlength=self.n)

    def advance(self, g, dt):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('Positive finite elapsed time required')
        p = self.params
        incoming = np.full(self.n, p.supply * dt)
        old = self.free
        if p.clearance:
            decay = np.exp(-p.clearance * dt)
            self.free = old * decay + p.supply * (-np.expm1(-p.clearance * dt)) / p.clearance
        else:
            self.free = old + incoming
        self.imported += incoming
        self.cleared += old + incoming - self.free
        demand = p.maintenance * dt * (self.total(g) + self.reserved/p.cost)
        paid = np.minimum(self.free, demand)
        self.free -= paid
        self.spent += paid
        self.unpaid += demand - paid

    def transfer(self, old, requested):
        """Apply a local structural proposal using only available donor stock.

        Negative changes dismantle first. Simultaneous positive demands on a
        donor share its stock proportionally, avoiding edge-order priority.
        This limits a material flux, not a neural output rate or vector norm.
        """
        old, requested = np.asarray(old), np.asarray(requested)
        if old.shape != self.post.shape or requested.shape != old.shape:
            raise ValueError('Structure arrays must match the immutable edge layout')
        if not np.isfinite(requested).all() or (requested < 0).any():
            raise ValueError('Requested structure must be finite and nonnegative')
        p = self.params
        removed = np.maximum(old - requested, 0)
        removed_material = p.cost * self.total(removed)
        self.free += p.recycle * removed_material
        self.discarded += (1 - p.recycle) * removed_material
        self.dismantled += removed_material
        addition = np.maximum(requested - old, 0)
        demand = p.cost * self.total(addition)
        scale = np.ones(self.n)
        nonzero = demand > 0
        scale[nonzero] = np.minimum(1, self.free[nonzero] / demand[nonzero])
        added = addition * scale[self.post]
        used = p.cost * self.total(added)
        # Roundoff after a fully exhausted donor is not a source of material.
        self.free = np.maximum(self.free - used, 0)
        self.synthesized += used
        return old - removed + added

    def balance(self, g):
        return (self.free + self.reserved + self.params.cost * self.total(g) + self.cleared
                + self.spent + self.discarded - self.initial - self.imported)

    def reserve(self, requested_material):
        """Pay for non-synaptic arbor material from the same local donor.

        Reserved material remains in the balance and pays maintenance. This
        operation allocates material; it does not invent a topology-growth law.
        """
        requested = np.asarray(requested_material, dtype=float)
        if requested.shape != (self.n,) or not np.isfinite(requested).all() or (requested < 0).any():
            raise ValueError('One finite nonnegative material request per donor required')
        paid = np.minimum(self.free, requested)
        self.free -= paid
        self.reserved += paid
        return paid

    def state_dict(self):
        return {'params': asdict(self.params), 'n': self.n, 'post': self.post.copy(),
                **{name: getattr(self, name).copy() for name in self._fields}}

    @classmethod
    def from_state_dict(cls, state):
        obj = cls(state['n'], state['post'], np.zeros(len(state['post'])),
                  MaterialParams(**state['params']))
        for name in cls._fields:
            # V2f archives predate the optional non-synaptic material account.
            setattr(obj, name, np.asarray(state.get(name, np.zeros(obj.n)), dtype=float).copy())
        return obj
