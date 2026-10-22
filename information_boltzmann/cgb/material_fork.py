"""Lossless boundary-time material forks with shared immutable constitutions."""
from __future__ import annotations

import copy

from .mobile_exchange import MobileExchange


def fork_material(model: MobileExchange) -> MobileExchange:
    """Copy all physical/RNG state without rebuilding the identity genome.

    Disposable observation callbacks are detached, as with archive restore.
    Only the immutable identity program/genes and frozen parameter dataclass
    are shared. O(N+E) physical payload, no autograd or historical Jacobian.
    This is a diagnostic/research fork, not a different evolution rule.
    """
    if not isinstance(model, MobileExchange):
        raise TypeError('A complete mobile material state is required')
    if (hasattr(model, '_motion_bins') or hasattr(model, '_motion_neighbors')
            or getattr(model, '_material_step_active', False)):
        raise ValueError('Fork only at a complete operator boundary')
    fork = copy.copy(model)
    for name in model._arrays:
        value = getattr(model, name)
        if name.startswith('gene_'):
            if value.flags.writeable:
                raise ValueError('Shared constitution must be immutable')
            continue
        copied = value.copy()
        if not value.flags.writeable:
            copied.setflags(write=False)
        setattr(fork, name, copied)
    for name in ('pre', 'post', 'capacity', 'port_nodes'):
        setattr(fork, name, getattr(model, name).copy())
    for name in ('rng_reaction', 'rng_motion', 'rng_field'):
        setattr(fork, name, copy.deepcopy(getattr(model, name)))
    fork.reaction_counts = dict(model.reaction_counts)
    if hasattr(model, '_ledger_roundoff'):
        fork._ledger_roundoff = dict(model._ledger_roundoff)
    if hasattr(model, '_pair_cache'):
        fork._pair_cache = type(model._pair_cache)()
    fork.event_observer = fork.source_observer = None
    if hasattr(model, 'domain_observer'):
        fork.domain_observer = None
    fork._validate()
    return fork
