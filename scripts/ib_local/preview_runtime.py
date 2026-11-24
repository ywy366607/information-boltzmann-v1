"""Ephemeral read reuse inside ONE functional, no-grad imagination call.

No optimizer, real observation, archive write or world step occurs in this
scope. Entries cannot survive the call, enter a checkpoint, or bypass a
differentiable evaluation. Branch outcomes and U decisions stay independent.
"""
from contextlib import contextmanager
from contextvars import ContextVar

import torch


_scope = ContextVar('causal_preview_reads', default=None)


@contextmanager
def preview_scope(model):
    active = _scope.get()
    if active is not None and active[0] is model:
        # Nested functional branches belong to the same immutable evaluation.
        # Their cache keys retain branch-specific inputs and strong anchors.
        yield
        return
    token = _scope.set((model, {}))
    try:
        yield
    finally:
        _scope.reset(token)


def read(model, namespace, key, compute, *, anchors=()):
    active = _scope.get()
    if torch.is_grad_enabled() or active is None or active[0] is not model:
        return compute()
    cache = active[1]
    identity = namespace, key
    if identity not in cache:
        # Keep key inputs alive: Python IDs / tensor addresses must not be
        # recycled while an entry uses them as its identity.
        cache[identity] = (compute(), anchors)
    return cache[identity][0]


def tensor_key(tensor):
    return (tensor.data_ptr(), tensor._version, tuple(tensor.shape),
            tuple(tensor.stride()), tensor.dtype, tensor.device)


def encoder_read(encoder, namespace, key, compute, *, anchors=()):
    """Reuse fixed encoder-side data for the active owning body's preview.

    A memory reader receives its encoder rather than the enclosing model.
    Resolve that ownership explicitly; derivative calls and unrelated encoders
    execute normally. Parameter/input versions also invalidate in-place edits.
    """
    active = _scope.get()
    if (torch.is_grad_enabled() or active is None or
            getattr(active[0], 'encoder', None) is not encoder):
        return compute()
    signature = tuple(tensor_key(p) for p in encoder.parameters())
    return read(active[0], namespace, (signature, key), compute,
                anchors=(encoder, *anchors))


def read_many(model, namespace, keys, compute, *, anchors=()):
    """The same scoped cache, with one computation for all missing keys."""
    active = _scope.get()
    if torch.is_grad_enabled() or active is None or active[0] is not model:
        return compute(keys)
    cache = active[1]
    missing = list(dict.fromkeys(k for k in keys if (namespace, k) not in cache))
    if missing:
        values = compute(missing)
        if len(values) != len(missing):
            raise ValueError('one result per original cache key required')
        for key, value in zip(missing, values):
            cache[namespace, key] = value, anchors
    return [cache[namespace, k][0] for k in keys]
