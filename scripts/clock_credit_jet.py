"""Ordered time/credit jets for explicitly supplied kernels and event traces.

clock=-d_eta, credit=d_delta, mixed=-d_eta d_delta at zero counting fields.
The arrays are supplied operators, never a graph inferred from phase names.
"""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Jet:
    mass: torch.Tensor
    clock: torch.Tensor
    credit: torch.Tensor
    mixed: torch.Tensor


def mark(mass, duration, credit=None, duration_credit=None):
    credit = torch.zeros_like(mass) if credit is None else credit
    duration_credit = torch.zeros_like(mass) if duration_credit is None else duration_credit
    return Jet(mass, mass*duration, credit, credit*duration+mass*duration_credit)


def compose(a, b):
    return Jet(a.mass@b.mass,
        a.clock@b.mass+a.mass@b.clock,
        a.credit@b.mass+a.mass@b.credit,
        a.mixed@b.mass+a.credit@b.clock+a.clock@b.credit+a.mass@b.mixed)


def add(a, b):
    return Jet(*(x+y for x,y in zip(vars(a).values(),vars(b).values())))


def first_arrival_prefix(continuing, arriving):
    """Sum Q0...Q(k-1) Rk and keep the full Q0...Q(N-1) tail jet."""
    if not continuing or len(continuing) != len(arriving):
        raise ValueError('equally long nonempty operator sequences required')
    n = continuing[0].mass.shape[0]
    shape = arriving[0].mass.shape
    identity = torch.eye(n, dtype=continuing[0].mass.dtype, device=continuing[0].mass.device)
    zero = torch.zeros_like(identity)
    tail = Jet(identity, zero, zero, zero)
    zero_hit = arriving[0].mass.new_zeros(shape)
    hit = Jet(zero_hit, zero_hit, zero_hit, zero_hit)
    for q,r in zip(continuing, arriving):
        hit = add(hit, compose(tail, r))
        tail = compose(tail, q)
    return hit, tail
