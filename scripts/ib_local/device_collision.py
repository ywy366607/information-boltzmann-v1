"""Strict local collisions without per-candidate device/host synchronization.

Candidate sampling and dependency scheduling run on CPU ahead of evolution.
All candidates are scheduled, including geometrically inactive ones. This only
adds dependency edges: it cannot reorder two events sharing a particle.
"""
import math

import torch
from torch.nn import functional as F

from .schedule import schedule_dependencies


def prepare_layers(table, particles, device):
    """Transfer a complete, uncapped candidate table in dependency order."""
    _, permutation, offsets = schedule_dependencies(table.i, table.j, particles)
    layers = []
    for start, end in zip(offsets.tolist()[:-1], offsets.tolist()[1:]):
        order = permutation[start:end]
        layers.append(tuple(a[order].to(device) for a in
                            (table.i, table.j, table.normal, table.uniform)))
    return layers


def collision_layer(x, v, context_x, key, value, kernel, layer, width):
    """One disjoint layer; no mutable background, detached geometry or epsilon leak."""
    i, j, normal, uniform = layer[:4]
    vi, vj = v[i], v[j]
    center = (x[i] + x[j]) * .5
    change = ((vi - vj) * normal).sum(-1, keepdim=True) * normal
    vp, wp = vi - change, vj + change
    pairs = torch.stack((torch.cat((vi, vj), -1), torch.cat((vj, vi), -1),
                         torch.cat((vp, wp), -1), torch.cat((wp, vp), -1)), 1)
    orbit = torch.cat((pairs[:, :, None, :].expand(-1, -1, 2, -1),
                       torch.stack((normal, -normal), 1)[:, None].expand(-1, 4, -1, -1)), -1)
    q = kernel.query(orbit.flatten(1, 2))
    factors = 1 - (context_x[None] - center[:, None]).abs() / width
    supported = (factors > 0).all(-1)
    # Never evaluate log(0), including branches later excluded by a mask.
    log_weight = torch.where(factors > 0, factors, torch.ones_like(factors)).log().sum(-1)
    mask = log_weight.masked_fill(~supported, float('-inf'))
    has_context = supported.any(-1)
    # Empty rows use a finite softmax then explicitly zero their attention.
    mask = torch.where(has_context[:, None], mask, torch.zeros_like(mask))
    attended = F.scaled_dot_product_attention(
        q[:, None], key[None, None].expand(q.shape[0], 1, -1, -1),
        value[None, None].expand(q.shape[0], 1, -1, -1),
        attn_mask=mask[:, None, None, :], dropout_p=0.)[:, 0]
    # Key's center translation cancels in softmax; value's does not.
    attended = attended - F.linear(center, kernel.value.weight[:, :x.shape[-1]])[:, None]
    attended = torch.where(has_context[:, None, None], attended, torch.zeros_like(attended))
    raw = kernel.output(torch.tanh(q + attended)).mean((1, 2)).clamp(-12., 12.)
    pair_factors = 1 - (x[i] - x[j]).abs() / width
    active = (pair_factors > 0).all(-1)
    if len(layer) == 5:
        active = active & layer[4]
    log_geom = torch.where(pair_factors > 0, pair_factors, torch.ones_like(pair_factors)).log().sum(-1)
    lp = log_geom + F.logsigmoid(raw)
    # Inactive events use an interior dummy probability and contribute zero.
    safe_lp = torch.where(active, lp, torch.full_like(lp, -1.))
    choose = active & (uniform.log() < safe_lp)
    low = safe_lp < -math.log(2.)
    a = torch.where(low, safe_lp, torch.full_like(safe_lp, -1.))
    b = torch.where(low, torch.full_like(safe_lp, -.5), safe_lp)
    reject = torch.where(low, torch.log1p(-a.exp()), torch.log(-torch.expm1(b)))
    event_lp = torch.where(active, torch.where(choose, safe_lp, reject), torch.zeros_like(lp))
    delta = torch.where(choose[:, None], change, torch.zeros_like(change))
    nv = v.index_add(0, i, -delta).index_add(0, j, delta)
    return nv, event_lp.sum(), choose.sum()


def collision_device(x, v, kernel, layers, width=1., layer_fn=collision_layer):
    context = torch.cat((x, v), -1)
    key, value = kernel.key(context), kernel.value(context)
    nv, log_prob, accepted = v, v.sum() * 0., v.new_zeros(())
    for layer in layers:
        nv, lp, count = layer_fn(x, nv, x, key, value, kernel, layer, width)
        log_prob = log_prob + lp
        accepted = accepted + count
    return nv, log_prob, accepted
