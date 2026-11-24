"""Bounded S/P event-boundary adapter for the existing packed Delta state.

This correctness-first adapter replays one already-observed layer input to
recover the *actual* innovation, including any content-budget correction.
It is deliberately not a replacement for the fused training recurrence.
No token history, optimizer, table, or full-body snapshot is retained.
"""
from __future__ import annotations

import os
import torch
from torch.nn import functional as F

from scripts.ib_local.bfss_gdn_reference import quiet_step
from scripts.ib_local.gated_deltanet import admit_association_content


def matrix_view(layer, packed):
    """Return the S portion; packed convolution history is not reinterpreted."""
    if layer.recurrence != 'decay_then_delta' or layer.share_size:
        raise ValueError('only the existing decay_then_delta state is supported')
    if layer.conv_kernel:
        return packed[..., :layer.matrix_size].reshape(
            packed.shape[0], layer.heads, layer.dk, layer.dv)
    return packed


@torch.no_grad()
def observed_momentum(layer, x, packed_before, momentum, *, contact=None, layer_index=None, impedance=None):
    """Replay actual inputs once, returning (S_after, P_after, corrections).

    P is kicked by the pre-admission delta innovation, as specified by the
    candidate.  S is admitted after each write.  Inputs are already the actual
    layer inputs, including any upstream experience or fast-table injection.
    """
    if x.ndim != 3 or x.shape[1] == 0:
        raise ValueError('a nonempty observed layer sequence is required')
    s = matrix_view(layer, packed_before).clone()
    if s.shape != momentum.shape:
        raise ValueError('momentum must match the corresponding matrix state')
    p = momentum.clone()
    batch, length, width = x.shape
    normalized = layer.norm(x)
    q = layer.q_proj(normalized).reshape(batch, length, layer.heads, layer.dk)
    k = layer.k_proj(normalized).reshape(batch, length, layer.heads, layer.dk)
    v = layer.v_proj(normalized).reshape(batch, length, layer.heads, layer.dv)
    if layer.conv_kernel:
        history = packed_before[:, layer.matrix_size:].reshape(
            batch, 3*width, layer.conv_kernel-1)
        projected = torch.cat((q.reshape(batch, length, width),
                               k.reshape(batch, length, width),
                               v.reshape(batch, length, width)), -1).transpose(1, 2)
        mixed, _ = layer._mix_qkv(projected, history)
        _, k, v = [part.reshape(batch, length, layer.heads, layer.dk)
                   for part in mixed.transpose(1, 2).split(width, dim=-1)]
    k = F.normalize(k, dim=-1)
    alpha = layer.retention(normalized)
    beta = layer.beta_proj(normalized).sigmoid()
    if contact is not None:
        return contact.observe(layer_index, k, v, alpha, beta, s, p,
                               impedance, layer.content_budget)
    if (x.is_cuda and s.dtype == torch.float32 and layer.dk == layer.dv and
            os.environ.get('CGB_MOMENTUM_BACKEND', 'fused') != 'reference'):
        from scripts.ib_local.bfss_momentum_fused import capture
        return capture(k, v, alpha, beta, s, p, layer.content_budget)
    correction_sq = s.new_zeros(())
    for t in range(length):
        a = alpha[:, t, :, None, None]
        b = beta[:, t, :, None, None]
        key = k[:, t, :, :, None]
        prior = a*s
        residual = v[:, t, :, :, None] - prior.transpose(-1, -2) @ key
        innovation = b * (key @ residual.transpose(-1, -2))
        proposed = prior + innovation
        p = a*p + innovation
        s = admit_association_content(proposed, layer.content_budget) if layer.content_budget else proposed
        correction_sq += (s-proposed).square().sum()
    return s, p, correction_sq.sqrt()


@torch.no_grad()
def quiet_packed(base, recurrent, momenta, *, dt, frequency, coupling, steps):
    """Evolve only S and P; leave convolution and all other state untouched."""
    if type(steps) is not int or steps < 0 or len(momenta) != len(base.layers):
        raise ValueError('invalid fixed internal budget or number of layers')
    adaptive = getattr(base, '_adaptive_quiet', None)
    if adaptive is not None:
        return adaptive.quiet(base, recurrent, momenta, dt=dt, steps=steps, capture=True)
    first=base.layers[0]
    if first.conv_kernel and all(layer.recurrence=='decay_then_delta' and not layer.share_size
            and (layer.matrix_size,layer.heads,layer.dk,layer.dv)==
                (first.matrix_size,first.heads,first.dk,first.dv) for layer in base.layers):
        # Layer is a batch axis here: forces couple heads within a layer only.
        # All 32 layers evolve in shared batched GEMMs, not 32 launch sequences.
        result=recurrent.clone()
        s=result[...,:first.matrix_size].reshape(len(base.layers),recurrent.shape[1],
                                               first.heads,first.dk,first.dv)
        p=torch.stack(momenta)
        for _ in range(steps):
            s,p=quiet_step(s,p,dt=dt,frequency=frequency,coupling=coupling)
        result[...,:first.matrix_size]=s.flatten(2)
        return result,list(p.unbind(0))
    result, next_p = recurrent.clone(), []
    for i, (layer, momentum) in enumerate(zip(base.layers, momenta)):
        s, p = matrix_view(layer, result[i]).clone(), momentum.clone()
        for _ in range(steps):
            s, p = quiet_step(s, p, dt=dt, frequency=frequency, coupling=coupling)
        if layer.conv_kernel:
            result[i, :, :layer.matrix_size] = s.flatten(1)
        else:
            result[i] = s
        next_p.append(p)
    return result, next_p
