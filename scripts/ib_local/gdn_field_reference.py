"""NumPy CPU references for bounded GDN field directions and local gain rows.

Sources: FLA 0.2.2 fused_recurrent.py:81-97, gated_deltanet.py:260-261,
and pretrained_gdn.streaming_short_convolution. Real-algebra counterparts:
gdn_operator_tags_20261108.lean (normalization, gates, convolution, delta/read)
and gdn_joint_step_20261108.lean::read_direction_gain (fixed-support read).

Gains are analytic, conditional real-arithmetic norm bounds, NOT floating-point
certificates or bounds on finite displacements. No full block, MLP, encoder,
output norm, projections, Top-k support switches, or whole 24-layer replay is
implemented. Directions are one local RHS; no dense Jacobian is constructed.
"""
from __future__ import annotations

import numpy as np


def _sigmoid(x):
    return np.exp(-np.logaddexp(0.0, -np.asarray(x, dtype=float)))


def normalize_jvp(x, dx, *, branch='recurrent', eps=1e-6):
    """Normalize one vector; return value, JVP, and a local operator-norm bound.

    Recurrent uses norm+eps, chunk sqrt(dot+eps), recall max(norm,eps).
    The recurrent vector map has derivative dx/eps at zero even though the
    scalar norm is not differentiable there. Recall equality is not admitted.
    The returned gain is 1/denominator (an upper bound, not generally tight).
    """
    x, dx = np.asarray(x, dtype=float), np.asarray(dx, dtype=float)
    if x.ndim != 1 or dx.shape != x.shape:
        raise ValueError('x and dx must be matching vectors')
    if eps <= 0:
        raise ValueError('eps must be positive')
    norm = float(np.linalg.norm(x))
    if branch == 'recurrent':
        denominator = norm + eps
        ddenominator = float(x @ dx) / norm if norm > 0 else 0.0
    elif branch == 'chunk':
        denominator = float(np.sqrt(x @ x + eps))
        ddenominator = float(x @ dx) / denominator
    elif branch == 'recall':
        if norm == eps:
            raise ValueError('recall norm == eps is a branch boundary')
        denominator = max(norm, eps)
        ddenominator = float(x @ dx) / norm if norm > eps else 0.0
    else:
        raise ValueError(f'Unknown normalization branch {branch!r}')
    value = x / denominator
    tangent = dx / denominator - value * (ddenominator / denominator)
    return value, tangent, 1.0 / denominator


def gates_jvp(a, b, A_log, dt_bias, da=0, db=0, dA=0, ddt=0):
    """Elementwise retention/write gates, with jointly varying inputs/parameters.

    g=-exp(A_log)*softplus(a+dt_bias); alpha=exp(g); beta=sigmoid(b).
    Match Torch's default softplus threshold=20: smooth strictly below, linear
    strictly above. Exact 20 is rejected: the piecewise real map has a jump
    there, however small; no derivative across that boundary is claimed.
    """
    z = np.asarray(a, dtype=float) + np.asarray(dt_bias, dtype=float)
    if np.any(z == 20):
        raise ValueError('softplus input == 20 is a branch boundary')
    softplus = np.where(z > 20, z, np.logaddexp(0.0, z))
    slope = np.where(z > 20, 1.0, _sigmoid(z))
    rate = np.exp(np.asarray(A_log, dtype=float))
    g = -rate * softplus
    dg = -rate * (softplus * np.asarray(dA) + slope *
                  (np.asarray(da) + np.asarray(ddt)))
    alpha, beta = np.exp(g), _sigmoid(b)
    return {'alpha': alpha, 'beta': beta, 'dalpha': alpha * dg,
            'dbeta': beta * (1.0 - beta) * np.asarray(db)}


def delta_read_jvp(S, q, k, v, alpha, beta, dS, dq, dk, dv,
                   dalpha=0, dbeta=0, *, scale=1.):
    """Decay, erase/write, then query read, for one key-by-value state matrix.

    S/dS have shape [key_dim,value_dim], q/k/dq/dk [key_dim], v/dv
    [value_dim]. alpha/beta and their directions are scalars; scale is fixed.
    Gain rows use Frobenius state norms, Euclidean vector norms and absolute
    scalar norms, conditional on these actual forward fields. The state-S
    coefficient uses the exact rank-one erase operator norm, including 1-D.
    """
    S, dS = np.asarray(S, dtype=float), np.asarray(dS, dtype=float)
    q, k, v, dq, dk, dv = (np.asarray(x, dtype=float)
                           for x in (q, k, v, dq, dk, dv))
    if (S.ndim != 2 or dS.shape != S.shape or
            q.shape != (S.shape[0],) or k.shape != q.shape or
            dq.shape != q.shape or dk.shape != q.shape or
            v.shape != (S.shape[1],) or dv.shape != v.shape):
        raise ValueError('Expected matching key-by-value state and vector fields')
    alpha, beta = float(alpha), float(beta)
    A = alpha * S
    r = v - k @ A
    state = A + beta * np.outer(k, r)
    output = scale * (q @ state)
    dA = alpha * dS + dalpha * S
    dr = dv - dk @ A - k @ dA
    dstate = (dA + np.outer(dbeta * k + beta * dk, r) +
              beta * np.outer(k, dr))
    doutput = scale * (dq @ state + q @ dstate)

    nk, nr = float(np.linalg.norm(k)), float(np.linalg.norm(r))
    parallel_gain = abs(1.0 - beta * (k @ k))
    erase_gain = (max(1.0, parallel_gain) if S.shape[0] >= 2
                  else parallel_gain)
    state_row = {
        'S': abs(alpha) * erase_gain,
        'q': 0.,
        'k': abs(beta) * (nr + nk * float(np.linalg.norm(A))),
        'v': abs(beta) * nk,
        'alpha': float(np.linalg.norm(S - beta * np.outer(k, k @ S))),
        'beta': nk * nr,
    }
    read_gain = abs(scale) * float(np.linalg.norm(q))
    output_row = {name: read_gain * gain for name, gain in state_row.items()}
    output_row['q'] = abs(scale) * float(np.linalg.norm(state))
    return {'state': state, 'output': output, 'dstate': dstate,
            'doutput': doutput,
            'gain_rows': {'state': state_row, 'output': output_row}}


def conv_step_jvp(raw, cache, weight, draw, dcache, *, bias=None, dbias=None,
                  activation='silu'):
    """One depthwise convolution step, with frozen weights and live cache JVP.

    cache/dcache/weight are [channels,width], oldest to newest; raw/draw are
    [channels]. The last width-1 raw cache entries precede the new raw value.
    Both returned caches are new arrays; incoming arrays are never mutated.
    Bias directions may be supplied independently (base bias defaults to zero).
    Supported activations match streaming_short_convolution: None/silu/swish.
    """
    raw, cache, weight, draw, dcache = (np.asarray(x, dtype=float)
                                      for x in (raw, cache, weight, draw, dcache))
    if (cache.ndim != 2 or cache.shape[1] < 1 or
            weight.shape != cache.shape or dcache.shape != cache.shape or
            raw.shape != (cache.shape[0],) or draw.shape != raw.shape):
        raise ValueError('Expected [channels,width] buffers and channel vectors')
    next_cache = np.concatenate((cache[:, 1:], raw[:, None]), axis=1)
    next_direction = np.concatenate((dcache[:, 1:], draw[:, None]), axis=1)
    value = np.sum(next_cache * weight, axis=1)
    dvalue = np.sum(next_direction * weight, axis=1)
    if bias is not None:
        value = value + np.asarray(bias)
    if dbias is not None:
        dvalue = dvalue + np.asarray(dbias)
    if activation in ('silu', 'swish'):
        sigmoid = _sigmoid(value)
        dvalue = dvalue * (sigmoid + value * sigmoid * (1.0 - sigmoid))
        value = value * sigmoid
    elif activation is not None:
        raise ValueError(f'Unsupported pretrained activation {activation!r}')
    return {'value': value, 'cache': next_cache,
            'dvalue': dvalue, 'dcache': next_direction}


def softmax_read_jvp(scores, values, dscores, dvalues):
    """Read over caller-supplied fixed support, including any explicit null row.

    scores/dscores are [rows]; values/dvalues are [rows,value_dim]. A null row
    has zero value and value direction, but its score remains active. No null
    is appended or removed, and support is never selected or renormalized over
    history-only rows. Coefficients bound the two distinct derivative legs.
    """
    scores, values, dscores, dvalues = (np.asarray(x, dtype=float)
                                      for x in (scores, values, dscores, dvalues))
    if (scores.ndim != 1 or scores.size == 0 or dscores.shape != scores.shape or
            values.ndim != 2 or values.shape[0] != scores.size or
            dvalues.shape != values.shape):
        raise ValueError('Expected fixed-support scores and matching value rows')
    weights = np.exp(scores - np.max(scores))
    weights = weights / np.sum(weights)
    value = weights @ values
    centered = values - value
    dvalue = weights @ dvalues + (weights * dscores) @ centered
    return {'weights': weights, 'value': value, 'dvalue': dvalue,
            'value_coefficients': weights.copy(),
            'score_coefficients': weights * np.linalg.norm(centered, axis=1)}
