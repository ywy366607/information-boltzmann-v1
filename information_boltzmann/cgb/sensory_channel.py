"""Exact conditional channel of one existing OU field substep.

This is an instrument, not a decoder installed in the material. It conditions
on ALL pre-field charges and current structure. Sensory evidence after arrival
must never be reported as a prediction before arrival.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


def field_channel(model, dt, observed_nodes=None):
    """Return y | q_before, V_source ~ N(offset + gain V_source, covariance).

    Reverse propagation uses only k observation rows, O(k(N+ports)+k^2)
    workspace. It follows the actual ordered OU updates; no dense global
    covariance, historical Jacobian, target byte, or RNG sample is retained.
    The field is held as it actually is at entry to model._field.
    """
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('Positive finite field duration required')
    nodes = np.asarray(model.port_nodes if observed_nodes is None else observed_nodes, dtype=np.int64)
    if (nodes.ndim != 1 or not len(nodes) or len(np.unique(nodes)) != len(nodes)
            or np.any(nodes < 0) or np.any(nodes >= model.n)):
        raise ValueError('Distinct valid observed brain nodes required')
    n, ports, k = model.n, len(model.port_nodes), len(nodes)
    c = np.r_[model.capacitance, model.world_capacitance]
    rows = np.zeros((k, n+ports))
    rows[np.arange(k), nodes] = 1.
    covariance = np.zeros((k, k))
    gain = np.zeros((k, ports))

    def pair(i, j, conductance):
        inverse_c = 1/c[i]+1/c[j]
        rate = dt*conductance*inverse_c
        transfer = -np.expm1(-rate)/inverse_c
        variance = -np.expm1(-2*rate)/inverse_c
        difference = rows[:, i]-rows[:, j]
        covariance[:] += variance*np.outer(difference, difference)
        rows[:, i] -= transfer*difference/c[i]
        rows[:, j] += transfer*difference/c[j]

    conductance = model.bridge_conductance()
    for edge in range(len(model.pre)-1, -1, -1):
        pair(int(model.pre[edge]), int(model.post[edge]), conductance[edge])
    for port in range(ports-1, -1, -1):
        pair(int(model.port_nodes[port]), n+port, model.port_conductance[port])
    for port in range(ports):
        rate = dt*model.source_conductance[port]/model.world_capacitance[port]
        response = rows[:, n+port].copy()
        gain[:, port] = model.world_capacitance[port]*(-np.expm1(-rate))*response
        covariance += (-model.world_capacitance[port]*np.expm1(-2*rate))*np.outer(response, response)
        rows[:, n+port] *= np.exp(-rate)
    return dict(offset=rows@np.r_[model.charge, model.world_charge],
                gain=gain, covariance=covariance, observed_nodes=nodes,
                rhs_rows=k, workspace_numeric_bytes=rows.nbytes+gain.nbytes+covariance.nbytes,
                workspace_scope='three operator buffers only; excludes capacities, edge conductances, small temporaries and Python overhead',
                scope='one field substep conditional on complete pre-field state; post-arrival evidence')


def byte_posterior(channel, observed_charge, amplitude=3., prior=None):
    """Instrument posterior over source bytes, after observing this substep."""
    if channel['gain'].shape[1] != 8 or not np.isfinite(amplitude) or amplitude <= 0:
        raise ValueError('Eight source ports and positive finite amplitude required')
    y = np.asarray(observed_charge, dtype=float)
    if y.shape != channel['offset'].shape or not np.isfinite(y).all():
        raise ValueError('One finite measured charge per observed node required')
    p = np.full(256, 1/256.) if prior is None else np.asarray(prior, float)
    if p.shape != (256,) or not np.isfinite(p).all() or np.any(p <= 0) or not np.isclose(p.sum(), 1.):
        raise ValueError('Positive normalized byte prior required')
    bits = ((np.arange(256)[:, None] >> np.arange(8)) & 1)
    means = channel['offset']+(amplitude*(2*bits-1))@channel['gain'].T
    residual = y-means
    # Cholesky failure explicitly exposes unobservable/degenerate channels;
    # no arbitrary noise floor is added to make an instrument look informative.
    chol = np.linalg.cholesky(channel['covariance'])
    white = np.linalg.solve(chol, residual.T).T
    score = -.5*np.sum(white*white, axis=1)+np.log(p)
    return score-logsumexp(score)


def uniform_information_bound(channel, amplitude=3.):
    """Gaussian-input upper bound for uniform independent source bits only.

    Not an estimate for natural text or a bound on all later perception.
    """
    if not np.isfinite(amplitude) or amplitude <= 0:
        raise ValueError('Positive finite amplitude required')
    chol = np.linalg.cholesky(channel['covariance'])
    white_gain = np.linalg.solve(chol, amplitude*channel['gain'])
    _, logdet = np.linalg.slogdet(np.eye(len(chol))+white_gain@white_gain.T)
    return min(channel['gain'].shape[1]*np.log(2.), .5*float(logdet))
