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
    return frozen_field_channel(model, [dt], observed_nodes)


def frozen_field_channel(model, durations, observed_nodes=None, observed_world_ports=()):
    """Field-only channel; structure frozen, the same source held throughout.

    This counterfactual excludes intervening reactions and motion. It is not
    the channel of the full adaptive material.step sequence.
    """
    if getattr(model, 'material_scheme', 'sequential') == 'symmetric':
        raise ValueError('Random edge-order OU is a Gaussian mixture; this instrument needs an explicit conditional order')
    durations = np.asarray(durations, float)
    if durations.ndim != 1 or not len(durations) or not np.isfinite(durations).all() or np.any(durations <= 0):
        raise ValueError('Positive finite field durations required')
    nodes = np.asarray(model.port_nodes if observed_nodes is None else observed_nodes, dtype=np.int64)
    world = np.asarray(observed_world_ports, dtype=np.int64)
    if (nodes.ndim != 1 or len(np.unique(nodes)) != len(nodes)
            or np.any(nodes < 0) or np.any(nodes >= model.n)):
        raise ValueError('Distinct valid observed brain nodes required')
    if (world.ndim != 1 or len(np.unique(world)) != len(world) or np.any(world < 0)
            or np.any(world >= len(model.port_nodes)) or not len(nodes)+len(world)):
        raise ValueError('Distinct valid world ports and nonempty observation required')
    n, ports, k = model.n, len(model.port_nodes), len(nodes)+len(world)
    observed = np.r_[nodes, n+world]
    c = np.r_[model.capacitance, model.world_capacitance]
    rows = np.zeros((k, n+ports))
    rows[np.arange(k), observed] = 1.
    covariance = np.zeros((k, k))
    gain = np.zeros((k, ports))

    def pair(i, j, conductance, dt):
        inverse_c = 1/c[i]+1/c[j]
        rate = dt*conductance*inverse_c
        transfer = -np.expm1(-rate)/inverse_c
        variance = -np.expm1(-2*rate)/inverse_c
        difference = rows[:, i]-rows[:, j]
        covariance[:] += variance*np.outer(difference, difference)
        rows[:, i] -= transfer*difference/c[i]
        rows[:, j] += transfer*difference/c[j]

    def source(dt):
        for port in range(ports):
            rate = dt*model.source_conductance[port]/model.world_capacitance[port]
            response = rows[:, n+port].copy()
            gain[:, port] += model.world_capacitance[port]*(-np.expm1(-rate))*response
            covariance[:] += (-model.world_capacitance[port]*np.expm1(-2*rate))*np.outer(response, response)
            rows[:, n+port] *= np.exp(-rate)
    conductance = model.bridge_conductance()
    port_g = model.effective_port_conductance()
    scheme = getattr(model, 'field_scheme', 'sequential_ou')
    for dt in durations[::-1]:
        if scheme == 'symmetric_ou':
            half = dt/2
            source(half)
            for port in range(ports):
                pair(int(model.port_nodes[port]), n+port, port_g[port], half)
            for edge in range(len(model.pre)):
                pair(int(model.pre[edge]), int(model.post[edge]), conductance[edge], half)
            for edge in range(len(model.pre)-1, -1, -1):
                pair(int(model.pre[edge]), int(model.post[edge]), conductance[edge], half)
            for port in range(ports-1, -1, -1):
                pair(int(model.port_nodes[port]), n+port, port_g[port], half)
            source(half)
        elif scheme == 'sequential_ou':
            for edge in range(len(model.pre)-1, -1, -1):
                pair(int(model.pre[edge]), int(model.post[edge]), conductance[edge], dt)
            for port in range(ports-1, -1, -1):
                pair(int(model.port_nodes[port]), n+port, port_g[port], dt)
            source(dt)
        else:
            raise ValueError('Unknown field integration scheme')
    return dict(offset=rows@np.r_[model.charge, model.world_charge],
                gain=gain, covariance=covariance, observed_nodes=nodes, observed_world_ports=world,
                observed_indices=observed, field_durations_ms=durations.copy(),
                rhs_rows=k, workspace_numeric_bytes=rows.nbytes+gain.nbytes+covariance.nbytes,
                workspace_scope='three operator buffers only; excludes capacities, edge conductances, small temporaries and Python overhead',
                scope='field-only channel with frozen structure; post-arrival evidence, not full adaptive sequence')


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


def source_innovation(model, record):
    """Sufficient source increment of the ACTUAL path, before body transfer.

    q_after - a*q_before = C(1-a)*V + independent Gaussian noise.
    All other transitions depend on the byte only through physical state;
    their factors cancel when comparing source hypotheses on the full path.
    This instrument observes the source contact, not just brain end charges.
    """
    dt = float(record['dt_ms'])
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('Positive source exposure required')
    before = np.asarray(record['before_charge'], float)
    after = np.asarray(record['after_charge'], float)
    if before.shape != model.world_charge.shape or after.shape != before.shape or not np.isfinite(before+after).all():
        raise ValueError('Finite complete source-contact charge observation required')
    rate = dt*model.source_conductance/model.world_capacitance
    return dict(observed=after-np.exp(-rate)*before,
                gain=-model.world_capacitance*np.expm1(-rate),
                variance=-model.world_capacitance*np.expm1(-2*rate))


def source_path_posterior(innovations, amplitude=3., prior=None):
    """Full-path instrument: multiply exact source factors over actual steps."""
    if not innovations or not np.isfinite(amplitude) or amplitude <= 0:
        raise ValueError('Nonempty source path and positive finite amplitude required')
    statistic, precision = np.zeros(8), np.zeros(8)
    for row in innovations:
        y, gain, variance = (np.asarray(row[name], float) for name in ('observed', 'gain', 'variance'))
        if any(value.shape != (8,) or not np.isfinite(value).all() for value in (y, gain, variance)) or np.any(variance < 0):
            raise ValueError('Eight finite source-contact channels required')
        active = variance > 0
        if np.any(gain[~active] != 0) or np.any(np.abs(y[~active]) > 1e-12):
            raise ValueError('Inconsistent deterministic source channel')
        statistic[active] += gain[active]*y[active]/variance[active]
        precision[active] += gain[active]**2/variance[active]
    p = np.full(256, 1/256.) if prior is None else np.asarray(prior, float)
    if p.shape != (256,) or not np.isfinite(p).all() or np.any(p <= 0) or not np.isclose(p.sum(), 1.):
        raise ValueError('Positive normalized byte prior required')
    signs = 2*((np.arange(256)[:, None] >> np.arange(8)) & 1)-1
    scores = amplitude*(signs@statistic)+np.log(p)
    return dict(logposterior=scores-logsumexp(scores), snr=amplitude**2*precision)


def uniform_binary_information(snr, order=128):
    """BPSK/AWGN information for the stated uniform bit design, in nats.

    Deterministic quadrature of the actual Gaussian likelihood. This is not
    an information estimate from a small sample of natural text bytes.
    """
    snr = np.asarray(snr, float)
    if not np.isfinite(snr).all() or np.any(snr < 0) or not isinstance(order, int) or order < 8:
        raise ValueError('Nonnegative finite SNR and quadrature order >=8 required')
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    loss = np.logaddexp(0., -2*snr[..., None]-2*np.sqrt(2*snr[..., None])*nodes)
    value = np.log(2.)-loss@weights/np.sqrt(np.pi)
    return np.maximum(value, 0.)  # roundoff at zero SNR only


def exact_field_reference(model, duration, observed_nodes=None, observed_world_ports=()):
    """Small, bounded continuous-time reference, never the production solver.

    Whitened charge x=C^-1/2 q obeys dx=(-M x+B V)dt+sqrt(2M)dW.
    Eigensolving the <=128 physical coordinates gives exact frozen linear
    propagation and covariance. This diagnostic size admission does not cap
    the material, prune its branches, or replace its sparse running dynamics.
    """
    n, ports = model.n, len(model.port_nodes)
    if n+ports > 128:
        raise ValueError('Dense diagnostic only supports <=128 coordinates; material is unchanged')
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError('Positive finite duration required')
    # Reuse the observation inventory checks; these buffers remain bounded.
    inventory = frozen_field_channel(model, [duration], observed_nodes, observed_world_ports)
    observed = inventory['observed_indices']
    capacitance = np.r_[model.capacitance, model.world_capacitance]
    root_c = np.sqrt(capacitance)
    laplacian = np.zeros((n+ports, n+ports))
    source = np.zeros((n+ports, ports))
    for i, j, g in zip(model.pre, model.post, model.bridge_conductance()):
        laplacian[i, i] += g
        laplacian[j, j] += g
        laplacian[i, j] -= g
        laplacian[j, i] -= g
    port_g = model.effective_port_conductance()
    for port, i in enumerate(model.port_nodes):
        j, g = n+port, port_g[port]
        laplacian[i, i] += g
        laplacian[j, j] += g+model.source_conductance[port]
        laplacian[i, j] -= g
        laplacian[j, i] -= g
        source[j, port] = model.source_conductance[port]/root_c[j]
    matrix = laplacian/root_c[:, None]/root_c[None, :]
    eigenvalue, eigenvector = np.linalg.eigh(matrix)
    tolerance = 1e-12*max(1., float(np.linalg.norm(matrix, 2)))
    if eigenvalue.min() < -tolerance:
        raise ValueError('Non-passive field reference')
    eigenvalue = np.maximum(eigenvalue, 0.)  # PSD eigensolver roundoff only
    decay = np.exp(-eigenvalue*duration)
    integrated = np.full_like(eigenvalue, duration)
    np.divide(-np.expm1(-eigenvalue*duration), eigenvalue, out=integrated, where=eigenvalue>0)
    propagator = (eigenvector*decay)@eigenvector.T
    transition = root_c[:, None]*propagator/root_c[None, :]
    gain_all = root_c[:, None]*((eigenvector*integrated)@eigenvector.T@source)
    covariance_all = root_c[:, None]*((eigenvector*(-np.expm1(-2*eigenvalue*duration)))@eigenvector.T)*root_c[None, :]
    q = np.r_[model.charge, model.world_charge]
    return dict(offset=(transition@q)[observed], gain=gain_all[observed],
                covariance=covariance_all[np.ix_(observed, observed)],
                observed_indices=observed, state_transition=transition,
                full_covariance=covariance_all, full_gain=gain_all,
                scope='bounded diagnostic: exact continuous-time frozen linear field; not adaptive material or production execution')
