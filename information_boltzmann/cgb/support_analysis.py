"""Read-only frozen-rate support diagnostics; not part of the living update.

Eliminating terminal and soma residence pools gives s=q+W s. Sparse W is
dimensionless survival-weighted return flow, not a new plasticity objective.
The direct steady solve may have fill-in; it is deliberately diagnostic only.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix, eye
from scipy.sparse.linalg import spsolve


def reverse_distances(n, pre, post, target, enabled=None):
    """Shortest support-direction distance, without imposing module labels."""
    pre, post = np.asarray(pre), np.asarray(post)
    mask = np.ones(len(pre), bool) if enabled is None else np.asarray(enabled, bool)
    if (pre.ndim != 1 or post.shape != pre.shape or mask.shape != pre.shape
            or not np.issubdtype(pre.dtype, np.integer)
            or not np.issubdtype(post.dtype, np.integer)
            or np.any(pre < 0) or np.any(pre >= n)
            or np.any(post < 0) or np.any(post >= n) or not 0 <= target < n):
        raise ValueError('Valid endpoints, edge mask and boundary required')
    adjacent = [[] for _ in range(n)]
    for a, b in zip(pre[mask], post[mask]):
        adjacent[b].append(int(a))
    distance = np.full(n, -1, dtype=np.int64)
    distance[target] = 0
    todo = [int(target)]
    for node in todo:
        for upstream in adjacent[node]:
            if distance[upstream] < 0:
                distance[upstream] = distance[node] + 1
                todo.append(upstream)
    return distance


def choose_support_boundary(n, pre, post, enabled):
    """Preregistered graph-only assay selection, not a learned action choice."""
    best, selected = None, None
    for target in range(n):
        d = reverse_distances(n, pre, post, target, enabled)
        score = (int((d >= 2).sum()), int((d >= 0).sum()), int(d.max()), -target)
        if best is None or score > best:
            best, selected = score, target
    if selected is None:
        raise ValueError('A nonempty graph is required')
    return selected


def frozen_support_flow(pre, post, carrier, source, *, kappa,
                        free_clearance, terminal_clearance, params,
                        max_cells=4096):
    """Exact linear steady reduction for fixed carriers, phi=0, positive loss.

    Caller supplies actual carrier (positive tags or inherited activity).
    This does not freeze or replace the living model's learning dynamics.
    Positive free clearance guarantees a strictly substochastic return matrix.
    """
    q = np.asarray(source, dtype=float)
    carrier = np.asarray(carrier, dtype=float)
    pre, post = np.asarray(pre), np.asarray(post)
    n = len(q)
    if q.ndim != 1 or not 0 < n <= max_cells:
        raise ValueError('A bounded nonempty diagnostic panel is required')
    reverse_distances(n, pre, post, 0)
    if carrier.shape != pre.shape:
        raise ValueError('One capture carrier per edge required')
    if (not np.isfinite(q).all() or not np.isfinite(carrier).all()
            or np.any(q < 0) or np.any(carrier < 0)
            or not np.isfinite(kappa) or kappa < 0
            or not np.isfinite(free_clearance) or free_clearance <= 0
            or not np.isfinite(terminal_clearance) or terminal_clearance < 0):
        raise ValueError('Nonnegative finite sources/rates and positive free clearance required')
    nu, rho = params.transport_rate, params.release_rate
    exit_t = nu + terminal_clearance
    exit_h = rho + params.cargo_clearance
    if exit_t <= 0 or exit_h <= 0:
        raise ValueError('Both residence pools need a positive exit rate for a steady reduction')
    a = kappa * carrier
    d = free_clearance + np.bincount(post, weights=a, minlength=n)
    survival = nu / exit_t * rho / exit_h
    weights = survival * a / d[post]
    w = coo_matrix((weights, (pre, post)), shape=(n, n)).tocsr()
    column_sums = np.asarray(w.sum(axis=0)).ravel()
    if not np.isfinite(d).all() or column_sums.max() >= 1:
        raise ValueError('No strict lossy contraction at this numerical operating point')
    arrivals = np.asarray(spsolve(eye(n, format='csc') - w.tocsc(), q))
    free = arrivals / d
    bound = a * free[post] / exit_t
    cargo = nu * np.bincount(pre, weights=bound, minlength=n) / exit_h
    loss = (free_clearance * free.sum() + terminal_clearance * bound.sum()
            + params.cargo_clearance * cargo.sum())
    return dict(return_matrix=w, capture_rate=a, incoming_flux=arrivals,
                free=free, bound=bound, cargo=cargo,
                total_clearance=float(loss), source_total=float(q.sum()),
                residence_survival=float(survival),
                max_one_hop_fraction=float(column_sums.max()),
                direct_return=w @ q, two_hop_return=w @ (w @ q),
                flow_residual=float(np.max(np.abs(arrivals - q - w @ arrivals))))
