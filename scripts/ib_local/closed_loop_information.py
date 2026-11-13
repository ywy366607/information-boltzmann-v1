"""Low-dimensional Gaussian information diagnostics, not IIT 3/4 Phi.

Phi_G follows Oizumi, Tsuchiya & Amari (2016), equations 17--23.
We cut cross-part lagged coefficients but retain correlated innovations.
Stochastic interaction instead cuts both lagged and instantaneous dependence.
All quantities refer to the declared observable projection, not the full body.
"""
from __future__ import annotations

import itertools
import numpy as np
from scipy.optimize import minimize


def logdet(matrix):
    sign, value = np.linalg.slogdet(matrix)
    if sign <= 0 or not np.isfinite(value):
        raise ValueError('positive definite finite covariance required')
    return value


def partitions(groups):
    """Each unordered bipartition once; the first group is always on the left."""
    for mask in range(2 ** (len(groups) - 1) - 1):
        left = [0] + [i + 1 for i in range(len(groups) - 1) if mask & (1 << i)]
        right = [i for i in range(len(groups)) if i not in left]
        yield left, right


def geometric_cut(A, sx, innovation, blocks):
    """Minimize KL to a Gaussian transition with block-diagonal lag matrix.

    Explicit zero initialization makes mutual information an attainable upper
    bound. Three deterministic starts reduce local-optimizer dependence; every
    result reports convergence/stationarity rather than asserting exactness.
    """
    d = len(A)
    allowed = np.zeros((d, d), bool)
    for idx in blocks:
        allowed[np.ix_(idx, idx)] = True
    base = logdet(innovation)

    def objective(v):
        cut = np.zeros_like(A)
        cut[allowed] = v
        delta = cut - A
        covariance = innovation + delta @ sx @ delta.T
        value = .5 * (logdet(covariance) - base)
        grad = np.linalg.solve(covariance, delta @ sx)
        return value, grad[allowed]

    marginal = np.zeros_like(A)
    for idx in blocks:
        cyx = A @ sx
        marginal[np.ix_(idx, idx)] = np.linalg.solve(
            sx[np.ix_(idx, idx)], cyx[np.ix_(idx, idx)].T).T
    starts = (np.zeros(int(allowed.sum())), A[allowed], marginal[allowed])
    answers = [minimize(objective, x, jac=True, method='L-BFGS-B',
                        options=dict(maxiter=1000, ftol=1e-12, gtol=1e-8)) for x in starts]
    best = min(answers, key=lambda a: a.fun)
    cut = np.zeros_like(A)
    cut[allowed] = best.x
    return dict(phi_g_nats=float(best.fun), converged=bool(best.success),
                max_gradient=float(np.max(np.abs(best.jac))),
                start_spread=float(max(a.fun for a in answers)-best.fun),
                iterations=int(best.nit)), cut


def gaussian_transition(x, y, regularization=1e-4):
    """One self-consistent regularized joint Gaussian, including cross-covariance."""
    d = x.shape[1]
    cov = np.cov(np.column_stack((x, y)), rowvar=False) + regularization*np.eye(2*d)
    sx, sy, cyx = cov[:d, :d], cov[d:, d:], cov[d:, :d]
    A = np.linalg.solve(sx, cyx.T).T
    innovation = sy - A @ sx @ A.T
    return A, sx, innovation, .5*(logdet(sy)-logdet(innovation))


def block_covariance(A, sx, innovation, blocks):
    """Optimal blockwise marginal regressions and independent innovations (SI)."""
    cyx, sy = A @ sx, innovation + A @ sx @ A.T
    cut, residual = np.zeros_like(A), np.zeros_like(A)
    for idx in blocks:
        b = np.linalg.solve(sx[np.ix_(idx, idx)], cyx[np.ix_(idx, idx)].T).T
        cut[np.ix_(idx, idx)] = b
        residual[np.ix_(idx, idx)] = sy[np.ix_(idx, idx)]-b @ sx[np.ix_(idx, idx)] @ b.T
    return cut, residual


def gaussian_nll(y, prediction, covariance):
    error = y-prediction
    return .5*(len(covariance)*np.log(2*np.pi) + logdet(covariance) +
               np.sum(error*np.linalg.solve(covariance, error.T).T, axis=1))


def mean_interval(values, block=8, draws=1000, seed=71):
    """Moving-block bootstrap interval for a fixed already-selected score."""
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    count = int(np.ceil(len(values)/block))
    starts = rng.integers(0, len(values), (draws, count))
    indices = (starts[..., None]+np.arange(block)) % len(values)
    averages = values[indices.reshape(draws, -1)[:, :len(values)]].mean(1)
    return dict(mean=float(values.mean()), ci95=np.quantile(averages, [.025, .975]).tolist(),
                samples=len(values), bootstrap_block=block)


def analyze_trace(trace, groups, lag=1, regularization=1e-4):
    """Training selects a Gaussian; validation selects a cut; final test is untouched.

    The raw minimum Phi_G is a fitted projected-Gaussian quantity, *not* the
    held-out score or an IIT minimum-information partition of the whole body.
    """
    trace = np.asarray(trace, np.float64)
    if trace.ndim != 2 or len(trace) < 96 or not np.isfinite(trace).all():
        raise ValueError('at least 96 finite multivariate observations required')
    pairs = len(trace)-lag
    ntrain, nvalid = pairs//2, pairs//4
    center, scale = trace[:ntrain].mean(0), trace[:ntrain].std(0)
    active = scale > 1e-10
    scale = np.where(active, scale, 1.)
    z = (trace-center)/scale
    x, y = z[:-lag], z[lag:]
    mx, my = x[:ntrain].mean(0), y[:ntrain].mean(0)
    A, sx, noise, mi = gaussian_transition(x[:ntrain]-mx, y[:ntrain]-my, regularization)
    val = slice(ntrain, ntrain+nvalid)
    test = slice(ntrain+nvalid, None)
    full_val = gaussian_nll(y[val], (x[val]-mx) @ A.T+my, noise)
    full_test = gaussian_nll(y[test], (x[test]-mx) @ A.T+my, noise)
    rows, models = [], []
    names = list(groups)
    for left, right in partitions(names):
        blocks = [list(itertools.chain.from_iterable(groups[names[i]] for i in side))
                  for side in (left, right)]
        result, cut = geometric_cut(A, sx, noise, blocks)
        _, independent = block_covariance(A, sx, noise, blocks)
        result['si_gaussian_nats'] = .5*(logdet(independent)-logdet(noise))
        # Preserve the full model's noise law: isolates loss of cross-part lag predictors.
        val_gap = gaussian_nll(y[val], (x[val]-mx) @ cut.T+my, noise)-full_val
        result.update(left=[names[i] for i in left], right=[names[i] for i in right],
                      validation_predictive_cut_penalty=float(val_gap.mean()))
        rows.append(result)
        models.append(cut)
    selected = int(np.argmin([r['validation_predictive_cut_penalty'] for r in rows]))
    cut_test = gaussian_nll(y[test], (x[test]-mx) @ models[selected].T+my, noise)
    gap = cut_test-full_test
    # A no-history mean predictor: full prediction must beat a meaningful baseline.
    marginal = noise+A @ sx @ A.T
    mean_test = gaussian_nll(y[test], np.broadcast_to(my, y[test].shape), marginal)
    bounds_ok = all(-1e-7 <= r['phi_g_nats'] <= mi+1e-6 and
                    r['phi_g_nats'] <= r['si_gaussian_nats']+1e-6 for r in rows)
    return dict(lag=lag, ntrain=ntrain, nvalid=nvalid, ntest=len(gap),
                dimensions=trace.shape[1], inactive_channels=np.flatnonzero(~active).tolist(),
                mutual_information_gaussian_nats=float(mi),
                minimum_projected_phi_g=min(rows, key=lambda r:r['phi_g_nats']),
                validation_selected_partition=rows[selected],
                heldout_predictive_cut_penalty=mean_interval(gap),
                full_vs_no_history_nll_improvement=mean_interval(mean_test-full_test),
                all_converged=all(r['converged'] for r in rows), gaussian_bounds_ok=bounds_ok,
                partitions=rows,
                meaning='Projected Gaussian Phi_G; held-out cut score is separate; not IIT3/4 Phi.')
