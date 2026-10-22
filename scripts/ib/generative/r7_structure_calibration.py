"""Infer one existing formation energy and clock from real R7 lifetimes.

Counts do not fit physical parameters. No fitted parameter is installed in
the physical individual. The fixed-material result remains a separate file.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from scipy.stats import binom

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from information_boltzmann.cgb.exchange_checkpoint import load_exchange
from scripts.ib.generative.native_learning_stream import atomic_json
from scripts.ib.generative.r7_lifetime_constraints import observed_intervals
from scripts.ib.generative.r7_structure_constraints import (
    LOG_CLOCK_BOUNDS, structure_generator, fit_clock, structure_score,
    observation_probabilities, count_closure)


ENERGY_BOUNDS = (-10., 30.)


def shift_structure_energy(generator, rho, receipts, delta):
    """Exact dependence of native rates and weights on formation energy.

    Only assembly changes installed material. It adds sign*delta to the
    original free-energy increment; binding/thermal and attempt rates stay
    unchanged. This caches algebra, not a substituted dynamical rule.
    """
    result = np.asarray(generator).copy()
    for row in receipts:
        if row['kind'] == 'assembly':
            attempt = row['rate']/expit(-row['delta_g'])
            result[row['source'], row['target']] = attempt*expit(
                -row['delta_g']-row['sign']*delta)
    np.fill_diagonal(result, 0.)
    result[np.diag_indices(len(result))] = -result.sum(axis=1)
    weights = np.log(rho)-delta*np.array([0., 1., 1., 1., 1.])
    return result, np.exp(weights-logsumexp(weights))


def fit_material(params, support, duration, ended):
    base, rho, receipts = structure_generator(params, support)

    def matrix(energy):
        return shift_structure_energy(base, rho, receipts, energy-params.structure_energy)

    def objective(theta):
        q, _ = matrix(theta[0])
        value = structure_score(q[1:, 1:], np.exp(theta[1]), duration, ended)['mean_nll']
        return value if np.isfinite(value) else 1e12

    fits = []
    bounds = [ENERGY_BOUNDS, tuple(LOG_CLOCK_BOUNDS)]
    for energy in (-4., 2., 8., 14., 24.):
        q, _ = matrix(energy)
        seed = fit_clock(q[1:, 1:], duration, ended)
        initial = np.array([energy, np.log(seed['clock_reference_ms_per_minute'])])
        curve = []

        def tracked(theta):
            value = objective(theta)
            curve.append(value)
            return value

        fit = minimize(tracked, initial, method='L-BFGS-B', bounds=bounds,
                       options=dict(maxiter=500, ftol=1e-13, gtol=1e-7,
                                    maxls=40, eps=1e-5))
        fits.append(dict(initial=initial.tolist(), parameters=fit.x.tolist(),
                         mean_nll=float(fit.fun), success=bool(fit.success),
                         iterations=int(fit.nit), message=str(fit.message), curve=curve))
    chosen = min(fits, key=lambda row: row['mean_nll'])
    # Independent derivative-free continuation checks small finite-difference
    # plateaus without changing the likelihood or taking a new data sample.
    extra = minimize(objective, chosen['parameters'], method='Nelder-Mead', bounds=bounds,
                     options=dict(maxiter=1500, xatol=1e-7, fatol=1e-13))
    theta = np.asarray(extra.x if extra.fun < chosen['mean_nll'] else chosen['parameters'])
    q, fitted_rho = matrix(theta[0])
    # Call actual native constructors again at the fitted energy to verify
    # every cached optimization rate against event_rate and native energies.
    actual_q, actual_rho, _ = structure_generator(replace(params, structure_energy=float(theta[0])), support)
    np.testing.assert_allclose(q, actual_q, rtol=1e-13, atol=1e-16)
    np.testing.assert_allclose(fitted_rho, actual_rho, rtol=1e-13, atol=1e-16)
    return dict(structure_energy=float(theta[0]), clock_reference_ms_per_minute=float(np.exp(theta[1])),
                generator=q.tolist(), stationary_distribution=fitted_rho.tolist(),
                native_fitted_rate_recheck=True,
                parameter_boundary_hits=[i for i, (lo, hi) in enumerate(bounds)
                                         if min(abs(theta[i]-lo), abs(theta[i]-hi)) < 1e-4],
                multistart=fits, final_mean_nll=objective(theta),
                derivative_free_success=bool(extra.success),
                derivative_free_improvement=max(0., chosen['mean_nll']-float(extra.fun)))


def main(args):
    args.out.mkdir(parents=True, exist_ok=False)
    parent, _ = load_exchange(args.archive)
    original = parent.state_dict()
    observations_path = args.audit/'observations.json'
    observations = json.loads(observations_path.read_text())
    control = sorted([w for w in observations['workbooks']
                      if w['kind'] == 'filopodium' and w['genotype'] == 'Control'],
                     key=lambda w: w['file'])
    assert len(control) == 3 and all(w['fit_admitted'] for w in control)
    data = {key: observed_intervals(group)
            for key, group in [('train', control[:2]), ('test', control[2:])]}
    counts = {key: np.concatenate([w['counts'] for w in group])
              for key, group in [('train', control[:2]), ('test', control[2:])]}
    rows = []
    for support in (1, 2, 4, 8, 16):
        fit = fit_material(parent.params, support, *data['train'])
        q = np.asarray(fit['generator'])
        rho = np.asarray(fit['stationary_distribution'])
        clock = fit['clock_reference_ms_per_minute']
        survival, event, entry, visible = observation_probabilities(q[1:, 1:], clock)
        rows.append(dict(total_support=support, fit=fit,
                         first_visible_state_distribution=entry.tolist(),
                         first_frame_visibility_probability=visible,
                         survival_by_frame=survival.tolist(), event_by_frame=event.tolist(),
                         train_score=structure_score(q[1:, 1:], clock, *data['train']),
                         count_closure=count_closure(q, rho, clock, visible, counts['train'])))
        print(json.dumps(dict(completed_support=support, energy=fit['structure_energy'],
                              clock=clock, train_nll=fit['final_mean_nll'])), flush=True)
    source = dict(parent_archive=str(args.archive),
                  parent_sha256=hashlib.sha256(args.archive.read_bytes()).hexdigest(),
                  observations_sha256=hashlib.sha256(observations_path.read_bytes()).hexdigest(),
                  prereg_sha256=hashlib.sha256((ROOT/'docs/information_boltzmann/CGB_R7_STRUCTURE_CALIBRATION_20261022.md').read_bytes()).hexdigest(),
                  lifetime_fit='Control CG1/2 only; counts do not tune formation energy or clock',
                  physical_parameters_installed=False)
    atomic_json(args.out/'training_locked.json', dict(source=source, rows=rows))
    for row in rows:
        q = np.asarray(row['fit']['generator'])
        row['test_score'] = structure_score(q[1:, 1:], row['fit']['clock_reference_ms_per_minute'], *data['test'])
        closure = row['count_closure']
        closure['test_mean_composite_nll'] = -float(binom.logpmf(
            counts['test'], closure['candidate_locations'], closure['installed_probability']).mean())
        closure['test_observed_mean'] = float(counts['test'].mean())
    from scripts.ib.generative.compare_support_transport import same
    same(original, parent.state_dict())
    report = dict(source=source, rows=rows, physical_individual_advanced=False,
                  scope='Exploratory constitutive-parameter inversion in a declared fixed-geometry sector')
    atomic_json(args.out/'report.json', report)
    print(json.dumps([dict(support=r['total_support'], energy=r['fit']['structure_energy'],
                           train_nll=r['train_score']['mean_nll'], test_nll=r['test_score']['mean_nll'],
                           p_on=r['count_closure']['installed_probability'],
                           predicted_mean=r['count_closure']['predicted_mean_count'],
                           locations=r['count_closure']['candidate_locations']) for r in rows], indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args())
