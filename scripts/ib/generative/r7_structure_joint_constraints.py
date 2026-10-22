"""Use distinct real observations to constrain the same native generator.

Stationary count marginals identify formation free energy; incident-run
timing identifies a common clock. Scores are not summed as independent data.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import logit
from scipy.stats import binom

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from information_boltzmann.cgb.exchange_checkpoint import load_exchange
from scripts.ib.generative.native_learning_stream import atomic_json
from scripts.ib.generative.r7_lifetime_constraints import observed_intervals
from scripts.ib.generative.r7_structure_constraints import (
    structure_generator, fit_clock, structure_score, observation_probabilities, count_closure)


def fit_locations(counts):
    counts = np.asarray(counts)
    mean = float(counts.mean())
    grid = list(range(int(counts.max()), 129))
    values = [-float(binom.logpmf(counts, k, mean/k).mean()) for k in grid]
    chosen = int(np.argmin(values))
    k = grid[chosen]
    return dict(locations=k, installed_probability=mean/k, train_mean=mean,
                location_grid=grid, train_composite_nll_grid=values,
                at_boundary=chosen in (0, len(grid)-1),
                score_scope='Correlated-frame marginal composite likelihood, not a trajectory likelihood')


def formation_energy_for_occupancy(params, support, installed_probability):
    _, rho, _ = structure_generator(params, support)
    p0 = float(1-rho[0])
    return float(params.structure_energy+logit(p0)-logit(installed_probability))


def empirical_lag_correlation(counts, lag):
    x = np.asarray(counts, float)
    centered = x-x.mean()
    variance = float(np.mean(centered*centered))
    return float(np.mean(centered[:-lag]*centered[lag:])/variance) if variance else None


def main(args):
    args.out.mkdir(parents=True, exist_ok=False)
    parent, _ = load_exchange(args.archive)
    original = parent.state_dict()
    path = args.audit/'observations.json'
    observations = json.loads(path.read_text())
    control = sorted([w for w in observations['workbooks']
                      if w['kind'] == 'filopodium' and w['genotype'] == 'Control'],
                     key=lambda w: w['file'])
    assert len(control) == 3 and all(w['fit_admitted'] for w in control)
    data = {key: observed_intervals(group)
            for key, group in [('train', control[:2]), ('test', control[2:])]}
    counts = {key: np.concatenate([w['counts'] for w in group])
              for key, group in [('train', control[:2]), ('test', control[2:])]}
    location_fit = fit_locations(counts['train'])
    rows = []
    for support in (1, 2, 4, 8, 16):
        energy = formation_energy_for_occupancy(parent.params, support, location_fit['installed_probability'])
        assert -10 <= energy <= 30
        q, rho, receipts = structure_generator(replace(parent.params, structure_energy=energy), support)
        np.testing.assert_allclose(1-rho[0], location_fit['installed_probability'], rtol=1e-11)
        fit = fit_clock(q[1:, 1:], *data['train'])
        clock = fit['clock_reference_ms_per_minute']
        survival, events, entry, visible = observation_probabilities(q[1:, 1:], clock)
        closure = count_closure(q, rho, clock, visible, counts['train'])
        assert closure['candidate_locations'] == location_fit['locations']
        closure['training_mean_is_fitted_not_validation'] = True
        rows.append(dict(total_support=support, structure_energy=energy, fit=fit,
                         generator=q.tolist(), native_events=receipts,
                         stationary_distribution=rho.tolist(),
                         first_visible_state_distribution=entry.tolist(),
                         first_frame_visibility_probability=visible,
                         survival_by_frame=survival.tolist(), event_by_frame=events.tolist(),
                         train_score=structure_score(q[1:, 1:], clock, *data['train']),
                         count_closure=closure))
    source = dict(parent_archive=str(args.archive),
                  parent_sha256=hashlib.sha256(args.archive.read_bytes()).hexdigest(),
                  observations_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  prereg_sha256=hashlib.sha256((ROOT/'docs/information_boltzmann/CGB_R7_JOINT_OBSERVABLES_20261022.md').read_bytes()).hexdigest(),
                  physical_individual_advanced=False, fitted_parameters_installed=False)
    atomic_json(args.out/'training_locked.json', dict(source=source, location_fit=location_fit, rows=rows))
    for row in rows:
        q = np.asarray(row['generator'])
        row['test_score'] = structure_score(q[1:, 1:], row['fit']['clock_reference_ms_per_minute'], *data['test'])
        closure = row['count_closure']
        closure['test_mean_composite_nll'] = -float(binom.logpmf(
            counts['test'], closure['candidate_locations'], closure['installed_probability']).mean())
        closure['test_observed_mean'] = float(counts['test'].mean())
    report = dict(source=source, location_fit=location_fit, rows=rows,
                  score_scope='Separate constraints from shared observations, not independent joint likelihood',
                  actual_cones=[dict(file=w['file'], mean_count=w['mean_count'], fano=w['fano'],
                                     counts=w['counts'], incident_records=w['incident_records'],
                                     lag_correlation={str(lag): empirical_lag_correlation(w['counts'], lag)
                                                      for lag in (1, 2, 4, 8, 16)}) for w in control])
    from scripts.ib.generative.compare_support_transport import same
    same(original, parent.state_dict())
    atomic_json(args.out/'report.json', report)
    print(json.dumps([dict(support=r['total_support'], energy=r['structure_energy'],
                           clock=r['fit']['clock_reference_ms_per_minute'],
                           train_nll=r['train_score']['mean_nll'], test_nll=r['test_score']['mean_nll'],
                           p_on=r['count_closure']['installed_probability'],
                           mean=r['count_closure']['predicted_mean_count'],
                           arrivals=r['count_closure']['predicted_visible_arrivals_per_minute'])
                      for r in rows], indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args())
