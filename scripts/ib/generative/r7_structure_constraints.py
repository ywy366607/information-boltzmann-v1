"""Real R7 observations of paid structure, using a conditional native generator.

The fixed-geometry finite sector is an observation hypothesis, not a new
physical update or a complete filopodium model. No third-party code is run.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from scipy.stats import binom

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from information_boltzmann.cgb.exchange_checkpoint import load_exchange
from information_boltzmann.cgb.mobile_exchange import MobileExchange
from scripts.ib.generative.native_learning_stream import atomic_json
from scripts.ib.generative.r7_lifetime_constraints import observed_intervals


# Tuples are actual installed, trace, bound configurations. State 0 is absent.
STATES = ((0, 0, 0), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1))
LOG_CLOCK_BOUNDS = np.log([1e-4, 1e8])


def physical_state(params, support: int, configuration: tuple) -> MobileExchange:
    """One bridge's finite B/P/D and support inventories, including off state."""
    if params.polarized_storage:
        raise ValueError('Polarized storage requires retaining charge states')
    if support < 1 or int(support) != support or configuration not in STATES:
        raise ValueError('Positive integer support and declared state required')
    installed, trace, bound = configuration
    return MobileExchange(
        [[1., 1., 1.], [1. + params.rest_length, 1., 1.]],
        material=[0, 1-installed], free=[0, support-bound],
        charged=[0, 1-installed], depleted=[0, installed],
        pre=[0] if installed else [], post=[1] if installed else [],
        trace=trace, bound=bound, end_capacity=3, params=params)


def structure_generator(params, support: int):
    """Project admitted native events; assemble nothing from fitted rates."""
    generator = np.zeros((len(STATES), len(STATES)))
    states = [physical_state(params, support, state) for state in STATES]
    receipts = []
    for i, model in enumerate(states):
        for event in model.events():
            if event.kind not in ('assembly', 'binding', 'thermal'):
                continue
            installed, trace, bound = STATES[i]
            if event.kind == 'assembly':
                target = (installed + event.sign, 0, 0)
            elif event.kind == 'binding':
                target = (installed, trace, bound + event.sign)
            else:
                target = (installed, trace + event.sign, bound)
            j = STATES.index(target)
            rate = model.event_rate(event)
            du, dg = model.event_delta(event)
            np.testing.assert_allclose(du, states[j].energy()-model.energy(), atol=1e-13)
            np.testing.assert_allclose(dg, states[j].free_energy()-model.free_energy(), atol=1e-13)
            generator[i, j] += rate
            receipts.append(dict(source=i, target=j, kind=event.kind,
                                 sign=event.sign, rate=rate, delta_u=du, delta_g=dg))
    generator[np.diag_indices(len(STATES))] = -generator.sum(axis=1)
    energies = np.array([model.free_energy() for model in states])
    rho = np.exp(-energies-logsumexp(-energies))
    np.testing.assert_allclose(rho @ generator, 0., atol=1e-15)
    np.testing.assert_allclose(rho[:, None]*generator,
                               (rho[:, None]*generator).T, atol=1e-15)
    return generator, rho, receipts


def sampled_entry(transient, clock: float):
    """Integrate survival of a birth before its first one-minute observation.

    Births are stationary within the sampling interval, not fixed at its
    beginning. T is killed on actual disassembly, including thermal reentry.
    """
    if not np.isfinite(clock) or clock <= 0:
        raise ValueError('A positive finite clock conversion is required')
    scaled = np.asarray(transient)*clock
    # The protected structure can live much longer than internal exchanges.
    # T^-1(exp(T)-I) loses accuracy in that regime, especially for short frames.
    # A block exponential computes the same integral without cancellation.
    size = len(scaled)
    augmented = np.zeros((2*size, 2*size))
    augmented[:size, :size] = scaled
    augmented[:size, size:] = np.eye(size)
    propagator = expm(augmented)
    step = propagator[:size, :size]
    integral = propagator[:size, size:]
    mass = integral[0]  # a newly assembled structure starts idle
    visible = float(mass.sum())
    if visible <= 0 or np.any(mass < -1e-12) or visible > 1+1e-10:
        raise FloatingPointError('Invalid first-visible birth probabilities')
    return mass/visible, visible, step


def observation_probabilities(transient, clock: float, frames=60):
    entry, visible, step = sampled_entry(transient, clock)
    row = entry.copy()
    # Compute a rare actual removal as an absorbing transition, rather than
    # subtracting nearly equal row sums from one.
    killed = np.zeros((len(step)+1, len(step)+1))
    killed[:-1, :-1] = transient
    killed[:-1, -1] = -np.asarray(transient).sum(axis=1)
    exit_probability = expm(killed*clock)[:-1, -1]
    survival, event = [], []
    for _ in range(frames):
        survival.append(float(row.sum()))
        event.append(float(row @ exit_probability))
        row = row @ step
    return np.array(survival), np.array(event), entry, visible


def structure_score(transient, clock: float, duration, ended):
    survival, event, _, _ = observation_probabilities(transient, clock)
    probabilities = np.where(ended, event[duration.astype(int)],
                              survival[duration.astype(int)])
    if np.any(probabilities <= 0) or not np.isfinite(probabilities).all():
        return dict(mean_nll=float('inf'))
    values = -np.log(probabilities)
    return dict(incident_tracks=len(duration), observed_endings=int(ended.sum()),
                right_censored=int(len(ended)-ended.sum()), mean_nll=float(values.mean()),
                total_nll=float(values.sum()), per_track_nll=values.tolist())


def fit_clock(transient, duration, ended):
    def objective(log_clock):
        return structure_score(transient, np.exp(log_clock), duration, ended)['mean_nll']
    grid = np.linspace(*LOG_CLOCK_BOUNDS, 241)
    values = [objective(value) for value in grid]
    fits = []
    for i in range(1, len(grid)-1):
        if np.isfinite(values[i]) and values[i] <= min(values[i-1], values[i+1]):
            result = minimize_scalar(objective, method='bounded',
                                     bounds=(grid[i-1], grid[i+1]),
                                     options=dict(xatol=1e-10, maxiter=500))
            fits.append(dict(log_clock=float(result.x), objective=float(result.fun),
                             success=bool(result.success), evaluations=int(result.nfev)))
    candidates = fits + [dict(log_clock=float(grid[i]), objective=values[i],
                              success=False, endpoint=True) for i in (0, -1)]
    best = min(candidates, key=lambda row: row['objective'])
    return dict(clock_reference_ms_per_minute=float(np.exp(best['log_clock'])),
                train_nll=best['objective'], optimizer_success=best['success'],
                clock_on_boundary=bool(best.get('endpoint', False)),
                grid_log_clock=grid.tolist(),
                grid_objective=[v if np.isfinite(v) else None for v in values],
                nonfinite_grid_objectives=sum(not np.isfinite(v) for v in values),
                local_fits=fits)


def count_closure(generator, rho, clock, visible_probability, train_counts):
    """A declared independent, stationary location limit; no shared competition."""
    installed = np.array([0., 1., 1., 1., 1.])
    p_on = float(rho @ installed)
    lower = int(np.max(train_counts))
    grid = list(range(lower, 129))
    scores = [-float(binom.logpmf(train_counts, k, p_on).mean()) for k in grid]
    best = int(np.argmin(scores))
    k = grid[best]
    covariance = {}
    for lag in (1, 2, 4, 8, 16, 32):
        both = float((rho*installed) @ expm(generator*clock*lag) @ installed)
        covariance[str(lag)] = (both-p_on*p_on)/(p_on*(1-p_on))
    return dict(closure='Independent identical fixed locations, stationary marginal counts',
                installed_probability=p_on, candidate_locations=k,
                location_grid=grid, train_composite_nll_grid=scores,
                locations_at_grid_boundary=best in (0, len(grid)-1),
                predicted_mean_count=k*p_on, predicted_stationary_fano=1-p_on,
                predicted_visible_arrivals_per_minute=k*clock*rho[0]*generator[0, 1]*visible_probability,
                predicted_count_autocorrelation=covariance,
                train_mean_composite_nll=scores[best],
                train_observed_mean=float(np.mean(train_counts)),
                score_scope='Marginal composite score; frames correlated, not a trajectory likelihood')


def main(args):
    args.out.mkdir(parents=True, exist_ok=False)
    parent_hash = hashlib.sha256(args.archive.read_bytes()).hexdigest()
    parent, _ = load_exchange(args.archive)
    original = parent.state_dict()
    observations_path = args.audit/'observations.json'
    observations = json.loads(observations_path.read_text())
    control = sorted([w for w in observations['workbooks']
                      if w['kind'] == 'filopodium' and w['genotype'] == 'Control'],
                     key=lambda row: row['file'])
    assert len(control) == 3 and all(w['fit_admitted'] for w in control)
    data = {key: observed_intervals(group)
            for key, group in [('train', control[:2]), ('test', control[2:])]}
    counts = {key: np.concatenate([w['counts'] for w in group])
              for key, group in [('train', control[:2]), ('test', control[2:])]}
    rows = []
    for support in [1, 2, 4, 8, 16]:
        generator, rho, receipts = structure_generator(parent.params, support)
        transient = generator[1:, 1:]
        fit = fit_clock(transient, *data['train'])
        clock = fit['clock_reference_ms_per_minute']
        survival, event, entry, visible = observation_probabilities(transient, clock)
        closure = count_closure(generator, rho, clock, visible, counts['train'])
        rows.append(dict(total_support=support, generator=generator.tolist(),
                         native_events=receipts, stationary_distribution=rho.tolist(),
                         fit=fit, first_visible_state_distribution=entry.tolist(),
                         first_frame_visibility_probability=visible,
                         train_score=structure_score(transient, clock, *data['train']),
                         survival_by_frame=survival.tolist(), event_by_frame=event.tolist(),
                         count_closure=closure))
    lock = dict(rows=rows, selection='All support conditions retained; no test selection',
                source_archive=str(args.archive), parent_sha256=parent_hash,
                observations_sha256=hashlib.sha256(observations_path.read_bytes()).hexdigest(),
                prereg_sha256=hashlib.sha256((ROOT/'docs/information_boltzmann/CGB_R7_STRUCTURE_PREREG_20261022.md').read_bytes()).hexdigest(),
                native_source_sha256=hashlib.sha256(Path(inspect.getfile(MobileExchange)).read_bytes()).hexdigest())
    atomic_json(args.out/'training_locked.json', lock)
    # Every scoring and location choice above used CG1/CG2 only.
    for row in rows:
        transient = np.asarray(row['generator'])[1:, 1:]
        row['test_score'] = structure_score(transient, row['fit']['clock_reference_ms_per_minute'], *data['test'])
        closure = row['count_closure']
        closure['test_mean_composite_nll'] = -float(binom.logpmf(
            counts['test'], closure['candidate_locations'], closure['installed_probability']).mean())
        closure['test_observed_mean'] = float(counts['test'].mean())
    from scripts.ib.generative.compare_support_transport import same
    same(original, parent.state_dict())
    assert parent_hash == hashlib.sha256(args.archive.read_bytes()).hexdigest()
    report = dict(rows=rows, source={key: value for key, value in lock.items() if key != 'rows'},
                  physical_individual_advanced=False, native_parameters_modified=False,
                  evidence_level='Exploratory fixed-geometry finite-sector observation constraint',
                  observation_hypothesis='Installed paid bridge is a visible structure; not yet a geometric filopodium',
                  actual_control=[{key: w[key] for key in ('file', 'mean_count', 'fano', 'counts', 'incident_records')}
                                  for w in control])
    atomic_json(args.out/'report.json', report)
    print(json.dumps([dict(support=r['total_support'], clock=r['fit']['clock_reference_ms_per_minute'],
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
