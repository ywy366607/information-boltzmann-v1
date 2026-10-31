"""Audit discrete observational equivalence using locked R7 fits, without fitting.

The rational checks verify algebra, not biological capability. Source workbooks
and prior results are read-only. Bulbous is used only as a positive/zero marker.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction as F
import hashlib
import itertools
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def capture_survival(u, v, k, steps: int):
    """Independent recurrence: U->U:u, U->C:k, C->C:v, else absorption."""
    unbound, captured = 1, 0
    survival, captured_mass = [], []
    for _ in range(steps + 1):
        survival.append(unbound + captured)
        captured_mass.append(captured)
        unbound, captured = u * unbound, k * unbound + v * captured
    return survival, captured_mass


def closed_survival(u, v, k, n: int):
    if n == 0:
        return 1
    if u == v:
        return u**n + k * n * u**(n - 1)
    return u**n + k * (v**n - u**n) / (v - u)


def count_path_law(survival, births=(0, 2, 4), frames=6):
    """Exact pushforward of independent duration laws to complete count paths."""
    choices = []
    for birth in births:
        exposure = frames - 1 - birth
        choices.append([(n, survival[n] - survival[n + 1])
                        for n in range(exposure)]
                       + [(exposure, survival[exposure])])
    law = defaultdict(F)
    for outcome in itertools.product(*choices):
        probability = math.prod(item[1] for item in outcome)
        counts = tuple(sum(b <= t <= b + item[0]
                           for b, item in zip(births, outcome))
                       for t in range(frames))
        law[counts] += probability
    return dict(law)


def exact_algebra_checks():
    parameter_cases = 0
    survival_checks = 0
    mixture_cases = 0
    # Includes coincident eigenvalues and negative exponential coefficients.
    for a in range(8):
        for b in range(8):
            u, v = F(a, 8), F(b, 8)
            for j in range(9 - a):
                k = F(j, 8)
                survival, _ = capture_survival(u, v, k, 24)
                for n, actual in enumerate(survival):
                    require(actual == closed_survival(u, v, k, n),
                            'Closed form differs from exact recurrence')
                    require(0 <= actual <= 1, 'Invalid survival probability')
                    if n:
                        require(actual <= survival[n - 1], 'Survival increased')
                    survival_checks += 1
                parameter_cases += 1
    for a in range(8):
        for b in range(a + 1, 8):
            for j in range(9):
                u, v, p = F(a, 8), F(b, 8), F(j, 8)
                k = p * (v - u)
                require(0 <= k <= 1 - u, 'Mixture mapping is not stochastic')
                survival, _ = capture_survival(u, v, k, 24)
                require(all(s == (1 - p) * u**n + p * v**n
                            for n, s in enumerate(survival)),
                        'Positive mixture differs from capture')
                mixture_cases += 1
    u, v, p = F(1, 4), F(3, 4), F(1, 3)
    dynamic, _ = capture_survival(u, v, p * (v - u), 6)
    static = [(1 - p) * u**n + p * v**n for n in range(7)]
    dynamic_counts, static_counts = count_path_law(dynamic), count_path_law(static)
    require(dynamic_counts == static_counts, 'Joint count laws differ')
    require(sum(dynamic_counts.values()) == 1, 'Count law is not normalized')
    # Enumerate hidden paths independently to check the marker conditioning.
    # C is irreversible; a visible C at any sampled time implies C at last alive.
    k = p * (v - u)
    paths = [(('U',), F(1))]
    transitions = {'U': [('U', u), ('C', k), ('D', 1-u-k)],
                   'C': [('C', v), ('D', 1-v)], 'D': [('D', F(1))]}
    for _ in range(5):
        paths = [(history + (state,), prob * chance)
                 for history, prob in paths
                 for state, chance in transitions[history[-1]]]
    survival, captured = capture_survival(u, v, k, 5)
    marker_checks = 0
    for n in range(5):
        ended_paths = [(h, q) for h, q in paths if h[n] != 'D' and h[n+1] == 'D']
        ended_total = sum(q for _, q in ended_paths)
        ended_marked = sum(q for h, q in ended_paths if 'C' in h[:n+1])
        require(ended_total == survival[n] - survival[n+1]
                and ended_marked == captured[n] * (1-v),
                'Marker event formula differs from hidden-path enumeration')
        alive_paths = [(h, q) for h, q in paths if h[n] != 'D']
        require(sum(q for _, q in alive_paths) == survival[n]
                and sum(q for h, q in alive_paths if 'C' in h[:n+1]) == captured[n],
                'Marker censor formula differs from hidden-path enumeration')
        marker_checks += 2
    # A selected capture intervention holds U-death probability and v fixed.
    k, extra = p * (v - u), F(1, 24)
    death = 1 - u - k
    u2, k2 = u - extra, k + extra
    p2 = k2 / (v - u2)
    require(1 - u2 - p2 * (v - u2) == death,
            'Capture-only perturbation did not preserve U-death')
    # A selected protection intervention holds u and k fixed.
    v2 = v + F(1, 16)
    require((k / (v2 - u)) * (v2 - u) == k,
            'Protection-only perturbation did not preserve capture')
    return dict(parameter_cases=parameter_cases,
                exact_survival_checks=survival_checks,
                exact_positive_mixture_mappings=mixture_cases,
                exact_count_paths=len(dynamic_counts),
                count_path_laws_identical=True,
                exact_marker_event_and_censor_checks=marker_checks,
                equal_eigenvalue_and_negative_coefficient_cases_included=True,
                conditional_intervention_identities_checked=True)


def audit(base: Path):
    obs_path = base / 'cgb_r7_observation_audit_20261022/observations.json'
    lock_path = base / 'cgb_r7_lifetime_constraints_20261022/parameters_locked.json'
    report_path = base / 'cgb_r7_lifetime_constraints_20261022/report.json'
    author_root = base / 'cgb_inverse_sources_20261022/Filo2'
    observed = json.loads(obs_path.read_text(encoding='utf-8'))
    locked = json.loads(lock_path.read_text(encoding='utf-8'))
    prior = json.loads(report_path.read_text(encoding='utf-8'))
    require(sha256(obs_path) == locked['observations_sha256'],
            'Observation audit differs from locked fit input')
    require(observed['source']['commit'] == locked['source_commit'],
            'Source revision mismatch')
    model = locked['parameters']['mixture']
    require(model == prior['models']['mixture']['parameters'],
            'Prior report and locked parameters differ')
    slow, fast = model['rates_per_minute']
    p, fast_weight = model['weights_at_first_observation']
    require(0 < slow < fast and 0 < p < 1 and abs(p + fast_weight - 1) < 1e-14,
            'Expected locked two-component mixture')
    u, v = math.exp(-fast), math.exp(-slow)
    k = p * (v - u)
    dynamic, captured = capture_survival(u, v, k, 60)
    static = [(1 - p) * u**n + p * v**n for n in range(61)]
    max_survival_error = max(abs(a - b) for a, b in zip(dynamic, static))
    require(max_survival_error < 1e-13, 'Numerical equivalence failed')
    cones = sorted((w for w in observed['workbooks']
                    if w['kind'] == 'filopodium' and w['genotype'] == 'Control'),
                   key=lambda w: w['file'])
    expected_files = [row['file'] for row in locked['train'] + locked['test']]
    require([w['file'] for w in cones] == expected_files,
            'Control split differs from locked manifest')
    source_files = {row['path']: row for row in observed['source']['files']}
    rows = []
    split_scores = {'train': [[], []], 'historical_test': [[], []]}
    for index, cone in enumerate(cones):
        raw = author_root / cone['file']
        expected_hash = source_files[cone['file']]['sha256']
        require(sha256(raw) == cone['sha256'] == expected_hash
                == (locked['train'] + locked['test'])[index]['sha256'],
                'Raw workbook hash mismatch')
        records = [r for r in cone['records'] if r['first_frame'] > 0]
        static_losses, dynamic_losses, marker_expectations = [], [], []
        for record in records:
            n = record['last_frame'] - record['first_frame']
            ended = record['last_frame'] < 59
            ps = static[n] - static[n + 1] if ended else static[n]
            unbound = u**n
            pd = (unbound * (1 - u - k) + captured[n] * (1 - v)
                  if ended else dynamic[n])
            static_losses.append(-math.log(ps))
            dynamic_losses.append(-math.log(pd))
            # Extra observation hypothesis: C is exactly the recorded Bulbous
            # mark, with perfect detection, irreversible C, and initial U.
            marked = captured[n] * (1 - v) if ended else captured[n]
            marker_expectations.append(marked / pd)
        split = 'train' if index < 2 else 'historical_test'
        split_scores[split][0].extend(static_losses)
        split_scores[split][1].extend(dynamic_losses)
        rows.append(dict(file=cone['file'], sha256=expected_hash, split=split,
                         incident_tracks=len(records),
                         observed_bulbous_positive=sum(
                             r['bulbous_frame_count'] > 0 for r in records),
                         expected_positive_if_C_is_Bulbous_given_lifetimes=sum(
                             marker_expectations),
                         max_track_nll_difference=max(abs(a - b) for a, b in
                                                      zip(static_losses, dynamic_losses))))
    scores = {}
    for split, (static_losses, dynamic_losses) in split_scores.items():
        count = len(static_losses)
        ref_key = 'train' if split == 'train' else 'test'
        original = prior['models']['mixture']['scores'][ref_key]['mean_nll']
        static_mean, dynamic_mean = sum(static_losses)/count, sum(dynamic_losses)/count
        require(max(abs(static_mean - original), abs(dynamic_mean - original)) < 1e-12,
                'Historical locked score was not reproduced')
        scores[split] = dict(tracks=count, static_mean_nll=static_mean,
                             dynamic_mean_nll=dynamic_mean,
                             historical_locked_mean_nll=original)
    require('Control_P60_GC4.xlsx' not in '\n'.join(expected_files),
            'Unexpected additional complete-Control cone')
    return dict(
        schema_version=1, stage='inverse_development_identifiability_audit',
        status='paper_derivation_with_exact_and_numerical_checks_not_Lean_proof',
        source_commit=locked['source_commit'],
        input_sha256={str(path): sha256(path)
                      for path in (obs_path, lock_path, report_path)},
        source_workbooks_sha256_checked=True,
        script_sha256=sha256(Path(__file__)),
        exact_algebra=exact_algebra_checks(),
        locked_fit_mapping=dict(u=u, v=v, k=k, death_from_U=1-u-k,
                                mixture_slow_coefficient=p,
                                ever_observed_C_probability=k/(1-u),
                                continuous_capture_rate=p*(fast-slow),
                                continuous_capture_before_loss_probability=p*(fast-slow)/fast),
        max_survival_difference=max_survival_error, scores=scores, cones=rows,
        marker_check_scope='Exploratory conditional expectations under an extra unreviewed '
            'C_equals_Bulbous observation hypothesis; no intervention or biological identification',
        marker_column_semantics='Cached bulbous_frame_count is raw column C (Bulbous); '
            'only positive/zero is used, consistent with author DataImport.m logical(T.Bulbous)',
        conclusions=dict(lifetime_mixture_does_not_identify_capture=True,
                         same_independent_arrivals_and_lifetimes_imply_same_count_and_turnover_law=True,
                         core_axioms_derive_capture=False,
                         new_parameters_fitted=False,
                         fresh_blind_test=False,
                         complete_Control_CG4_exists_in_locked_split=False,
                         independent_mechanism_review_completed=False,
                         biological_validation_complete=False,
                         physical_individual_or_training_advanced=False))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), 'Refusing to overwrite an existing audit')
    result = audit(args.base)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n',
                        encoding='utf-8')
    print(json.dumps(dict(passed=True, exact=result['exact_algebra'],
                          scores=result['scores'], marker_checks=result['cones']),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
