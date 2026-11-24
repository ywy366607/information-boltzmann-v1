"""Actual scalar-bank spectral filter: event-group holdout and root-use account."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from scripts.native_value_residual_filter import coefficients, spectral_report


def mse(prediction, target):
    return float((prediction-target).square().mean())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--outcomes', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    data = torch.load(a.capture/'capsule.pt', weights_only=False, map_location='cpu')
    bank = data['bank']; records = bank['records']; ridge = bank['ridge']
    x, d = bank['x'].double(), bank['scale'].double()
    kernel = (1+x@x.T).square()
    gram = kernel*d[:, None]*d[None]
    target, prior = bank['target'].double(), bank['prior'].double()
    r = d*(target-prior)
    factor = torch.linalg.cholesky(gram+ridge*torch.eye(len(x), dtype=x.dtype))
    alpha1 = coefficients(factor, r, ridge, 1)
    alpha2 = coefficients(factor, r, ridge, 2)
    production_reconstruction = float((alpha1-bank['coefficient'].double()).abs().max())
    fitted = [prior+(kernel*d[None])@alpha[:, 0] for alpha in (alpha1, alpha2)]
    folds = []
    for revision in sorted(set(row['revision'] for row in records)):
        test = torch.tensor([row['revision'] == revision for row in records])
        train = ~test
        kinds = [row['kind'] for row, yes in zip(records, train) if yes]
        counts = {kind:kinds.count(kind) for kind in set(kinds)}
        scale = torch.tensor([len(kinds)/(len(counts)*counts[kind]) for kind in kinds], dtype=x.dtype).sqrt()
        g = kernel[train][:, train]*scale[:, None]*scale[None]
        l = torch.linalg.cholesky(g+ridge*torch.eye(len(g), dtype=x.dtype))
        residual = scale*(target[train]-prior[train])
        cross = kernel[test][:, train]*scale[None]
        first = prior[test]+(cross@coefficients(l, residual, ridge, 1))[:, 0]
        second = prior[test]+(cross@coefficients(l, residual, ridge, 2))[:, 0]
        folds.append(dict(revision=revision, rows=int(test.sum()),
            kinds=[row['kind'] for row, yes in zip(records, test) if yes],
            target=target[test].tolist(), original=first.tolist(), two_fit=second.tolist()))
    errors = {}
    for kind in ('all', 'native', 'world'):
        rows = [(f['target'][j], f['original'][j], f['two_fit'][j]) for f in folds
                for j,k in enumerate(f['kinds']) if kind == 'all' or k == kind]
        values = torch.tensor(rows, dtype=x.dtype)
        errors[kind] = dict(rows=len(rows), original_mse=mse(values[:, 1], values[:, 0]),
            two_fit_mse=mse(values[:, 2], values[:, 0]))
    shadow = json.loads((a.capture/'root_shadow.json').read_text(encoding='utf8'))
    outcomes = json.loads(a.outcomes.read_text(encoding='utf8'))
    original = torch.tensor(shadow['original_probability'], dtype=x.dtype)
    alternative = torch.tensor(shadow['alternative_probability'], dtype=x.dtype)
    habit = torch.tensor(shadow['habit_probability'], dtype=x.dtype)
    legal = shadow['legal']
    assert all(str(j) in outcomes for j in legal)
    assert all(float(original[j]) == 0 and float(alternative[j]) == 0
               for j in range(len(original)) if j not in legal)
    segments = len(outcomes[str(legal[0])]['segment_G2'])
    gains = torch.tensor([outcomes[str(j)]['G2'] if j in legal else 0.
                         for j in range(len(original))], dtype=x.dtype)
    segment_gains = torch.tensor([outcomes[str(j)]['segment_G2'] if j in legal else [0.]*segments
                                 for j in range(len(original))], dtype=x.dtype)
    # Signed root functional on alpha. Each physical root takes the outer
    # response mean and inner successor weighted expectation exactly once.
    root = torch.zeros(len(original), len(x), dtype=x.dtype)
    for key, options in data['root_options'].items():
        calls = [c for c in data['calls'] if c['key'] == key]
        row = sum((c['weights'].double()@c['cross'].double()) for c in calls)/len(calls)
        for option in options:
            root[option] = 10*float(json.loads((a.capture/'action.json').read_text())['gamma'])*row
    eig, vec = torch.linalg.eigh(gram)
    centered = root-(original[:, None]*root).sum(0)
    modal_root = centered@vec
    q = ridge/(eig+ridge)
    modal_delta_alpha = ridge*(vec.T@r)/(eig+ridge).square()
    modal_critic = modal_root*modal_delta_alpha[None]
    actual_delta = torch.tensor(shadow['critic_alternative'], dtype=x.dtype)-torch.tensor(shadow['critic_original'], dtype=x.dtype)
    centered_delta = actual_delta-(original*actual_delta).sum()
    transport_error = float((modal_critic.sum(-1)-centered_delta).abs().max())
    assert transport_error < 2e-5
    # This gradient describes the fixed candidate AFTER its definition; G2
    # enters this account only and never the residual fit or features.
    sensitivity = float(shadow['precision'])*(original*(gains-(original*gains).sum()))@root
    modal_use = (sensitivity@vec)*modal_delta_alpha
    anchors = []
    for teacher in json.loads((a.capture/'teacher_checks.json').read_text(encoding='utf8')):
        increments = []
        for call in data['calls']:
            if call['key'] != teacher['root_key']:
                continue
            matched = (call['baseline']-teacher['predicted']).abs() < 1e-7
            increments.extend((call['cross'].double()[matched]@(
                ridge*torch.cholesky_solve(bank['coefficient'].double(), bank['factor'].double())))[:, 0].tolist())
        if increments:
            anchors.append(dict(root_key=teacher['root_key'], matched_nodes=len(increments),
                original_prediction=teacher['predicted'], current_native_target=teacher['native'],
                increment_min=min(increments), increment_max=max(increments),
                original_error=teacher['predicted']-teacher['native'],
                candidate_error_min=teacher['predicted']+min(increments)-teacher['native'],
                candidate_error_max=teacher['predicted']+max(increments)-teacher['native']))
    timing = []
    for _ in range(100):
        start = time.perf_counter()
        ridge*torch.cholesky_solve(bank['coefficient'].double(), bank['factor'].double())
        timing.append(1e3*(time.perf_counter()-start))
    result = dict(schema=1, capture=str(a.capture),
        capsule_sha256=hashlib.sha256((a.capture/'capsule.pt').read_bytes()).hexdigest(),
        active_target='native_amortized_decision scalar value consumed by persistent_rhythm',
        candidate='exactly two ridge residual fits, predeclared before this capture',
        bank=dict(rows=len(records), revision=bank['revision'], ridge=ridge,
            original_training_mse=mse(fitted[0], target), two_fit_training_mse=mse(fitted[1], target),
            float64_coefficient_reconstruction_max_abs=production_reconstruction,
            spectrum=spectral_report(gram, r, ridge)),
        holdout=dict(unit='one complete commit revision', folds=len(folds), scores=errors, details=folds,
            coordinates='current frozen encoder and current prior; complete bank revisions held out'),
        current_native_anchors=anchors,
        use=dict(boundary=json.loads((a.capture/'action.json').read_text())['unknown'], legal=legal,
            original_probability=original.tolist(), two_fit_probability=alternative.tolist(),
            G2=gains.tolist(), original_C=float((original-habit)@gains),
            two_fit_C=float((alternative-habit)@gains), delta_C=float((alternative-original)@gains),
            per_segment_delta_C=((alternative-original)@segment_gains).tolist(),
            mode_transport_max_abs_error=transport_error,
            weak_mode_linearized_use=float(modal_use[eig<ridge].sum()),
            strong_mode_linearized_use=float(modal_use[eig>=ridge].sum()),
            total_linearized_use=float(modal_use.sum()),
            reuse='same complete menu and already executed three conditional two-round outcomes'),
        cost=dict(extra_fit_cpu_median_ms=sorted(timing)[len(timing)//2],
            extra_matrix_elements=len(x)**2, extra_teacher_calls=0, new_world_calls=0,
            new_optimizer_updates=0),
        decision='candidate evaluated by real selection contribution; label holdout describes its distinct prediction target')
    a.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf8')
    print(json.dumps(dict(bank={k:v for k,v in result['bank'].items() if k!='spectrum'},
        spectrum={k:v for k,v in result['bank']['spectrum'].items() if isinstance(v,(int,float))},
        holdout=errors, use=result['use'], cost=result['cost']), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
