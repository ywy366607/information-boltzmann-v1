"""Offline event-group holdout and contrast accounting on captured native data."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from scripts.option_basis_calibration import evaluate, fit


def error_parts(prediction, target):
    error = prediction.double()-target.double()
    centered = error-error.mean()
    n = len(error)
    return dict(mse=float(error.square().mean()), common_error=float(error.mean()),
        common_mse=float(error.mean().square()), centered_mse=float(centered.square().mean()),
        all_pair_difference_mse=float(2*n/(n-1)*centered.square().mean()) if n > 1 else 0.)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ports', type=Path, required=True)
    parser.add_argument('--measurement', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    source = torch.load(args.ports, weights_only=False, map_location='cpu')
    bank = source['engine']['latent_action_bank']['records']
    measurement = json.loads(args.measurement.read_text(encoding='utf-8'))
    native = []
    capsule = None
    for case in measurement['cases']:
        capsule = torch.load(case['option_calibration']['capsule'], weights_only=False, map_location='cpu')
        native.append(dict(role=case['role'],
            original=error_parts(torch.tensor(case['vectors']['student_q']), capsule['teacher_q']),
            calibrated=error_parts(torch.tensor(case['option_calibration']['q']), capsule['teacher_q'])))
    # Reuse the captured original factor. Its product defines the same solved
    # SPD system to floating factorization accuracy; verify full prediction.
    factor, scale = capsule['factor'].double(), capsule['scale'].double()
    ridge = float(source['config']['relation_wood']['ridge'])
    kernel = (factor@factor.T-ridge*torch.eye(len(bank)))/scale[:, None]/scale[None]
    target, prior = capsule['target'].double(), capsule['bank_prior'].double()
    options = capsule['options']
    assert len(bank) == len(target)
    assert options.tolist() == [r['packet']['option'] for r in bank]
    torch.testing.assert_close(target, torch.stack([r['target'] for r in bank])[:, -1].double())
    original_alpha = torch.cholesky_solve((scale*(target-prior))[:, None], factor)[:, 0]
    reconstructed = capsule['query_prior'].double()+capsule['cross'].double()@original_alpha
    reconstruction_error = float((reconstructed-capsule['original_q'].double()).abs().max())
    assert reconstruction_error < 1e-4
    folds = []
    for revision in sorted(set(r['revision'] for r in bank)):
        test = torch.tensor([r['revision'] == revision for r in bank])
        train = ~test
        if not train.any():
            continue
        kinds = [r['kind'] for r, yes in zip(bank, train) if yes]
        counts = {k: kinds.count(k) for k in set(kinds)}
        d = torch.tensor([len(kinds)/(len(counts)*counts[k]) for k in kinds]).double().sqrt()
        k = kernel[train][:, train]
        l = torch.linalg.cholesky(k*d[:, None]*d[None]+ridge*torch.eye(len(k)))
        y = target[train]-prior[train]
        alpha = torch.cholesky_solve((d*y)[:, None], l)[:, 0]
        cross = kernel[test][:, train]*d[None]
        baseline = prior[test]+cross@alpha
        calibrated = evaluate(fit(l, d, options[train], y), prior[test], options[test], cross)
        folds.append(dict(revision=revision, rows=int(test.sum()),
            options=options[test].tolist(), kinds=[r['kind'] for r, yes in zip(bank, test) if yes],
            target=target[test].tolist(), original_q=baseline.tolist(), calibrated_q=calibrated.tolist(),
            original=error_parts(baseline, target[test]),
            calibrated=error_parts(calibrated, target[test])))
    total = sum(f['rows'] for f in folds)
    holdout = {name: sum(f['rows']*f[name]['mse'] for f in folds)/total
               for name in ('original', 'calibrated')}
    by_kind = {}
    for kind in ('native', 'world'):
        t, a, b = [], [], []
        for f in folds:
            for i, k in enumerate(f['kinds']):
                if k == kind:
                    t.append(f['target'][i]); a.append(f['original_q'][i]); b.append(f['calibrated_q'][i])
        by_kind[kind] = dict(rows=len(t), original=error_parts(torch.tensor(a), torch.tensor(t)),
                            calibrated=error_parts(torch.tensor(b), torch.tensor(t)))
    result = dict(schema=1, measurement=str(args.measurement),
        source_sha256=measurement['source_sha256'],
        capsule_sha256=hashlib.sha256(Path(measurement['cases'][-1]['option_calibration']['capsule']).read_bytes()).hexdigest(),
        ridge=ridge, native= native, factor_reconstruction_error=reconstruction_error,
        holdout=dict(unit='one complete bank commit revision', folds=len(folds), rows=total,
                     mse=holdout, by_kind=by_kind, details=folds),
        world_calls=0, optimizer_updates=0)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(native=native, holdout=holdout, by_kind=by_kind, folds=len(folds))))


if __name__ == '__main__':
    main()
