"""Known-type spatial program, selected on held-out real cells (not types)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from information_boltzmann.cgb.morphology import SpatialIdentityProgram, symmetric_log


def main(args):
    started = time.perf_counter()
    z = np.load(args.data/'morphology.npz', allow_pickle=False)
    keep = np.isfinite(z['soma']).all(1) & np.isin(z['side'], ['L', 'R']) & (z['length'] > 0)
    tc, body, side = z['cell_type'][keep], z['body'][keep], z['side'][keep]
    n_types = len(z['types'])
    rng = np.random.default_rng(20261022)
    split = {key: [] for key in ('train', 'val', 'test')}
    for t in np.unique(tc):
        rows = np.flatnonzero(tc == t)
        rows = rows[rng.permutation(len(rows))]
        n_hold = max(1, int(len(rows)*.1)) if len(rows) >= 3 else 0
        split['val'].extend(rows[:n_hold])
        split['test'].extend(rows[n_hold:2*n_hold])
        split['train'].extend(rows[2*n_hold:])
    split = {k: np.asarray(v, dtype=int) for k, v in split.items()}
    tr, va, te = (split[k] for k in ('train', 'val', 'test'))
    soma = z['soma'][keep].copy()
    left = np.bincount(tc[tr], weights=(side[tr] == 'L')*soma[tr, 0], minlength=n_types)
    right = np.bincount(tc[tr], weights=(side[tr] == 'R')*soma[tr, 0], minlength=n_types)
    nl = np.bincount(tc[tr][side[tr] == 'L'], minlength=n_types)
    nr = np.bincount(tc[tr][side[tr] == 'R'], minlength=n_types)
    both = (nl > 0) & (nr > 0)
    midline = float(np.median((left[both]/nl[both]+right[both]/nr[both])/2))
    sign = np.where(side == 'L', 1., -1.)
    reflection = np.column_stack([sign, np.ones((len(tc), 2))])
    offset = (z['centroid'][keep]-soma)*reflection
    soma[:, 0] = sign*(soma[:, 0]-midline)
    covariance = z['covariance'][keep]*reflection[:, :, None]*reflection[:, None, :]
    y = np.column_stack([soma, offset, np.log(z['length'][keep]), symmetric_log(covariance),
                         np.log((z['branches'][keep]+1)/z['length'][keep])])
    counts = np.bincount(tc[tr], minlength=n_types)
    known = counts > 0
    sums = np.column_stack([np.bincount(tc[tr], weights=y[tr, j], minlength=n_types) for j in range(14)])
    means = sums/np.maximum(counts[:, None], 1)
    mean = means[known].mean(0)
    scale = np.maximum(means[known].std(0), 1e-8)

    def error(pred, rows):
        losses = ((pred[tc[rows]]-y[rows])/scale)**2
        loss_by_type = np.column_stack([np.bincount(tc[rows], weights=losses[:, j], minlength=n_types) for j in range(14)])
        n = np.bincount(tc[rows], minlength=n_types)
        return (loss_by_type[n > 0]/n[n > 0, None]).mean(0)

    grid = []
    for alpha in (0., .1, .3, 1., 3., 10., 30., 100.):
        program = (sums+alpha*mean)/(np.maximum(counts[:, None]+alpha, 1e-30))
        program[~known] = mean
        grid.append({'alpha': alpha, 'train_mse': float(error(program, tr).mean()),
                     'val_mse': float(error(program, va).mean())})
    alpha = min(grid, key=lambda r: r['val_mse'])['alpha']
    code = (sums+alpha*mean)/np.maximum(counts[:, None]+alpha, 1e-30)
    code[~known] = mean
    lock = {'alpha': alpha, 'split_body_sha256': {k: hashlib.sha256(body[v].tobytes()).hexdigest() for k, v in split.items()}}
    (args.data/'spatial_identity_selection_locked.json').write_text(json.dumps(lock, indent=2), encoding='utf-8')
    test, baseline = error(code, te), error(np.broadcast_to(mean, code.shape), te)
    # Remove fitted type means before estimating seed noise, using training only.
    residual = y[tr, :3]-means[tc[tr], :3]
    jitter = residual.T@residual/max(1, len(residual))
    path = args.data/'spatial_identity_program.npz'
    np.savez(path, spatial_code=code, known=known, types=z['types'], midline=midline,
             soma_noise_covariance=jitter, counts=counts)
    deployed = SpatialIdentityProgram.load(path).describe(tc[te], side[te])
    deploy_error = float(np.abs(deployed['soma_mean']-z['soma'][keep][te]).mean())
    report = {'status': 'completed', 'scope': 'Known-type held-out cells, one animal; descriptor estimation, not tree topology',
              'cells': len(tc), 'known_types': int(known.sum()), 'parameters': int(code.size+10),
              'split': {k: {'cells': len(v), 'types': len(np.unique(tc[v]))} for k, v in split.items()},
              'alpha': alpha, 'midline_um': midline, 'grid': grid,
              'test_mse': float(test.mean()), 'baseline_test_mse': float(baseline.mean()),
              'gain': float(1-test.mean()/baseline.mean()),
              'test_mse_by_dimension': test.tolist(), 'baseline_by_dimension': baseline.tolist(),
              'heldout_soma_mean_absolute_error_um': deploy_error,
              'length_quantiles_um': np.quantile(np.exp(code[known, 6]), [.1, .5, .9]).tolist(),
              'program': str(path), 'seconds': time.perf_counter()-started,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.data/'spatial_identity_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--data', type=Path, required=True)
    main(p.parse_args())
