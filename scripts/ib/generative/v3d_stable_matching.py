"""V3d: stable type-matching fits, validation-only selection, executable codes.

The historical E1 split is reused and explicitly labelled; no new individual
is trained. Run --prepare-only for data checks or --calibrate for a bounded
resource/optimizer calibration before the preregistered multi-seed sweep.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from information_boltzmann.cgb.matching import TypeMatching, PairBatch, linear_fit, optimize, objective


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def prepare(work):
    cache = np.load(work/'e1_types3000.npz', allow_pickle=True)
    sizes, names = cache['sizes'], cache['names'].astype(str)
    sides = np.load(work/'e1_types3000_sides.npz')
    wr, nr, nl = sides['WR'], sides['nr'], sides['nl']
    n = len(sizes)
    valid = (wr >= 3) & ((nl > 0) & (nr > 0))[:, None]
    pre, post = np.nonzero(valid)
    u = torch.rand((n, n), generator=torch.Generator().manual_seed(1)).numpy()[pre, post]
    splits = {'train': np.flatnonzero(u >= .10), 'val': np.flatnonzero(u < .05),
              'test': np.flatnonzero((u >= .05) & (u < .10))}
    target = np.log(wr[pre, post].astype(float) / (nr[pre] * sizes[post]))
    wl = sides['WL']
    twin = np.stack((np.log(np.maximum(wl[pre, post], 1) / (nl[pre] * sizes[post])),
                     wl[pre, post] >= 3), axis=1)
    del wr, wl, valid, sides
    features = []
    paths = [work/'e1_types3000.npz', work/'e1_types3000_sides.npz']
    for scale in (4, 2):
        path = work/f'e1c_overlap_trainlabels_{scale}um.npy'
        paths.append(path)
        matrix = np.load(path, mmap_mode='r')
        # V3c floors AFTER normalization over all type pairs. Bound workspace.
        floor = np.inf
        for start in range(0, n, 128):
            e = matrix[start:start+128] / (sizes[start:start+128, None] * sizes[None, :])
            positive = e[e > 0]
            if len(positive):
                floor = min(floor, float(positive.min()) * .5)
        for a, b in ((pre, post), (post, pre)):
            encounter = matrix[a, b] / (sizes[pre] * sizes[post])
            features.extend((np.log(np.maximum(encounter, floor)), (encounter == 0).astype(float)))
    features = np.stack(features, axis=1)
    mean, scale = features[splits['train']].mean(0), features[splits['train']].std(0)
    scale = np.maximum(scale, 1e-8)
    features = (features-mean)/scale
    split_digest = {key: hashlib.sha256(np.stack((pre[idx], post[idx]), axis=1).tobytes()).hexdigest()
                    for key, idx in splits.items()}
    return {'n': n, 'names': names, 'pre': pre, 'post': post, 'target': target,
        'features': features, 'twin': twin, 'splits': splits, 'feature_mean': mean,
        'feature_scale': scale, 'files': {str(p): digest(p) for p in paths},
        'split_sha256': split_digest}


def main(args):
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    data = prepare(args.work)
    train = data['splits']['train']
    baselines, linear = {}, {}
    for name, features in (('degree', data['features'][:, :0]),
                          ('geometry', data['features']), ('replicate', data['twin'])):
        coefficients, prediction, info = linear_fit(data['n'], data['pre'], data['post'],
            features, data['target'], train)
        baselines[name] = {'solver': info, 'mse_train': float(np.mean((prediction[train]-data['target'][train])**2)),
                           'mse_val': float(np.mean((prediction[data['splits']['val']]-data['target'][data['splits']['val']])**2))}
        linear[name] = (coefficients, prediction)
    device = args.device
    batch = PairBatch(torch.tensor(data['pre'], device=device), torch.tensor(data['post'], device=device),
        torch.tensor(data['features'], device=device, dtype=torch.float32),
        torch.tensor(data['target'], device=device, dtype=torch.float32))
    batches = {key: batch.selected(torch.tensor(idx, device=device)) for key, idx in data['splits'].items() if key != 'test'}
    provenance = {'created_at': datetime.now(timezone.utc).isoformat(), 'types': data['n'],
        'pairs': {key: len(value) for key, value in data['splits'].items()},
        'data_sha256': data['files'], 'split_sha256': data['split_sha256'],
        'code_sha256': {str(p.relative_to(ROOT)): digest(p) for p in (Path(__file__), ROOT/'information_boltzmann/cgb/matching.py')},
        'baselines': baselines, 'test_scope': 'Reused historical E1 pair split, not a fresh independent biological replication',
        'preregistration': 'docs/information_boltzmann/CGB_V3D_PREREG_20261021.md',
        'device': device, 'gpu': torch.cuda.get_device_name() if device == 'cuda' else None,
        'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    write_json(args.out/'provenance.json', provenance)
    print(json.dumps({'prepared': provenance['pairs'], 'baselines': baselines}), flush=True)
    if args.prepare_only:
        return
    families = [('directed64', 64, False), ('symmetric64', 64, True), ('directed256', 256, False)]
    if args.calibrate:
        families = families[:1]
    seeds = [int(x) for x in args.seeds.split(',')]
    lambdas = sorted(float(x) for x in args.lambdas.split(','))
    rows = []
    fits = args.out/'fits'
    fits.mkdir(exist_ok=True)

    def fit_one(family, rank, symmetric, seed, regularization, warm=None, suffix='', steps=None):
        key = f'{family}_s{seed}_l{regularization:g}{suffix}'
        path = fits/(key+'.pt')
        record_path = fits/(key+'.json')
        if path.exists() and record_path.exists():
            record = json.loads(record_path.read_text())
            if (record['code_sha256'] != provenance['code_sha256']
                    or record['data_sha256'] != provenance['data_sha256']
                    or record['adam_updates'] != (args.steps if steps is None else steps)):
                raise ValueError('Resuming a different fitting implementation requires a new output directory')
            archive = torch.load(path, map_location=device, weights_only=False)
            return record, archive['state']
        model = TypeMatching(data['n'], rank, batch.features.shape[1], symmetric=symmetric, seed=seed, device=device)
        model.initialize_linear(linear['geometry'][0])
        if warm is not None:
            model.load_state_dict(warm)
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        result = optimize(model, batches['train'], batches['val'], regularization,
                          steps=args.steps if steps is None else steps, polish=args.polish)
        state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        record = {**result, 'family': family, 'rank': rank, 'symmetric': symmetric, 'seed': seed,
            'lambda': regularization, 'warm_start': warm is not None, 'suffix': suffix,
            'parameters': sum(p.numel() for p in model.parameters()),
            'seconds': time.perf_counter()-t0, 'model_path': str(path), 'code_sha256': provenance['code_sha256'],
            'data_sha256': provenance['data_sha256'],
            'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20 if device == 'cuda' else None}
        temporary = path.with_suffix('.tmp')
        torch.save({'specification': model.specification(), 'state': state,
            'feature_mean': data['feature_mean'], 'feature_scale': data['feature_scale'],
            'types': data['names'].tolist(), 'data_sha256': data['files']}, temporary)
        temporary.replace(path)
        write_json(record_path, record)
        print(json.dumps({k: record[k] for k in ('family','seed','lambda','train_mse','val_mse','gradient_l2','seconds','peak_allocated_mib')}), flush=True)
        del model
        gc.collect()
        return record, state

    # Each seed has its own path through the SAME grid. No test metric is read.
    for family, rank, symmetric in families:
        for seed in seeds:
            warm = None
            for regularization in lambdas:
                record, warm = fit_one(family, rank, symmetric, seed, regularization, warm)
                rows.append(record)
                write_json(args.out/'progress.json', {'status': 'fitting_validation_only', 'fits_completed': len(rows),
                    'last_fit': {k: record[k] for k in ('family','seed','lambda','val_mse')}, 'seconds': time.perf_counter()-started})
    if args.calibrate:
        write_json(args.out/'calibration.json', {'rows': rows, 'provenance': provenance,
            'seconds': time.perf_counter()-started, 'test_evaluated': False})
        return

    chosen = {}
    for family, rank, symmetric in families:
        candidates = [r for r in rows if r['family'] == family]
        means = {l: float(np.mean([r['val_mse'] for r in candidates if r['lambda'] == l])) for l in lambdas}
        best = min(means, key=means.get)
        # Preregistered outward extension only when the optimum hits an edge.
        extra = 1.0 if best == lambdas[0] and best > 1 else 100.0 if best == lambdas[-1] and best < 100 else None
        if extra is not None:
            for seed in seeds:
                anchor = next(r for r in candidates if r['seed'] == seed and r['lambda'] == best)
                state = torch.load(anchor['model_path'], map_location='cpu', weights_only=False)['state']
                record, _ = fit_one(family, rank, symmetric, seed, extra, state)
                rows.append(record); candidates.append(record)
            means[extra] = float(np.mean([r['val_mse'] for r in candidates if r['lambda'] == extra]))
            best = min(means, key=means.get)
        chosen[family] = {'lambda': best, 'validation_means': means,
                          'edge_of_searched_grid': best in (min(means), max(means)),
                          'fits': [r for r in candidates if r['lambda'] == best]}
        cold, _ = fit_one(family, rank, symmetric, seeds[0], best, suffix='_cold')
        chosen[family]['cold_start'] = cold
        extended = []
        for original in chosen[family]['fits']:
            state = torch.load(original['model_path'], map_location='cpu', weights_only=False)['state']
            refined, _ = fit_one(family, rank, symmetric, original['seed'], best,
                warm=state, suffix='_extended', steps=1000)
            # Select by the fixed TRAINING objective, never by a test score.
            extended.append({'seed': original['seed'],
                'objective_improvement': original['objective']-refined['objective'],
                'relative_objective_improvement': (original['objective']-refined['objective'])/abs(original['objective']),
                'validation_change': refined['val_mse']-original['val_mse'],
                'refined': refined})
        chosen[family]['extension_check'] = extended
        chosen[family]['fits'] = [check['refined'] if check['objective_improvement'] > 0 else original
            for check, original in zip(extended, chosen[family]['fits'])]
        chosen[family]['numerically_settled'] = all(
            abs(check['relative_objective_improvement']) < 1e-4 for check in extended)
    write_json(args.out/'selection_locked.json', {'chosen': chosen, 'criterion': 'mean validation MSE over all specified seeds',
        'test_evaluated': False, 'locked_at': datetime.now(timezone.utc).isoformat()})

    # Test is touched only AFTER all hyperparameters, seeds and cold controls are locked.
    test = data['splits']['test']
    target = data['target'][test]
    null_sq = (linear['degree'][1][test]-target)**2
    geo_sq = (linear['geometry'][1][test]-target)**2
    null_mse = float(null_sq.mean())
    baseline_scores = {name: 1-float(np.mean((prediction[test]-target)**2))/null_mse
                       for name, (_, prediction) in linear.items()}
    tb = batch.selected(torch.tensor(test, device=device))
    selected_predictions = {}
    for family, rank, symmetric in families:
        selected = chosen[family]
        scores, predictions = [], []
        for record in selected['fits'] + [selected['cold_start']]:
            archive = torch.load(record['model_path'], map_location=device, weights_only=False)
            model = TypeMatching(**archive['specification'], device=device)
            model.load_state_dict(archive['state'])
            with torch.no_grad():
                prediction = model(tb.pre, tb.post, tb.features).double().cpu().numpy()
            error_sq = (prediction-target)**2
            score = 1-float(error_sq.mean())/null_mse
            item = {'seed': record['seed'], 'test_explained': score, 'test_gain_over_geometry': score-baseline_scores['geometry']}
            if record is selected['cold_start']:
                selected['cold_test'] = item
            else:
                scores.append(item); predictions.append(prediction)
            del model
        selected['test_by_seed'] = scores
        selected['test_mean'] = float(np.mean([s['test_explained'] for s in scores]))
        selected['test_std'] = float(np.std([s['test_explained'] for s in scores], ddof=1)) if len(scores)>1 else 0.
        # Mean loss across individual fits, not the improvement from an ensemble.
        sq = np.mean([(prediction-target)**2 for prediction in predictions], axis=0)
        rng = np.random.default_rng(20261021)
        gains = []
        for _ in range(1000):
            draw = rng.integers(0, len(test), len(test))
            gains.append(float((geo_sq[draw]-sq[draw]).sum()/null_sq[draw].sum()))
        selected['gain_pair_bootstrap_95'] = np.quantile(gains, [.025,.975]).tolist()
        selected_predictions[family] = predictions
        # Export the seed closest to mean validation error, not the best test seed.
        center = np.mean([r['val_mse'] for r in selected['fits']])
        representative = min(selected['fits'], key=lambda r: abs(r['val_mse']-center))
        archive = torch.load(representative['model_path'], map_location='cpu', weights_only=False)
        arrays = {k: value.numpy() for k, value in archive['state'].items()}
        np.savez(args.out/(family+'_generator.npz'), **arrays, feature_mean=data['feature_mean'],
                 feature_scale=data['feature_scale'], types=data['names'])
        selected['representative_generator'] = str(args.out/(family+'_generator.npz'))
    d64, s64, d256 = (chosen[n]['test_mean'] for n in ('directed64','symmetric64','directed256'))
    report = {'provenance': provenance, 'baselines_test_explained': baseline_scores,
        'selected': chosen, 'all_fits': [{k:v for k,v in r.items() if k != 'curve'} for r in rows],
        'predictions': {'P1_directed64_gain_at_least_8pt': d64-baseline_scores['geometry'] >= .08,
            'P2_directed64_seed_std_at_most_2pt': chosen['directed64']['test_std'] <= .02,
            'P3_directional_gain_at_least_2pt': d64-s64 >= .02,
            'P4_256_gain_under_2pt': d256-d64 < .02},
        'bootstrap_scope': 'Conditional resampling of held-out type pairs, not independent flies; shared-type dependence remains',
        'seconds': time.perf_counter()-started, 'status': 'completed',
        'unified_living_brain_claim': False}
    write_json(args.out/'report.json', report)
    write_json(args.out/'progress.json', {'status': 'completed', 'report': str(args.out/'report.json')})
    print(json.dumps({'status': 'completed', 'baselines': baseline_scores,
        'selected': {name:{k:r[k] for k in ('lambda','test_mean','test_std')} for name,r in chosen.items()},
        'predictions': report['predictions'], 'seconds': report['seconds']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--work', type=Path, default=Path('F:/fly_stream_overflow/cgb_v1'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu','cuda'), default='cuda')
    parser.add_argument('--seeds', default='0,1,2')
    parser.add_argument('--lambdas', default='3,5,7,10,14,20,30')
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--polish', type=int, default=200)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--calibrate', action='store_true')
    main(parser.parse_args())
