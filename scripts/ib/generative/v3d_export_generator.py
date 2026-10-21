"""Export selected matching rules and verify CPU reconstruction on real pairs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from information_boltzmann.cgb.type_generator import TypeStrengthGenerator
from information_boltzmann.cgb.matching import TypeMatching
from scripts.ib.generative.v3d_stable_matching import prepare, digest, write_json


def main(args):
    report_path = args.run/args.report
    report = json.loads(report_path.read_text())
    if report['status'] != 'completed':
        raise ValueError('Finish validation selection before publishing a generator')
    data = prepare(args.work)
    if data['files'] != report['provenance']['data_sha256']:
        raise ValueError('Real data changed after fitting')
    verification = {}
    for family, selected in report['selected'].items():
        center = np.mean([r['val_mse'] for r in selected['fits']])
        chosen = min(selected['fits'], key=lambda r: abs(r['val_mse']-center))
        archive = torch.load(chosen['model_path'], map_location='cpu', weights_only=False)
        arrays = {k: value.numpy() for k, value in archive['state'].items()}
        path = args.run/(family+'_deploy.npz')
        np.savez(path, **arrays, symmetric=np.array(archive['specification']['symmetric']),
            feature_mean=archive['feature_mean'], feature_scale=archive['feature_scale'], types=np.array(archive['types']))
        generator = TypeStrengthGenerator.load(path)
        # No refitting; verify all observed real type pairs in bounded batches.
        idx = np.arange(len(data['pre']))
        raw = data['features']*data['feature_scale']+data['feature_mean']
        prediction = generator.log_strength(data['pre'], data['post'], raw)
        model = TypeMatching(**archive['specification'])
        model.load_state_dict(archive['state'])
        maximum = 0.
        with torch.no_grad():
            expected = model(torch.tensor(data['pre']), torch.tensor(data['post']),
                torch.tensor(data['features'], dtype=torch.float32)).double().numpy()
            maximum = float(np.max(np.abs(expected-prediction)))
        passed = maximum < 1e-4
        verification[family] = {'source_fit': chosen['model_path'], 'generator': str(path),
            'sha256': digest(path), 'bytes': path.stat().st_size,
            'verified_real_pairs': len(idx), 'max_cpu_reconstruction_error': maximum,
            'passed': passed, 'scope': 'Conditional type-pair log strength; connection occurrence is separate'}
        # Gauge-invariant dimension of the actual interaction, not the raw code
        # coordinates. QR keeps this in a rank x rank matrix, never 3000 x 3000.
        code = arrays['code'].astype(np.float64)
        q, r = np.linalg.qr(code, mode='reduced')
        a = generator.affinity.astype(np.float64)
        interaction = r @ a @ r.T
        u, spectrum, vt = np.linalg.svd(interaction, full_matrices=False)
        power = spectrum**2
        skew = .5*(interaction-interaction.T)
        verification[family]['interaction_antisymmetric_power_share'] = float(np.sum(skew*skew)/power.sum())
        verification[family]['interaction_singular_values'] = spectrum.tolist()
        verification[family]['interaction_power_effective_rank'] = float(power.sum()**2/(power@power))
        left, right = q @ u, q @ vt.T
        base = prediction.copy()
        for start in range(0,len(idx),4096):
            take = idx[start:start+4096]
            base[take] -= np.sum(generator._pre_code[data['pre'][take]]*code[data['post'][take]],axis=1)
        val = data['splits']['val']
        truncation = []
        for rank in (4,8,16,32,64,128,256):
            if rank > len(spectrum):
                continue
            approx = base[val].copy()
            for start in range(0,len(val),4096):
                take=val[start:start+4096]
                approx[start:start+len(take)] += np.sum(
                    left[data['pre'][take],:rank]*spectrum[:rank]*right[data['post'][take],:rank],axis=1)
            truncation.append({'rank':rank, 'interaction_power_retained':float(power[:rank].sum()/power.sum()),
                'validation_mse_without_refitting':float(np.mean((approx-data['target'][val])**2))})
        verification[family]['posthoc_validation_rank_truncation'] = truncation
        if not passed:
            raise RuntimeError('Deployment differs from the selected matching model')
    write_json(args.run/'generator_verification.json', verification)
    print(json.dumps({name:{k:row[k] for k in ('generator','bytes','verified_real_pairs',
        'max_cpu_reconstruction_error','passed','interaction_power_effective_rank')}
        for name,row in verification.items()}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--report', default='report.json')
    parser.add_argument('--work', type=Path, default=Path('F:/fly_stream_overflow/cgb_v1'))
    main(parser.parse_args())
