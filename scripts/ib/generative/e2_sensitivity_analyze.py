"""Curate immutable E2 sensitivity runs, with paired block descriptions."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


RUNS = {
    'newborn_128': 'cgb_e2_sensitivity_20261021',
    'newborn_512': 'cgb_e2_sensitivity_replication_20261021',
    'learned_128': 'cgb_e2_mature_admission_20261021',
    'learned_512': 'cgb_e2_mature_replication_20261021',
    'sensory_256': 'cgb_e2_sensory_localization_20261021',
}


def paired_description(delta, block_size=32):
    """Contiguous-block resampling, not independent-token confidence claims."""
    delta = np.asarray(delta, dtype=float)
    if delta.ndim != 1 or not np.isfinite(delta).all():
        raise ValueError('A finite paired sequence is required')
    if len(delta) % block_size:
        raise ValueError('Use complete blocks; do not silently drop exposures')
    blocks = delta.reshape(-1, block_size).mean(axis=1)
    rng = np.random.default_rng(20261021)
    means = blocks[rng.integers(len(blocks), size=(5000, len(blocks)))].mean(axis=1)
    return {'paired_mean': float(delta.mean()),
            'block_size': block_size, 'blocks': len(blocks),
            'descriptive_block_resample_95pct': np.quantile(means, [.025, .975]).tolist(),
            'second_half_mean': float(delta[len(delta)//2:].mean()),
            'scope': 'Within this finite stream segment; not convergence or cross-stream generalization'}


def analyze(root):
    runs = {}
    for name, directory in RUNS.items():
        folder = root / directory
        report = json.loads((folder/'report.json').read_text(encoding='utf-8'))
        if not report['checkpoint_unchanged'] or not report['sham_pass']:
            raise ValueError(f'Failed source/restore control: {folder}')
        real = np.load(folder/'real.npz', allow_pickle=False)['nll']
        with np.load(folder/'exposure.npz', allow_pickle=False) as exposure:
            if len(exposure['targets']) != len(real):
                raise ValueError('Score and exposure length differ')
            ticks = int(exposure['ticks'].sum())
        if ticks != report['arms']['real']['physical_ticks']:
            raise ValueError('Clock and report disagree')
        comparisons = {}
        for arm in report['arms']:
            values = np.load(folder/(arm+'.npz'), allow_pickle=False)['nll']
            if values.shape != real.shape:
                raise ValueError('Unpaired scores')
            comparisons[arm] = paired_description(values-real)
        for baseline in ('bias_baseline', 'constant_baseline'):
            if (folder/(baseline+'.npz')).exists():
                values = np.load(folder/(baseline+'.npz'), allow_pickle=False)['nll']
                if values.shape != real.shape:
                    raise ValueError('Baseline score length mismatch')
                comparisons[baseline] = paired_description(values-real)
        runs[name] = {'raw_directory': str(folder), 'report': report, 'comparisons': comparisons}
    return {'created_utc': datetime.now(timezone.utc).isoformat(),
            'scope': 'Historical frozen-parameter causal diagnostics, not generated-brain learning acceptance',
            'analysis_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'runs': runs}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: v['comparisons'] for k, v in result['runs'].items()}, indent=2))
