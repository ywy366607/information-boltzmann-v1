"""Compare two independently resumed, cache-matched matrix-momentum states."""
import argparse
import gc
import gzip
import json
from pathlib import Path

import torch

from scripts.ib_local.audit_internal_rollout import state_digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--left', type=Path, required=True)
    p.add_argument('--right', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('output already exists')
    keys = ('model', 'optimizer', 'engine', 'table', 'world', 'torch_rng',
            'cuda_rng', 'numpy_rng', 'python_rng')
    hashes, cycles, metrics, world_keys = [], [], [], []
    for directory in (args.left, args.right):
        with gzip.open(directory/'last.pt.gz', 'rb') as f:
            state = torch.load(f, map_location='cpu', weights_only=False)
        hashes.append({k: state_digest(state[k]) for k in keys})
        world_keys.append({k: state_digest(v) for k, v in state['world'].items()})
        cycles.append(state['cycle'])
        del state
        gc.collect()
        rows = (directory/'metrics.jsonl').read_text(encoding='utf-8').splitlines()
        metrics.append(json.loads(rows[-1]))
    different = [k for k in metrics[0] if metrics[0][k] != metrics[1][k]]
    report = dict(cycles=cycles, state_equal={k: hashes[0][k] == hashes[1][k] for k in keys},
                  state_sha256=hashes, differing_metrics=different,
                  differing_world_keys=[k for k in world_keys[0] if world_keys[0][k] != world_keys[1].get(k)])
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report['state_equal'], ensure_ascii=False))
    if cycles[0] != cycles[1] or not all(report['state_equal'].values()) or set(different)-{'elapsed_seconds'}:
        raise RuntimeError('restored transition differs; evidence retained in report')


if __name__ == '__main__':
    main()
