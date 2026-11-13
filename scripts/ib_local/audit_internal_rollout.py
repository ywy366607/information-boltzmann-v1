"""Audit real recorded transitions and sequentially restored complete states."""
import argparse
import gc
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def state_digest(value):
    h = hashlib.sha256()

    def visit(x):
        h.update(type(x).__name__.encode())
        if torch.is_tensor(x):
            visit(x.detach().cpu().contiguous().numpy())
        elif isinstance(x, np.ndarray):
            h.update(str((x.shape, x.dtype)).encode())
            array = np.ascontiguousarray(x)
            view = memoryview(array).cast('B')
            for start in range(0, len(view), 1024*1024):
                h.update(view[start:start+1024*1024])
        elif isinstance(x, dict):
            for k in sorted(x, key=repr):
                visit(k); visit(x[k])
        elif isinstance(x, (tuple, list)):
            for v in x: visit(v)
        elif isinstance(x, torch.nn.Module):
            visit(x.state_dict())
        elif isinstance(x, np.random.Generator):
            visit(x.bit_generator.state)
        elif hasattr(x, '__dict__'):
            visit(vars(x))
        else:
            h.update(repr(x).encode())
    visit(value)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--arms', nargs=2, default=('resume_a', 'resume_b'))
    p.add_argument('--repair-initial-roundoff', action='store_true',
                   help='remove the one observed spurious calibration sample, retaining the individual')
    args = p.parse_args()
    torch.set_num_threads(1)
    columns = ('model', 'optimizer', 'engine', 'table', 'world',
               'torch_rng', 'cuda_rng', 'numpy_rng', 'python_rng')
    hashes, cycles = {}, {}
    left, right = args.arms
    for name in args.arms:
        with gzip.open(args.root/name/'last.pt.gz', 'rb') as stream:
            state = torch.load(stream, map_location='cpu', weights_only=False)
        hashes[name] = {k: state_digest(state[k]) for k in columns}
        cycles[name] = state['cycle']
        del state
        gc.collect()
    rows = {}
    for name in ('admission4', left, right):
        rows[name] = [json.loads(l) for l in (args.root/name/'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
    differing = [k for k, v in rows[left][0].items() if v != rows[right][0][k]]
    report = dict(
        state_equal={k: hashes[left][k] == hashes[right][k] for k in columns},
        state_sha256=hashes, cycles=cycles, differing_metric_fields=differing,
        transitions=[{k: r.get(k) for k in ('cycle', 'choice', 'internal_transition_loss',
            'internal_transition_samples', 'internal_rollout', 'policy_replay_fallback')}
            for r in rows['admission4']+rows[left]],
        scope='real-interface and complete restoration only; no learning-benefit claim')
    (args.root/'readiness.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('state_equal', 'cycles', 'differing_metric_fields')}, indent=2))
    assert all(report['state_equal'].values()) and differing == ['elapsed_seconds'], report['state_equal']
    if args.repair_initial_roundoff:
        # Specific repair of this recorded admission, never a general reset.
        for r in report['transitions']:
            assert not any(r['internal_rollout']['proposed_offset'])
        with gzip.open(args.root/right/'last.pt.gz', 'rb') as stream:
            state = torch.load(stream, map_location='cpu', weights_only=False)
        calibration = state['engine']['internal_gain_state']
        assert calibration['count'].tolist() == [0, 1, 0]
        assert 0 < calibration['second'][1] < 1e-15
        before = {k: np.array(calibration[k], copy=True).tolist()
                  for k in ('cross', 'second', 'error', 'count')}
        without = lambda: state_digest({k: v for k, v in state['engine'].items()
                                        if k != 'internal_gain_state'})
        carried_sha = without()
        for k in before:
            calibration[k].fill(0)
        assert without() == carried_sha
        repair = dict(at_cycle=state['cycle'], removed=before,
            reason='one equal-value weighted-sum roundoff sample; all delivered offsets were zero',
            retained='all parameters, optimizer, S, tables, RNG, world, real transition tags and pending settlement')
        state['config']['internal_rollout']['initial_roundoff_repair'] = repair
        output = args.root/'selected'
        output.mkdir(exist_ok=False)
        from scripts.ib_local.mother_checkpoint import ChunkedWriter
        with gzip.open(output/'last.tmp.gz', 'wb', compresslevel=1) as stream:
            torch.save(state, ChunkedWriter(stream))
        (output/'last.tmp.gz').replace(output/'last.pt.gz')
        report['selected'] = dict(path=str(output/'last.pt.gz'), repair=repair)
        (args.root/'readiness.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
