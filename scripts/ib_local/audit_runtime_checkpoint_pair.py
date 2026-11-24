"""Check full world/RNG and pending delayed actor after matched real events."""
import argparse
import gc
import gzip
import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
import torch


def digest(value):
    h = hashlib.sha256()
    def feed(v):
        if torch.is_tensor(v) or isinstance(v, np.ndarray):
            array = v.detach().contiguous().numpy() if torch.is_tensor(v) else np.ascontiguousarray(v)
            h.update(str((array.dtype.str, array.shape)).encode())
            data = memoryview(array).cast('B')
            for i in range(0, len(data), 1024*1024):
                h.update(data[i:i+1024*1024])
        elif isinstance(v, dict):
            h.update(b'dict')
            for k in sorted(v):
                feed(k); feed(v[k])
        elif isinstance(v, (tuple, list)):
            h.update(str(type(v)).encode())
            for item in v: feed(item)
        else:
            h.update(repr(v).encode())
    feed(value)
    return h.hexdigest()


def read(path, *, exact_state=False):
    with gzip.open(path, 'rb') as f:
        checkpoint = torch.load(f, map_location='cpu', weights_only=False)
    actor = checkpoint['engine']['prev']['actor']
    hashes = {k:digest(checkpoint[k]) for k in ('world','torch_rng','cuda_rng','numpy_rng','python_rng')}
    if exact_state:
        for key in ('model', 'optimizer', 'table'):
            hashes[key] = digest(checkpoint[key])
        hashes['recurrent'] = digest(checkpoint['engine']['recurrent'])
        snapshot = checkpoint['config']['causal_archive']['snapshot']
        with sqlite3.connect(Path(snapshot).resolve().as_uri()+'?mode=ro', uri=True) as db:
            hashes['causal_archive_sql'] = digest(list(db.iterdump()))
    cycle = checkpoint['cycle']
    del checkpoint
    gc.collect()
    return actor, hashes, cycle


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--before', type=Path, required=True)
    p.add_argument('--after', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--exact-state', action='store_true')
    a = p.parse_args()
    old, old_hash, cycle = read(a.before, exact_state=a.exact_state)
    new, new_hash, new_cycle = read(a.after, exact_state=a.exact_state)
    maximum, squared, reference = 0., 0., 0.
    for u, v in zip(old, new):
        if u is None or v is None:
            if u is not None or v is not None:
                raise RuntimeError('delayed actor participation changed')
            continue
        error = u.double()-v.double()
        maximum = max(maximum, float(error.abs().max()))
        squared += float(error.square().sum())
        reference += float(u.double().square().sum())
    relative = (squared/max(reference, 1e-30))**.5
    report = dict(schema=1, cycle=cycle, same_cycle=cycle == new_cycle,
        equal={k:old_hash[k] == new_hash[k] for k in old_hash},
        pending_actor_max_error=maximum, pending_actor_relative_l2=relative,
        scope='full saved world, RNGs and future delayed actor; zero learning')
    if (not report['same_cycle'] or not all(report['equal'].values()) or
            len(old) != len(new) or relative > 1e-4):
        raise RuntimeError(f'complete continuation mismatch: {report}')
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
