"""Extract small, hash-bound supply ports from a trusted complete body.

Reads the rolling archive without restoring an engine or making a table copy.
All retained features are real stored episode/current-history observations.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
import time

import numpy as np
import torch


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--events', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    started = time.perf_counter()
    metadata = json.loads(a.checkpoint.with_name('last.metadata.json').read_text(encoding='utf8'))
    sha = digest(a.checkpoint)
    assert sha == metadata['checkpoint_sha256']
    manifest = metadata['config']['causal_archive']
    archive = Path(manifest['snapshot'])
    assert digest(archive) == manifest['sha256']
    events = []
    for file in a.events:
        for line in file.read_text(encoding='utf8').splitlines():
            r = json.loads(line)
            if r.get('autonomous_event', {}).get('kind') != 'external':
                continue
            e = r.get('causal_use', {}).get('finite_U', {})
            if not e.get('verified'):
                continue
            events.append(dict(cycle=r['cycle'], episode=e['observed_episode'],
                unknown=r['causal_use']['unknown'], choice=r['causal_use']['choice']))
    db = sqlite3.connect(archive.as_uri()+'?mode=ro', uri=True)
    schema = json.loads(db.execute("SELECT value FROM meta WHERE key='schema'").fetchone()[0])
    layouts, offset = {}, 0
    for k, spec in sorted(schema.items()):
        dtype = np.dtype(spec['dtype'])
        count = int(np.prod(spec['shape'])) if spec['shape'] else 1
        layouts[k] = spec, offset, count
        offset += count*dtype.itemsize
    episodes = []
    for receipt in events:
        row = db.execute('SELECT payload FROM episodes WHERE id=?', (receipt['episode'],)).fetchone()
        if row is None:
            raise ValueError('declared real event missing from source archive')
        blob = row[0]
        assert len(blob) == offset
        values = {}
        for name in ('source', 'action', 'mother_next', 'mother_valid', 'mother_stamp', 'last_write', 'count'):
            spec, start, count = layouts[name]
            values[name] = torch.from_numpy(np.frombuffer(blob, dtype=spec['dtype'],
                count=count, offset=start).reshape(spec['shape']).copy())
        assert int(values['count']) == 1 and bool(values['mother_valid'])
        assert int(values['mother_stamp']) == int(values['last_write'])
        episodes.append(dict(receipt=receipt, **values))
    db.close()
    with gzip.open(a.checkpoint, 'rb') as f:
        saved = torch.load(f, map_location='cpu', weights_only=False)
    model, engine = saved['model'], saved['engine']
    prefixes = ('encoder.', 'long_predictor.', 'mother_from_boundary.', 'relation_basis.', 'kanli.')
    result = dict(schema=1, cycle=saved['cycle'], checkpoint=str(a.checkpoint), checkpoint_sha256=sha,
        archive=str(archive), archive_sha256=manifest['sha256'],
        events=[dict(path=str(f), sha256=digest(f)) for f in a.events],
        model={k:v for k,v in model.items() if k.startswith(prefixes)},
        history=model['memory.history'], history_valid=model['memory.history_valid'],
        directions=torch.randn(model['encoder.weight'].shape[0], 32,
            generator=torch.Generator().manual_seed(20261122)),
        diagnostic_direction_seed=20261122, episodes=episodes,
        records=engine['native_value_bank']['records'],
        extractor_seconds=time.perf_counter()-started)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, a.output)
    print(json.dumps(dict(cycle=result['cycle'], episodes=len(episodes),
        history=list(result['history'].shape), directions=list(result['directions'].shape),
        encoder=list(model['encoder.weight'].shape), output=str(a.output),
        bytes=a.output.stat().st_size, seconds=result['extractor_seconds'])))


if __name__ == '__main__':
    main()
