"""Read recorded optimizer tensor payloads and native edge weights, offline."""
import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path
import zlib
import numpy as np


def tensor(handle, ref):
    if ref is None:
        return None
    r = ref['tensor_ref']
    handle.seek(r['offset'])
    raw = zlib.decompress(handle.read(r['compressed_bytes']))
    if hashlib.sha256(raw).hexdigest() != r['sha256']:
        raise ValueError('tensor checksum mismatch')
    return np.broadcast_to(np.frombuffer(raw, dtype=r['dtype']).reshape(r['stored_shape']), r['shape'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('work', type=Path)
    ap.add_argument('output', type=Path)
    args = ap.parse_args()
    steps, events = [], []
    for seg, directory in enumerate(sorted(args.work.glob('collect_observations*'))):
        kinds = {}
        with gzip.open(directory/'operations.jsonl.gz', 'rt', encoding='utf8') as stream:
            rows = [json.loads(line) for line in stream]
        for row in rows:
            if row['kind'] == 'event':
                kinds[row['event']] = row['data'].get('autonomous_event', {}).get('kind')
        with (directory/'tensors.bin').open('rb') as handle:
            for row in rows:
                if row['kind'] != 'optimizer':
                    continue
                d = row['data']; before = tensor(handle,d['theta_before']); after = tensor(handle,d['theta_after'])
                delta = tensor(handle,d['applied_delta']); grad = tensor(handle,d['actual_gradient'])
                gate = tensor(handle,d['budget_gate'])
                if not np.array_equal(after-before,delta):
                    raise ValueError('recorded theta delta mismatch')
                steps.append({'segment':seg,'event':int(row['event']),'update_clock':int(row['update_clock']),
                    'kind':kinds.get(row['event']),'gradient_present':grad is not None,
                    'gradient_l2':float(np.linalg.norm(grad)) if grad is not None else None,
                    'gate':float(gate) if gate is not None and gate.size == 1 else None,
                    'frequency_delta_l2':float(np.linalg.norm(delta[:,:8])),
                    'context_delta_l2':float(np.linalg.norm(delta[:,8])),
                    'strength_delta_l2':float(np.linalg.norm(delta[:,9])),
                    'edge_content_delta_l2':float(np.linalg.norm(delta[:,10:12])),
                    'changed_layers':int(np.count_nonzero(np.any(delta != 0,axis=1)))})
        before = {}
        for path in sorted(directory.glob('physical_*.npz')):
            with np.load(path,allow_pickle=False) as data:
                for i,kind in enumerate(data['kind']):
                    e = int(data['event'][i]); key=(seg,e)
                    if kind == 'event_before':
                        before[key]=(data['edge_weight'][i].copy(),data['theta'][i].copy())
                    if kind == 'event_after' and key in before:
                        oldw,oldtheta=before.pop(key)
                        events.append({'segment':seg,'event':e,'kind':kinds.get(e),
                            'edge_weight_delta_l2':float(np.linalg.norm(data['edge_weight'][i]-oldw)),
                            'edge_theta_delta_l2':float(np.linalg.norm(data['theta'][i,:,10:12]-oldtheta[:,10:12]))})
    active = [s for s in steps if s['edge_content_delta_l2']>0]
    out = {'optimizer_steps':len(steps),'edge_coefficient_nonzero_steps':len(active),
           'gradient_present_steps':sum(s['gradient_present'] for s in steps),
           'event_pairs':len(events),'steps':steps,'events':events,
           'note':'Same event edge-weight differences include S and theta changes; do not attribute to either without counterfactual reconstruction. Theta gates and steps are actual recorded values.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in out.items() if k not in ('steps','events')},ensure_ascii=False))


if __name__ == '__main__':
    main()
