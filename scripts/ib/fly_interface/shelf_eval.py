"""Curious shelf v1: where did each selector spend its time, and what did it learn?  Reads <dir>/bookshelf.jsonl.
Per arm: share of segments by zone (first half / second half), learning progress measured per zone, probe NLL of each
zone at its first and last visit.
Usage: python shelf_eval.py <dir> [<dir> ...]
"""
import sys, json, numpy as np
from pathlib import Path
report = {}
for d in map(Path, sys.argv[1:]):
    rows = [json.loads(l) for l in open(d / 'bookshelf.jsonl')]
    zones = sorted(set(r['zone'] for r in rows))
    half = len(rows) // 2
    share = {z: [float(np.mean([r['zone'] == z for r in part])) for part in (rows[:half], rows[half:])] for z in zones}
    first, last = {}, {}
    for r in rows:
        first.setdefault((r['zone'], r['read']), r['probe_nll'])
        last[(r['zone'], r['read'])] = r['probe_nll']
    change = {}
    for z in zones:
        ks = [k for k in first if k[0] == z]
        change[z] = float(np.mean([last[k] - first[k] for k in ks]))
    report[d.name] = {'segments': len(rows), 'share_first_half_second_half': share, 'probe_nll_last_minus_first_by_zone': change}
    print(d.name, len(rows), 'segments')
    for z in zones:
        print(f'  {z:14s} share {share[z][0]:.2f} -> {share[z][1]:.2f}   probe change {change[z]:+.3f}')
json.dump(report, open('scratch/fly_stream/shelf_eval.json', 'w'), indent=1, ensure_ascii=False)
