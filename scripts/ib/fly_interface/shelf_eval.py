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
    whole = {z: float(np.mean([r['zone'] == z for r in rows])) for z in zones}
    gains, prev = {}, {}
    for r in rows:                                   # single-visit gains: the raw progress measurements
        if r['read'] in prev:
            gains.setdefault(r['zone'], []).append(prev[r['read']] - r['probe_nll'])
        prev[r['read']] = r['probe_nll']
    gain_stats = {z: [float(np.mean(g)), float(np.std(g)), len(g)] for z, g in gains.items()}
    report[d.name] = {'segments': len(rows), 'share_whole': whole, 'share_first_half_second_half': share,
                      'probe_nll_last_minus_first_by_zone': change, 'single_visit_gain_mean_sd_n': gain_stats}
    print(d.name, len(rows), 'segments')
    for z in zones:
        g = gain_stats.get(z, [float('nan')] * 2 + [0])
        print(f'  {z:14s} share {whole[z]:.2f} ({share[z][0]:.2f} -> {share[z][1]:.2f})   probe change {change[z]:+.3f}   gain {g[0]:+.3f} +- {g[1]:.3f} (n {g[2]})')
json.dump(report, open(sys.argv[1].rstrip('/') + '_eval.json' if len(sys.argv) > 1 else 'shelf_eval.json', 'w'), indent=1, ensure_ascii=False)
