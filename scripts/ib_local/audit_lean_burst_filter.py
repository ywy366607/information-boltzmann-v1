"""Exploratory walk-forward audit of a Lean-inspired innovation filter.

Uses only previously observed innovations to scale each new one. The real
historical actions and outcomes are fixed; no model, optimizer or policy is run.
The mixture parameters are explicitly chosen hypotheses, not Lean premises.
"""
from __future__ import annotations

import json
import math
import statistics

from scripts.ib_local.audit_shadow_gain_trace import BASE, RUNS


def run(rows):
    rows = sorted(rows, key=lambda r: r['cycle'])
    if [r['cycle'] for r in rows] != list(range(2945, 2994)):
        raise ValueError('requires the complete recorded 2945--2993 sequence')
    means = [float(v[3]) for v in rows[0]['wood_all_bases']]
    history, results = [], []
    for r in rows:
        option = {'silence': 0, 'generate': 1, 'reuse': 2}[r['choice']]
        target = float(r['wood_target'][3])
        innovation = target-means[option]
        weight = 1.
        if len(history) >= 8:
            center = statistics.median(history)
            scale = max(1e-6, 1.4826*statistics.median(abs(v-center) for v in history))
            standardized = (innovation-center)/scale
            # Posterior narrow-component probability: zero-mean Gaussian
            # residual mixture, broad:narrow SD 10:1, broad prior 0.1.
            z = math.log(1/90) + .495*min(standardized**2, 2000.)
            if z >= 0:
                exp_neg = math.exp(-z)
                weight = exp_neg/(1+exp_neg)
            else:
                weight = 1/(1+math.exp(z))
        results.append(dict(cycle=r['cycle'], option=option, target=target,
                            original=float(r['wood_base'][3]), candidate=means[option],
                            innovation=innovation, weight=weight))
        # Same joint common/action geometry as SharedActionEvidence.
        means = [v+.05*(1. if i == option else .4)*weight*innovation
                 for i,v in enumerate(means)]
        history.append(innovation)
        history = history[-32:]
    phases = []
    for lo, hi in ((2945,2958), (2960,2968), (2969,2976),
                   (2977,2984), (2985,2993), (2960,2993)):
        subset = [r for r in results if lo <= r['cycle'] <= hi]
        phases.append(dict(cycles=[lo,hi], n=len(subset),
            original_mae=statistics.mean(abs(r['target']-r['original']) for r in subset),
            candidate_mae=statistics.mean(abs(r['target']-r['candidate']) for r in subset),
            original_mse=statistics.mean((r['target']-r['original'])**2 for r in subset),
            candidate_mse=statistics.mean((r['target']-r['candidate'])**2 for r in subset)))
    return dict(scope='exploratory fixed realized trajectory; not an interactive policy comparison',
                assumptions=dict(broad_prior=.1, broad_sd_ratio=10, history=32,
                                 warmup=8, robust_scale_floor=1e-6),
                phases=phases, rows=results)


def main():
    rows = []
    for name in RUNS:
        rows.extend(json.loads(line) for line in (BASE/name/'metrics.jsonl').read_text(
            encoding='utf-8').splitlines() if line.strip())
    print(json.dumps(run(rows), indent=2))


if __name__ == '__main__':
    main()
