"""Prequential fixed-history audit of fast action-conditioned gain estimates.

All means use only earlier committed, unmerged, version-valid observed rewards.
This inspects the individual's existing water; no training or policy is changed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from scripts.ib_local.run_causal_water_reply import build


@torch.no_grad()
def audit(engine, n_eval=32, horizons=(8, 32, 128)):
    m = engine.model.memory
    eligible = (m.occupied & (m.count == 1) & (m.wood_target_count == 1) &
                (m.wood_option >= 0) & engine.reward_label_valid(wood=True))
    slots = torch.nonzero(eligible).flatten()
    slots = slots[m.last_write[slots].argsort()]
    events = [(int(m.last_write[i]), int(m.wood_option[i]),
               float(m.wood_target[i, 3]), float(m.wood_baseline[i, 3])) for i in slots]
    if len(events) <= n_eval:
        raise ValueError('not enough retained real events for prequential audit')
    if len({x[0] for x in events}) != len(events):
        raise ValueError('ambiguous event versions; do not infer temporal order')
    history = [[], [], []]
    # Score candidates only after each real target arrives; bounded 32-event
    # errors. The small denominator floor has units of squared gain.
    scores = {str(k): [] for k in horizons}
    observed = []
    for pos, (stamp, option, target, original) in enumerate(events):
        if not 0 <= option < 3:
            raise ValueError('invalid observed action label')
        past = history[option]
        predictions = {str(k): sum(past[-k:])/min(len(past), k)
                       for k in horizons if past}
        supported = [k for k in predictions if scores[k]]
        weights = {}
        mixture = None
        if supported:
            precision = {k: 1/(sum(scores[k])/len(scores[k])+1e-6) for k in supported}
            total = sum(precision.values())
            weights = {k: value/total for k,value in precision.items()}
            mixture = sum(weights[k]*predictions[k] for k in supported)
        if pos >= len(events)-n_eval:
            observed.append(dict(stamp=stamp, option=option, target=target,
                                 original=original, estimates=predictions,
                                 adaptive_mixture=mixture, adaptive_weights=weights,
                                 available_history=len(past)))
        for k,p in predictions.items():
            scores[k].append((p-target)**2)
            scores[k] = scores[k][-32:]
        past.append(target)
    metrics = {}
    for name in ('original', 'adaptive_mixture', *(str(k) for k in horizons)):
        pairs = []
        for r in observed:
            p = r[name] if name in ('original', 'adaptive_mixture') else r['estimates'].get(name)
            if p is not None:
                pairs.append((p, r['target']))
        if pairs:
            metrics[name] = dict(n=len(pairs),
                mae=sum(abs(p-y) for p,y in pairs)/len(pairs),
                mse=sum((p-y)**2 for p,y in pairs)/len(pairs),
                bias=sum(p-y for p,y in pairs)/len(pairs))
    return dict(eligible=len(events), evaluated=len(observed), horizons=list(horizons),
                metrics=metrics, events=observed,
                scope='selected retained history only; observed actions fixed; no policy rollout')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('refusing to overwrite existing audit')
    torch.set_num_threads(1)
    engine, _, _, provenance = build(args.checkpoint, device='cpu')
    output = dict(source=provenance, audit=audit(engine))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=True)+'\n', encoding='utf-8')
    print(json.dumps({'source_cycle': provenance['cycle'], 'metrics': output['audit']['metrics']},
                     ensure_ascii=True))


if __name__ == '__main__':
    main()
