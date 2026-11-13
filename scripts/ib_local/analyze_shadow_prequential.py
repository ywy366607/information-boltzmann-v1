"""Summarize actual decision-time shadow forecasts against later outcomes."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def summarize(path, *, skip_first_score=False):
    rows = [json.loads(line) for line in path.read_text(encoding='utf8').splitlines() if line.strip()]
    score_rows = rows[1:] if skip_first_score else rows
    scored = [r['shadow_preview']['settled_candidate_prediction'] for r in score_rows
              if r['shadow_preview']['settled_candidate_prediction'] is not None]
    fields = ('baseline_mse', 'candidate_mean_mse', 'candidate_weighted_mse',
              'candidate_best_mse', 'candidate_spread')
    means = {k: float(np.mean([x[k] for x in scored])) for k in fields} if scored else {}
    deltas = [x['candidate_mean_mse']-x['baseline_mse'] for x in scored]
    offsets = [max(map(abs, r['shadow_preview']['delivered_offset'])) for r in rows]
    return dict(cycles=len(rows), first_cycle=rows[0]['cycle'] if rows else None,
        last_cycle=rows[-1]['cycle'] if rows else None,
        choices=dict(Counter(r['choice'] for r in rows)), settled_candidates=len(scored),
        first_score_excluded=skip_first_score,
        means=means, candidate_mean_wins=sum(d < 0 for d in deltas),
        paired_mean_delta=float(np.mean(deltas)) if deltas else None,
        nonzero_delivered=sum(x > 0 for x in offsets),
        max_delivered=max(offsets, default=0.),
        last_calibration_counts=rows[-1]['shadow_preview']['calibration_counts'] if rows else None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('metrics', type=Path)
    p.add_argument('--skip-first-score', action='store_true',
                   help='exclude a settlement inherited from the previous candidate mechanism')
    args = p.parse_args()
    print(json.dumps(summarize(args.metrics, skip_first_score=args.skip_first_score), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
