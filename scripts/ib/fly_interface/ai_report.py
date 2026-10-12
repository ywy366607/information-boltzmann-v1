"""Paired comparisons for the active-inference milestones (same text, same intervals per mu, so words pair one to one).
Usage: python ai_report.py <a.jsonl>:<mode>:<label> <b.jsonl>:<mode>:<label> [metric]
Prints, per condition present in both, mean(a) - mean(b) of the per-word metric (default survival; also arrival or
answer_ce) with a 95% bootstrap interval, plus min | median | mean | max of each side.
"""
import sys, json, numpy as np


def load(spec):
    path, mode, label = spec.rsplit(':', 2)
    rows = {}
    for line in open(path):
        r = json.loads(line)
        if r['mode'] == mode:
            rows[r.get('condition') or f"geo{r['mu']}"] = r      # the last run of a condition wins
    return rows, label


def main():
    specs = [a for a in sys.argv[1:] if a.count(':') >= 2]
    metric = next((a for a in sys.argv[1:] if a.count(':') < 2), 'survival')
    (A, la), (B, lb) = load(specs[0]), load(specs[1])
    rng = np.random.default_rng(0)
    for cond in A:
        if cond not in B:
            continue
        a = np.asarray(A[cond][f'per_word_{metric}'], dtype=float)
        b = np.asarray(B[cond][f'per_word_{metric}'], dtype=float)
        keep = np.isfinite(a) & np.isfinite(b)                 # words that are questions in both (M3: main words)
        a, b = a[keep], b[keep]
        diff = a - b
        boot = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        q = lambda x: f'{x.min():.2f} | {np.median(x):.2f} | {x.mean():.3f} | {x.max():.2f}'
        print(f'{cond:>14} {metric}: {la} - {lb} = {diff.mean():+.4f}  [95% {lo:+.4f}, {hi:+.4f}]   '
              f'{la}: {q(a)}   {lb}: {q(b)}')


if __name__ == '__main__':
    main()
