"""Living causal gate: an ablated arm against the living brain on the same stream (same words, same world, learning on).
Each file is a --score-log JSONL (one row per window: pre-update word scores and the unigram on the same words).
Paired by window: d = NLL(arm) - NLL(brain); > 0 means the removed part carries prediction.
Usage: python live_gate_report.py <brain scores.jsonl> <arm scores.jsonl> [label]   -> one JSON line"""
import sys, json, numpy as np


def windows(path):
    rows = [json.loads(l) for l in open(path, encoding='utf-8') if l.strip()]
    return {r['trained']: (float(np.mean(r['scores'])), float(np.mean(r['unigram']))) for r in rows}


def compare(brain_path, arm_path):
    b, a = windows(brain_path), windows(arm_path)
    keys = sorted(set(b) & set(a))
    if len(keys) < 2:
        return {'windows': len(keys)}
    d = np.array([a[k][0] - b[k][0] for k in keys])
    return {'windows': len(keys), 'words': len(keys) * 32,
            'mean': round(float(d.mean()), 4), 'se': round(float(d.std(ddof=1) / np.sqrt(len(d))), 4),
            'min_median_mean_max': [round(float(x), 4) for x in (d.min(), np.median(d), d.mean(), d.max())],
            'brain_nll': round(float(np.mean([b[k][0] for k in keys])), 4),
            'arm_nll': round(float(np.mean([a[k][0] for k in keys])), 4),
            'unigram': round(float(np.mean([b[k][1] for k in keys])), 4)}


if __name__ == '__main__':
    rep = compare(sys.argv[1], sys.argv[2])
    rep['label'] = sys.argv[3] if len(sys.argv) > 3 else 'arm'
    print(json.dumps(rep))
