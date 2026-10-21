"""E2 report: newborns with different mushroom-body wiring, the same text stream, word by word.

Reads <root>/<arm>/scores.jsonl (per word: the brain readout's NLL 'scores' and the fixed unigram's 'unigram').
Per arm: NLL [min | median | mean | max], mean NLL - unigram by block (learning curve), and the paired difference to
the real-wiring newborn on the same words with a block-bootstrap 95% interval (blocks of 32 words; words are
identical across arms, so the difference cancels word difficulty).
Usage: python e2_report.py [root] [--arms real,typelevel,generator,null]
"""
import argparse, json
from pathlib import Path
import numpy as np


def load(path):
    s, u = [], []
    for line in path.read_text(encoding='utf-8').splitlines():
        r = json.loads(line)
        s += r['scores']; u += r['unigram']
    return np.array(s), np.array(u)


def boot(d, block=32, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    m = len(d) // block
    b = d[:m * block].reshape(m, block).mean(1)
    idx = rng.integers(0, m, (n, m))
    means = b[idx].mean(1)
    return float(d[:m * block].mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root', nargs='?', default='F:/fly_stream_overflow/cgb_v1/e2')
    ap.add_argument('--arms', default='real,typelevel,generator,null')
    ap.add_argument('--block', type=int, default=256)
    a = ap.parse_args()
    root = Path(a.root)
    data = {arm: load(root / arm / 'scores.jsonl') for arm in a.arms.split(',') if (root / arm / 'scores.jsonl').exists()}
    n = min(len(s) for s, _ in data.values())
    rep = {'words': n, 'arms': {}}
    for arm, (s, u) in data.items():
        s, u = s[:n], u[:n]
        blocks = [round(float((s[i:i + a.block] - u[i:i + a.block]).mean()), 3) for i in range(0, n, a.block)]
        e = {'nll': {'min': round(float(s.min()), 3), 'median': round(float(np.median(s)), 3), 'mean': round(float(s.mean()), 3),
                     'max': round(float(s.max()), 3)},
             'minus_unigram_by_block': blocks}
        if arm != 'real' and 'real' in data:
            d = s - data['real'][0][:n]
            mean, lo, hi = boot(d)
            e['minus_real'] = {'mean': round(mean, 4), 'ci95': [round(lo, 4), round(hi, 4)]}
            late = d[n // 2:]
            mean, lo, hi = boot(late)
            e['minus_real_second_half'] = {'mean': round(mean, 4), 'ci95': [round(lo, 4), round(hi, 4)]}
        rep['arms'][arm] = e
    print(json.dumps(rep, indent=1))
    (root / 'e2_report.json').write_text(json.dumps(rep, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
