"""Living gates in the mother world, on her normal replies (the words tagged 'mother_reply' in a --score-log).
The two arms of a gate live different conversations (the mouth answers what each fly said), so words are compared
through their own fixed word frequency: per window, excess = mean over her words of (pre-update NLL - unigram NLL).
  brain only:  is the brain better than the fixed word frequency on her replies?  mean excess (< 0 = better), its
               standard error over windows, the share of windows below 0, per-1024-word block [min | median | mean | max]
  brain + arm: brain contribution = excess(arm) - excess(brain) (> 0 = the removed part carries prediction), SE
               from both arms' windows.
Usage: python live_mother_gate_report.py <brain scores.jsonl> [<arm scores.jsonl> [label]]   -> one JSON line"""
import sys, json, os, numpy as np
TAGS = tuple(os.environ.get('MOTHER_TAGS', 'mother_reply').split(','))     # her normal replies (not calls or the opening)


def excess(path):
    out = []
    for line in open(path, encoding='utf-8'):
        if not line.strip():
            continue
        r = json.loads(line)
        src = r.get('sources')
        if not src:
            continue
        d = [(s - u, s, u) for s, u, tag in zip(r['scores'], r['unigram'], src) if tag in TAGS]
        if d:
            a = np.array(d)
            out.append((float(a[:, 0].mean()), len(d), float(a[:, 1].mean()), float(a[:, 2].mean())))
    return out


def summary(rows):
    if len(rows) < 2:
        return {'windows': len(rows)}
    x = np.array([r[0] for r in rows])
    words = int(sum(r[1] for r in rows))
    rep = {'windows': len(x), 'mother_words': words, 'excess': round(float(x.mean()), 4),
           'nll': round(float(np.mean([r[2] for r in rows])), 4), 'unigram': round(float(np.mean([r[3] for r in rows])), 4),
           'se': round(float(x.std(ddof=1) / np.sqrt(len(x))), 4), 'share_below_unigram': round(float((x < 0).mean()), 3)}
    per = max(1, 1024 // max(1, words // len(x)))          # windows per ~1024 of her words
    blocks = [float(x[i:i + per].mean()) for i in range(0, len(x), per) if len(x[i:i + per]) == per]
    if blocks:
        rep['per_block_min_median_mean_max'] = [round(float(f(blocks)), 4) for f in (np.min, np.median, np.mean, np.max)]
    return rep


if __name__ == '__main__':
    line = '--line' in sys.argv
    argv = [a for a in sys.argv[1:] if a != '--line']
    b_rows = excess(argv[0])
    brain = summary(b_rows)
    rep = {'brain': brain}
    if len(argv) > 1:
        a_rows = excess(argv[1])
        arm = summary(a_rows)
        rep['arm'], rep['label'] = arm, (argv[2] if len(argv) > 2 else 'arm')
        if 'excess' in brain and 'excess' in arm:
            rep['contribution'] = round(arm['excess'] - brain['excess'], 4)
            rep['contribution_se'] = round(float(np.hypot(arm['se'], brain['se'])), 4)
            m = min(len(a_rows), len(b_rows))           # windows paired by their order (both start at c-1)
            d = np.array([a_rows[i][0] - b_rows[i][0] for i in range(m)])
            rep['paired_min_median_mean_max'] = [round(float(f(d)), 4) for f in (np.min, np.median, np.mean, np.max)]
    if line and 'contribution' in rep:
        # the dashboard's gate line (scripts/ib/dashboard/server.py GATE)
        mm = rep['paired_min_median_mean_max']
        print(f"(1) brain contribution {rep['contribution']:+.3f} +- {rep['contribution_se']:.3f} [min {mm[0]:+.2f} med "
              f"{mm[1]:+.2f} max {mm[3]:+.2f}] on her replies, excess over unigram (brain {brain['nll']}, "
              f"{rep['label']} {arm['nll']}, unigram {brain['unigram']}, {brain['mother_words']} words)")
    else:
        print(json.dumps(rep))
