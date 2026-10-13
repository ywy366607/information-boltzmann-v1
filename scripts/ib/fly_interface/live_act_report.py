"""Living causal gate (3): do the acts read uncertainty, in the learning individual's own stream?  From the trainer's
ai_trace.jsonl (one row per window: per word its interval and arrival CE, per tick the commit hazard of the motor acts).
Per word, from its ticks' hazards h_1..h_k (exact, no sampling): P(commit at t) = h_t prod_{s<t} (1 - h_s);
answered = sum_t P(t); expected latency among answered; latency entropy (diversity of timing).
Reports per 1024-word block [min | median | mean | max] of: answered fraction, expected latency, latency entropy,
Spearman(expected latency, the word's CE) (adaptivity: harder -> later), and the same for the previous word's CE.
Usage: python live_act_report.py <ai_trace.jsonl> [first_window_trained]"""
import sys, json, numpy as np


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float('nan')
    ra, rb = np.argsort(np.argsort(a[ok])), np.argsort(np.argsort(b[ok]))
    return float(np.corrcoef(ra, rb)[0, 1]) if ra.std() > 0 and rb.std() > 0 else float('nan')


rows = [json.loads(l) for l in open(sys.argv[1], encoding='utf-8') if l.strip()]
start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
rows = [r for r in rows if r.get('hazard') is not None and r['bptt_train_tokens'] > start]
words = []                                   # (answered, latency, entropy, ce, prev_ce)
for r in rows:
    hz = np.asarray(r['hazard'], float).reshape(-1)
    if hz.ndim > 1:
        hz = hz[:, 0]
    pos, prev = 0, float('nan')
    for k, ce in zip(r['interval'], r['ce']):
        h = np.clip(hz[pos:pos + k], 0.0, 1.0); pos += k
        survive = np.concatenate([[1.0], np.cumprod(1.0 - h)[:-1]])
        p = h * survive
        ans = float(p.sum())
        lat = float((p * np.arange(1, k + 1)).sum() / ans) if ans > 1e-9 else float('nan')
        q = p / ans if ans > 1e-9 else p
        ent = float(-(q[q > 0] * np.log(q[q > 0])).sum())
        words.append((ans, lat, ent, float(ce), prev))
        prev = float(ce)
w = np.array(words)
q = lambda v: [round(float(x), 4) for x in (np.nanmin(v), np.nanmedian(v), np.nanmean(v), np.nanmax(v))]
blocks = [w[i:i + 1024] for i in range(0, len(w) - 1023, 1024)] or [w]
per = {'answered': [np.nanmean(b[:, 0]) for b in blocks], 'latency': [np.nanmean(b[:, 1]) for b in blocks],
       'latency_entropy': [np.nanmean(b[:, 2]) for b in blocks],
       'rho_latency_ce': [spearman(b[:, 1], b[:, 3]) for b in blocks],
       'rho_latency_prev_ce': [spearman(b[:, 1], b[:, 4]) for b in blocks]}
print(json.dumps({'words': int(len(w)), **{k: q(v) for k, v in per.items()},
                  'rho_latency_ce_all': round(spearman(w[:, 1], w[:, 3]), 4)}))
