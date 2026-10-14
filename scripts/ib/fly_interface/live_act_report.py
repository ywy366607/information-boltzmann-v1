"""Living causal gate (3): do the acts read uncertainty, in the learning individual's own stream?  From the trainer's
ai_trace.jsonl (one row per window: per word its interval and arrival CE, per tick the commit hazard of the motor acts).
Per word, from its ticks' hazards h_1..h_k (exact, no sampling): P(commit at t) = h_t prod_{s<t} (1 - h_s);
answered = sum_t P(t); expected latency among answered; latency entropy (diversity of timing).
Reports per 1024-word block [min | median | mean | max] of: answered fraction, expected latency, latency entropy,
Spearman(expected latency, the word's CE) (adaptivity: harder -> later), and the same for the previous word's CE.
Usage: python live_act_report.py <ai_trace.jsonl> [first_window_trained]"""
import sys, json, numpy as np


from scipy.stats import rankdata


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float('nan')
    ra, rb = rankdata(a[ok]), rankdata(b[ok])          # average ranks: ties carry no order
    return float(np.corrcoef(ra, rb)[0, 1]) if ra.std() > 0 and rb.std() > 0 else float('nan')


def within_interval(x, y, k, min_n=30):
    """Spearman within words of the same interval, weighted by count: a longer interval allows both a later commit
    and a later best tick, so the pooled correlation holds even for a constant hazard (2026-10-14: null +0.66)."""
    rs, ns = [], []
    for kk in np.unique(k):
        m = k == kk
        if m.sum() >= min_n:
            r = spearman(x[m], y[m])
            if np.isfinite(r):
                rs.append(r); ns.append(m.sum())
    return float(np.average(rs, weights=ns)) if rs else float('nan')


rows = [json.loads(l) for l in open(sys.argv[1], encoding='utf-8') if l.strip()]
start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
rows = [r for r in rows if r.get('hazard') is not None and r['bptt_train_tokens'] > start]
words = []   # (answered, latency, entropy, ce, prev_ce, wait_gain, best_tick, ce_at_commit - ce_tick1, k, null value)
for r in rows:
    hz = np.asarray(r['hazard'], float).reshape(-1)
    tce = np.asarray(r['tick_ce'], float) if r.get('tick_ce') is not None else None
    pos, prev = 0, float('nan')
    for k, ce in zip(r['interval'], r['ce']):
        h = np.clip(hz[pos:pos + k], 0.0, 1.0)
        c = tce[pos:pos + k] if tce is not None else None
        pos += k
        survive = np.concatenate([[1.0], np.cumprod(1.0 - h)[:-1]])
        p = h * survive
        ans = float(p.sum())
        lat = float((p * np.arange(1, k + 1)).sum() / ans) if ans > 1e-9 else float('nan')
        q = p / ans if ans > 1e-9 else p
        ent = float(-(q[q > 0] * np.log(q[q > 0])).sum())
        if c is not None and k > 1:
            gain, best = float(c[0] - c.min()), float(np.argmin(c) + 1)
            value = float((q * c).sum() - c[0]) if ans > 1e-9 else float('nan')
        else:
            gain = best = value = float('nan')
        words.append([ans, lat, ent, float(ce), prev, gain, best, value, k, float('nan')])
        if c is not None and k > 1:
            words[-1].append(c)
        else:
            words[-1].append(None)
        prev = float(ce)
# the null policy: the same mean hazard on every tick (no state), scored on the same words
hz_all = np.concatenate([np.clip(np.asarray(r['hazard'], float).reshape(-1), 0, 1) for r in rows]) if rows else np.zeros(1)
hbar = float(hz_all.mean())
for wd in words:
    c = wd[10]
    if c is not None:
        k = len(c)
        p0 = hbar * (1 - hbar) ** np.arange(k)
        wd[9] = float((p0 / p0.sum() * c).sum() - c[0])
w = np.array([wd[:10] for wd in words], dtype=float)
q = lambda v: [round(float(x), 4) for x in (np.nanmin(v), np.nanmedian(v), np.nanmean(v), np.nanmax(v))]
blocks = [w[i:i + 1024] for i in range(0, len(w) - 1023, 1024)] or [w]
per = {'answered': [np.nanmean(b[:, 0]) for b in blocks], 'latency': [np.nanmean(b[:, 1]) for b in blocks],
       'latency_entropy': [np.nanmean(b[:, 2]) for b in blocks],
       'rho_latency_ce': [spearman(b[:, 1], b[:, 3]) for b in blocks],
       'rho_latency_prev_ce': [spearman(b[:, 1], b[:, 4]) for b in blocks],
       # the definition's test (needs tick_ce, words with >= 2 ticks): later where waiting pays, earlier where it does not
       'rho_latency_wait_gain': [spearman(b[:, 1], b[:, 5]) for b in blocks],
       'rho_latency_best_tick': [spearman(b[:, 1], b[:, 6]) for b in blocks],
       'ce_at_commit_minus_tick1': [np.nanmean(b[:, 7]) for b in blocks],
       # the real tests: within words of the same interval, and the timing's value against the constant-hazard null
       'rho_latency_wait_gain_within_k': [within_interval(b[:, 1], b[:, 5], b[:, 8]) for b in blocks],
       'rho_latency_best_tick_within_k': [within_interval(b[:, 1], b[:, 6], b[:, 8]) for b in blocks],
       'timing_value_over_null': [np.nanmean(b[:, 9] - b[:, 7]) for b in blocks]}
print(json.dumps({'words': int(len(w)), **{k: q(v) for k, v in per.items()},
                  'rho_latency_ce_all': round(spearman(w[:, 1], w[:, 3]), 4)}))
