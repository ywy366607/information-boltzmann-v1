"""Are the fly's acts heavy-tailed (Levy-like, as spontaneous fly behaviour) or exponential (a memoryless simple
strategy)?  From odour_acts.jsonl (one row per act: tick, act, phase) of a training run.
Intervals between successive acts (all, and within gaps only).  Fits: exponential (MLE) and a discrete power law above
x_min (Clauset et al.: x_min by minimal KS distance); the log-likelihood ratio R (power law vs exponential, same tail)
with its normalized significance (Vuong).  Also the act hazard against the time since the last act (a flat hazard is
memoryless; a falling hazard is the signature of heavy tails).
Usage: python act_intervals.py <odour_acts.jsonl>
"""
import sys, json, math, numpy as np

rows = [json.loads(l) for l in open(sys.argv[1])]
ticks = np.array(sorted({r['tick'] for r in rows}))


def fit(x):
    x = np.asarray(x, dtype=float)
    x = x[x >= 1]
    if len(x) < 30:
        return {'n': int(len(x)), 'note': 'too few intervals'}
    best = None
    for xmin in np.unique(np.quantile(x, np.linspace(0, 0.8, 17))):
        t = x[x >= xmin]
        if len(t) < 20:
            continue
        alpha = 1 + len(t) / np.sum(np.log(t / (xmin - 0.5)))
        cdf_emp = np.arange(1, len(t) + 1) / len(t)
        cdf_fit = 1 - (np.sort(t) / xmin) ** (1 - alpha)
        ks = float(np.max(np.abs(cdf_emp - cdf_fit)))
        if best is None or ks < best[0]:
            best = (ks, float(xmin), float(alpha))
    ks, xmin, alpha = best
    t = x[x >= xmin]
    lam = 1.0 / np.mean(t - xmin + 1)                       # shifted geometric/exponential on the same tail
    ll_pow = np.log(alpha - 1) - np.log(xmin) - alpha * np.log(t / xmin)
    ll_exp = np.log(lam) - lam * (t - xmin)
    d = ll_pow - ll_exp
    R = float(d.sum()); z = float(d.sum() / (math.sqrt(len(d)) * d.std() + 1e-12))
    return {'n': int(len(x)), 'tail_n': int(len(t)), 'x_min': xmin, 'alpha': alpha, 'ks': ks, 'R_pow_minus_exp': R,
            'vuong_z': z, 'mean': float(x.mean()), 'max': float(x.max()), 'cv': float(x.std() / x.mean())}


gaps = np.diff(ticks)
rep = {'acts': int(len(ticks)), 'all_intervals': fit(gaps)}
gap_ticks = np.array(sorted({r['tick'] for r in rows if r['phase'] == 'gap'}))
rep['gap_intervals'] = fit(np.diff(gap_ticks)) if len(gap_ticks) > 2 else None
# hazard against time since the last act
if len(gaps) > 50:
    edges = np.unique(np.quantile(gaps, np.linspace(0, 1, 9)))
    haz = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        at_risk = np.sum(gaps >= lo); ended = np.sum((gaps >= lo) & (gaps < hi))
        haz.append({'from': float(lo), 'to': float(hi), 'hazard_per_tick': float(ended / max(at_risk, 1) / max(hi - lo, 1))})
    rep['hazard_vs_time_since_act'] = haz
print(json.dumps(rep, indent=1))
