"""Inter-commit intervals from a run's own commit hazards (ai_trace.jsonl, motor acts): is the waiting heavy-tailed?
The trained windows are consecutive in the stream, so the per-tick hazards concatenate into the life's hazard sequence;
commits are drawn from it (20 seeds; the realized commits were draws from the same hazards).  Reports the interval
distribution [min | median | mean | max], CV, q99, the Hill tail exponent over the top 5% (geometric: large, ~3+ here;
heavy tail: < 2), the same for a geometric of equal mean, and -- with dopamine-encoded precision -- the windows' gamma
distribution and the correlation of a window's commit rate with its gamma (the explore / exploit switching).
Usage: python commit_tail.py <run dir> [out.json]
"""
import sys, json, numpy as np
from pathlib import Path

run = Path(sys.argv[1])
rows = [json.loads(x) for x in open(run / 'ai_trace.jsonl', encoding='utf-8')]
rows = [r for r in rows if r.get('hazard')]
h = np.clip(np.concatenate([r['hazard'] for r in rows]), 0.0, 1.0)


def hill(x, frac=0.05):
    x = np.sort(x)[::-1]
    k = max(5, int(len(x) * frac))
    return float(1.0 / np.mean(np.log(x[:k] / x[k])))


def stats(iv):
    return {'n': int(len(iv)), 'min_median_mean_max': [float(np.min(iv)), float(np.median(iv)), float(np.mean(iv)), float(np.max(iv))],
            'cv': float(np.std(iv) / np.mean(iv)), 'q99': float(np.quantile(iv, 0.99)), 'hill_top5': hill(iv)}


ivs = []
for seed in range(20):
    c = np.nonzero(np.random.default_rng(seed).random(len(h)) < h)[0]
    ivs.append(np.diff(c))
iv = np.concatenate(ivs)
geo = np.random.default_rng(99).geometric(1.0 / iv.mean(), len(iv))
rep = {'run': str(run), 'ticks': int(len(h)), 'hazard_mean': float(h.mean()), 'intervals': stats(iv), 'geometric_same_mean': stats(geo)}
gam = np.array([r.get('gamma') if r.get('gamma') is not None else np.nan for r in rows], float)
if np.isfinite(gam).any():
    rate = np.array([np.mean(r['hazard']) for r in rows])
    ok = np.isfinite(gam)
    rep['gamma_min_median_mean_max'] = [float(np.min(gam[ok])), float(np.median(gam[ok])), float(np.mean(gam[ok])), float(np.max(gam[ok]))]
    rep['gamma_cv'] = float(np.std(gam[ok]) / np.mean(gam[ok]))
    rep['corr_window_rate_gamma'] = float(np.corrcoef(rate[ok], gam[ok])[0, 1])
print(json.dumps(rep, indent=1))
json.dump(rep, open(sys.argv[2] if len(sys.argv) > 2 else run / 'commit_tail.json', 'w'), indent=1)
