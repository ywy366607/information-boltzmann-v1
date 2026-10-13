"""M3 acceptance table from the act probes of every arm (GOAL: docs/information_boltzmann/FLY_ACTIVE_INFERENCE_GOAL_20261013.md).
Usage: python m3_report.py <root with arm dirs> [arms...]
Per arm and condition: eyes (open fraction at word onsets, within words, on noise vs text, closures when a question is
open / by its difficulty), output (answered, latency, rho with the word's noise and with its tick-1 CE, answer value),
and paired language and answer comparisons against the passive (language) and open (forced-eyes) arms.
"""
import sys, json, numpy as np
from pathlib import Path

root = Path(sys.argv[1])
arms = sys.argv[2:] or ['full', 'open', 'passive', 'noepi', 'noprag', 'depth0']


def load(arm):
    rows = {}
    f = root / arm / 'act_probe.jsonl'
    if not f.exists():
        return rows
    for line in open(f):
        r = json.loads(line)
        rows[(r['condition'], r['mode'])] = r
    return rows


def paired(a, b, key):
    x = np.asarray(a[f'per_word_{key}'], dtype=float); y = np.asarray(b[f'per_word_{key}'], dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    d = x[ok] - y[ok]
    rng = np.random.default_rng(0)
    boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(1000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return f'{d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]'


R = {arm: load(arm) for arm in arms}
f3 = lambda v: '-' if v is None else (f'{v:.3f}' if isinstance(v, float) else str(v))
for arm in arms:
    print(f'\n=== {arm}')
    for (cond, mode), r in sorted(R[arm].items()):
        line = (f"{cond:>12} {mode:>4} | eye onset {f3(r.get('eye_open_frac'))} word {f3(r.get('eye_word_mean'))}"
                f" closed|open-q {f3(r.get('closed_frac_when_open_question'))} none {f3(r.get('closed_frac_when_none_open'))}"
                f" hard {f3(r.get('closed_frac_open_hard_half'))} easy {f3(r.get('closed_frac_open_easy_half'))}"
                f" | answered {f3(r['answered_frac'])} latency {r['latency_min_median_mean_max']}"
                f" rho(lat,sigma) {f3(r.get('rho_latency_sigma'))} rho(lat,ce1) {f3(r.get('rho_latency_ce1'))}"
                f" value {r['answer_value_mean']:.3f} ext {f3(r.get('questions_extended_frac'))}"
                f" | arrival {r['arrival_ce_min_median_mean_max'][2]:.3f} survival {r['survival_ce_min_median_mean_max'][2]:.3f}")
        if cond.startswith('noisy'):
            line += (f" | noise open {f3(r.get('eye_open_frac_noise'))} late {f3(r.get('eye_open_frac_noise_late'))}"
                     f" text {f3(r.get('eye_open_frac_text'))} resume {f3(r.get('eye_open_frac_resume'))}")
        print(line)
    # same weights: own eyes vs forced open (answer value and language)
    for cond in sorted({c for c, m in R[arm] if m == 'own'}):
        if (cond, 'open') in R[arm]:
            a, b = R[arm][(cond, 'own')], R[arm][(cond, 'open')]
            print(f'  {cond:>12} own - forced open: answer CE {paired(a, b, "answer_ce")}  survival {paired(a, b, "survival")}')
    # language and answers vs the reference arms
    for ref in ('passive', 'open'):
        if ref == arm or not R.get(ref):
            continue
        for cond in sorted({c for c, m in R[arm]}):
            mode = 'own' if (cond, 'own') in R[arm] else 'open'
            if (cond, 'open') in R[ref]:
                a, b = R[arm][(cond, mode)], R[ref][(cond, 'open')]
                print(f'  {cond:>12} {arm}({mode}) - {ref}: survival {paired(a, b, "survival")}  arrival {paired(a, b, "arrival")}'
                      f'  answer CE {paired(a, b, "answer_ce")}')
