"""Check the ignition predictions I1-I7 (prereg CGB_GENERATED_IGNITION_PREREG_20261026) on held-out worlds.

usage: python scripts/analyze_cgb_ignition.py <run json> <out json>
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np
from scipy.stats import skew, kurtosis, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "information_boltzmann" / "core"))
import cgb_ignition as I  # noqa: E402

HELD = tuple(range(101, 109))
TICKS = 20000
G24_MAX, G24_MEAN = 0.436, 0.206  # prereg section 3 (196 independent units)


def trials_of(g, seed, **kw):
    w = I.make_world(seed, TICKS)
    (surv, intake, rate), tr = I.live(g, seed, TICKS, trace=True, world=w, **kw)
    rows = []
    for s0, a, d, w0 in w[4]:
        s0, d, w0 = int(s0), int(d), int(w0)
        if w0 + I.WIN > surv:
            break
        g0 = w0 - 1
        rows.append(dict(present=a > 0, a=float(a), d=d, reported=float(tr[w0:w0 + I.WIN, 0].mean()) > 0.5,
                         occ=tr[g0, 2:].copy(), fuel=float(tr[s0 + I.STIM:g0, 1].mean())))
    return surv, rows


def hit_fa(rows, lo=0, hi=10**9):
    r = [x for x in rows if lo <= x["d"] <= hi]
    p = [x["reported"] for x in r if x["present"]]
    n = [x["reported"] for x in r if not x["present"]]
    h = float(np.mean(p)) if p else float("nan")
    f = float(np.mean(n)) if n else float("nan")
    return h, f, len(p), len(n)


def bimodality(x):
    n = len(x)
    if n < 8 or np.std(x) == 0:
        return float("nan")
    g, k = skew(x), kurtosis(x)
    return float((g * g + 1) / (k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))))


def dist(v):
    return dict(min=min(v), median=statistics.median(v), mean=statistics.mean(v), max=max(v))


def main():
    run = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    g = run["best"]
    out = {}
    arms = {"organism": {}, "calm_fuel": {"calm_fuel": True}, "blind": {"blind": True}}
    rows, survival = {}, {}
    for arm, kw in arms.items():
        rows[arm], survival[arm] = [], []
        for s in HELD:
            surv, r = trials_of(g, s, **kw)
            survival[arm].append(surv)
            rows[arm] += r
    out["survival"] = {k: dist(v) for k, v in survival.items()}
    for arm in arms:
        out[f"{arm}_hit_fa_D>=80"] = hit_fa(rows[arm], 80)
        out[f"{arm}_hit_fa_by_D"] = {f"{lo}-{hi}": hit_fa(rows[arm], lo, hi)
                                     for lo, hi in [(10, 40), (41, 80), (81, 120), (121, 150)]}
    h, f, *_ = out["organism_hit_fa_D>=80"]
    out["I1_pass"] = bool(h - f > G24_MAX)
    h2, f2, *_ = out["calm_fuel_hit_fa_D>=80"]
    out["I2_pass"] = bool(h2 - f2 <= G24_MEAN and statistics.mean(survival["calm_fuel"]) < statistics.mean(survival["organism"]))
    hs, fs, *_ = hit_fa(rows["organism"], 10, 40)
    hl, fl, *_ = hit_fa(rows["organism"], 100, 150)
    out["I3_hitminusfa_short_long"] = (hs - fs, hl - fl)
    out["I3_pass"] = bool(hs - fs >= 0.2 and hl - fl >= 0.8 * (hs - fs))
    R = rows["organism"]
    if R and len(R[0]["occ"]):
        occ = np.array([x["occ"] for x in R], dtype=float)
        lab = np.array([x["present"] for x in R], dtype=float)
        cors = [abs(np.corrcoef(occ[:, p], lab)[0, 1]) if occ[:, p].std() > 0 else 0.0 for p in range(occ.shape[1])]
        p = int(np.argmax(cors))
        mid = [x["occ"][p] for x in R if x["present"] and 1.5 <= x["a"] <= 2.5]
        ign = [(x["occ"][p], x["a"]) for x in R if x["present"] and x["reported"]]
        rho = spearmanr([u for u, _ in ign], [v for _, v in ign])[0] if len(ign) > 8 else float("nan")
        bc = bimodality(np.array(mid, dtype=float))
        out["I4_page"], out["I4_corr"], out["I4_bimodality"], out["I4_rho_occ_a"] = p, cors, bc, float(rho)
        # guard: both modes must hold at least 10% of the trials (split at the Otsu threshold)
        m = np.array(mid, dtype=float)
        best, thr = -1.0, float("nan")
        for t in np.unique(m)[:-1]:
            lo, hi = m[m <= t], m[m > t]
            v = len(lo) * len(hi) * (lo.mean() - hi.mean()) ** 2
            if v > best:
                best, thr = v, float(t)
        minor = min(np.mean(m <= thr), np.mean(m > thr)) if np.isfinite(thr) else 0.0
        out["I4_minor_mode_fraction"] = float(minor)
        out["I4_pass"] = bool(cors[p] >= 0.3 and bc > 0.555 and minor >= 0.1 and abs(rho) < 0.3)
        hist = np.histogram(mid, bins=12)
        out["I4_hist"] = [hist[0].tolist(), [float(x) for x in hist[1]]]
    else:
        out["I4_pass"] = False
        out["I4_note"] = "no interior page"
    after_p = [R[i]["reported"] for i in range(1, len(R)) if R[i - 1]["present"] and not R[i]["present"]]
    after_a = [R[i]["reported"] for i in range(1, len(R)) if not R[i - 1]["present"] and not R[i]["present"]]
    fa_p, fa_a = float(np.mean(after_p)) if after_p else float("nan"), float(np.mean(after_a)) if after_a else float("nan")
    out["I5_fa_after_present_absent"] = (fa_p, fa_a)
    out["I5_pass"] = bool(fa_a < 0.5 and fa_p <= 1.5 * max(fa_a, 0.02))
    fi = [x["fuel"] for x in R if x["reported"]]
    fn = [x["fuel"] for x in R if not x["reported"]]
    out["I6_fuel_reported_unreported"] = (float(np.mean(fi)) if fi else float("nan"), float(np.mean(fn)) if fn else float("nan"))
    out["I6_pass"] = bool(fi and fn and np.mean(fi) > np.mean(fn))
    out["I7_pass"] = bool(abs(statistics.median(survival["blind"]) - 7500) <= 1500)
    Path(sys.argv[2]).write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    for k, v in out.items():
        print(k, v)


if __name__ == "__main__":
    main()
