"""Test the preregistered chemotaxis predictions E1-E5 on a generated organism (held-out seeds)."""

from __future__ import annotations

import json
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "information_boltzmann" / "core"))
from cgb_chemotaxis import live, make_world, respond  # noqa: E402

TICKS = 20000
HELD = tuple(range(101, 109))


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if rx.std() == 0 or ry.std() == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    rep = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    g = rep["best"]
    up, down, rho_rows = [], [], {f"I{p - 2}": ([], []) for p in range(2, 2 + g["interior"])}
    for s in HELD:
        (surv, _, _), tr = live(g, s, TICKS, trace=True, world=make_world(s, TICKS))
        tr = tr[:surv]
        tumble, mu = tr[:, 0], tr[:, 1]
        for t in range(2, len(tr)):
            if tumble[t - 1]:
                continue  # no move on the previous tick
            d = mu[t - 1] - mu[t - 2]
            (up if d > 0.02 else down if d < -0.02 else []).append(tumble[t])
        if len(tr) > 20:
            avg = np.convolve(mu, np.ones(10) / 10, mode="full")[:len(mu)]
            for p in range(2, 2 + g["interior"]):
                rho_rows[f"I{p - 2}"][0].extend(tr[10:, p])
                rho_rows[f"I{p - 2}"][1].extend(avg[10:])
    p_up = float(np.mean(up)) if up else float("nan")
    p_down = float(np.mean(down)) if down else float("nan")
    rho = {k: spearman(np.array(v[0]), np.array(v[1])) for k, v in rho_rows.items() if v[0]}
    steps = {}
    for mu0 in (-1.0, 1.0, 3.0):
        mus = np.full(140, mu0)
        mus[100:] = mu0 + 2.0
        f = respond(g, mus)
        base, dip, late = float(f[60:100].mean()), float(f[100:110].min()), float(f[120:140].mean())
        steps[str(mu0)] = dict(baseline=base, dip=dip, late=late,
                               adapts=bool(base > 0.01 and dip < 0.7 * base and abs(late - base) <= 0.3 * base))
    ho = rep["held_out"]
    surv = {k: [x["survival"] for x in v] for k, v in ho.items()}
    out = dict(
        genome=g,
        E1_p_tumble_after_rise=p_up, E1_p_tumble_after_fall=p_down, E1_n=(len(up), len(down)),
        E1_pass=bool(p_up < 0.7 * p_down),
        E2_step_response=steps, E2_pass=any(v["adapts"] for v in steps.values()),
        E3_spearman_occupancy_vs_mu_avg10=rho, E3_pass=bool(rho) and max(abs(v) for v in rho.values()) > 0.5,
        survival={k: dict(min=min(v), median=statistics.median(v), mean=statistics.mean(v), max=max(v))
                  for k, v in surv.items()},
    )
    out["E4_blind_vs_fixed_tumble"] = (out["survival"]["blind"]["mean"], out["survival"]["fixed_tumble_0p1"]["mean"])
    out["E5_pass"] = out["survival"]["organism"]["mean"] > 2 * max(out["survival"]["never_tumble"]["mean"],
                                                                     out["survival"]["fixed_tumble_0p1"]["mean"])
    Path(sys.argv[2]).write_text(json.dumps(out, indent=1), encoding="utf-8")
    for k, v in out.items():
        if k != "genome":
            print(k, v)


if __name__ == "__main__":
    main()
