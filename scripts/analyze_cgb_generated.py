"""Test the preregistered predictions P1-P5 on a generated organism (held-out seeds)."""

from __future__ import annotations

import json
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "information_boltzmann" / "core"))
from cgb_table_fast import live, make_world  # noqa: E402

TICKS = 20000
HELD = tuple(range(101, 109))


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if rx.std() == 0 or ry.std() == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def misses(tr):
    """Consecutive non-travel ticks without intake before each decision."""
    m = np.zeros(len(tr), dtype=int)
    c = 0
    for t in range(len(tr)):
        m[t] = c
        intake, travelled = tr[t, 1], tr[t, 2]
        c = 0 if (intake or travelled) else c + 1
    return m


def main():
    rep = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    g = rep["best"]
    rows, flux = [], []
    for s in HELD:
        (surv, intake, sw, fl), tr = live(g, s, TICKS, trace=True, world=make_world(s, TICKS))
        tr = tr[:surv]
        rows.append((tr, misses(tr)))
        flux.append([float(x) for x in fl[:3]])
    tr = np.concatenate([r[0] for r in rows])
    m = np.concatenate([r[1] for r in rows])
    rho = {f"I{p - 3}": spearman(tr[:, p], m) for p in range(3, 3 + g["interior"])}
    curve = {}
    for k in range(0, 7):
        sel = m == k
        if sel.sum() >= 20:
            curve[k] = float(tr[sel, 0].mean())
    out = dict(
        genome=g,
        P1_spearman_occupancy_vs_misses=rho,
        P1_pass=bool(rho) and max(abs(v) for v in rho.values()) > 0.5,
        P2_switch_given_misses=curve,
        P2_pass=(curve.get(0, 1) < 0.05 and curve.get(3, 0) > 3 * curve.get(1, 1)),
        P3_net_flux_per_tick=dict(F=statistics.mean(f[0] for f in flux), A=statistics.mean(f[1] for f in flux),
                                  M=statistics.mean(f[2] for f in flux)),
        held_out=rep["held_out"],
    )
    surv = {k: [x["survival"] for x in v] for k, v in rep["held_out"].items()}
    out["P4_blind_mean_survival"] = statistics.mean(surv["blind"])
    out["P5_loop_cut_mean_survival"] = statistics.mean(surv["loop_cut"])
    out["organism_mean_survival"] = statistics.mean(surv["organism"])
    Path(sys.argv[2]).write_text(json.dumps(out, indent=1), encoding="utf-8")
    for k, v in out.items():
        if k not in ("genome", "held_out"):
            print(k, v)


if __name__ == "__main__":
    main()
