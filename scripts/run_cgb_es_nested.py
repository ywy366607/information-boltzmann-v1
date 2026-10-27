"""Evolution strategies for the numbers of the nested organism (prereg
docs/information_boltzmann/CGB_NESTED_ORGANISM_PREREG_20261027.md; structure and physics fixed).

The algorithm of Qiu et al. 2025 (arXiv 2509.24372, "Evolution Strategies at Scale"), i.e.
OpenAI-ES reduced to its core: each iteration samples N Gaussian perturbations of the center,
scores every perturbed genome, z-scores the scores, and moves the center by the score-weighted
average of the noise (no mirrored sampling, rank transform, weight decay or Adam). Our additions:
the genome lives in [0, 1] coordinates of the physics ranges (clipped when decoded, time-scale
premise enforced); all perturbations of an iteration live in the same fresh worlds (common random
numbers: same rain, same birth place); the center is scored with them and validated on seeds
500-515 every `vevery` iterations. Score = mean survival over the iteration's worlds.

Usage: python scripts/run_cgb_es_nested.py --init <record.json> --iters 30 --out <out.json>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "information_boltzmann" / "core"))
import cgb_nested_world as NW  # noqa: E402
import cgb_nested_organism as NO  # noqa: E402
import cgb_nested_organism_triton as NX  # noqa: E402

T = NO.LEGACY      # the organism table (set by --table before anything else runs)


def keys():
    return (("phi", len(T.SELECTED_PHI)), ("logk_rule", len(T.RULES)), ("logk_x", T.N_SELECTED_X),
            ("log_flip", 0), ("log_move", 0))


def bounds():
    plo, phi_hi = NO.phi_range(T)
    lo, hi = [plo], [phi_hi]
    lo.append(np.array([r[0] for r in T.RULE_RANGE]))
    hi.append(np.array([r[1] for r in T.RULE_RANGE]))
    lo.append(np.full(T.N_SELECTED_X, T.X_RANGE[0]))
    hi.append(np.full(T.N_SELECTED_X, T.X_RANGE[1]))
    for _ in range(2):
        lo.append(np.full(1, NO.LOGK_RANGE[0]))
        hi.append(np.full(1, NO.LOGK_RANGE[1]))
    return np.concatenate(lo), np.concatenate(hi)


def set_table(name):
    global T, KEYS, LO, HI
    T = NO.table_of(name)
    KEYS = keys()
    LO, HI = bounds()


set_table("legacy")
WITNESS = dict(phi=np.array([3.0, 6.0, 3.0, 3.0, 3.0, 1.0]), logk_rule=np.array([-6.0, -9.0, -9.0, -2.0]),
               logk_x=np.array([-3.0, -9.0, -1.0, -1.0]) - 4.7, log_flip=np.array(-9.0), log_move=np.array(-9.0))
MOVER = dict(WITNESS, logk_rule=np.array([-6.0, -3.0, -3.0, -2.0]), log_flip=np.array(-3.0), log_move=np.array(-4.0))


def encode(g):
    v = np.concatenate([np.atleast_1d(np.asarray(g[k], dtype=float)) for k, _ in KEYS])
    return (v - LO) / (HI - LO)


def decode(U):
    """Rows of [0, 1] coordinates -> genome dict of arrays (clipped, premise enforced)."""
    U = np.atleast_2d(U)
    V = LO + np.clip(U, 0.0, 1.0) * (HI - LO)
    if T.name != "legacy":   # a prefactor at coordinate 0 is an absent event (conductance 0)
        nphi = len(T.SELECTED_PHI)
        V[:, nphi:] = np.where(U[:, nphi:] <= 0.0, -np.inf, V[:, nphi:])
    g, i = {}, 0
    for k, n in KEYS:
        if n == 0:
            g[k] = V[:, i].copy()
            i += 1
        else:
            g[k] = V[:, i:i + n].copy()
            i += n
    return NO.premise(g, T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="selection record whose best genome starts the center, or 'witness'")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--N", type=int, default=30)
    ap.add_argument("--worlds", type=int, default=8)
    ap.add_argument("--sigma", type=float, default=0.08)
    ap.add_argument("--alpha", type=float, default=0.08)
    ap.add_argument("--ticks", type=int, default=8000)
    ap.add_argument("--vevery", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--table", default="legacy", help="organism table: legacy or grammar_v1")
    a = ap.parse_args()
    set_table(a.table)
    tbl = NW.load("rain_torus_w0")
    rng = np.random.default_rng(a.seed)
    if a.init in ("witness", "mover"):   # declared before any selection (still / moving), in table T
        w0 = WITNESS if a.init == "witness" else MOVER
        theta = encode(w0 if T.name == "legacy" else NO.map_genome(w0, "legacy", T))
    else:
        init = json.loads(Path(a.init).read_text(encoding="utf-8"))
        theta = encode(init["best"]["genome"])
    theta = np.clip(theta, 0.0, 1.0)
    hist, vals, best = [], [], None
    out = Path(a.out)
    for it in range(a.iters):
        t0 = time.time()
        eps = rng.standard_normal((a.N, theta.size))
        U = np.vstack([theta[None], theta[None] + a.sigma * eps])          # row 0: the center
        g = decode(U)
        rows = {k: np.repeat(v, a.worlds, axis=0) for k, v in g.items()}   # lane = row * W + world
        seeds = 200000 * (a.seed + 1) + 1000 * it + np.arange(a.worlds)
        r = NX.run_exact(tbl, rows, seeds, ticks=a.ticks, copies=a.N + 1, table=T)
        # a lane that needed more than K events in a leap is not exact (overflow): it scores as dead at
        # birth, so selection stays where the simulator is exact (total event rate <= K / dt per tick)
        surv = np.where(r["overflow"] > 0, 0, r["survival"]).reshape(a.N + 1, a.worlds)
        score = surv.mean(1)
        z = (score[1:] - score[1:].mean()) / (score[1:].std() + 1e-9)
        theta = theta + a.alpha * (z[:, None] * eps).mean(0)
        theta = np.clip(theta, 0.0, 1.0)
        hist.append(dict(iter=it, center_mean=float(score[0]), center_median=float(np.median(surv[0])),
                         pert_mean=[float(x) for x in np.percentile(score[1:], [0, 50, 100])],
                         overflow_lanes=int((r["overflow"] > 0).sum()), seconds=round(time.time() - t0, 1)))
        print(json.dumps(hist[-1]), flush=True)
        if it % a.vevery == a.vevery - 1 or it == a.iters - 1:
            gc = decode(theta[None])
            vr = NX.run_exact(tbl, {k: np.repeat(v, len(VAL), axis=0) for k, v in gc.items()}, VAL, ticks=a.ticks,
                               table=T)
            vs = vr["survival"]
            row = dict(iter=it, val=[float(x) for x in np.percentile(vs, [0, 50, 100])], val_mean=float(vs.mean()))
            vals.append(row)
            print("VAL", json.dumps(row), flush=True)
            cand = {k: (np.asarray(v)[0].tolist() if np.ndim(v) > 1 else float(np.asarray(v)[0])) for k, v in gc.items()}
            if best is None or (row["val"][1], row["val_mean"]) > (best["val"][1], best["val_mean"]):
                best = dict(gen=it, genome=cand, val=row["val"], val_mean=row["val_mean"])
        out.write_text(json.dumps(dict(prereg="docs/information_boltzmann/CGB_NESTED_ORGANISM_PREREG_20261027.md",
                                       method="ES (Qiu et al. 2025, arXiv 2509.24372), common random numbers; overflow lanes score 0",
                                       world="rain_torus_w0", table=T.name, init=a.init, args=vars(a), history=hist,
                                       validation=vals, theta=theta.tolist(), best=best), indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
