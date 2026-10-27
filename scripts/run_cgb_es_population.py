"""ES on lineages: the score of a genome is what its lineage does in the world (prereg
docs/information_boltzmann/CGB_GENERATED_ECOLI_GOAL_20261027.md, step 5 bootstrap).

Each perturbed genome founds K bodies in each of W worlds (the same worlds, birth cells and rain
for every genome: common random numbers); bodies live, divide (no mutation during scoring: the
genome is scored, not its descendants' variants), die and dissolve in the population engine
(`cgb_nested_population`, checked against Lean). Score of a world = log(1 + living bodies at the
end) + (last tick a body of the lineage was alive) / ticks; a genome's score is the mean over its
worlds. The ES update is that of Qiu et al. 2025 (z-scored scores, score-weighted noise average).
This only bootstraps a viable lineage; the species comes from evolution inside the world.

Usage: python scripts/run_cgb_es_population.py --init <record.json | witness | mover> --out <out.json>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "information_boltzmann" / "core"))
sys.path.insert(0, str(ROOT / "scripts"))
import cgb_nested_world as NW  # noqa: E402
import cgb_nested_world_triton as NT  # noqa: E402
import cgb_nested_organism as NO  # noqa: E402
import cgb_nested_population as NP  # noqa: E402

VAL = np.arange(500, 516)


def lineage_scores(tbl, thetas, world_seeds, ticks, founders, SL, warm=1500, seed=0, table=None):
    """Score every genome (rows of `thetas`) in the same worlds. Returns (scores (G, W), info)."""
    Gn, Wn = len(thetas), len(world_seeds)
    NWt = Gn * Wn
    W = NT.FusedWorld(tbl, B=NWt, seed=int(world_seeds[0]))
    wea = NW.Weather(tbl, B=Wn, device="cuda", seed=0)
    wea.rng = np.random.default_rng(list(map(int, world_seeds)))
    wea.X = wea.rng.normal(0.0, wea.s["rate_sigma"], size=Wn)
    tick_mu = lambda: wea.tick().repeat(Gn, 1, 1)   # noqa: E731
    buf, replay = W.captured()
    for _ in range(warm):
        buf.copy_(tick_mu())
        replay()
    P = NP.Population(W, tbl, SL=SL, table=table, seed=seed, pmut=0.0)
    G2 = P.G * P.G
    crng = np.random.default_rng(list(map(int, world_seeds)) + [7])
    cells = [crng.choice(G2, size=founders, replace=False) for _ in range(Wn)]
    for g in range(Gn):
        for w in range(Wn):
            for c in cells[w]:
                P.place(g * Wn + w, int(c), thetas[g])
    P._sync_free()
    last = np.zeros(NWt)
    capped = 0
    nleap = max(1, int(round(1.0 / W.dt)))
    for t in range(ticks):
        W.mu32.copy_(tick_mu())
        for _ in range(nleap):
            W.leap()
            P.leap()
        P.end_tick()
        if t % 50 == 49 or t == ticks - 1:
            a, living = P.counts()
            last = np.where(living > 0, t + 1, last)
            capped += int((P.free.cpu().numpy() == 0).sum())
    a, living = P.counts()
    score = np.log1p(living) + last / ticks
    info = dict(living=living.reshape(Gn, Wn), births=P.births_total.reshape(Gn, Wn), capped_checks=capped)
    return score.reshape(Gn, Wn), info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--N", type=int, default=30)
    ap.add_argument("--worlds", type=int, default=4)
    ap.add_argument("--founders", type=int, default=4)
    ap.add_argument("--SL", type=int, default=64)
    ap.add_argument("--sigma", type=float, default=0.08)
    ap.add_argument("--alpha", type=float, default=0.08)
    ap.add_argument("--ticks", type=int, default=6000)
    ap.add_argument("--vevery", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--log_div0", type=float, default=-4.5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tbl = NW.load("rain_torus_w0")
    T = NP.DIV_GRAMMAR
    coder = NP.Coder(T)
    rng = np.random.default_rng(a.seed)
    import run_cgb_es_nested as E
    if a.init in ("witness", "mover"):
        g0 = NO.map_genome(E.WITNESS if a.init == "witness" else E.MOVER, "legacy", T)
    else:
        rec = json.loads(Path(a.init).read_text(encoding="utf-8"))
        g0 = {k: np.asarray(v, dtype=float) for k, v in rec["best"]["genome"].items()}
    g0.setdefault("log_div", np.array(a.log_div0))
    theta = coder.encode(g0)
    hist, vals, best = [], [], None
    out = Path(a.out)
    for it in range(a.iters):
        t0 = time.time()
        eps = rng.standard_normal((a.N, theta.size))
        U = np.clip(np.vstack([theta[None], theta[None] + a.sigma * eps]), 0.0, 1.0)
        seeds = 300000 * (a.seed + 1) + 1000 * it + np.arange(a.worlds)
        sc, info = lineage_scores(tbl, U, seeds, a.ticks, a.founders, a.SL, seed=it, table=T)
        score = sc.mean(1)
        z = (score[1:] - score[1:].mean()) / (score[1:].std() + 1e-9)
        theta = np.clip(theta + a.alpha * (z[:, None] * eps).mean(0), 0.0, 1.0)
        hist.append(dict(iter=it, center=float(score[0]), center_living=info["living"][0].tolist(),
                         center_births=info["births"][0].tolist(),
                         pert=[float(x) for x in np.percentile(score[1:], [0, 50, 100])],
                         pert_births_max=int(info["births"][1:].sum(1).max()), capped_checks=info["capped_checks"],
                         seconds=round(time.time() - t0, 1)))
        print(json.dumps(hist[-1]), flush=True)
        if it % a.vevery == a.vevery - 1 or it == a.iters - 1:
            vs, vinfo = lineage_scores(tbl, theta[None], VAL, a.ticks, a.founders, a.SL, seed=999, table=T)
            row = dict(iter=it, val=[float(x) for x in np.percentile(vs[0], [0, 50, 100])], val_mean=float(vs[0].mean()),
                       living=vinfo["living"][0].tolist(), births=vinfo["births"][0].tolist())
            vals.append(row)
            print("VAL", json.dumps(row), flush=True)
            gc = coder.decode(theta[None])
            cand = {k: (np.asarray(v)[0].tolist() if np.ndim(v) > 1 else float(np.asarray(v)[0])) for k, v in gc.items()}
            if best is None or (row["val_mean"], row["val"][1]) > (best["val_mean"], best["val"][1]):
                best = dict(gen=it, genome=cand, theta=theta.tolist(), val=row["val"], val_mean=row["val_mean"])
        out.write_text(json.dumps(dict(prereg="docs/information_boltzmann/CGB_GENERATED_ECOLI_GOAL_20261027.md",
                                       method="ES on lineage scores (Qiu et al. 2025), common random numbers",
                                       world="rain_torus_w0", table=T.name, init=a.init, args=vars(a), history=hist,
                                       validation=vals, theta=theta.tolist(), best=best), indent=1,
                                  default=lambda o: None if isinstance(o, float) and not np.isfinite(o) else o),
                       encoding="utf-8")


if __name__ == "__main__":
    main()
