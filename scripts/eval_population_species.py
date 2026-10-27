"""Held-out evaluation of a dividing species selected by lineage ES (prereg
docs/information_boltzmann/CGB_ESPOP2_SPECIES_PREREG_20261027.md).

Arms (the genome in [0, 1] coordinates; a coordinate at 0 is an absent event, inherited by every
newborn because the scoring runs without mutation):
  organism  the selected genome;
  blind     every selected exchange and crossing rule that binds food on the sensing page at 0
            (the receptor definition of `cgb_nested_organism.Organisms(blind=True)`);
  still     movement at 0;
  nodiv     division at 0.
Each held-out world (seeds 9000 + i, never used in selection or validation) gets `founders` bodies
on the same cells for every arm (common random numbers). Reported per arm: living bodies at the
end, last tick a body was alive, births, lineage score log(1 + living) + last / ticks, all as
[min | median | mean | max] over worlds.

--persist runs the organism alone with mutation on (pmut 0.1, the population engine's physics)
for the given ticks and records extinction, living counts every 1000 ticks and genotype drift.

Usage:
  python scripts/eval_population_species.py <espop record> <out.json> [--worlds 16] [--ticks 20000]
  python scripts/eval_population_species.py <espop record> <out.json> --persist --ticks 100000
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
import cgb_nested_world as NW  # noqa: E402
import cgb_nested_world_triton as NT  # noqa: E402
import cgb_nested_organism as NO  # noqa: E402
import cgb_nested_population as NP  # noqa: E402

HELD = 9000


def dist(x):
    x = np.asarray(x, dtype=float)
    return dict(min=float(x.min()), median=float(np.median(x)), mean=float(x.mean()), max=float(x.max()))


def arm_theta(theta, arm, coder):
    T = coder.T
    t = np.array(theta, dtype=float)
    nphi, nrule, nx = len(T.SELECTED_PHI), len(T.RULES), T.N_SELECTED_X
    if arm == "blind":
        for x, (p, a, b, food, gated) in enumerate(T.EXCH[:nx]):
            if p == NO.F and food:
                t[nphi + nrule + x] = 0.0
        for r, rule in enumerate(T.RULES):
            if NO.F in (rule[0], rule[1]) and rule[6] != 0:
                t[nphi + r] = 0.0
    elif arm == "still":
        t[nphi + nrule + nx + 1] = 0.0      # coordinates after logk_x: log_flip, log_move, log_div
    elif arm == "nodiv":
        t[nphi + nrule + nx + 2] = 0.0
    return t


def run(tbl, thetas, seeds, ticks, founders, SL, pmut, every=50, track=False):
    """One world per (theta, seed) pair, same weather and founder cells for equal seeds."""
    Wn = len(seeds)
    Gn = len(thetas)
    W = NT.FusedWorld(tbl, B=Gn * Wn, seed=int(seeds[0]))
    wea = NW.Weather(tbl, B=Wn, device="cuda", seed=0)
    wea.rng = np.random.default_rng(list(map(int, seeds)))
    wea.X = wea.rng.normal(0.0, wea.s["rate_sigma"], size=Wn)
    tick_mu = lambda: wea.tick().repeat(Gn, 1, 1)   # noqa: E731
    buf, replay = W.captured()
    for _ in range(1500):
        buf.copy_(tick_mu())
        replay()
    P = NP.Population(W, tbl, SL=SL, seed=11, pmut=pmut)
    crng = np.random.default_rng(list(map(int, seeds)) + [7])
    G2 = P.G * P.G
    cells = [crng.choice(G2, size=founders, replace=False) for _ in range(Wn)]
    for g in range(Gn):
        for w in range(Wn):
            for c in cells[w]:
                P.place(g * Wn + w, int(c), thetas[g])
    P._sync_free()
    last = np.zeros(Gn * Wn)
    nleap = max(1, int(round(1.0 / W.dt)))
    traj, drift = [], []
    t0 = time.time()
    for t in range(ticks):
        W.mu32.copy_(tick_mu())
        for _ in range(nleap):
            W.leap()
            P.leap()
        P.end_tick()
        if t % every == every - 1 or t == ticks - 1:
            _, living = P.counts()
            last = np.where(living > 0, t + 1, last)
        if track and (t % 1000 == 999 or t == ticks - 1):
            _, living = P.counts()
            alive = P.alive.cpu().numpy().astype(bool)
            th = P.theta[alive] if alive.any() else np.zeros((0, P.coder.D))
            traj.append(dict(tick=t + 1, living=living.tolist(), seconds=round(time.time() - t0, 1)))
            drift.append(dict(tick=t + 1, genotypes=int(len({tuple(np.round(r, 3)) for r in th})),
                              mean_abs_change=float(np.abs(th - thetas[0]).mean()) if len(th) else None,
                              max_gen=int(np.asarray(P.gen)[alive].max()) if alive.any() else 0))
            print(json.dumps(dict(living_min_med_max=[int(living.min()), float(np.median(living)), int(living.max())],
                                  extinct=int((living == 0).sum()), **drift[-1], seconds=traj[-1]["seconds"])), flush=True)
    _, living = P.counts()
    births = P.births_total.cpu().numpy() if torch.is_tensor(P.births_total) else np.asarray(P.births_total)
    score = np.log1p(living) + last / ticks
    return dict(living=living.reshape(Gn, Wn), last=last.reshape(Gn, Wn), births=births.reshape(Gn, Wn),
                score=score.reshape(Gn, Wn), traj=traj, drift=drift)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("record")
    ap.add_argument("out")
    ap.add_argument("--worlds", type=int, default=16)
    ap.add_argument("--ticks", type=int, default=20000)
    ap.add_argument("--founders", type=int, default=4)
    ap.add_argument("--SL", type=int, default=128)
    ap.add_argument("--persist", action="store_true")
    a = ap.parse_args()
    rec = json.loads(Path(a.record).read_text(encoding="utf-8"))
    tbl = NW.load(rec.get("world", "rain_torus_w0"))
    theta = np.asarray(rec["best"]["theta"], dtype=float)
    coder = NP.Coder(NO.table_of(NP.DIV_GRAMMAR))
    seeds = HELD + np.arange(a.worlds)
    out = dict(prereg="docs/information_boltzmann/CGB_ESPOP2_SPECIES_PREREG_20261027.md", record=a.record,
               world=rec.get("world", "rain_torus_w0"), held_out_seeds=[int(seeds[0]), int(seeds[-1])],
               ticks=a.ticks, founders=a.founders, SL=a.SL)
    if a.persist:
        r = run(tbl, [theta], seeds, a.ticks, a.founders, a.SL, pmut=NP.PMUT, every=50, track=True)
        out.update(mode="persist", pmut=NP.PMUT, living=dist(r["living"][0]), last=dist(r["last"][0]),
                   extinct=int((r["living"][0] == 0).sum()), births=dist(r["births"][0]),
                   living_per_world=r["living"][0].tolist(), trajectory=r["traj"], drift=r["drift"])
    else:
        arms = ("organism", "blind", "still", "nodiv")
        r = run(tbl, [arm_theta(theta, k, coder) for k in arms], seeds, a.ticks, a.founders, a.SL, pmut=0.0)
        out.update(mode="arms", pmut=0.0, arms={k: dict(score=dist(r["score"][i]), living=dist(r["living"][i]),
                                                       last=dist(r["last"][i]), births=dist(r["births"][i]),
                                                       extinct=int((r["living"][i] == 0).sum()),
                                                       score_per_world=r["score"][i].tolist())
                                               for i, k in enumerate(arms)})
        for k in arms:
            print(k, json.dumps({m: out["arms"][k][m] for m in ("score", "living", "extinct")}), flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
