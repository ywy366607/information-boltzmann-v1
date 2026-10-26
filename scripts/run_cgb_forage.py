"""B' test: does the LG-1 forager keep itself alive through its own actions?

Same world law and seeds for every agent. Reports survival ticks and intake
rate as [min | median | mean | max] over seeds. Controls: loop cut (ghost
actions at the organism's own switch rate), frozen learning, reflex, E. coli
integrator, random, oracle.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "information_boltzmann" / "core"))
from cgb_observation_stream import load_owt_sources  # noqa: E402
from cgb_forage_world import ForageWorld  # noqa: E402
from cgb_lg1_forager import LG1Forager, Reflex, Integrator, RandomPolicy, Oracle  # noqa: E402
from cgb_pg_forager import PGForager  # noqa: E402


def stats(v):
    return dict(min=min(v), median=statistics.median(v), mean=statistics.mean(v), max=max(v))


def live(agent_factory, sources, wkw, ticks, seed, cut=False):
    world = ForageWorld(sources, cut=cut, seed=seed, **wkw)
    agent = agent_factory(world)
    death, intake, stays = ticks, 0, 0
    for t in range(ticks):
        tk = world.step(agent.act())
        agent.sense(tk.byte, tk.intake, tk.travelled)
        intake += tk.intake
        if not tk.alive:
            death = t + 1
            break
    out = dict(survival=death, intake_rate=intake / death, switches=world.requested / death)
    if isinstance(agent, LG1Forager) and agent.byte_nll:
        n = agent.byte_nll[-20000:]
        out["byte_bits_last"] = sum(n) / len(n) / math.log(2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=100_000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--cost", type=float, default=0.6)
    ap.add_argument("--hazard", type=float, default=0.005)
    ap.add_argument("--p-rich", type=float, default=0.9)
    ap.add_argument("--p-poor", type=float, default=0.1)
    ap.add_argument("--reserve", type=float, default=150.0)
    ap.add_argument("--sources", type=int, default=3)
    ap.add_argument("--stage", type=int, default=1, help="2: hidden state = digit-rich document rule")
    ap.add_argument("--agents", default="", help="comma list; empty = all")
    ap.add_argument("--pg-eta", type=float, default=0.1)
    ap.add_argument("--pg-trace", type=float, default=0.9)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    d = ROOT / "data" / "ib_owt_10000"
    train, _, _, prov = load_owt_sources(d / "train.npy", d / "validation.npy", "byte")
    wkw = dict(hazard=args.hazard, cost=args.cost, reserve=args.reserve, capacity=2 * args.reserve,
               p_rich=args.p_rich, p_poor=args.p_poor)

    def forager(frozen=False):
        def make(world):
            a = LG1Forager(args.hidden, seed=world.seed + 7919, eta=0.05, relax=1.0,
                           diffuse=0.3, feedback=0.1, recur=2.0)
            a.learning = not frozen
            return a
        return make

    def pg(frozen=False):
        def make(world):
            a = PGForager(eta=args.pg_eta, trace=args.pg_trace, seed=world.seed + 7919)
            a.learning = not frozen
            return a
        return make

    agents = {
        "g15_forager": (pg(), False),
        "g15_forager_loop_cut": (pg(), True),
        "g15_forager_frozen": (pg(frozen=True), False),
        "lg1_forager": (forager(), False),
        "lg1_forager_loop_cut": (forager(), True),
        "lg1_forager_frozen": (forager(frozen=True), False),
        "integrator_ecoli": (lambda w: Integrator(), False),
        "reflex": (lambda w: Reflex(), False),
        "random": (lambda w: RandomPolicy(seed=7), False),
        "oracle": (lambda w: Oracle(w), False),
    }
    if args.agents:
        agents = {k: v for k, v in agents.items() if k in args.agents.split(",")}
    results = {k: [] for k in agents}
    for seed in range(1, args.seeds + 1):
        off = seed * 3_000_011
        spans = [(off + k * 500_000, off + k * 500_000 + 400_000) for k in range(args.sources)]
        sources = [train[a:b] for a, b in spans]
        if args.stage == 2:
            import bisect
            import numpy as np
            ends, rich = [], []
            for (a, b), src in zip(spans, sources):
                e = [x - a for x in train.ends[bisect.bisect_right(train.ends, a):bisect.bisect_left(train.ends, b)]]
                e.append(b - a)
                cuts = [0] + e
                frac = [float(np.mean((src[i:j] >= 48) & (src[i:j] <= 57))) if j > i else 0.0
                        for i, j in zip(cuts, cuts[1:])]
                ends.append(e)
                rich.append(frac)
            med = float(np.median([f for fr in rich for f in fr]))
            rich = [[f >= med for f in fr] for fr in rich]
            wkw_s = dict(wkw, doc_ends=ends, doc_rich=rich)
        else:
            wkw_s = wkw
        for name, (factory, cut) in agents.items():
            r = live(factory, sources, wkw_s, args.ticks, seed, cut=cut)
            results[name].append(r)
            print(seed, name, json.dumps(r), flush=True)
    summary = {k: {m: stats([r[m] for r in v]) for m in v[0]} for k, v in results.items()}
    report = dict(world=dict(wkw, ticks=args.ticks, sources=args.sources, stage=args.stage,
                             stage2_rule="document digit fraction >= median over sources"), hidden=args.hidden,
                  seeds=args.seeds, g15=dict(eta=args.pg_eta, trace=args.pg_trace), provenance=prov, per_seed=results, summary=summary, admitted=False)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    for k, v in summary.items():
        s = v["survival"]
        print(f"{k:24s} survival [{s['min']:.0f} | {s['median']:.0f} | {s['mean']:.0f} | {s['max']:.0f}]"
              f"  intake {v['intake_rate']['mean']:.3f}")


if __name__ == "__main__":
    main()
