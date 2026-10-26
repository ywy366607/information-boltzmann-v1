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
from cgb_lg1_forager import LG1Forager, G15Organism, Reflex, Integrator, RandomPolicy, Oracle  # noqa: E402
from cgb_pg_forager import PGForager, LangPGForager  # noqa: E402


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
        out["bytes_heard"] = len(agent.byte_nll)
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
    ap.add_argument("--taus", default="16,300")
    ap.add_argument("--window", type=int, default=32, help="stage 3: rich iff a digit in the last W bytes")
    ap.add_argument("--shuffle-bytes", type=int, default=0, help="1: permute each source (destroys text, keeps byte counts)")
    ap.add_argument("--pg-eta", type=float, default=0.1)
    ap.add_argument("--pg-trace", type=float, default=0.9)
    ap.add_argument("--pg-relax", type=float, default=0.0)
    ap.add_argument("--pg-state-baseline", type=int, default=0)
    ap.add_argument("--pg-base-rate", type=float, default=0.01)
    ap.add_argument("--pg-couple", type=float, default=0.0)
    ap.add_argument("--pg-slow-c", type=float, default=10.0)
    ap.add_argument("--pg-slow-leak", type=float, default=0.0)
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
            a = PGForager(eta=args.pg_eta, trace=args.pg_trace, relax=args.pg_relax,
                          state_baseline=bool(args.pg_state_baseline), base_rate=args.pg_base_rate,
                          couple=args.pg_couple, slow_c=args.pg_slow_c, slow_leak=args.pg_slow_leak,
                          seed=world.seed + 7919)
            a.learning = not frozen
            return a
        return make

    def lang(use=True, cut=False):
        def make(world):
            return LangPGForager(eta=args.pg_eta, trace=args.pg_trace, taus=tuple(float(t) for t in args.taus.split(",")), lang=use,
                                 seed=world.seed + 7919)
        return make

    def org(use_hidden=True, lang_learn=True, policy_learn=True, grown_R=False):
        def make(world):
            a = G15Organism(args.hidden, seed=world.seed + 7919, eta=0.05, relax=1.0, diffuse=0.3,
                            feedback=0.1, recur=2.0, pg_eta=args.pg_eta, pg_trace=args.pg_trace,
                            use_hidden=use_hidden, grown_R=grown_R)
            a.learning, a.policy_learning = lang_learn, policy_learn
            return a
        return make

    agents = {
        "org": (org(), False),
        "org_grown": (org(grown_R=True), False),
        "org_counter_policy": (org(use_hidden=False), False),
        "org_lang_frozen": (org(lang_learn=False), False),
        "org_frozen": (org(lang_learn=False, policy_learn=False), False),
        "org_loop_cut": (org(), True),
        "g15_lang": (lang(), False),
        "g15_lang_loop_cut": (lang(), True),
        "g15_counter_only": (lang(use=False), False),
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
        if args.shuffle_bytes:
            import numpy as np
            prng = np.random.default_rng(seed + 55)
            sources_text = [np.asarray(s) for s in sources]  # labels below use the real text
            sources_shuf = [prng.permutation(s) for s in sources_text]
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
        elif args.stage == 3:
            import numpy as np
            masks = []
            for src in sources:
                dig = ((np.asarray(src) >= 48) & (np.asarray(src) <= 57)).astype(np.int64)
                c = np.concatenate([[0], np.cumsum(dig)])
                i = np.arange(len(dig))
                masks.append((c[i + 1] - c[np.maximum(i + 1 - args.window, 0)]) > 0)
            wkw_s = dict(wkw, rich_mask=masks)
        else:
            wkw_s = wkw
        heard = sources_shuf if args.shuffle_bytes else sources
        for name, (factory, cut) in agents.items():
            r = live(factory, heard, wkw_s, args.ticks, seed, cut=cut)
            results[name].append(r)
            print(seed, name, json.dumps(r), flush=True)
    summary = {k: {m: stats([r[m] for r in v]) for m in v[0]} for k, v in results.items()}
    report = dict(world=dict(wkw, ticks=args.ticks, shuffle_bytes=args.shuffle_bytes, taus=args.taus, window=args.window, sources=args.sources, stage=args.stage,
                             stage2_rule="document digit fraction >= median over sources",
                             stage3_rule="a digit in the last `window` bytes of the source text (including the current byte)"), hidden=args.hidden,
                  seeds=args.seeds, g15=dict(eta=args.pg_eta, trace=args.pg_trace, relax=args.pg_relax,
                                                 state_baseline=args.pg_state_baseline, base_rate=args.pg_base_rate,
                                                 couple=args.pg_couple, slow_c=args.pg_slow_c, slow_leak=args.pg_slow_leak), provenance=prov, per_seed=results, summary=summary, admitted=False)
    sys.path.insert(0, str(ROOT / "scripts"))
    from cgb_gate import gate
    g = gate("g15_organism")
    report["gate"] = {"g15_organism": dict(admitted=g["admitted"], failures=g["failures"])}
    report["admitted"] = False  # no agent here has passed scripts/cgb_gate.py
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    for k, v in summary.items():
        s = v["survival"]
        print(f"{k:24s} survival [{s['min']:.0f} | {s['median']:.0f} | {s['mean']:.0f} | {s['max']:.0f}]"
              f"  intake {v['intake_rate']['mean']:.3f}")


if __name__ == "__main__":
    main()
