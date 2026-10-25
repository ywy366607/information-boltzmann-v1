"""Exact finite-state semantic review, not a learning/capability experiment.

The Lean review file proves the body kernel table and update ordering used here.
The cut samples its ghost from the evolving unconditional cut body marginal;
recovery uses that same future cut schedule, including off-support starts.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction as Q
import hashlib
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
COST = (0, 1, 1, 2)
STATES = tuple((x, w) for x in range(4) for w in range(6))


def advance(w: int, x: int) -> int:
    if w == 1:
        return 1
    if w == 2:
        return 2
    if w in (3, 4):
        return 1 if x == 3 else 3
    return x


def world_moves(w: int, x: int) -> dict[int, Q]:
    if w == 0:
        return {1: Q(1, 2), 2: Q(1, 2)}
    if w in (1, 2):
        return {w + 2: Q(1)}
    if w == 3:
        return {3 if x in (1, 3) else 5: Q(1)}
    if w == 4:
        return {4 if x in (2, 3) else 5: Q(1)}
    return {5: Q(1)}


def replay(cut: bool, steps: int) -> tuple[list[dict], list[dict]]:
    laws = [{(1, 0): Q(1)}]
    kernels = []
    for _ in range(steps):
        current = laws[-1]
        marginal = defaultdict(Q)
        for (x, _w), mass in current.items():
            marginal[x] += mass
        kernel = {}
        for x, w in STATES:
            row = defaultdict(Q)
            ghosts = marginal.items() if cut else [(x, Q(1))]
            for ghost, mass in ghosts:
                for target, probability in world_moves(w, ghost).items():
                    row[advance(w, x), target] += mass * probability
            assert sum(row.values()) == 1 and all(p >= 0 for p in row.values())
            kernel[x, w] = {z: p for z, p in row.items() if p}
        next_law = defaultdict(Q)
        for source, mass in current.items():
            for target, probability in kernel[source].items():
                next_law[target] += mass * probability
        assert sum(next_law.values()) == 1
        kernels.append(kernel)
        laws.append(dict(next_law))
    return laws, kernels


def recovery(kernels: list[dict], time: int, horizon: int, state: tuple) -> Q:
    if horizon == 0:
        return Q(0)
    return sum((p * (1 if COST[state[0]] < COST[target[0]] else
                    recovery(kernels, time + 1, horizon - 1, target))
                for target, p in kernels[time][state].items()), Q(0))


def survival(law: dict, kernels: list[dict], time: int) -> Q:
    return sum((mass for state, mass in law.items() if COST[state[0]] > 0
                and recovery(kernels, time, 2, state) >= Q(1, 2)), Q(0))


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path,
                        default=PROJECT / ".lake/active_loop_semantics_20261024.json")
    args = parser.parse_args()
    normal, normal_kernels = replay(False, 11)
    cut, cut_kernels = replay(True, 11)
    rows = [{"time": t,
             "natural_survival": str(survival(normal[t], normal_kernels, t)),
             "cut_survival": str(survival(cut[t], cut_kernels, t)),
             "natural_law": {f"{x},{w}": str(p) for (x, w), p in normal[t].items()},
             "cut_law": {f"{x},{w}": str(p) for (x, w), p in cut[t].items()}}
            for t in range(10)]
    assert rows[0]["natural_survival"] == "0"
    assert rows[2]["natural_survival"] == rows[2]["cut_survival"] == "1"
    assert rows[5]["natural_survival"] == "1/2"
    assert rows[5]["cut_survival"] == "3/8"
    assert normal[5] == {(3, 3): Q(1, 2), (3, 5): Q(1, 2)}
    violations = [r["time"] for r in rows if r["time"] >= 2 and
                  Q(r["cut_survival"]) + Q(1, 4) > Q(r["natural_survival"])]
    assert 2 in violations and 5 in violations
    theorem_inventory = json.loads((PROJECT /
        ".lake/active_loop_review_20261024.json").read_text(encoding="utf-8"))
    authored = [d for d in theorem_inventory["declarations"] if
                d["module"] == "CGBAxioms.ActiveLoop" and d["kind"] == "theorem"
                and d["source_declared"]]
    assert len(authored) == 8
    assert not any("SelfMaintains" in d["signature"] or "survival" in d["signature"]
                   for d in authored)
    source_paths = [PROJECT / "CGBAxioms/ActiveLoop.lean", Path(__file__),
                    PROJECT / "reviews/active_loop_20261024.lean"]
    output = {"date": "2026-10-24", "status": "reported_claims_refuted",
              "scope": "Exact rational replay of a finite mathematical model; no training.",
              "lean_table_check": "reviews/active_loop_20261024.lean",
              "arithmetic": "fractions.Fraction, no sampling or floating-point tolerance",
              "rows": rows, "claimed_quarter_gap_counterexample_times": violations,
              "active_loop_authored_theorems": [d["name"] for d in authored],
              "source_hashes": {p.relative_to(PROJECT).as_posix():
                                hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"passed": True, "claim_counterexamples": violations,
                      "time_5": rows[5], "output": str(args.output)}))


if __name__ == "__main__":
    main()
