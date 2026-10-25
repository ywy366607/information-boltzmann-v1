"""Exact semantic cross-check of the Lean maintenance witness; no training."""

from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction as Q
import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
STATES = tuple((a, w) for a in range(3) for w in range(6))


def sense(w):
    return {1: 1, 2: 2, 3: 3, 4: 3, 5: 4}.get(w, 0)


def advance(s, a):
    if a == 0:
        return 0
    return 1 if s == 1 else 2 if s == 2 else 0 if s == 4 else a


def incoming(s, a):
    return int(s == 3 and a != 0)


def outgoing(s, a):
    return int(s in (3, 4) and a != 0)


def moves(w, a):
    if w == 0:
        return {1: Q(1, 2), 2: Q(1, 2)}
    if w in (1, 2):
        return {w + 2: Q(1)}
    if w == 3:
        return {3 if a == 1 else 5: Q(1)}
    if w == 4:
        return {4 if a == 2 else 5: Q(1)}
    return {5: Q(1)}


def replay(cut, steps):
    laws = [{(1, 0): Q(1)}]
    kernels = []
    for _ in range(steps):
        current = laws[-1]
        marginal = defaultdict(Q)
        for (a, _w), p in current.items():
            marginal[a] += p
        kernel = {}
        for a, w in STATES:
            row = defaultdict(Q)
            for ghost, mass in (marginal.items() if cut else [(a, Q(1))]):
                for target, p in moves(w, ghost).items():
                    row[advance(sense(w), a), target] += mass * p
            assert sum(row.values()) == 1 and all(p >= 0 for p in row.values())
            kernel[a, w] = {z: p for z, p in row.items() if p}
        new = defaultdict(Q)
        for z, p in current.items():
            for target, q in kernel[z].items():
                new[target] += p * q
        assert sum(new.values()) == 1
        laws.append(dict(new))
        kernels.append(kernel)
    return laws, kernels


def recovery(kernels, t, horizon, z):
    if horizon == 0:
        return Q(0)
    if incoming(sense(z[1]), z[0]):
        return Q(1)
    return sum((p * recovery(kernels, t + 1, horizon - 1, z2)
                for z2, p in kernels[t][z].items()), Q(0))


def owners(a, p):
    return tuple("body" if c == p and a else "source" if c == (p + 1) % 3
                 else "sink" for c in range(3))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path,
                        default=PROJECT / ".lake/active_maintenance_semantics_20261024.json")
    args = parser.parse_args()
    carrier_checks = 0
    for s in range(5):
        for a in range(3):
            b = advance(s, a)
            registered_event = s != 0 and a != 0
            if not registered_event:
                assert b == a and incoming(s, a) == outgoing(s, a) == 0
            assert int(b != 0) + outgoing(s, a) == int(a != 0) + incoming(s, a)
            for phase in range(3):
                new_phase = (phase + 1) % 3 if incoming(s, a) else phase
                before, after = owners(a, phase), owners(b, new_phase)
                transported = list(zip(before, after))
                assert transported.count(("source", "body")) == incoming(s, a)
                assert transported.count(("body", "sink")) == outgoing(s, a)
                recharge = transported.count(("sink", "source"))
                energy = lambda places: sum(p in ("source", "body") for p in places)
                assert energy(after) + outgoing(s, a) == energy(before) + recharge
                carrier_checks += 1
    assert all(advance(4, a) == 0 for a in range(3))
    assert all(advance(s, 0) == 0 for s in range(5))
    runs = {}
    for cut in (False, True):
        laws, kernels = replay(cut, 12)
        rows, cumulative_in, cumulative_out = [], Q(0), Q(0)
        for t in range(10):
            current = laws[t]
            reserve = sum((p * int(a != 0) for (a, _w), p in current.items()), Q(0))
            assert reserve + cumulative_out == 1 + cumulative_in
            alive = sum((p for z, p in current.items() if z[0] != 0
                         and recovery(kernels, t, 3, z) >= Q(1, 2)), Q(0))
            if not cut:
                assert alive == 1
            elif t >= 3:
                assert alive <= Q(1, 2)
            input_flux = sum((p * incoming(sense(w), a) for (a, w), p in current.items()), Q(0))
            use_flux = sum((p * outgoing(sense(w), a) for (a, w), p in current.items()), Q(0))
            rows.append({"time": t, "survival": str(alive), "mean_reserve": str(reserve),
                         "input_flux": str(input_flux), "consumption_flux": str(use_flux),
                         "cumulative_input_before_tick": str(cumulative_in),
                         "cumulative_consumption_before_tick": str(cumulative_out),
                         "law": {f"{a},{w}": str(p) for (a, w), p in current.items()}})
            cumulative_in += input_flux
            cumulative_out += use_flux
        runs["cut" if cut else "natural"] = rows
    assert runs["cut"][3]["survival"] == "1/2"
    sources = [PROJECT / f"CGBAxioms/{name}.lean" for name in
               ("ActiveMaintenance", "ActiveMaintenanceEnergy", "ActiveMaintenanceCarrier")]
    sources.append(Path(__file__))
    output = {"date": "2026-10-24", "scope": "Exact finite mathematical-model cross-check; no training.",
              "arithmetic": "fractions.Fraction", "carrier_checks": carrier_checks,
              "zero_reserve_disables_registered_channels": True,
              "budget_prefix_checks": 20, "runs": runs,
              "source_hashes": {p.relative_to(PROJECT).as_posix():
                                hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": True, "carrier_checks": carrier_checks,
                      "budget_prefix_checks": 20, "output": str(args.output)}))


if __name__ == "__main__":
    main()
