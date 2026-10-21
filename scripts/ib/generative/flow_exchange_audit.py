"""Matched physical work/structure audit on an existing archived graph.

No text training, original-state migration, or world-learning claim. The
electrical realization and stimulus are reference constitutive choices.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from information_boltzmann.cgb.checkpoint import load_medium
from information_boltzmann.cgb.exchange_checkpoint import load_exchange, save_exchange
from information_boltzmann.cgb.flow_exchange import FlowExchange, FlowExchangeParams
from scripts.ib.generative.compare_support_transport import digest, same


def construct(medium, params, capacity=None):
    return FlowExchange(medium.pre, medium.post,
        np.ones(medium.E) if capacity is None else capacity,
        medium.Nf, medium.T, np.zeros(medium.N),
        np.full(medium.N, .5), np.full(medium.N, .5),
        np.full(medium.E, 2.), np.full(medium.N, .5),
        np.zeros(medium.N), 1., .01, params)


def run(medium, params, current, dt, duration=200.):
    state = construct(medium, params)
    peak_errors = {key: 0. for key in state.balances()}
    max_iterations, least_drop = 0, 0.
    started = time.perf_counter()
    for _ in range(round(duration/dt)):
        update = state.step(dt, current)
        least_drop = min(least_drop, update['free_energy_drop'])
        max_iterations = max(max_iterations, state.last_cg_iterations)
        for name, value in state.balances().items():
            peak_errors[name] = max(peak_errors[name], float(np.max(np.abs(value), initial=0.)))
    report = dict(dt_ms=dt, elapsed_ms=state.t, seconds=time.perf_counter()-started,
        maximum_balance_errors=peak_errors, minimum_irreversible_g_drop=least_drop,
        cg_iterations_max=max_iterations, charged=float(state.charged.sum()),
        installed=float(state.capacity.sum()), material=float(state.material.sum()),
        field_energy=state.field_energy(), internal_energy=state.energy(),
        free_energy=state.free_energy(), work_in=state.work_in, heat_out=state.heat_out,
        entropy_balance=state.entropy_balance, conduction_heat=state.conduction_heat,
        conduction_numerical_heat=state.conduction_numerical_heat,
        conduction_heat_split_error=state.conduction_energy_drop-state.conduction_heat
                                    -state.conduction_numerical_heat)
    assert max(peak_errors.values()) < 1e-8
    assert least_drop >= -1e-10
    return state, report


def main(args):
    args.out.mkdir(parents=True, exist_ok=False)
    source_hash = digest(args.archive)
    medium, inherited = load_medium(args.archive)
    paths = [Path(__file__), ROOT/'information_boltzmann/cgb/flow_exchange.py',
             ROOT/'information_boltzmann/cgb/structural_exchange.py',
             ROOT/'information_boltzmann/cgb/finite_exchange.py',
             ROOT/'information_boltzmann/cgb/material_reactions.py',
             ROOT/'information_boltzmann/cgb/exchange_checkpoint.py',
             ROOT/'docs/information_boltzmann/CGB_FAST_WORK_PREREG_20261022.md']
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in paths}
    snapshots = args.out/'source_snapshot'
    snapshots.mkdir()
    for path in paths:
        (snapshots/path.name).write_bytes(path.read_bytes())
    params = FlowExchangeParams()
    drive = np.zeros(medium.N)
    # Port fixed before execution: same node used in the earlier reverse
    # support experiment. Its usefulness is not selected from this outcome.
    drive[args.port] = .1
    states, reports, archives = {}, {}, {}
    for label, active, powered in (
        ('coupled_driven', True, True), ('coupled_quiet', True, False),
        ('uncoupled_driven', False, True), ('uncoupled_quiet', False, False)):
        p = params if active else replace(params, conversion_rate=0.)
        state, reports[label] = run(medium, p, drive if powered else np.zeros(medium.N), 1.)
        states[label] = state
        path = args.out/(label+'.npz')
        provenance = dict(source_sha256=source_hash, label=label,
            reference_units=True, not_a_migrated_brain=True, port=args.port,
            current=.1 if powered else 0., params=asdict(p))
        save_exchange(path, state, provenance)
        restored, saved = load_exchange(path)
        assert saved == provenance
        same(state.state_dict(), restored.state_dict())
        archives[label] = dict(path=str(path), sha256=digest(path))
    a, b, c, d = (states[k] for k in ('coupled_driven', 'coupled_quiet',
                                     'uncoupled_driven', 'uncoupled_quiet'))
    contrasts = {}
    for name in ('charged', 'capacity', 'material'):
        on = getattr(a, name)-getattr(b, name)
        off = getattr(c, name)-getattr(d, name)
        contrasts[name] = dict(coupled_delta_sum=float(on.sum()),
            uncoupled_delta_absmax=float(np.max(np.abs(off), initial=0.)),
            difference_in_differences_absmax=float(np.max(np.abs(on-off), initial=0.)))
        assert np.array_equal(getattr(c, name), getattr(d, name))
        assert contrasts[name]['difference_in_differences_absmax'] > 1e-9

    # Conditional acute response: only installed occupancy varies; same q and
    # C, no intervening conversion/assembly. This is not a behavioral test.
    probe_a, probe_b = construct(medium, params, a.capacity), construct(medium, params, b.capacity)
    probe_a.charge[args.port] = probe_b.charge[args.port] = 1.
    probe_a._conduction(1.)
    probe_b._conduction(1.)
    response_delta = float(np.max(np.abs(probe_a.charge-probe_b.charge)))
    assert response_delta > 1e-10

    # Step refinement of the complete coupled model, not a fixed-clock probe.
    refinements, refined = {}, {}
    for dt in (.5, .25):
        refined[dt], refinements[str(dt)] = run(medium, params, drive, dt)
    fields = ('charge', 'charged', 'capacity', 'free', 'bound', 'carried', 'material')
    def distance(x, y):
        return float(np.sqrt(sum(np.sum((getattr(x, name)-getattr(y, name))**2) for name in fields)))
    coarse_difference = distance(a, refined[.5])
    fine_difference = distance(refined[.5], refined[.25])
    assert fine_difference < coarse_difference
    restored, _ = load_exchange(args.out/'coupled_driven.npz')
    for _ in range(20):
        a.step(1., drive)
        restored.step(1., drive)
    same(a.state_dict(), restored.state_dict())
    save_exchange(args.out/'coupled_after_restore.npz', restored,
                  dict(source_sha256=source_hash, elapsed_ms=220., restore_future_bitwise=True))
    assert digest(args.archive) == source_hash
    assert hashes == {str(path.relative_to(ROOT)): digest(path) for path in paths}
    state_bytes = sum(getattr(a, name).nbytes for name in a._arrays)
    state_bytes += sum(x.nbytes for x in (a.pre, a.post, a.capacity))
    report = dict(date='2026-10-22', scope='physical two-way work/structure delivery; not learning benefit',
        baseline_git=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        source_archive=str(args.archive), source_sha256=source_hash, source_unchanged=True,
        inherited_provenance=inherited, code_sha256=hashes,
        preregistered='docs/information_boltzmann/CGB_FAST_WORK_PREREG_20261022.md',
        cells=medium.N, contacts=medium.E, current_port=args.port,
        params=asdict(params), initial_installed=medium.E, initial_charged=.5*medium.N,
        arms=reports, contrast=contrasts, acute_fast_response_charge_absmax=response_delta,
        refinement=dict(arms=refinements, dt1_vs_dt05=coarse_difference,
            dt05_vs_dt025=fine_difference, ratio=coarse_difference/fine_difference),
        source_archives=archives, complete_restore_future_bitwise=True, restore_future_ms=20.,
        state_array_bytes=state_bytes, solver_workspace='matrix-free O(N+E), no dense factorization',
        delivery_observed=True, learning_or_behavior_benefit='unmeasured',
        directed_active_spiking='not implemented in this passive realization',
        world_consequence_loop='not coupled in this audit', language_training=False, passed=True)
    (args.out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('contrast', 'acute_fast_response_charge_absmax',
        'state_array_bytes', 'complete_restore_future_bitwise', 'passed')}, ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--port', type=int, default=88)
    main(p.parse_args())
