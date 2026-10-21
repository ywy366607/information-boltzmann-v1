"""Finite reversible exchange on the existing saved graph/pool panel.

Numerical/physical interface only: no new text training, rebirth, source
mutation, structural-conductance update or functional-capability claim.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from information_boltzmann.cgb.checkpoint import load_medium
from information_boltzmann.cgb.exchange_checkpoint import load_exchange, save_exchange
from information_boltzmann.cgb.finite_exchange import ExchangeParams, FiniteExchange
from information_boltzmann.cgb.structural_exchange import StructuralExchange, StructuralExchangeParams
from scripts.ib.generative.compare_support_transport import digest, same


def main(args):
    args.out.mkdir(parents=True, exist_ok=False)
    source_hash = digest(args.archive)
    medium, inherited = load_medium(args.archive)
    paths = [Path(__file__), ROOT/'information_boltzmann/cgb/finite_exchange.py',
             ROOT/'information_boltzmann/cgb/exchange_checkpoint.py',
             ROOT/'information_boltzmann/cgb/structural_exchange.py',
             ROOT/'information_boltzmann/cgb/material_reactions.py']
    hashes = {str(path.relative_to(ROOT)):digest(path) for path in paths}
    snapshots = args.out/'source_snapshot'
    snapshots.mkdir()
    for path in paths:
        (snapshots/path.name).write_bytes(path.read_bytes())
    params = StructuralExchangeParams() if args.structural else ExchangeParams()
    # Explicit reference units for an interface experiment, not a calibrated
    # conversion of the old support signal to physical contact or fuel count.
    values = (medium.pre, medium.post, np.ones(medium.E), medium.Nf, medium.T,
              np.zeros(medium.N), np.full(medium.N, .5), np.full(medium.N, .5))
    state = (StructuralExchange(*values, np.full(medium.E,2.), np.full(medium.N,.5), params)
             if args.structural else FiniteExchange(*values, params))
    initial_mass = float(state.owned().sum())
    initial_charged = float(state.charged.sum())
    initial_g = state.free_energy()
    initial_u = state.energy()
    initial_installed = float(state.capacity.sum())
    initial_material = float(state.material.sum()) if args.structural else None
    max_error = dict(support=0., drive=0., energy=0., entropy=0.)
    if args.structural:
        max_error['material'] = 0.
    largest_g_increase = 0.
    trajectory = []
    started = time.perf_counter()
    for step in range(200):
        update = state.step(1.)
        largest_g_increase = max(largest_g_increase, -update['free_energy_drop'])
        for key, values in state.balances().items():
            max_error[key] = max(max_error[key], float(np.max(np.abs(values), initial=0.)))
        if (step+1) % 20 == 0:
            row = dict(ms=state.t, free_energy=state.free_energy(),
                                   internal_energy=state.energy(), heat_out=state.heat_out,
                                   charged=float(state.charged.sum()),
                                   bound=float(state.bound.sum()), carried=float(state.carried.sum()),
                                   entropy_balance=state.entropy_balance)
            if args.structural:
                row.update(installed=float(state.capacity.sum()), material=float(state.material.sum()))
            trajectory.append(row)
    elapsed = time.perf_counter()-started
    provenance = dict(scope='Finite exchange physical/interface audit, not migrated brain',
                      source_sha256=source_hash, params=asdict(params),
                      capacity='one reference occupancy per existing edge, uncalibrated',
                      drive='P=D=.5 per node, finite closed stock, uncalibrated')
    if args.structural:
        provenance.update(structure='n=1,K=2 per edge; B=.5 per node; no mapping to old g')
    path = args.out/'exchange_complete.npz'
    save_exchange(path, state, provenance)
    restored, saved_provenance = load_exchange(path)
    assert saved_provenance == provenance
    same(state.state_dict(), restored.state_dict())
    for _ in range(20):
        state.step(1.)
        restored.step(1.)
    same(state.state_dict(), restored.state_dict())
    save_exchange(args.out/'exchange_after_restore.npz', state, provenance)
    array_bytes = sum(getattr(state, name).nbytes for name in state._arrays)
    array_bytes += sum(a.nbytes for a in (state.pre, state.post, state.capacity))
    assert largest_g_increase < 1e-10
    assert max(max_error.values()) < 1e-8
    assert digest(args.archive) == source_hash
    assert hashes == {str(path.relative_to(ROOT)):digest(path) for path in paths}
    report = dict(date='2026-10-22', scope=provenance['scope'],
        preregistered=('docs/information_boltzmann/CGB_STRUCTURE_EXCHANGE_PREREG_20261022.md'
                       if args.structural else 'docs/information_boltzmann/CGB_FINITE_EXCHANGE_PREREG_20261022.md'),
        source_archive=str(args.archive), source_sha256=source_hash, inherited_provenance=inherited,
        source_unchanged=True, cells=medium.N, edges=medium.E,
        params=asdict(params), reference_units=provenance,
        actual_integration_ms=200., restore_future_ms=20., integration_dt_ms=1.,
        initial_support=initial_mass, initial_charged=initial_charged,
        initial_free_energy=initial_g, initial_internal_energy=initial_u,
        initial_installed=initial_installed, initial_material=initial_material,
        integration_final=trajectory[-1], largest_free_energy_increase=largest_g_increase,
        maximum_balance_error=max_error, trajectory=trajectory,
        state_array_bytes=array_bytes, archive_bytes=path.stat().st_size,
        seconds_for_200_steps=elapsed, complete_restore_future_bitwise=True,
        archive=str(path), archive_sha256=digest(path), code_sha256=hashes,
        external_material_input=0., external_work_input=0.,
        whole_brain_energy_closed=False, structural_conductance_evolved=False,
        structural_occupation_evolved=args.structural,
        language_training=False, biological_parameters_calibrated=False, passed=True)
    (args.out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--structural',action='store_true')
    main(p.parse_args())
