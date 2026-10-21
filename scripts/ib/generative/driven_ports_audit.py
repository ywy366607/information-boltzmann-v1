"""Reference physical interfaces driven by retained real UTF-8 text.

Tests conservative delivery and finite world reaction, not language learning.
All source voltages, material constants and byte clocks are declared here.
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
from information_boltzmann.cgb.driven_exchange import DrivenExchange, DrivenExchangeParams
from information_boltzmann.cgb.exchange_checkpoint import load_exchange, save_exchange
from scripts.ib.generative.compare_support_transport import digest, same


def construct(m, active=True):
    return DrivenExchange(m.pre, m.post, np.ones(m.E), m.Nf, m.T, np.zeros(m.N),
        np.full(m.N, .5), np.full(m.N, .5), np.full(m.E, 2.), np.full(m.N, .5),
        np.zeros(m.N), params=DrivenExchangeParams(active_rate=.001 if active else 0.),
        port_nodes=np.arange(8), world_charge=np.zeros(8), world_capacitance=2.,
        port_conductance=.05, source_conductance=.1)


def input_voltage(data, step, dt):
    byte = data[min(len(data)-1, int(step*dt/4.))]
    return np.array([3. if (byte >> bit) & 1 else -3. for bit in range(8)])


def reset_world_control(m):
    """Explicit energy-extracting diagnostic intervention, never a live reset."""
    s = m.state_dict()
    removed = m.world_energy()
    s['arrays']['world_injected'] -= s['arrays']['world_charge']
    s['arrays']['world_charge'][:] = 0.
    s['work_in'] -= removed
    s['source_work'] -= removed
    return DrivenExchange.from_state_dict(s), removed


def main(args):
    args.out.mkdir(parents=True, exist_ok=False)
    source_hash = digest(args.archive)
    medium, inherited = load_medium(args.archive)
    shelf_path = ROOT/'data/harness/reading/shelf.json'
    shelf = json.loads(shelf_path.read_text(encoding='utf-8'))
    story = shelf['entries'][0]
    data = story['text'].encode('utf-8')[:24]
    assert len(data) == 24 and medium.N >= 8
    source_paths = [Path(__file__), ROOT/'information_boltzmann/cgb/driven_exchange.py',
        ROOT/'information_boltzmann/cgb/flow_exchange.py',
        ROOT/'information_boltzmann/cgb/structural_exchange.py',
        ROOT/'information_boltzmann/cgb/finite_exchange.py',
        ROOT/'information_boltzmann/cgb/material_reactions.py',
        ROOT/'information_boltzmann/cgb/exchange_checkpoint.py',
        ROOT/'docs/information_boltzmann/CGB_DRIVEN_PORTS_PREREG_20261022.md']
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in source_paths}
    snapshot = args.out/'source_snapshot'
    snapshot.mkdir()
    for p in source_paths:
        (snapshot/p.name).write_bytes(p.read_bytes())
    (args.out/'reading_shelf_snapshot.json').write_bytes(shelf_path.read_bytes())
    states, reports = {}, {}
    for label, active, text in (('active_text', True, True), ('passive_text', False, True),
                                ('active_quiet', True, False), ('passive_quiet', False, False)):
        state = construct(medium, active)
        max_errors = {k: 0. for k in state.balances()}
        steps = len(data)*4
        fork, reset, reset_energy = None, None, None
        least_drop = 0.
        started = time.perf_counter()
        for k in range(steps):
            voltage = input_voltage(data, k, 1.) if text else np.zeros(8)
            update = state.step(1., world_voltage=voltage)
            least_drop = min(least_drop, update['free_energy_drop'])
            if fork is not None:
                fork.step(1., world_voltage=voltage)
            if reset is not None:
                reset.step(1., world_voltage=voltage)
            for name, value in state.balances().items():
                max_errors[name] = max(max_errors[name], float(np.max(np.abs(value), initial=0.)))
            if k+1 == steps//2:
                provenance = dict(source_sha256=source_hash, arm=label, next_step=k+1,
                    byte_stream_hex=data.hex(), byte_interval_ms=4., integration_dt_ms=1.,
                    stimulus='source voltage +/-3 on eight fixed UTF-8 bit ports',
                    language_training=False)
                save_exchange(args.out/(label+'_half.npz'), state, provenance)
                fork, meta = load_exchange(args.out/(label+'_half.npz'))
                assert meta == provenance
                if label == 'active_text':
                    reset, reset_energy = reset_world_control(state)
        same(state.state_dict(), fork.state_dict())
        assert max(max_errors.values()) < 1e-8 and least_drop >= -1e-10
        path = args.out/(label+'_complete.npz')
        save_exchange(path, state, dict(source_sha256=source_hash, arm=label,
            next_step=steps, byte_stream_hex=data.hex(), byte_interval_ms=4., integration_dt_ms=1.))
        states[label] = state
        reports[label] = dict(elapsed_ms=state.t, seconds_with_full_restore_replay=time.perf_counter()-started,
            active_net=float(state.active_net.sum()), active_absolute_net=float(np.abs(state.active_net).sum()),
            charged=float(state.charged.sum()), installed=float(state.capacity.sum()),
            world_voltage=state.world_voltage.tolist(), source_work=state.source_work,
            source_heat=state.source_heat, active_heat=state.active_heat, port_heat=state.port_heat,
            total_work_in=state.work_in, total_heat_out=state.heat_out,
            maximum_balance_errors=max_errors, minimum_irreversible_g_drop=least_drop,
            archive=str(path), sha256=digest(path), full_restore_future_bitwise=True,
            active_matching_groups=len(state._edge_groups))
        if reset is not None:
            reports[label]['world_reset_control'] = dict(
                explicitly_extracted_world_energy=reset_energy,
                subsequent_brain_charge_absmax=float(np.max(np.abs(state.charge-reset.charge))),
                subsequent_structure_absmax=float(np.max(np.abs(state.capacity-reset.capacity))),
                intervention_only='diagnostic external state reset; energy and charge recorded; not a live reset')
            assert reports[label]['world_reset_control']['subsequent_brain_charge_absmax'] > 1e-9
        print(json.dumps(dict(arm=label, **reports[label]), ensure_ascii=False), flush=True)
    a, b = states['active_text'], states['passive_text']
    contrast = dict(brain_charge_absmax=float(np.max(np.abs(a.charge-b.charge))),
        world_charge_absmax=float(np.max(np.abs(a.world_charge-b.world_charge))),
        structure_absmax=float(np.max(np.abs(a.capacity-b.capacity))))
    assert a.active_net.sum() > 0 and np.array_equal(b.active_net, np.zeros(medium.E))
    assert min(contrast.values()) > 1e-9
    # Refine complete model with the same continuous source schedule.
    refinements, previous, differences = {}, a, []
    fields = ('charge', 'charged', 'depleted', 'capacity', 'world_charge', 'bound', 'free', 'carried', 'material')
    for dt in (.5, .25):
        state = construct(medium)
        started = time.perf_counter()
        for k in range(round(len(data)*4/dt)):
            state.step(dt, world_voltage=input_voltage(data, k, dt))
        difference = float(np.sqrt(sum(np.sum((getattr(state, f)-getattr(previous, f))**2) for f in fields)))
        differences.append(difference)
        refinements[str(dt)] = dict(elapsed_ms=state.t, seconds=time.perf_counter()-started,
            difference_from_previous=difference,
            maximum_balance_error=max(float(np.max(np.abs(v), initial=0.)) for v in state.balances().values()))
        previous = state
    assert differences[1] < differences[0]
    assert hashes == {str(p.relative_to(ROOT)): digest(p) for p in source_paths}
    assert digest(args.archive) == source_hash
    report = dict(date='2026-10-22', scope='finite fueled directional and world-state interface, not learning capability',
        baseline_git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        preregistered='docs/information_boltzmann/CGB_DRIVEN_PORTS_PREREG_20261022.md',
        source_archive=str(args.archive), source_sha256=source_hash, source_unchanged=True,
        inherited_provenance=inherited, cells=medium.N, contacts=medium.E,
        story=story['title'], story_source=story['source'], reading_shelf_sha256=digest(shelf_path),
        byte_stream_hex=data.hex(), bytes_presented=len(data), source_clock_ms_per_byte=4.,
        fixed_encoding='UTF-8 bits -> +/-3 source volts, ports 0..7; not semantic embeddings',
        params=asdict(a.params), arms=reports, active_vs_passive=contrast,
        refinement=refinements, refinement_difference_ratio=differences[0]/differences[1],
        state_array_bytes=sum(getattr(a,f).nbytes for f in a._arrays)+sum(x.nbytes for x in (a.pre,a.post,a.capacity,a.port_nodes)),
        code_sha256=hashes, complete_restore_future_bitwise=True, new_language_training=False,
        learning_or_behavior_benefit='unmeasured', full_language_world_operation_closed=False, passed=True)
    (args.out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
