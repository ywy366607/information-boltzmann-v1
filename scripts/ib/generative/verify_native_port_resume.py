"""Independently recover complete native-port provenance and local adaptation."""
import argparse
import copy
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from information_boltzmann.cgb.exchange_checkpoint import load_exchange
from information_boltzmann.cgb.local_learning import local_learning_affinity
from information_boltzmann.cgb.material_fork import fork_material
from information_boltzmann.cgb.mobile_exchange import MaterialEvent
from scripts.ib.generative.compare_support_transport import digest, same
from scripts.ib.generative.domain_rate_audit import CarrierObservation
from scripts.ib.generative.native_learning_stream import advance_byte, atomic_json


def main(run, data):
    final, fmeta = load_exchange(run/'natural_complete.npz')
    final_hash = digest(run/'natural_complete.npz')
    m, meta = load_exchange(run/'natural_mid.npz')
    initial = m.state_dict()
    corpus = meta['sources']['corpus']
    assert digest(data/'train.npy') == corpus['files_sha256']['train.npy']
    tokens = np.load(data/'train.npy', mmap_mode='r')
    cursor = meta['experience_protocol']['next_train_byte']
    last = int(tokens[cursor-1])-corpus['tokenizer']['offset']
    assert 0 <= last <= 255
    session = copy.deepcopy(meta['session'])
    observer = CarrierObservation(m, meta['growth_observer'], meta['carrier_observer'])
    m.domain_observer = observer.domain
    affinity_samples = []
    decomposition_count = 0
    depletion_events = 0
    recovery_events = 0
    max_decomposition = 0.

    def native(row):
        nonlocal depletion_events, recovery_events
        observer.native(row)
        # The callback observes the already executed state and receipt. Rebuild
        # only the pre-event local composition, not a new physical history.
        for name, node, delta in row['composition_changes']:
            if name != 'charged' or row['kind'] not in ('active', 'write', 'conversion'):
                continue
            before_p = int(m.charged[node])-delta
            before_d = int(m.depleted[node])+delta
            after_p, after_d = int(m.charged[node]), int(m.depleted[node])
            if min(before_p, after_p) == 0:
                shift = None
            else:
                shift = (np.log(after_d+1)-np.log(after_p)
                         -np.log(before_d+1)+np.log(before_p)) / m.params.charge_per_event
                assert (shift > 0) if delta < 0 else (shift < 0)
            depletion_events += delta < 0
            recovery_events += delta > 0
            affinity_samples.append(dict(time_ms=row['time_ms'], kind=row['kind'],
                sign=row['sign'], material_id=observer.state['ids'][node],
                before_p=before_p, before_d=before_d, after_p=after_p, after_d=after_d,
                threshold_composition_shift=None if shift is None else float(shift),
                scope='Local composition contribution, electrical/support changes kept separate'))

    m.event_observer = native
    original = [json.loads(x) for x in (run/'natural_intervals.jsonl').read_text().splitlines()][128:]
    for expected in original:
        constitution = m.port_constitution().state_dict()
        assert constitution == expected['native_constitution']
        persistence = (m.world_charge/m.world_capacitance).copy()
        previous = np.array([3. if (last >> bit) & 1 else -3. for bit in range(8)])
        held = previous+(np.asarray(session['source_only_voltage'])-previous)*np.exp(
            -4*m.source_conductance/m.world_capacitance)
        np.testing.assert_array_equal(persistence, expected['world_persistence'])
        np.testing.assert_array_equal(held, expected['held_previous_source_mean'])
        for event in m.events():
            if event.kind in ('write', 'active') and event.sign == 1:
                value = m.learning_affinity(event)
                z = m.params.charge_per_event
                factored = z*(value.half_attempt_voltage_drop-value.voltage_drop)
                max_decomposition = max(max_decomposition, abs(factored-value.delta_g))
                assert abs(factored-value.delta_g) < 2e-12
                decomposition_count += 1
        target = int(tokens[expected['corpus_train_byte']])-corpus['tokenizer']['offset']
        row = advance_byte(m, session, target, 'fresh_H')
        assert all(expected[key] == value for key, value in row.items())
        actual = m.world_charge/m.world_capacitance
        risks = {key:.5*float(np.sum(m.world_capacitance*(value-actual)**2))
                 for key, value in [('native_matching',np.asarray(constitution['matching_world_voltage'])),
                                    ('world_persistence',persistence),('held_previous_source_mean',held)]}
        assert risks == expected['quadratic_sensory_risk']
        last = target
    same(m.state_dict(), final.state_dict())
    assert session == fmeta['session']
    assert observer.state == fmeta['growth_observer'] and observer.carriers == fmeta['carrier_observer']
    assert meta['inherited_external_observation'] == fmeta['inherited_external_observation']
    assert final_hash == digest(run/'natural_complete.npz')
    atomic_json(run/'full_metadata_resume.json',dict(passed=True,actual_diagnostic_replay_bytes=len(original),
        native_values_forecasts_risks_physics_rngs_session_observers_exact=True,
        baseline_lastbyte_source='Immutable corpus at saved next_train_byte minus one; no future target',
        inherited_external_observation_preserved=True,main_candidate_unchanged=True,last_actual_byte=last))
    atomic_json(run/'native_adaptation_checks.json',dict(passed=True,
        source='Actual native reactions in diagnostic replay of the saved final 128 experienced OWT bytes',
        new_primary_text_exposures=0,all_admitted_forward_transfer_decompositions=decomposition_count,
        maximum_decomposition_error=max_decomposition,depletion_events=depletion_events,
        recovery_events=recovery_events,actual_local_transitions=affinity_samples,
        scope='Finite composition causes endogenous adaptive affinity; not full V2 timing or learned prediction',
        main_candidate_unchanged=True))


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--data', type=Path, default=ROOT/'data/ib_owt_10000')
    args = p.parse_args()
    main(args.run, args.data)
