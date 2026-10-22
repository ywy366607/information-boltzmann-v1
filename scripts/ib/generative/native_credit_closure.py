"""Cross native activity configuration with actual downstream support return.

Full-state, matched-clock research forks of a retained individual. Forced
admitted events are conditional interventions, not a new autonomous policy.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from information_boltzmann.cgb.credit_observation import contact_roles, resource_bandwidth, support_return_paths
from information_boltzmann.cgb.exchange_checkpoint import load_exchange, save_exchange
from information_boltzmann.cgb.material_fork import fork_material
from information_boltzmann.cgb.memory_intervention import assert_matched_trace_states
from information_boltzmann.cgb.learning_observation import parameter_quench
from information_boltzmann.cgb.mobile_exchange import MaterialEvent
from scripts.ib.generative.compare_support_transport import same
from scripts.ib.generative.domain_rate_audit import CarrierObservation
from scripts.ib.generative.native_learning_stream import advance_byte, atomic_json


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def locate(model, observer, a, b):
    ids = observer.state['ids']
    try:
        i, j = ids.index(a), ids.index(b)
    except ValueError:
        return None
    edges = np.flatnonzero((model.pre == i) & (model.post == j))
    return int(edges[0]) if len(edges) else None


def target_panel(model, observer, target, original_carrier):
    key = 'i:'+str(target[0])+':'+str(target[1])
    edge = locate(model, observer, *target)
    if edge is None:
        return dict(installed=False, original_carrier_alive=False, rebuilt=False,
                    bound_support=0, activity_configuration=0, capture_rate=0.,
                    basal_installed_conductance=0., dismantle_rate=0.)
    current = observer.carriers['alive'][key]
    return dict(**contact_roles(model, edge), original_carrier_alive=current == original_carrier,
                rebuilt=current != original_carrier, carrier_id=current)


def save(path, model, observer, session, budget, source):
    save_exchange(path, model, dict(growth_observer=copy.deepcopy(observer.state),
                  carrier_observer=copy.deepcopy(observer.carriers), session=copy.deepcopy(session),
                  resource_budget=copy.deepcopy(budget), source=source,
                  is_main_candidate=False, conditional_diagnostic=True))


def main(args):
    args.out.mkdir(parents=True, exist_ok=False)
    parent, metadata = load_exchange(args.archive)
    parent_hash = digest(args.archive)
    original = parent.state_dict()
    if parent.params.polarized_storage:
        raise ValueError('This preregistered comparison uses the retained scalar material')
    paths = support_return_paths(parent)
    if not paths:
        atomic_json(args.out/'blocked.json', dict(reason='No admitted support return path; no synthetic path created'))
        return
    # Present-state eligibility only. No future score, target text, or result
    # enters path selection, and no failed delayed path is replaced.
    path = paths[0]
    ids = metadata['growth_observer']['ids']
    target = (ids[path['pre']], ids[path['middle']])
    downstream = (ids[path['middle']], ids[path['downstream']])
    target_key = 'i:'+str(target[0])+':'+str(target[1])
    carrier = metadata['carrier_observer']['alive'][target_key]
    offset = metadata['experience_protocol']['next_train_byte']
    train_path = args.data/'train.npy'
    corpus = metadata['sources']['corpus']
    assert digest(train_path) == corpus['files_sha256']['train.npy']
    raw = np.asarray(np.load(train_path, mmap_mode='r')[offset:offset+24])
    assert len(raw) == 24 and np.all((raw >= 4) & (raw < 260))
    block = np.asarray(raw-4, np.uint8)
    np.save(args.out/'actual_input.npy', block, allow_pickle=False)
    source = dict(parent_archive=str(args.archive), parent_sha256=parent_hash,
                  corpus=corpus, train_byte_range=[offset, offset+24],
                  actual_input_sha256=hashlib.sha256(block.tobytes()).hexdigest(),
                  target_stable_ids=list(target), downstream_stable_ids=list(downstream),
                  original_carrier_id=carrier, native_path=path,
                  delays_ms=args.delays_ms, future_noise_groups=8, post_return_ms=32,
                  response_control=args.response_control,
                  main_individual_advanced=False)
    paths_to_save = [Path(__file__), ROOT/'information_boltzmann/cgb/credit_observation.py',
                     ROOT/'docs/information_boltzmann/CGB_NATIVE_CREDIT_CLOSURE_20261022.md']
    if args.response_control == 'flat':
        paths_to_save.append(ROOT/'docs/information_boltzmann/CGB_NATIVE_CREDIT_RESPONSE_CONTROL_20261022.md')
    paths_to_save += sorted((ROOT/'information_boltzmann/cgb').glob('*.py'))
    paths_to_save += [ROOT/'scripts/ib/generative'/name for name in
                      ('domain_birth_audit.py', 'domain_rate_audit.py', 'native_learning_stream.py',
                       'compare_support_transport.py')]
    for file in set(paths_to_save):
        dest = args.out/'source_snapshot'/file.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, dest)
    source['code_sha256'] = {str(p.relative_to(ROOT)): digest(p) for p in set(paths_to_save)}
    atomic_json(args.out/'sources.json', source)

    prepared = fork_material(parent)
    # Preserve the inherited ledger residual while preventing new cumulative
    # summation loss. This changes arithmetic only, not a physical rate/state.
    prepared.enable_compensated_ledgers()
    if args.response_control == 'flat':
        response_intervention = parameter_quench(prepared, gate_high=prepared.params.gate_low)
        atomic_json(args.out/'response_intervention.json', response_intervention)
    obs = CarrierObservation(prepared, metadata['growth_observer'], metadata['carrier_observer'])
    setup = []
    def record_setup(row):
        obs.native(row)
        setup.append(row)
    prepared.event_observer = record_setup
    edge = locate(prepared, obs, *target)
    i, j = int(prepared.pre[edge]), int(prepared.post[edge])
    if prepared.trace[edge]:
        prepared.apply_event(MaterialEvent('thermal', i, j, edge, -1))
    if prepared.charged[i] == 0:
        prepared.apply_event(MaterialEvent('conversion', i))
    prepared.event_observer = None
    atomic_json(args.out/'common_setup.json', setup)
    summaries = []
    started = time.perf_counter()
    with gzip.open(args.out/'events.jsonl.gz', 'wt', encoding='utf-8') as stream:
        for group in range(8):
            for delay in args.delays_ms:
                pair = []
                for marked in (False, True):
                    m = fork_material(prepared)
                    edge = locate(m, obs, *target)
                    e = MaterialEvent('write' if marked else 'active', i, j, edge)
                    m.apply_event(e)
                    for order, name in enumerate(('rng_reaction', 'rng_motion', 'rng_field')):
                        setattr(m, name, np.random.default_rng(20261022+group*100+order))
                    pair.append(m)
                assert_matched_trace_states(pair[1], pair[0])
                for marked in (False, True):
                    for returned in (False, True):
                        label = f'g{group}_d{delay}_m{int(marked)}_r{int(returned)}'
                        m = fork_material(pair[int(marked)])
                        observer = CarrierObservation(m, obs.state, obs.carriers)
                        session = copy.deepcopy(metadata['session'])
                        session.update(phase='conditional_credit', next_phase_byte=0)
                        budget = dict(start_charged=int(m.charged.sum()), used=0, recovered=0)
                        target_events = dict(binding=0, unbinding=0)
                        def native(row):
                            observer.native(row)
                            for name, node, change in row['composition_changes']:
                                if name == 'charged':
                                    budget['recovered' if change > 0 else 'used'] += abs(change)
                            if (row['kind'] == 'binding' and
                                (row['observed_pre_id'], row.get('observed_post_id')) == target):
                                target_events['binding' if row['sign'] > 0 else 'unbinding'] += 1
                            stream.write(json.dumps(dict(arm=label, event=row), separators=(',', ':'))+'\n')
                        def domain(row):
                            observer.domain(row)
                            if row['accepted']:
                                budget['used' if row['kind'] == 'domain_birth' else 'recovered'] += 1
                            stream.write(json.dumps(dict(arm=label, event=row), separators=(',', ':'))+'\n')
                        m.event_observer, m.domain_observer = native, domain
                        for index in range(delay//4):
                            advance_byte(m, session, int(block[index]), 'conditional_credit')
                        before_return = target_panel(m, observer, target, carrier)
                        return_receipts = []
                        applied = False
                        if returned:
                            down = locate(m, observer, *downstream)
                            if down is not None:
                                a, b = int(m.pre[down]), int(m.post[down])
                                transport = MaterialEvent('transport', a, b, down)
                                if transport in m.events():
                                    du, dg = m.event_delta(transport)
                                    m.apply_event(transport)
                                    release = MaterialEvent('release', a)
                                    assert release in m.events()
                                    u, g = m.event_delta(release)
                                    m.apply_event(release)
                                    return_receipts = [dict(kind='transport+', delta_u=du, delta_g=dg),
                                                       dict(kind='release+', delta_u=u, delta_g=g)]
                                    applied = True
                        after_return = target_panel(m, observer, target, carrier)
                        # All preparation jumps have been explicitly accounted.
                        resource0 = resource_bandwidth(m)
                        representative = group == 0 and delay == 0
                        if representative:
                            save(args.out/(label+'_start.npz'), m, observer, session, budget, source)
                        resume = resume_session = resume_observer = None
                        for index in range(delay//4, delay//4+8):
                            row = advance_byte(m, session, int(block[index]), 'conditional_credit')
                            if resume is not None:
                                assert row == advance_byte(resume, resume_session, int(block[index]), 'conditional_credit')
                            observer.validate(m)
                            if representative and index == delay//4+3:
                                file = args.out/(label+'_mid.npz')
                                save(file, m, observer, session, budget, source)
                                resume, restored = load_exchange(file)
                                resume_session = restored['session']
                                resume_observer = CarrierObservation(resume, restored['growth_observer'], restored['carrier_observer'])
                                resume.event_observer, resume.domain_observer = resume_observer.native, resume_observer.domain
                        if representative:
                            same(m.state_dict(), resume.state_dict())
                            assert session == resume_session
                            assert observer.state == resume_observer.state and observer.carriers == resume_observer.carriers
                            save(args.out/(label+'_complete.npz'), m, observer, session, budget, source)
                        assert int(m.charged.sum()) == budget['start_charged']+budget['recovered']-budget['used']
                        assert budget['used'] <= budget['start_charged']+budget['recovered']
                        final = target_panel(m, observer, target, carrier)
                        # Accounted diagnostic washout on a fork only. No history
                        # is erased in the evaluated full material evolution.
                        washed = fork_material(m)
                        heat0 = washed.heat_out
                        cleared = 0
                        for e in list(washed.events()):
                            if e.kind == 'thermal' and e.sign == -1:
                                washed.apply_event(e)
                                cleared += 1
                        wash = target_panel(washed, observer, target, carrier)
                        error = max(abs(float(v)) for v in m.balances().values())
                        assert error < 1e-8
                        result = dict(arm=label, group=group, delay_ms=delay, marked=marked,
                                      requested_return=returned, return_applied=applied,
                                      return_receipts=return_receipts, before_return=before_return,
                                      after_return=after_return, final=final, after_configuration_washout=wash,
                                      washed_configurations=cleared, washout_reaction_heat=washed.heat_out-heat0,
                                      actual_ms=m.t-prepared.t, actual_bytes=delay//4+8,
                                      actual_time_ms=m.t, resource_budget=budget,
                                      initial_conditional_bandwidth=resource0,
                                      target_events=target_events, maximum_error=error,
                                      resumed_future_exact=representative)
                        summaries.append(result)
                        atomic_json(args.out/(label+'_summary.json'), result)
                atomic_json(args.out/'progress.json', dict(status='running', completed=len(summaries),
                            group=group, delay_ms=delay, elapsed_seconds=time.perf_counter()-started))
                print(json.dumps(dict(completed=len(summaries), group=group, delay=delay,
                                      elapsed_seconds=time.perf_counter()-started)), flush=True)
                stream.flush()
    same(original, parent.state_dict())
    assert digest(args.archive) == parent_hash
    report = dict(source=source, rows=summaries, full_state_source_unchanged=True,
                  numerical_unit_tests='Separate JUnit file',
                  scope='Conditional native delivery/retention with real input; not predictive-credit or language capability proof')
    atomic_json(args.out/'report.json', report)
    atomic_json(args.out/'progress.json', dict(status='completed', completed=len(summaries)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--response-control', choices=('native', 'flat'), default='native')
    parser.add_argument('--delays-ms', nargs='+', type=int, choices=(0, 16, 64), default=[0, 16, 64])
    main(parser.parse_args())
