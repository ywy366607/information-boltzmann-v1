"""Continue real text with paid domain birth and the original local learning."""
from __future__ import annotations

import argparse
import copy
import gzip
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from information_boltzmann.cgb.exchange_checkpoint import load_exchange, save_exchange
from information_boltzmann.cgb.growing_material import GrowingIdentityMobileExchange
from scripts.ib.generative.compare_support_transport import digest, same
from scripts.ib.generative.native_learning_stream import advance_byte, atomic_json


class Observation:
    """Disposable IDs follow array compaction; no history enters physics."""
    def __init__(self, model, state=None):
        self.state = (copy.deepcopy(state) if state is not None else
                      dict(ids=list(range(model.n)), initial_n=model.n, next_id=model.n,
                           births=0, deaths=0, original_events={}, new_domain_events={},
                           geometry_rejections=0, acceptance_rejections=0))
        assert len(self.state['ids']) == model.n

    def native(self, row):
        ids = self.state['ids']
        row['observed_pre_id'] = ids[row['i']]
        if row['j'] >= 0:
            row['observed_post_id'] = ids[row['j']]
        key = ('port_' if row.get('interface', -1) >= 0 else '')+row['kind']+('+' if row['sign'] > 0 else '-')
        group = self.state['original_events']
        group[key] = group.get(key, 0)+1
        if any(row.get(name, -1) >= self.state['initial_n']
               for name in ('observed_pre_id', 'observed_post_id')):
            group = self.state['new_domain_events']
            group[key] = group.get(key, 0)+1

    def domain(self, row):
        state = self.state
        row['observed_parent_id'] = state['ids'][row['parent_before']]
        if not row['accepted']:
            key = ('geometry_rejections' if row['rejection'] == 'outside allowed physical geometry'
                   else 'acceptance_rejections')
            state[key] += 1
            return
        if row['kind'] == 'domain_birth':
            row['observed_child_id'] = state['next_id']
            state['ids'].insert(row['child_after'], state['next_id'])
            state['next_id'] += 1
            state['births'] += 1
        else:
            row['observed_child_id'] = state['ids'].pop(row['child_before'])
            state['deaths'] += 1

    def validate(self, model):
        s = self.state
        assert len(s['ids']) == model.n and len(set(s['ids'])) == model.n
        assert model.n == s['initial_n']+s['births']-s['deaths']
        assert s['births'] == model.domain_births and s['deaths'] == model.domain_deaths


def snapshot(path, model, session, observer, sources):
    observer.validate(model)
    save_exchange(path, model, dict(session=copy.deepcopy(session),
        growth_observer=copy.deepcopy(observer.state), sources=sources, byte_ms=4., dt_ms=2.))


def frame(model, observer, cursor):
    observer.validate(model)
    return dict(time_ms=model.t, next_phase_byte=cursor, coordinates=model.coordinates.tolist(),
                node_ids=list(observer.state['ids']), pre=model.pre.tolist(), post=model.post.tolist(),
                material=model.material.tolist(), charged=model.charged.tolist(),
                trace=model.trace.tolist(), support=model.bound.tolist(),
                port_nodes=model.port_nodes.tolist(), capacitance=model.capacitance.tolist(),
                port_installed=model.port_installed.tolist(),
                births=model.domain_births, deaths=model.domain_deaths,
                attempts=model.domain_attempts, source_work=model.work_in, heat_out=model.heat_out,
                maximum_balance_error=max(abs(float(v)) for v in model.balances().values()))


def run_arm(label, parent, metadata, block, out, sources):
    model = GrowingIdentityMobileExchange.from_boundary(parent, domain_reactions_enabled=label == 'growing')
    session = copy.deepcopy(metadata['session'])
    session.update(phase='fresh_C', next_phase_byte=0)
    session['interventions'].append(dict(time_ms=model.t, material_domain_phase_enabled=label == 'growing',
        material_created=0, baseline_account_refinement=dict(material_domains=model.n,
        added_constant_energy=model.n*model.domain_formation_energy),
        scope='Declared material phase space; reference measure and constitutive choices archived'))
    observer = Observation(model)
    snapshot(out/(label+'_start.npz'), model, session, observer, sources)
    source_work, heat = model.work_in, model.heat_out
    first_t = model.t
    frames = [frame(model, observer, 0)]
    rows = []
    fork = fork_meta = fork_session = fork_observer = None
    reference = type(parent).from_state_dict(parent.state_dict()) if label == 'closed_domain_barrier' else None
    reference_session = copy.deepcopy(session)
    errors = {key:0. for key in model.balances()}
    started = time.perf_counter()
    with gzip.open(out/(label+'_events.jsonl.gz'), 'wt', encoding='utf-8') as events, \
            gzip.open(out/(label+'_domains.jsonl.gz'), 'wt', encoding='utf-8') as domains, \
            (out/(label+'_scores.jsonl')).open('w', encoding='utf-8') as scores:
        def native_receipt(row):
            observer.native(row)
            events.write(json.dumps(row, separators=(',', ':'))+'\n')

        def domain_receipt(row):
            observer.domain(row)
            domains.write(json.dumps(row, separators=(',', ':'))+'\n')

        model.event_observer, model.domain_observer = native_receipt, domain_receipt
        for index, target in enumerate(block):
            row = advance_byte(model, session, int(target), 'fresh_C')
            row.update(phase='fresh_C', phase_byte=index, corpus_train_byte=1280+index)
            rows.append(row)
            scores.write(json.dumps(row, separators=(',', ':'))+'\n')
            if reference is not None:
                expected = advance_byte(reference, reference_session, int(target), 'fresh_C')
                assert row['scores'] == expected['scores']
            if fork is not None:
                expected = advance_byte(fork, fork_session, int(target), 'fresh_C')
                assert row['scores'] == expected['scores']
            observer.validate(model)
            for key, value in model.balances().items():
                errors[key] = max(errors[key], abs(float(value)))
            if (index+1) % 16 == 0:
                frames.append(frame(model, observer, index+1))
                atomic_json(out/(label+'_frames.json'), frames)
                events.flush(); domains.flush(); scores.flush()
                progress = dict(arm=label, next_phase_byte=index+1, time_ms=model.t, nodes=model.n,
                                births=model.domain_births, deaths=model.domain_deaths,
                                bridges=len(model.pre), maximum_error=max(errors.values()))
                atomic_json(out/'progress.json', dict(status='running', **progress))
                print(json.dumps(progress), flush=True)
            if index+1 == len(block)//2:
                path = out/(label+'_mid.npz')
                snapshot(path, model, session, observer, sources)
                fork, fork_meta = load_exchange(path)
                fork_session = copy.deepcopy(fork_meta['session'])
                fork_observer = Observation(fork, fork_meta['growth_observer'])
                fork.event_observer, fork.domain_observer = fork_observer.native, fork_observer.domain
        assert max(errors.values()) < 1e-8
        same(model.state_dict(), fork.state_dict())
        assert session == fork_session and observer.state == fork_observer.state
        if reference is not None:
            for name in parent._arrays:
                if name != 'initial_material':
                    np.testing.assert_array_equal(getattr(model, name), getattr(reference, name))
            np.testing.assert_array_equal(model.pre, reference.pre)
            np.testing.assert_array_equal(model.post, reference.post)
            for name in ('rng_reaction', 'rng_motion', 'rng_field'):
                assert getattr(model, name).bit_generator.state == getattr(reference, name).bit_generator.state
            assert model.work_in == reference.work_in and model.heat_out == reference.heat_out
        archive = out/(label+'_complete.npz')
        snapshot(archive, model, session, observer, sources)
    result = dict(archive=str(archive), sha256=digest(archive), first_time_ms=first_t, last_time_ms=model.t,
        bytes=len(block), nll_nats_per_byte={key:float(np.mean([row['scores'][key] for row in rows]))
                                           for key in rows[0]['scores']}, nodes=model.n,
        internal_bridges=len(model.pre), boundary_contacts=int(model.port_installed.sum()),
        initial_total_material=int(model.initial_material), final_free_material=int(model.material.sum()),
        final_domain_material=model.n, births=model.domain_births, deaths=model.domain_deaths,
        attempts=model.domain_attempts, initial_live_domains=parent.n,
        new_live_domains=sum(uid >= parent.n for uid in observer.state['ids']),
        actual_original_events=observer.state['original_events'],
        actual_events_touching_new_domains=observer.state['new_domain_events'],
        max_degree=int(model.degrees().max()), source_work=model.work_in-source_work,
        heat_out=model.heat_out-heat, domain_heat=model.domain_heat,
        maximum_errors=errors, restored_future_bytes=len(block)-len(block)//2,
        full_physics_session_observer_and_three_rngs_exact=True,
        closed_barrier_parent_physical_future_exact=reference is not None,
        seconds=time.perf_counter()-started)
    atomic_json(out/(label+'_summary.json'), result)
    return result


def main(args):
    args.out.mkdir(parents=True, exist_ok=False)
    parent, metadata = load_exchange(args.archive)
    source_hash = digest(args.archive)
    corpus = metadata['sources']['corpus']
    data = args.data/'train.npy'
    if digest(data) != corpus['files_sha256']['train.npy']:
        raise ValueError('Training corpus changed')
    values = np.asarray(np.load(data, mmap_mode='r')[1280:1408])
    if len(values) != 128 or np.any((values < 4) | (values >= 260)):
        raise ValueError('Expected a new, boundary-free continuation; revise protocol explicitly')
    block = np.asarray(values-4, np.uint8)
    np.save(args.out/'fresh_train_1280_1408.npy', block, allow_pickle=False)
    sources = dict(parent_archive=str(args.archive), parent_sha256=source_hash,
                   original_sources=metadata['sources'], split='train', token_offset=4,
                   fresh_byte_start=1280, fresh_byte_end=1408,
                   selected_sha256=__import__('hashlib').sha256(block.tobytes()).hexdigest())
    source_dir = args.out/'source_snapshot'
    source_dir.mkdir()
    paths = [Path(__file__), ROOT/'information_boltzmann/cgb/growing_material.py',
             ROOT/'information_boltzmann/cgb/mobile_exchange.py',
             ROOT/'information_boltzmann/cgb/identity_material.py',
             ROOT/'information_boltzmann/cgb/boundary_material.py',
             ROOT/'information_boltzmann/cgb/exchange_checkpoint.py',
             ROOT/'docs/information_boltzmann/CGB_DOMAIN_BIRTH_PREREG_20261022.md']
    hashes = {str(path.relative_to(ROOT)):digest(path) for path in paths}
    for path in paths:
        shutil.copy2(path, source_dir/path.name)
    atomic_json(args.out/'sources.json', sources)
    arms = {}
    for label in ('growing', 'closed_domain_barrier'):
        arms[label] = run_arm(label, parent, metadata, block, args.out, sources)
    assert digest(args.archive) == source_hash
    assert hashes == {str(path.relative_to(ROOT)):digest(path) for path in paths}
    atomic_json(args.out/'report.json', dict(passed=True, sources=sources,
        baseline_git=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        code_sha256=hashes, arms=arms, optimizer_updates=0, byte_ms=4., dt_ms=2.,
        original_source_unchanged=True, scope='Actual paid domain birth, original learning and full continuation; no predictive convergence, complete morphology or language-world capability claim'))
    atomic_json(args.out/'progress.json', dict(status='completed', report=str(args.out/'report.json')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--archive', type=Path, default=Path('F:/fly_stream_overflow/cgb_future_exchange_recovered_20261022/native_revisit_A_complete.npz'))
    parser.add_argument('--data', type=Path, default=ROOT/'data/ib_owt_10000')
    parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args())
