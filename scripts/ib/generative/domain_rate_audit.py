"""Exact marginal phase rates and continued native learning on fresh OWT."""
from __future__ import annotations

import argparse
import copy
import gzip
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from information_boltzmann.cgb.exchange_checkpoint import load_exchange, save_exchange
from information_boltzmann.cgb.growing_material import GrowingIdentityMobileExchange
from information_boltzmann.cgb.learning_observation import native_learning_drift, parameter_quench
from scripts.ib.generative.compare_support_transport import digest, same
from scripts.ib.generative.domain_birth_audit import Observation
from scripts.ib.generative.native_learning_stream import advance_byte, atomic_json


class CarrierObservation(Observation):
    """Track actual paid carriers across domain index changes and rebuilds."""
    def __init__(self, model, node_state, carrier_state=None):
        super().__init__(model, node_state)
        if carrier_state is None:
            keys = sorted(self.keys(model))
            self.carriers = dict(alive={key:i for i,key in enumerate(keys)},
                                 initial_count=len(keys), next_id=len(keys), deaths=0)
        else:
            self.carriers = copy.deepcopy(carrier_state)
        self.validate(model)

    def keys(self, model):
        ids = self.state['ids']
        return {'i:'+str(ids[int(i)])+':'+str(ids[int(j)]) for i,j in zip(model.pre,model.post)} | {
            'p:'+str(p) for p,n in enumerate(model.port_installed) if n}

    def change(self, key, sign, row):
        data = self.carriers
        if sign > 0:
            assert key not in data['alive']
            data['alive'][key] = data['next_id']
            data['next_id'] += 1
        assert key in data['alive']
        row['observed_carrier_id'] = data['alive'][key]
        if sign < 0:
            data['alive'].pop(key)
            data['deaths'] += 1

    def native(self, row):
        super().native(row)
        if row['kind'] in ('conversion', 'exchange', 'release'):
            return
        key = ('p:'+str(row['interface']) if row.get('interface',-1) >= 0 else
               'i:'+str(row['observed_pre_id'])+':'+str(row['observed_post_id']))
        self.change(key, row['sign'] if row['kind']=='assembly' else 0, row)

    def domain(self, row):
        super().domain(row)
        if row['accepted']:
            key = 'i:'+str(row['observed_parent_id'])+':'+str(row['observed_child_id'])
            self.change(key, 1 if row['kind']=='domain_birth' else -1, row)

    def validate(self, model):
        super().validate(model)
        if hasattr(self, 'carriers'):
            assert set(self.carriers['alive']) == self.keys(model)
            assert len(self.carriers['alive']) == self.carriers['next_id']-self.carriers['deaths']

    def summary(self, model):
        self.validate(model)
        return dict(live_carriers=len(self.carriers['alive']),
                    source_carriers_surviving=sum(uid < self.carriers['initial_count']
                                                 for uid in self.carriers['alive'].values()),
                    births=self.carriers['next_id']-self.carriers['initial_count'],
                    deaths=self.carriers['deaths'])


def verify_legacy(run):
    recorded = json.loads((run/'report.json').read_text())
    path = run/'source_snapshot/growing_material.py'
    code_hashes = {name.replace('\\', '/'):value for name,value in recorded['code_sha256'].items()}
    assert digest(path) == code_hashes['information_boltzmann/cgb/growing_material.py']
    spec = importlib.util.spec_from_file_location('information_boltzmann.cgb.growth_before_marginal', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    current, metadata = load_exchange(run/'growing_mid.npz')
    old = module.GrowingIdentityMobileExchange.from_state_dict(current.state_dict())
    session = copy.deepcopy(metadata['session']);old_session = copy.deepcopy(session)
    observer = Observation(current, metadata['growth_observer'])
    current.event_observer, current.domain_observer = observer.native, observer.domain
    block = np.load(run/'fresh_train_1280_1408.npy', allow_pickle=False)
    expected = [json.loads(line) for line in (run/'growing_scores.jsonl').read_text().splitlines()]
    for index in range(64,128):
        row = advance_byte(current,session,int(block[index]),'fresh_C')
        assert row == advance_byte(old,old_session,int(block[index]),'fresh_C')
        assert all(expected[index][key] == value for key,value in row.items())
    same(current.state_dict(),old.state_dict())
    final,final_meta = load_exchange(run/'growing_complete.npz')
    same(current.state_dict(),final.state_dict())
    assert session == old_session == final_meta['session']
    assert observer.state == final_meta['growth_observer']
    return dict(bytes=64,old_actual_source_sha256=digest(path),
                all_logits_scores_physical_states_three_rngs_and_session_exact=True)


def snapshot(path, model, session, observer, sources):
    observer.validate(model)
    save_exchange(path,model,dict(session=copy.deepcopy(session),growth_observer=copy.deepcopy(observer.state),
        carrier_observer=copy.deepcopy(observer.carriers),sources=sources,byte_ms=4.,dt_ms=2.))


def run_arm(label,parent,metadata,block,out,sources):
    model = GrowingIdentityMobileExchange.from_state_dict(parent.state_dict())
    model.domain_sampling = 'resolved' if label=='resolved_reference' else 'marginal'
    session = copy.deepcopy(metadata['session'])
    session.update(phase='fresh_D',next_phase_byte=0)
    session['interventions'].append(dict(time_ms=model.t,domain_sampling=model.domain_sampling,
        scope='Same phase generator, exact conditional factorization and Poisson thinning',
        state_erased=False,instantaneous_work=0.))
    before_rates = model.domain_generator_rates()
    before_native = native_learning_drift(model)
    intervention = None
    if label=='no_capture':
        intervention = parameter_quench(model,capture_coupling=0.)
        session['interventions'].append(intervention)
        assert before_rates == model.domain_generator_rates()
    observer = CarrierObservation(model,metadata['growth_observer'])
    initial_births,initial_deaths = model.domain_births,model.domain_deaths
    initial_events = dict(model.reaction_counts)
    work,heat = model.work_in,model.heat_out
    probe0 = dict(native=before_native,after_intervention=native_learning_drift(model),intervention=intervention)
    atomic_json(out/(label+'_initial.json'),probe0)
    snapshot(out/(label+'_start.npz'),model,session,observer,sources)
    rows=[];windows=[];fork=fork_session=fork_observer=None
    max_error=0.
    with gzip.open(out/(label+'_events.jsonl.gz'),'wt',encoding='utf-8') as events, \
            gzip.open(out/(label+'_domains.jsonl.gz'),'wt',encoding='utf-8') as domains, \
            (out/(label+'_scores.jsonl')).open('w',encoding='utf-8') as scores:
        def native(row):
            observer.native(row);events.write(json.dumps(row,separators=(',',':'))+'\n')
        def domain(row):
            observer.domain(row);domains.write(json.dumps(row,separators=(',',':'))+'\n')
        model.event_observer,model.domain_observer = native,domain
        for index,target in enumerate(block):
            row=advance_byte(model,session,int(target),'fresh_D')
            if fork is not None:
                assert row==advance_byte(fork,fork_session,int(target),'fresh_D')
            row.update(phase='fresh_D',phase_byte=index,corpus_train_byte=1408+index)
            rows.append(row);scores.write(json.dumps(row,separators=(',',':'))+'\n')
            observer.validate(model)
            max_error=max(max_error,max(abs(float(value)) for value in model.balances().values()))
            if (index+1)%32==0:
                recent=rows[-32:]
                info=dict(arm=label,next_phase_byte=index+1,time_ms=model.t,nodes=model.n,
                    births=model.domain_births-initial_births,deaths=model.domain_deaths-initial_deaths,
                    internal_bridges=len(model.pre),boundary_contacts=int(model.port_installed.sum()),
                    free_material=int(model.material.sum()),trace=int(model.trace.sum()),support=int(model.bound.sum()),
                    nll={key:float(np.mean([item['scores'][key] for item in recent])) for key in row['scores']},
                    carrier=observer.summary(model),generator=native_learning_drift(model),
                    source_work=model.work_in-work,heat_out=model.heat_out-heat,maximum_error=max_error)
                windows.append(info);atomic_json(out/(label+'_windows.json'),windows)
                atomic_json(out/'progress.json',dict(status='running',**{key:info[key] for key in (
                    'arm','next_phase_byte','time_ms','nodes','births','deaths','maximum_error')}))
                print(json.dumps({key:info[key] for key in ('arm','next_phase_byte','time_ms','nodes','births','deaths','maximum_error')}),flush=True)
                events.flush();domains.flush();scores.flush()
            if index+1==len(block)//2:
                path=out/(label+'_mid.npz');snapshot(path,model,session,observer,sources)
                fork,meta=load_exchange(path);fork_session=copy.deepcopy(meta['session'])
                fork_observer=CarrierObservation(fork,meta['growth_observer'],meta['carrier_observer'])
                fork.event_observer,fork.domain_observer=fork_observer.native,fork_observer.domain
        same(model.state_dict(),fork.state_dict())
        assert session==fork_session and observer.state==fork_observer.state and observer.carriers==fork_observer.carriers
        assert max_error<1e-8
        archive=out/(label+'_complete.npz');snapshot(archive,model,session,observer,sources)
    report=dict(archive=str(archive),sha256=digest(archive),bytes=len(block),time_ms=model.t,
        nll={key:float(np.mean([row['scores'][key] for row in rows])) for key in rows[0]['scores']},
        nodes=model.n,stage_births=model.domain_births-initial_births,stage_deaths=model.domain_deaths-initial_deaths,
        carrier=observer.summary(model),final_material=int(model.material.sum()),total_material=int(model.initial_material),
        internal_bridges=len(model.pre),boundary_contacts=int(model.port_installed.sum()),
        original_reactions={key:model.reaction_counts.get(key,0)-initial_events.get(key,0)
                            for key in set(model.reaction_counts)|set(initial_events)},
        initial_probe=probe0,final_probe=native_learning_drift(model),intervention=intervention,
        flow_source_work=model.work_in-work,heat_out=model.heat_out-heat,maximum_error=max_error,
        restored_future_bytes=len(block)//2,all_logits_physics_three_rngs_session_and_both_observers_exact=True,
        primary=label=='marginal')
    atomic_json(out/(label+'_summary.json'),report)
    return report


def main(args):
    args.out.mkdir(parents=True,exist_ok=False)
    parent,metadata=load_exchange(args.archive);source_hash=digest(args.archive)
    source=metadata['sources']
    while 'corpus' not in source:source=source['original_sources']
    data=args.data/'train.npy'
    if digest(data)!=source['corpus']['files_sha256']['train.npy']:raise ValueError('Corpus changed')
    raw=np.asarray(np.load(data,mmap_mode='r')[1408:1920])
    if len(raw)!=512 or np.any((raw<4)|(raw>=260)):raise ValueError('Declare boundary handling before continuation')
    block=np.asarray(raw-4,np.uint8);np.save(args.out/'fresh_train_1408_1920.npy',block,allow_pickle=False)
    prior=verify_legacy(args.prior_run)
    sources=dict(parent_archive=str(args.archive),parent_sha256=source_hash,original_sources=metadata['sources'],
        split='train',token_offset=4,fresh_byte_start=1408,fresh_byte_end=1920,
        block_sha256=__import__('hashlib').sha256(block.tobytes()).hexdigest())
    paths=[Path(__file__),ROOT/'information_boltzmann/cgb/growing_material.py',
           ROOT/'information_boltzmann/cgb/learning_observation.py',
           ROOT/'scripts/ib/generative/domain_birth_audit.py',
           ROOT/'docs/information_boltzmann/CGB_DOMAIN_RATE_DERIVATION_20261022.md']
    hashes={str(path.relative_to(ROOT)):digest(path) for path in paths}
    folder=args.out/'source_snapshot';folder.mkdir()
    for path in paths:shutil.copy2(path,folder/path.name)
    atomic_json(args.out/'legacy_equivalence.json',prior)
    arms={}
    for label in ('marginal','resolved_reference','no_capture'):
        arms[label]=run_arm(label,parent,metadata,block,args.out,sources)
    assert digest(args.archive)==source_hash
    assert hashes=={str(path.relative_to(ROOT)):digest(path) for path in paths}
    atomic_json(args.out/'report.json',dict(passed=True,baseline_git=subprocess.check_output(
        ['git','rev-parse','HEAD'],text=True).strip(),sources=sources,source_unchanged=True,
        code_sha256=hashes,legacy_equivalence=prior,arms=arms,optimizer_updates=0,byte_ms=4.,dt_ms=2.,
        scope='Exact conditional phase sampler and continued local physical learning. Single trajectories are not a convergence or predictive-benefit claim.'))
    atomic_json(args.out/'progress.json',dict(status='completed'))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--archive',type=Path,default=Path('F:/fly_stream_overflow/cgb_domain_birth_20261022/growing_complete.npz'))
    p.add_argument('--prior-run',type=Path,default=Path('F:/fly_stream_overflow/cgb_domain_birth_20261022'))
    p.add_argument('--data',type=Path,default=ROOT/'data/ib_owt_10000')
    p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
