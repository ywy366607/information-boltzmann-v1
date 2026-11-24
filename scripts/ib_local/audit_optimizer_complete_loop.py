"""Same complete body: stable real text -> new real source -> old-source revisit.

This is optimizer integration/relearning evidence, not a dialogue capability
exam. All language proposals, S/P/M, tables, U, real outcomes and delayed
actor credit continue. Only the explicit optimizer and private world-cursor
interventions differ. No mother calls, toy tasks or frozen neural parameters.
"""
import argparse
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from scripts.ib_local.run_mother_continuation import continuation_parser, prepare_continuation, ROOT
from scripts.ib_local.native_navigation_body import restore_birth_counters, record_birth_event
from scripts.ib_local.optimizer_policy import diagnostics
from scripts.ib_local.run_language_navigation_v2 import validate


def validation_data(config, size=8192):
    raw = np.load(ROOT/'data/ib_owt_10000/validation.npy', mmap_mode='r')
    offset = json.loads((ROOT/'data/ib_owt_10000/tokenizer.json').read_text())['offset']
    part = np.asarray(raw[:2*(262144+size+1)])
    owt = (part[part >= offset]-offset)[262144:262144+size+1].astype(np.uint8)
    fineweb = np.frombuffer(Path(config['natural_text_corpus']['validation_path']).read_bytes(),np.uint8)[:size+1]
    if min(len(owt), len(fineweb)) != size+1: raise ValueError('validation data too short')
    return dict(owt=owt, fineweb=fineweb)


def observe(row):
    search = row['joint_language_search']
    if len(search['candidates']) != 4 or row['world'].get('provider_calls') != 0:
        raise RuntimeError('complete four-proposal natural-text loop was not preserved')
    if max(row['policy_replay_max_error'], search['replay_max_error']) > 1e-3:
        raise RuntimeError('actual action credit lost its sampled path')
    if not all(math.isfinite(row[k]) for k in ('metal_g','response_nll','actor_norm')):
        raise RuntimeError('nonfinite actual learning/settlement')


def summarize(rows):
    def mean(key):
        values = [r[key] for r in rows if isinstance(r.get(key), (int,float))]
        return statistics.fmean(values) if values else None
    choices = {k:sum(r['choice']==k for r in rows) for k in ('silence','generate','reuse')}
    return dict(events=len(rows), updates=sum(r['updates'] for r in rows), choices=choices,
        mean_event_seconds=mean('event_seconds'), response_nll=mean('response_nll'), metal_g=mean('metal_g'),
        outcome_error=mean('outcome_error'), wood_used_error=mean('wood_used_error'),
        wood_base_error=mean('wood_base_error'), wood_policy_tv=mean('wood_policy_tv'),
        real_episode_error=mean('episodic_real_error'), shuffled_episode_error=mean('episodic_shuffled_error'),
        episodic_policy_tv=mean('episodic_policy_tv'), actor_norm=mean('actor_norm'),
        delayed_actor_events=sum(bool(r['delayed_actor_applied']) for r in rows),
        read_credit_records=mean('water_credit_records'),
        long_address_events=sum(r['world'].get('match',0)>=8 for r in rows),
        source_bytes={source:sum(r['world']['text_source_bytes'][source] for r in rows)
                      for source in ('inherited_prefix','natural_append')},
        coevolution_rms=statistics.fmean(r['past_present']['workspace_rms'] for r in rows))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--body', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--arm', choices=('adamw','global_anneal','budgeted'), required=True)
    p.add_argument('--phase-events', type=int, default=32)
    p.add_argument('--stop-file', type=Path)
    a = p.parse_args()
    if a.phase_events < 1: raise ValueError('positive actual event budget required')
    directory = a.output/a.arm; directory.mkdir(parents=True, exist_ok=False)
    opts = ['--checkpoint',str(a.body/'last.pt.gz'), '--output',str(a.body),
        '--cycles',str(3*a.phase_events), '--resume-output','--world-source','text',
        '--table-storage',str(directory/'table'), '--causal-storage',str(directory/'causal'),
        '--optimizer-policy','budgeted' if a.arm == 'budgeted' else 'adamw']
    loaded = prepare_continuation(continuation_parser().parse_args(opts), resume_output=True)
    e, config = loaded.engine, loaded.config
    from scripts.ib_local.text_source_availability import TextSourceAvailability
    e.world = TextSourceAvailability(e.world)
    restore_birth_counters(e,loaded.cycle,a.body/'metrics.jsonl')
    data = validation_data(config)
    def score(): return {k:validate(e.model,e.table,v) for k,v in data.items()}
    count = [0]; timings = []; original = e.optimizer.step
    selected = [(n,p) for n,p in e.model.named_parameters() if
                n.endswith('.o_proj.weight') or n.startswith(('kanli.choice.', 'kanli.critic.',
                    'binding_choice.', 'question_to_hidden.', 'handover.', 'body.coevolution.'))]
    writes = []
    def step(*args, **kw):
        samples = [p.detach().flatten()[::max(1,p.numel()//16)][:16].clone() for _,p in selected]
        torch.cuda.synchronize(); start = time.perf_counter()
        value = original(*args, **kw)
        torch.cuda.synchronize(); timings.append(time.perf_counter()-start)
        writes.append(torch.stack([(p.detach().flatten()[::max(1,p.numel()//16)][:16]-v).square().mean()
                                   for (_,p),v in zip(selected,samples)]))
        count[0] += 1
        return value
    e.optimizer.step = step
    rates = [g['lr'] for g in e.optimizer.param_groups]
    origin = int(e.world.cursor)
    prefix = config['natural_text_corpus']['prefix_bytes']
    phases = [('stable_owt',origin,(0,prefix)),
              ('new_fineweb',prefix+1048576,(prefix,len(e.world.data))),
              ('revisit_owt',origin,(0,prefix))]
    report = dict(schema=1, source_sha=loaded.source_sha, source_cycle=loaded.cycle,
        arm=a.arm, phase_events=a.phase_events, parameters=sum(p.numel() for p in e.model.parameters()),
        all_parameters_trainable=all(p.requires_grad for p in e.model.parameters()),
        intervention='private real-source availability changes; full navigation within each source; absolute address receipts',
        initial_validation=score(), phases=[], optimizer_cuda_synchronized_seconds=[], main_written=False)
    (directory/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    with (directory/'metrics.jsonl').open('w',encoding='utf-8') as log:
        index = 0
        for name, cursor, bounds in phases:
            e.world.bounds = bounds
            e.world.base.cursor = cursor
            phase = []; before = score()
            write_begin = len(writes)
            for j in range(a.phase_events):
                if a.stop_file is not None and a.stop_file.exists():
                    raise RuntimeError('private comparison stopped at a completed event boundary')
                if a.arm == 'global_anneal':
                    # Reference finite-life schedule: cosine from inherited LR
                    # to zero over the declared whole experiment, not per age.
                    scale = .5*(1+math.cos(math.pi*index/max(3*a.phase_events-1,1)))
                    for group,rate in zip(e.optimizer.param_groups,rates): group['lr'] = rate*scale
                updates = count[0]; start = time.perf_counter()
                row = e.step(torch.randn(e.model.encoder.out_features,32,device=e.device))
                torch.cuda.synchronize()
                row.update(updates=count[0]-updates, event_seconds=time.perf_counter()-start,
                           audit_event=index+1, phase=name)
                record_birth_event(e,row); observe(row)
                expected_source = 'natural_append' if name == 'new_fineweb' else 'inherited_prefix'
                if row['world']['text_source_bytes'][expected_source] != row['external_bytes']:
                    raise RuntimeError('actual source exposure did not follow the declared control')
                if j%8 == 0 or j+1 == a.phase_events:
                    row['optimizer_plasticity'] = diagnostics(e)
                log.write(json.dumps(row,ensure_ascii=True)+'\n'); log.flush()
                phase.append(row); index += 1
                print(json.dumps(dict(arm=a.arm, event=index, phase=name,
                    choice=row['choice'], response_nll=row['response_nll'], metal_g=row['metal_g'])),flush=True)
            report['phases'].append(dict(name=name, validation_before=before, validation_after=score(),
                loop=summarize(phase), optimizer=diagnostics(e),
                sampled_actual_write_rms=dict(zip((n for n,_ in selected),
                    torch.stack(writes[write_begin:]).mean(0).sqrt().cpu().tolist()))))
            report['optimizer_cuda_synchronized_seconds'] = list(timings)
            (directory/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    report['status'] = 'complete'
    report['optimizer_seconds_median'] = statistics.median(timings)
    (directory/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    e.model._causal_archive.db.close()


if __name__ == '__main__': main()
