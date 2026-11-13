"""Migrate the real complete individual, verify preservation, commit two events."""
from __future__ import annotations

import argparse
import gc
import gzip
import json
from pathlib import Path
import time

import torch

from scripts.ib_local.audit_joint_restore import canonical
from scripts.ib_local.checkpoint_digest import state_digest
from scripts.ib_local.run_mother_continuation import continuation_parser, prepare_continuation
from scripts.ib_local.mother_checkpoint import save
from scripts.ib_local.life_card import export_live


def old_fingerprints(path):
    with gzip.open(path,'rb') as f:
        state=torch.load(f,map_location='cpu',weights_only=False)
    result={section:{k:state_digest(canonical(v))['sha256'] for k,v in state[section].items()}
            for section in ('model','engine')}
    result['optimizer_state']={k:state_digest(canonical(v))['sha256'] for k,v in state['optimizer']['state'].items()}
    result['optimizer_groups']=[state_digest(canonical(g))['sha256'] for g in state['optimizer']['param_groups']]
    result['whole']={k:state_digest(canonical(state[k]))['sha256'] for k in
                    ('table','world','torch_rng','cuda_rng','numpy_rng','python_rng','cycle')}
    del state;gc.collect()
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mother-harness-url',default='http://127.0.0.1:8902')
    args=p.parse_args();torch.set_num_threads(1)
    before=old_fingerprints(args.checkpoint)
    options=continuation_parser().parse_args(['--checkpoint',str(args.checkpoint),
        '--output',str(args.output),'--cycles','2','--generative-relations',
        '--mother-harness-url',args.mother_harness_url])
    loaded=prepare_continuation(options)
    after=old_fingerprints(args.output/'last.pt.gz')
    preserved={section:all(after[section].get(k)==v for k,v in before[section].items())
               for section in ('model','engine','optimizer_state','whole')}
    preserved['optimizer_groups']=after['optimizer_groups'][:len(before['optimizer_groups'])]==before['optimizer_groups']
    report=dict(preserved=preserved,passed=all(preserved.values()),source_cycle=loaded.cycle,
        all_trainable=loaded.config['mother_parameter_audit'],events=0,
        scope='full CUDA trained individual; exact inherited values, new typed grammar state explicit')
    audit=report['all_trainable']
    report['all_parameters_in_optimizer']=(audit['total']==audit['trainable']==audit['trainable_in_optimizer']
        and not audit['trainable_missing_from_optimizer'] and not audit['frozen'])
    report['passed'] &= report['all_parameters_in_optimizer']
    if not report['passed']:
        report['changed']={section:[k for k,v in before[section].items() if after[section].get(k)!=v]
                           for section in ('model','engine','optimizer_state','whole')}
    path=args.output/'admission.json'
    def write():path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    write()
    if not report['passed']:
        raise RuntimeError('whole-body migration changed inherited state')
    engine=loaded.engine; started=time.perf_counter();updates=[0]
    original=engine.optimizer.step
    def count(*a,**kw):
        result=original(*a,**kw);updates[0]+=1;return result
    engine.optimizer.step=count
    with (args.output/'metrics.jsonl').open('w',encoding='utf-8') as log:
        for offset in (1,2):
            old_updates=updates[0]
            row=engine.step(torch.randn(64,32,device=engine.device))
            row.update(cycle=loaded.cycle+offset,updates=updates[0]-old_updates,
                       elapsed_seconds=time.perf_counter()-started)
            save(engine,args.output,row['cycle'],loaded.config)
            if loaded.life_card is not None:
                export_live(engine,checkpoint=args.output/'last.pt.gz',config=loaded.config,
                    cycle=row['cycle'],row=row,output=loaded.life_card['path'],branch_id=loaded.life_branch)
            log.write(json.dumps(row,ensure_ascii=False)+'\n');log.flush()
            report['events']=offset;report['cycle']=row['cycle'];report['updates']=updates[0]
            report['latest']=row['generative_relation'];write()
            print(json.dumps(dict(cycle=row['cycle'],choice=row['choice'],generative=row['generative_relation'])),flush=True)
            checks=dict(real_relation_commit=row['causal_program']['evidence_committed'],
                learning_updated=row['updates']>0,
                same_update=row['generative_relation']['same_U'],
                binding_replay=row['generative_relation']['binding_replay_error'] <= 1e-5,
                previous_program_settled=(offset==1 or row['generative_relation']['program_evidence_updated']))
            report.setdefault('event_checks',[]).append(checks);write()
            if not all(checks.values()):
                raise RuntimeError('committed real event failed the whole-cycle admission: '+str(checks))
    report['event_admission_passed']=True
    report['end_to_end_benefit']='not determined by admission; continuous comparison required'
    write()


if __name__=='__main__':
    main()
