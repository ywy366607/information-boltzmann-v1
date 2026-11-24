"""Explicit migration and real-event admission of the continuing full body."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from scripts.ib_local.admit_generative_cycle import old_fingerprints
from scripts.ib_local.mother_checkpoint import save
from scripts.ib_local.run_mother_continuation import continuation_parser,prepare_continuation


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--events',type=int,choices=(0,2),default=0)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    args=parser.parse_args()
    torch.set_num_threads(1)
    before=old_fingerprints(args.checkpoint)
    options=continuation_parser().parse_args(['--checkpoint',str(args.checkpoint),
        '--output',str(args.output),'--cycles','2','--device',args.device,'--shared-unknowns'])
    loaded=prepare_continuation(options)
    after=old_fingerprints(args.output/'last.pt.gz')
    preserved={name:all(after[name].get(k)==v for k,v in before[name].items())
               for name in ('model','engine','optimizer_state','whole')}
    preserved['optimizer_groups']=before['optimizer_groups']==after['optimizer_groups']
    preserved['no_added_parameters']=before['model'].keys()==after['model'].keys()
    report=dict(source_cycle=loaded.cycle,source_sha256=loaded.source_sha,
        preserved=preserved,passed=all(preserved.values()),events=0,updates=0,
        scope='actual full body, same world and original delayed participation; not an ability score')
    path=args.output/'admission.json'
    def write():
        path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    write()
    if not report['passed']:
        report['changed']={name:[k for k,v in before[name].items() if after[name].get(k)!=v]
                           for name in ('model','engine','optimizer_state','whole')}
        write();raise RuntimeError('inherited state changed in shared-unknown migration')
    engine=loaded.engine
    if args.events:
        updates=[0];original=engine.optimizer.step
        def counted(*a,**kw):
            result=original(*a,**kw);updates[0]+=1;return result
        engine.optimizer.step=counted
        started=time.perf_counter()
        with (args.output/'metrics.jsonl').open('w',encoding='utf8') as log:
            for offset in range(1,args.events+1):
                old=updates[0]
                row=engine.step(torch.randn(64,32,device=engine.device))
                row.update(cycle=loaded.cycle+offset,updates=updates[0]-old,
                           cumulative_optimizer_updates=updates[0],mode='real',
                           elapsed_seconds=time.perf_counter()-started)
                save(engine,args.output,row['cycle'],loaded.config)
                log.write(json.dumps(row,ensure_ascii=False)+'\n');log.flush()
                q=row['shared_unknown']
                checks=dict(actual_revision=q['revision']['committed'],
                    identity_in_program=q['identity']==row['causal_program']['roles']['unknown_identity'],
                    revised=q['revision'].get('version_after',0)>q['version'],
                    finite_budget=q['revision'].get('support',0)<=engine.model.causal_relations.evidence_budget,
                    actual_learning=row['updates']>0,
                    policy_replay=row['generative_relation']['binding_replay_error']<1e-5,
                    previous_program_settled=offset==1 or row['generative_relation']['program_evidence_updated'])
                report.update(events=offset,cycle=row['cycle'],updates=updates[0])
                report.setdefault('event_checks',[]).append(checks)
                report['latest_question']=q;write()
                print(json.dumps(dict(cycle=row['cycle'],checks=checks,question=q)),flush=True)
                if not all(checks.values()):raise RuntimeError('whole-event admission failed')
        report['event_admission_passed']=True;write()
        (args.output/'completed.json').write_text(json.dumps(dict(status='complete',cycles=args.events,
            requested_cycles=args.events,source_sha256=loaded.source_sha,
            final_cycle=loaded.cycle+args.events),indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(migration_passed=True,events=report['events'],cycle=report.get('cycle',loaded.cycle))),flush=True)


if __name__=='__main__':main()
