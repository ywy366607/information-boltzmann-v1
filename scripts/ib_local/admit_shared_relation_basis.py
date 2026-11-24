"""Migrate the actual continuous body, then commit two complete real events."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

import torch

from scripts.ib_local.admit_generative_cycle import old_fingerprints
from scripts.ib_local.audit_joint_restore import canonical
from scripts.ib_local.checkpoint_digest import state_digest
from scripts.ib_local.mother_checkpoint import save
from scripts.ib_local.run_mother_continuation import continuation_parser,prepare_continuation
from scripts.ib_local.run_causal_water_reply import build,sha256_file
from scripts.ib_local.self_action_observer import attach_self_action_observer


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--events',type=int,choices=(0,1,2),default=0)
    p.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    p.add_argument('--joint-relation-update',action='store_true')
    p.add_argument('--unified-relation-evidence',action='store_true')
    p.add_argument('--qualified-delayed-evidence',action='store_true')
    p.add_argument('--content-question',action='store_true')
    args=p.parse_args()
    torch.set_num_threads(1)
    before=old_fingerprints(args.checkpoint)
    source_sha=sha256_file(args.checkpoint)
    options=['--checkpoint',str(args.checkpoint),'--output',str(args.output),
             '--cycles','2','--device',args.device,'--shared-relation-basis']
    if args.joint_relation_update:options.append('--joint-relation-update')
    if args.unified_relation_evidence:options.append('--unified-relation-evidence')
    if args.qualified_delayed_evidence:options.append('--qualified-delayed-evidence')
    if args.content_question:options.append('--content-question')
    opts=continuation_parser().parse_args(options)
    loaded=prepare_continuation(opts)
    engine=loaded.engine
    after=old_fingerprints(args.output/'last.pt.gz')
    authorized={}
    if args.unified_relation_evidence or args.qualified_delayed_evidence:
        key='joint_relation_update_revision'
        h=lambda value:state_digest(canonical(value))['sha256']
        old,new,flag=(3,4,'qualified-delayed-evidence') if args.qualified_delayed_evidence else (2,3,'unified-relation-evidence')
        if before['engine'].get(key)==h(old) and after['engine'].get(key)==h(new):
            authorized[key]=dict(before=old,after=new,reason=f'explicit --{flag} migration')
    preserved={name:all(after[name].get(k)==v for k,v in before[name].items())
               for name in ('model','engine','optimizer_state','whole')}
    preserved['engine']=all(after['engine'].get(k)==v or k in authorized for k,v in before['engine'].items())
    preserved['old_optimizer_groups']=after['optimizer_groups'][:len(before['optimizer_groups'])]==before['optimizer_groups']
    new_params={n:p.numel() for n,p in engine.model.named_parameters() if n not in before['model']}
    s,a=engine._internal_source[None],engine._internal_action
    state=engine._program_question_state
    from scripts.ib_local.causal_relation_program import mechanism,Port
    dist=mechanism(engine.model,state,s.expand_as(a),a,ridge=engine.relation_ridge)
    x=dist['x'].double();c=x[:,:2];u=x[:,2:]
    conditioned=u-c@torch.linalg.pinv(c)@u
    checks=dict(four_shared_inputs=x.shape[1]==4,
        condition_rank=int(torch.linalg.matrix_rank(c,tol=1e-6))<=2,
        full_rank=int(torch.linalg.matrix_rank(x,tol=1e-6))==4,
        conditional_action_rank=int(torch.linalg.matrix_rank(conditioned,tol=1e-6))==2,
        finite=bool(torch.isfinite(dist['mean']).all() and torch.isfinite(dist['scale2']).all()),
        same_condition_across_actions=bool(torch.equal(dist['query'][:,:2],dist['query'][:1,:2].expand(len(a),-1))))
    report=dict(source_cycle=loaded.cycle,source_sha256=source_sha,preserved=preserved,
        new_parameters=new_params,new_parameter_count=sum(new_params.values()),
        initialization=loaded.config['shared_relation_basis'],geometry_checks=checks,
        parameter_audit=loaded.config['mother_parameter_audit'],events=0,updates=0,
        authorized_metadata_migrations=authorized,
        scope='migration and real whole-cycle admission, not capability or convergence')
    if args.content_question:
        report['content_question']=loaded.config['content_question']
    if args.joint_relation_update or args.unified_relation_evidence or args.qualified_delayed_evidence:
        report['joint_relation_update']=loaded.config['joint_relation_update']
    def write():
        (args.output/'admission.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    write()
    if not all(preserved.values()) or not all(checks.values()):
        report['changed']={name:[k for k,v in before[name].items() if after[name].get(k)!=v]
                           for name in ('model','engine','optimizer_state','whole')}
        write();raise RuntimeError('shared-coordinate migration admission failed')
    if args.events:
        attach_self_action_observer(engine)
        updates=[0];original=engine.optimizer.step
        def counted(*pos,**kw):
            value=original(*pos,**kw);updates[0]+=1;return value
        engine.optimizer.step=counted
        start=time.perf_counter()
        with (args.output/'metrics.jsonl').open('w',encoding='utf-8') as log:
            for i in range(1,args.events+1):
                old=updates[0]
                row=engine.step(torch.randn(64,32,device=engine.device))
                row.update(cycle=loaded.cycle+i,updates=updates[0]-old,
                    cumulative_optimizer_updates=updates[0],elapsed_seconds=time.perf_counter()-start)
                save(engine,args.output,row['cycle'],loaded.config)
                log.write(json.dumps(row,ensure_ascii=False)+'\n');log.flush()
                obs=row.get('self_action_observer',{})
                changed=row['shared_relation_basis']['parameter_update_norms']
                tests=dict(observed=obs.get('available',False),
                    reproduction=obs.get('actual_score_reproduction_error',1.)<1e-5,
                    real_revision=row['shared_unknown']['revision']['committed'],
                    same_U=row['generative_relation']['same_U'],
                    learning=row['updates']>0,
                    encoder_basis_learned=all(v>0 for v in changed.values()),
                    old_fact_replay=row['causal_program_replay']>0,
                    draft_replay=row['joint_language_search']['replay_max_error']<1e-3,
                    binding_replay=row['generative_relation']['binding_replay_error']<1e-5,
                    previous_settled=i==1 or row['generative_relation']['program_evidence_updated'])
                if args.content_question:
                    port=row['question_context']['conditional_transition']
                    tests.update(question_transition_learned=port['parameter_step']>0,
                        question_scope_clear=port['scope_cleared'],
                        no_direct_question_world_bias=port['direct_world_output_positions']==0,
                        own_state_matches_draft=row.get('draft_actual_action_l2',0.)<1e-3)
                if args.joint_relation_update or args.unified_relation_evidence or args.qualified_delayed_evidence:
                    joint=row['joint_relation_update']
                    tests.update(organization_in_U=joint['organization_in_shared_U'],
                        organization_shared_features=joint['feature_dimension']==4,
                        overlap_descends=all(a<=b+1e-5 for a,b in zip(
                            joint['overlap_objective_after'],joint['overlap_objective_before'])),
                        coordinated_continuation=joint['continuation_uses_coordinated_boundary'],
                        dependency_used=row['shared_unknown']['revision'].get('dependent_relations',0)>0)
                    if args.unified_relation_evidence or args.qualified_delayed_evidence:
                        tests['canonical_support']=joint['evidence_contract']['same_facts']
                    if args.qualified_delayed_evidence:
                        current=row['causal_program']['committed_episode_identity']
                        tests['committed_fact_recorded']=engine.internal_pending['causal_program'].get('episode_identity')==current and bool(current)
                        tests['delayed_fact_qualified']=bool(row.get('causal_program_delayed_fact'))
                report.setdefault('event_checks',[]).append(tests)
                report.update(events=i,updates=updates[0],cycle=row['cycle'])
                write();print(json.dumps(dict(cycle=row['cycle'],checks=tests)),flush=True)
                if not all(tests.values()):
                    raise RuntimeError('actual complete event admission failed')
        expected_model=state_digest(canonical(engine.model.state_dict()))['sha256']
        expected_optimizer=state_digest(canonical(engine.optimizer.state_dict()))['sha256']
        carried_names=('recurrent','bfss_momenta','pending','internal_pending','prev')
        expected_carried=state_digest(canonical({k:getattr(engine,k) for k in carried_names}))['sha256']
        archive=engine.model._causal_archive
        archive.close()
        for name in ('rows','count','tag','err','epoch','last_write'):
            v=getattr(engine.table,name,None)
            if hasattr(v,'_mmap'):v._mmap.close()
        del archive,loaded,engine,dist,original
        gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
        resumed,_,_,_=build(args.output/'last.pt.gz',device=args.device,
            table_storage=args.output/'restore_check',allow_external_archive=True)
        report['disk_restore']=dict(
            model_exact=state_digest(canonical(resumed.model.state_dict()))['sha256']==expected_model,
            optimizer_exact=state_digest(canonical(resumed.optimizer.state_dict()))['sha256']==expected_optimizer,
            carried_exact=state_digest(canonical({k:getattr(resumed,k) for k in carried_names}))['sha256']==expected_carried)
        report['source_unchanged']=sha256_file(args.checkpoint)==source_sha
        report['event_admission_passed']=all(report['disk_restore'].values()) and report['source_unchanged']
        write()
        if not report['event_admission_passed']:raise RuntimeError('full GPU restore differs')
    print(json.dumps(dict(migration_passed=True,events=report['events'],
                         cycle=report.get('cycle',report['source_cycle']))),flush=True)


if __name__=='__main__':main()
