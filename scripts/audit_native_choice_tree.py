"""Enumerate actual categorical selectors in complete private body restores.

Every leaf restores the same weights/Adam/S/P/archives/pending/world/RNG.
Generated draft sampling remains native; the audit conditions only question
and proposal categorical choices, consumes their original draws, and applies
the original actor and original finite-U expected payment. No world step or
optimizer update occurs. Byte draws are a recorded conditional boundary.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf8')


def leaf(args):
    import torch
    from scripts.ib_local.run_mother_continuation import continuation_parser, prepare_continuation
    from scripts.ib_local.run_causal_water_reply import sha256_file
    from scripts.ib_local.autonomous_cycle import dispatch
    from scripts.ib_local.finite_choice_contract import trace_choices
    from scripts.ib_local.conditional_actor_reference import encoded_view, evaluate
    from scripts.ib_local.causal_use_value import queries, calibrated
    from scripts.ib_local import epistemic_value_reference as ev
    torch.set_num_threads(1)
    started=time.perf_counter()
    source_hash=sha256_file(args.checkpoint)
    opts=continuation_parser().parse_args([
        '--checkpoint',str(args.checkpoint),'--output',str(args.checkpoint.parent),
        '--resume-output','--cycles','1','--world-source','text','--autonomous-events',
        '--optimizer-policy','budgeted','--table-storage',str(args.work/'table'),
        '--causal-storage',str(args.work/'causal')])
    loaded=prepare_continuation(opts,resume_output=True)
    e=loaded.engine;model=e.model
    archive=model._causal_archive
    updates=[0]
    def no_update(*unused, **kw):
        updates[0]+=1
        raise RuntimeError('zero-update choice audit reached optimizer')
    e.optimizer.step=no_update
    try:
        schedule=[]
        for _ in range(64):
            if e.autonomous_ready:
                break
            event=dispatch(e)
            assert event['autonomous_event']['kind']!='external'
            schedule.append(event['autonomous_event'])
        else:
            raise RuntimeError('natural external boundary not reached')
        archive.prepare()
        e._identified_event_fact=None
        torch.randn(model.encoder.out_features,32,device=e.device)
        capture={}
        original_plan=e.plan
        def plan(h):
            result=original_plan(h)
            capture.update(source=h.detach().clone(),habit=result[0].detach().clone(),
                           feeling=e.intero().detach().clone())
            return result
        e.plan=plan
        overrides={}
        if args.question is not None:overrides['question']=args.question
        if args.proposal is not None:overrides['proposal']=args.proposal
        with trace_choices(overrides) as trace:
            words,record=e._utter()
        del e.plan
        with torch.no_grad():
            source=capture['source'];z=model.encoder(source)
            q=z.new_tensor(e.epistemic_value_last['local_expected'])
            full=z.new_tensor(record['values'])/10
            view=encoded_view(model,z,capture['feeling'],reward_valid=e.reward_label_valid())
            logits=evaluate(view,full[None],precision=e.precision(),gain_units_schema=e.gain_units_schema)
            recorded=z.new_tensor(record['choice_logits'])
            replay_error=float((logits['logits'][0]-recorded).abs().max())
            assert replay_error < 3e-5
            probability=recorded.softmax(-1)
            state=e._program_question_state
            facts=queries(e,state)
            context=ev.from_body(e,state,facts,momentum=(e._momentum_decision_read[None]
                if getattr(e,'epistemic_value_schema',1)>=2 else None))
            recalled,detail=ev.continuation(model,context,z,reward_valid=e.reward_label_valid(wood=True),
                ridge=e.relation_ridge,calibration=float(calibrated(e,1.)))
            tensor_hash=lambda x:hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            out=dict(source_sha256=source_hash,source_cycle=loaded.cycle,
                epistemic_schema=e.epistemic_value_schema,settlement=e.settlement_kind,
                schedule=schedule,question=e._program_question_roles,
                selections=[dict(stage=x['stage'],sampled=x['sampled'],selected=x['selected'],
                    probability=x['probability'].tolist()) for x in trace.events],
                actor_probability=probability.tolist(),local_U_values=q.tolist(),
                actor_expected_U=float((probability*q).sum()),
                total_action_values=full.tolist(),actor_replay_error=replay_error,
                recalled_value=float(recalled[0]),recalled_detail=detail[0],
                query_facts=[x.fact_id for x in facts],
                action_hashes=[tensor_hash(x) for x in e._internal_action],
                action_latents=model.encoder(e._internal_action).tolist(),
                proposed_words=[x['words'].hex() for x in e._joint_proposals],
                selected_words=words.hex(),source_latent=z.tolist(),
                selected_proposal=e._joint_selection['selected'],
                proposal_count=len(e._joint_proposals),optimizer_updates=updates[0],world_calls=0,
                source_unchanged=sha256_file(args.checkpoint)==source_hash,
                seconds=time.perf_counter()-started)
            assert out['source_unchanged'] and updates[0]==0
            dump(args.output,out)
    finally:
        archive.close()


def run(args):
    import math
    from statistics import mean
    args.work.mkdir(parents=True,exist_ok=True)
    status=args.work/'status.json'
    start=time.perf_counter()
    def execute(name,q=None,d=None):
        output=args.work/(name+'.json')
        if output.exists():
            return json.loads(output.read_text(encoding='utf8'))
        dump(status,dict(status='running',stage=name,pid=os.getpid()))
        cmd=[sys.executable,'-m','scripts.audit_native_choice_tree','--leaf',
            '--checkpoint',str(args.checkpoint),'--work',str(args.work/'private_body'),
            '--output',str(output)]
        if q is not None:cmd+=['--question',str(q)]
        if d is not None:cmd+=['--proposal',str(d)]
        with (args.work/(name+'.log')).open('w',encoding='utf8') as log:
            result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'actual restored leaf failed: {name}; inspect its log')
        return json.loads(output.read_text(encoding='utf8'))
    try:
        natural=execute('natural')
        def event(row,stage):return next(x for x in row['selections'] if x['stage']==stage)
        # Contract every level in host float64. Keep the native float32 dot
        # separately, rather than mixing its rounded leaf sums into one side
        # of an otherwise float64 associativity/reconstruction check.
        def actor_mean(row):
            return sum(p*q for p,q in zip(row['actor_probability'],row['local_U_values']))
        qp=event(natural,'question')['probability']
        leaves=[];by_question=[];checks={}
        for r in range(len(qp)):
            branch=[]
            for d in range(natural['proposal_count']):
                row=execute(f'q{r}_d{d}',r,d)
                checks[f'{r}.{d}.question_probability']=event(row,'question')['probability']==qp
                checks[f'{r}.{d}.conditions']=(row['source_sha256']==natural['source_sha256'] and
                    row['source_cycle']==natural['source_cycle'] and
                    row['epistemic_schema']==natural['epistemic_schema'] and
                    row['settlement']==natural['settlement'] and row['source_unchanged'])
                checks[f'{r}.{d}.zero_world_update']=row['world_calls']==row['optimizer_updates']==0
                checks[f'{r}.{d}.actor_replay']=row['actor_replay_error']<3e-5
                checks[f'{r}.{d}.native_dot_roundoff']=abs(actor_mean(row)-row['actor_expected_U'])<3e-6
                if branch:
                    checks[f'{r}.{d}.same_drafts']=row['proposed_words']==branch[0]['proposed_words']
                    checks[f'{r}.{d}.same_proposal_probability']=(event(row,'proposal')['probability']==
                        event(branch[0],'proposal')['probability'])
                    checks[f'{r}.{d}.same_Q']=row['query_facts']==branch[0]['query_facts']
                branch.append(row)
            dp=event(branch[0],'proposal')['probability']
            expected=sum(w*actor_mean(x) for w,x in zip(dp,branch))
            by_question.append(dict(question_index=r,question_probability=qp[r],
                unknown=branch[0]['question']['unknown_identity'],open=branch[0]['question']['open'],
                query_facts=branch[0]['query_facts'],proposal_probability=dp,
                actor_values=[actor_mean(x) for x in branch],expected_U=expected,
                recalled_values=[x['recalled_value'] for x in branch]))
            leaves.append(branch)
        # Independent flattened joint sum and nested contraction.
        nested=sum(qp[r]*x['expected_U'] for r,x in enumerate(by_question))
        flat=sum(qp[r]*event(row,'proposal')['probability'][d]*p*v
            for r,branch in enumerate(leaves) for d,row in enumerate(branch)
            for p,v in zip(row['actor_probability'],row['local_U_values']))
        mass=sum(qp[r]*event(row,'proposal')['probability'][d]*sum(row['actor_probability'])
            for r,branch in enumerate(leaves) for d,row in enumerate(branch))
        checks['tower_reconstruction']=abs(nested-flat)<1e-10
        checks['joint_mass']=abs(mass-1)<1e-5
        nr=event(natural,'question')['selected'];nd=event(natural,'proposal')['selected']
        replay=leaves[nr][nd]
        checks['natural_leaf_replay']=(replay['selected_words']==natural['selected_words'] and
            replay['actor_probability']==natural['actor_probability'] and
            replay['local_U_values']==natural['local_U_values'])
        current_hash=hashlib.sha256()
        with args.checkpoint.open('rb') as source:
            for block in iter(lambda:source.read(1024*1024),b''):
                current_hash.update(block)
        checks['source_unchanged_at_publication']=current_hash.hexdigest()==natural['source_sha256']
        # These add exactly, separating categorical conditioning from recall.
        selected=actor_mean(natural);question=by_question[nr]['expected_U']
        components=dict(recalled_to_actual_leaf=natural['recalled_value']-selected,
            selected_draft_to_draft_average=selected-question,
            selected_question_to_question_average=question-nested)
        out=dict(valid=all(checks.values()),checks=checks,source_sha256=natural['source_sha256'],
            source_cycle=natural['source_cycle'],epistemic_schema=natural['epistemic_schema'],
            scope='actual complete boundary, categorical Q/draft/actor tree conditional on native draft byte draws',
            by_question=by_question,nested_expected_U=nested,flattened_expected_U=flat,total_mass=mass,
            selected_actor_expected_U=selected,recalled_value=natural['recalled_value'],
            reference_gap=natural['recalled_value']-nested,gap_components=components,
            gap_reconstruction=natural['recalled_value']-nested-sum(components.values()),
            question_value_range=max(x['expected_U'] for x in by_question)-min(x['expected_U'] for x in by_question),
            optimizer_updates=0,world_calls=0,leaves=sum(map(len,leaves)),
            seconds=time.perf_counter()-start)
        dump(args.output,out)
        dump(status,dict(status='completed' if out['valid'] else 'failed_checks',
                         output=str(args.output),checks=len(checks),seconds=out['seconds']))
        assert out['valid'],[k for k,v in checks.items() if not v]
        print(json.dumps(dict(valid=True,leaves=out['leaves'],value=nested,gap=out['reference_gap'])))
    except Exception as error:
        dump(status,dict(status='failed',error=str(error)))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--leaf',action='store_true')
    p.add_argument('--question',type=int)
    p.add_argument('--proposal',type=int)
    a=p.parse_args()
    leaf(a) if a.leaf else run(a)
