"""Measure the prospective word transport against the original known-word executor.

Same actual body boundary, same chosen Q, same stored words and finite U.
Only the action endpoint differs: recalled displacement or native execution.
No new decoding, optimizer update, world observation or source write occurs.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--native-self',action='store_true')
    a=p.parse_args()
    import torch
    from scripts.ib_local.run_mother_continuation import continuation_parser,prepare_continuation
    from scripts.ib_local.run_causal_water_reply import sha256_file
    from scripts.ib_local.autonomous_cycle import dispatch
    from scripts.ib_local import epistemic_value_reference as ev
    from scripts.ib_local import navigation_generative_cycle as cycle
    from scripts.ib_local.prospective_decision import program_actions,terminal
    from scripts.ib_local.known_utterance_runtime import known_film
    from scripts.ib_local.table_branch import table_branch
    from scripts.ib_local.causal_relation_program import compose
    from scripts.ib_local.causal_use_value import queries,calibrated
    torch.set_num_threads(1)
    start=time.perf_counter();digest=sha256_file(a.checkpoint)
    opts=continuation_parser().parse_args(['--checkpoint',str(a.checkpoint),
        '--output',str(a.checkpoint.parent),'--resume-output','--cycles','1',
        '--world-source','text','--autonomous-events','--optimizer-policy','budgeted',
        '--table-storage',str(a.work/'table'),'--causal-storage',str(a.work/'causal')])
    loaded=prepare_continuation(opts,resume_output=True);e=loaded.engine;model=e.model
    updates=[0]
    def forbidden(*args,**kw):
        updates[0]+=1
        raise RuntimeError('read-only operator audit reached optimizer')
    e.optimizer.step=forbidden
    capture={};original=cycle.question_choices
    def collect(*args,**kw):
        result=original(*args,**kw);capture['questions']=result
        capture['question_source']=args[2].reshape(1,-1).detach().clone()
        return result
    try:
        for _ in range(64):
            if e.autonomous_ready:break
            assert dispatch(e)['autonomous_event']['kind']!='external'
        else:raise RuntimeError('native release unavailable')
        model._causal_archive.prepare();e._identified_event_fact=None
        # Exactly the native pre-utterance direction draw.
        torch.randn(model.encoder.out_features,32,device=e.device)
        cycle.question_choices=collect
        try:words,record=e._utter()
        finally:cycle.question_choices=original
        selected=e._binding_choices[-1]['selected']
        proposal=capture['questions']['proposals'][selected]
        state=e._program_question_state
        source=e._internal_source.reshape(1,-1)
        archive=model._causal_archive;sql=archive.db.total_changes
        cpu=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state().clone()
        def copy_state(x):
            return x.clone() if torch.is_tensor(x) else type(x)(copy_state(t) for t in x)
        def same_state(x,y):
            return torch.equal(x,y) if torch.is_tensor(x) else len(x)==len(y) and all(same_state(a,b) for a,b in zip(x,y))
        recurrent=copy_state(e.recurrent);momenta=copy_state(e.bfss_momenta)
        with torch.no_grad():
            predicted,slots,reuse=program_actions(e,source,proposal)
            actual=[];payloads=[];execute_seconds=[]
            for slot in slots:
                payload=bytes(model.memory.words[slot,:int(model.memory.word_count[slot])].cpu().tolist())
                t=time.perf_counter()
                with table_branch(e.table) as branch:
                    known_film(e,payload,branch)
                actual.append(e._draft_actions.pop()[None]);payloads.append(payload)
                torch.cuda.synchronize();execute_seconds.append(time.perf_counter()-t)
            if not actual:raise RuntimeError('no actual word programs')
            actual=torch.cat(actual)
            context=ev.from_body(e,state,queries(e,state),momentum=e._momentum_decision_read[None])
            endpoints=torch.cat((predicted,actual));z=model.encoder(source)
            with terminal():
                path=compose(model,state,source.expand_as(endpoints),endpoints,
                    ridge=e.relation_ridge,reward_valid=e.reward_label_valid(wood=True),
                    momentum=e._momentum_decision_read[None].expand(len(endpoints),-1))
                u=10*float(calibrated(e,1.))*ev.expected(model,context,z,
                    model.encoder(endpoints),ridge=e.relation_ridge,distribution=path['mother'])
            before,after=u.split(len(slots))
            zp,za=model.encoder(predicted),model.encoder(actual)
            rows=[]
            for i,slot in enumerate(slots):
                rows.append(dict(slot=slot,stamp=int(model.memory.last_write[slot]),
                    words_hex=payloads[i].hex(),bytes=len(payloads[i]),
                    raw_endpoint_MSE=float((predicted[i]-actual[i]).square().mean()),
                    latent_endpoint_MSE=float((zp[i]-za[i]).square().mean()),
                    predicted_increment_norm=float((zp[i]-z[0]).norm()),
                    actual_increment_norm=float((za[i]-z[0]).norm()),
                    predicted_U=float(before[i]),executed_U=float(after[i]),
                    known_word_seconds=execute_seconds[i]))
            checks=dict(source_unchanged=sha256_file(a.checkpoint)==digest,
                optimizer_updates_zero=updates[0]==0,sql_read_only=archive.db.total_changes==sql,
                cpu_rng=torch.equal(cpu,torch.get_rng_state()),gpu_rng=torch.equal(gpu,torch.cuda.get_rng_state()),
                recurrent_unchanged=same_state(recurrent,e.recurrent),momentum_unchanged=same_state(momenta,e.bfss_momenta),
                finite=bool(torch.isfinite(u).all()))
            native_check=None
            if a.native_self:
                from scripts.ib_local.prospective_self_execution import execute,question_context
                pre=capture['question_source']
                q_error=float((question_context(e,pre,proposal)[0]-model._question_context).abs().max())
                corrected_source,corrected_actions,_=execute(e,pre,proposal,slots)
                native_check=dict(context_max_error=q_error,
                    post_source_max_error=float((corrected_source-source).abs().max()),
                    endpoint_max_error=float((corrected_actions-actual).abs().max()))
                checks['native_anchor_exact']=max(native_check.values())<3e-5
                checks['native_anchor_private']=(archive.db.total_changes==sql and
                    torch.equal(cpu,torch.get_rng_state()) and torch.equal(gpu,torch.cuda.get_rng_state()) and
                    same_state(recurrent,e.recurrent) and same_state(momenta,e.bfss_momenta))
            out=dict(valid=all(checks.values()),checks=checks,source_cycle=loaded.cycle,source_sha256=digest,
                unknown=e._program_question_roles,actual_generated_words=words.hex(),rows=rows,
                predicted_best_slot=slots[int(before.argmax())],executed_best_slot=slots[int(after.argmax())],
                U_MAE=float((before-after).abs().mean()),
                endpoint_mapping='transport=h_current+old_action-old_source; native=known_film(current recurrent/table/Q, same words)',
                payoff='original finite U on same real query scope and P',optimizer_updates=0,world_calls=0,
                boundary_ports=dict(
                    label_to_question_input_max_error=float((e._internal_boundary.reshape(1,-1)-capture['question_source']).abs().max()),
                    before_after_intention_raw_MSE=float((source-capture['question_source']).square().mean()),
                    before_after_intention_latent_MSE=float((model.encoder(source)-model.encoder(capture['question_source'])).square().mean())),
                witnesses=dict(pre_question_source=capture['question_source'].tolist(),post_question_source=source.tolist(),
                    question_context=model._question_context.tolist(),program_slots=slots,executed_endpoints=actual.tolist()),
                native_self_execution=native_check,
                seconds=time.perf_counter()-start)
            a.output.parent.mkdir(parents=True,exist_ok=True)
            a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
            assert out['valid'],[k for k,v in checks.items() if not v]
            print(json.dumps({k:out[k] for k in ('valid','source_cycle','U_MAE','predicted_best_slot','executed_best_slot','seconds')}))
    finally:cycle.question_choices=original;archive=model._causal_archive;archive.close()


if __name__=='__main__':main()
