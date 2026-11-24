"""Admit schema3 on a real complete body's actual hypothetical transition."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--native-word-execution',action='store_true')
    a=p.parse_args()
    import torch
    from scripts.ib_local.run_mother_continuation import continuation_parser,prepare_continuation
    from scripts.ib_local.run_causal_water_reply import sha256_file
    from scripts.ib_local.autonomous_cycle import dispatch
    from scripts.ib_local import epistemic_value_reference as ev
    from scripts.ib_local.causal_relation_program import Evidence,Port,revise,compose
    from scripts.ib_local.prospective_decision import preview_bind
    torch.set_num_threads(1)
    before=sha256_file(a.checkpoint);started=time.perf_counter()
    opts=continuation_parser().parse_args(['--checkpoint',str(a.checkpoint),
        '--output',str(a.checkpoint.parent),'--resume-output','--cycles','1',
        '--world-source','text','--autonomous-events','--optimizer-policy','budgeted',
        '--table-storage',str(a.work/'table'),'--causal-storage',str(a.work/'causal')])
    loaded=prepare_continuation(opts,resume_output=True)
    e=loaded.engine;model=e.model
    try:
        for _ in range(64):
            if e.autonomous_ready:break
            assert dispatch(e)['autonomous_event']['kind']!='external'
        else:raise RuntimeError('natural release unavailable')
        model._causal_archive.prepare();e._identified_event_fact=None
        torch.randn(model.encoder.out_features,32,device=e.device)
        utter_start=time.perf_counter()
        words,record=e._utter()
        baseline_seconds=time.perf_counter()-utter_start
        source=e._internal_source.reshape(1,-1);action=e._internal_action[record['choice']:record['choice']+1]
        state=e._program_question_state;planned=e._program_forecasts
        b=record['choice'];bank=model.causal_relations
        with torch.no_grad():
            y=planned['hypotheses'][b:b+1,0]+planned['path']['mother']['noise'].sqrt()*bank.observation_directions[0,-1]
            updated=revise(state,Evidence(source,action,y,Port.MOTHER,False,encoded=True),model=model)
            momentum=e._momentum_decision_read[None]
            path=compose(model,updated,source,action,ridge=e.relation_ridge,
                reward_valid=e.reward_label_valid(wood=True),momentum=momentum,mother=y)
            old=ev.branch_value(model,updated,path['decision']['mean'],momentum=momentum)
            buffers={n:t.clone() for n,t in bank.named_buffers()}
            sql=model._causal_archive.db.total_changes
            cpu=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state().clone()
            params=sum(p.numel() for p in model.parameters())
            ev.configure(e,schema=3,word_execution='native_anchor' if a.native_word_execution else 'historical')
            begin=time.perf_counter()
            new=ev.branch_value(model,updated,path['decision']['mean'],momentum=momentum,
                                source=source,action=action,mother=y)
            torch.cuda.synchronize();seconds=time.perf_counter()-begin
            detail=e.prospective_decision_last
            checks=dict(source_unchanged=sha256_file(a.checkpoint)==before,
                native_buffers=all(torch.equal(t,buffers[n]) for n,t in bank.named_buffers()),
                sql_read_only=model._causal_archive.db.total_changes==sql,
                cpu_rng=torch.equal(cpu,torch.get_rng_state()),cuda_rng=torch.equal(gpu,torch.cuda.get_rng_state()),
                same_parameters=params==sum(p.numel() for p in model.parameters()),
                finite=bool(torch.isfinite(new).all()),
                latent_coordinate=max(x['dynamics']['latent_residual'] for x in detail)<3e-5,
                has_native_queries=all(bool(x['cases']) for x in detail),
                bounded_programs=all(len(c['programs'])<=e.joint_proposal_budget for x in detail for c in x['cases']))
            out=dict(valid=all(checks.values()),checks=checks,source_cycle=loaded.cycle,
                source_sha256=before,old_value=old.tolist(),prospective_value=new.tolist(),
                prospective_seconds=seconds,native_utter_seconds=baseline_seconds,details=detail,
                schema=3,word_execution=e.epistemic_word_execution,
                optimizer_updates=0,world_calls=0,total_seconds=time.perf_counter()-started)
            a.output.parent.mkdir(parents=True,exist_ok=True)
            a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
            assert out['valid'],[k for k,v in checks.items() if not v]
            print(json.dumps(dict(valid=True,old=old.tolist(),new=new.tolist(),seconds=seconds)))
    finally:model._causal_archive.close()


if __name__=='__main__':main()
