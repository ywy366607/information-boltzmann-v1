"""Matched mean/distribution reads on three original, fully learning transitions.

The original schema3 life is replayed exactly. A read-only observer evaluates
the coordinated successor distribution at the same physical decision ports.
Later observations enter only the explicitly retrospective actual-response
case and final scores; no optimizer or world call receives them.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint','reference','work','output'):
        p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args()
    import torch
    from scripts.ib_local import run_mother_continuation as loader
    from scripts.ib_local import epistemic_value_reference as ev
    from scripts.ib_local import prospective_decision as pd
    from scripts.ib_local import successor_distribution as sd
    from scripts.ib_local import mother_checkpoint
    from scripts.ib_local.preview_runtime import preview_scope
    from scripts.ib_local.causal_relation_program import Evidence,Port,revise,compose,log_score
    from scripts.audit_native_future_boundary import archived_vectors
    from scripts.audit_native_transition_ensemble import checkpoint_without_observer
    from scripts.audit_native_value_transition import summarize
    from scripts.run_native_preview_restore_comparison import main as run
    args.work.mkdir(parents=True,exist_ok=False)
    reference=[json.loads(x) for x in (args.reference/'events.jsonl').read_text(encoding='utf8').splitlines()]
    external=[r for r in reference if 'causal_use' in r]
    archive,=list((args.reference/'body').glob('causal_archive_*.sqlite'))
    labels={int(r['causal_program']['committed_episode_identity'].split(':')[1]):archived_vectors(
        archive,int(r['causal_program']['committed_episode_identity'].split(':')[1])) for r in external[:-1]}
    captured=[];observations=[];utterances=[0]
    original_prepare,original_branch,original_save=loader.prepare_continuation,ev.branch_value,mother_checkpoint.save

    def branch(model,state,latent,**kw):
        result=original_branch(model,state,latent,**kw)
        if not pd.at_terminal():
            captured.append(dict(state=state,latent=latent,kw=kw,value=float(result[0]),
                                 trace=copy.deepcopy(model._epistemic_value_engine.prospective_decision_last)))
        return result

    def save(engine,*a,**kw):
        with checkpoint_without_observer(engine):return original_save(engine,*a,**kw)

    def prepare(*a,**kw):
        loaded=original_prepare(*a,**kw);e=loaded.engine;utter=e._utter
        def observed():
            captured.clear()
            words,record=utter()
            index=utterances[0];utterances[0]+=1
            if index>=3:
                captured.clear();return words,record
            source=e._internal_source.reshape(1,-1);actions=e._internal_action
            count=e._program_forecasts['hypotheses'].shape[1]
            episode=int(external[index]['causal_program']['committed_episode_identity'].split(':')[1])
            actual=labels[episode]
            tensor=lambda name:torch.as_tensor(actual[name],device=e.device,dtype=source.dtype).reshape(1,-1)
            actual_source=tensor('source');actual_action=tensor('action')
            errors=[float((source-actual_source).abs().max()),
                    float((actions[record['choice']:record['choice']+1]-actual_action).abs().max())]
            assert max(errors)<3e-5,errors
            bank={n:t.clone() for n,t in e.model.causal_relations.named_buffers()}
            sql=e.model._causal_archive.db.total_changes
            cpu,gpu=torch.get_rng_state().clone(),torch.cuda.get_rng_state().clone()
            names=('prospective_decision_last','successor_distribution_last')
            traces={n:copy.deepcopy(getattr(e,n,None)) for n in names}
            present={n:hasattr(e,n) for n in names}
            rows=[]
            try:
                with torch.no_grad(),preview_scope(e.model):
                    for j,action in enumerate(actions):
                        calls=[c for c in captured if torch.equal(c['kw']['source'],source) and
                               torch.equal(c['kw']['action'],action[None])][:count]
                        assert len(calls)==count,(j,len(calls),count)
                        branches=[]
                        for c in calls:
                            begin=time.perf_counter()
                            result=sd.value(e,c['state'],source,action[None],c['kw']['mother'],
                                            c['kw']['decision_distribution'],c['kw']['momentum'])
                            branches.append(dict(mean_value=c['value'],distribution_value=float(result[0]),
                                seconds=time.perf_counter()-begin,quadrature=copy.deepcopy(e.successor_distribution_last),
                                mean_address=c['trace'][0]['address'],
                                addresses=[r['address'] for r in e.prospective_decision_last],
                                max_lift_residual=max(r['dynamics']['latent_residual'] for r in e.prospective_decision_last)))
                        rows.append(dict(action=j,mean_value=sum(c['mean_value'] for c in branches)/count,
                            distribution_value=sum(c['distribution_value'] for c in branches)/count,branches=branches))
                    j=record['choice'];action=actions[j:j+1];mom=e._momentum_decision_read[None]
                    y=e.model.encoder(tensor('mother_next'))
                    updated=revise(e._program_question_state,Evidence(source,action,y,Port.MOTHER,
                        observed=False,encoded=True),model=e.model)
                    path=compose(e.model,updated,source,action,mother=y,momentum=mom,
                        ridge=e.relation_ridge,reward_valid=e.reward_label_valid(wood=True))
                    mean_value=pd.value(e,updated,source,action,y,path['decision']['mean'],mom)
                    mean_address=copy.deepcopy(e.prospective_decision_last[0]['address'])
                    begin=time.perf_counter()
                    distributed=sd.value(e,updated,source,action,y,path['decision'],mom)
                    oracle=dict(mean_value=float(mean_value[0]),distribution_value=float(distributed[0]),
                        mean_address=mean_address,addresses=[r['address'] for r in e.prospective_decision_last],
                        seconds=time.perf_counter()-begin,
                        actual_latent_log_score=float(log_score(path['decision'],e.model.encoder(tensor('internal_next'))).mean()),
                        quadrature=copy.deepcopy(e.successor_distribution_last))
            finally:
                for n in names:
                    if present[n]:setattr(e,n,traces[n])
                    elif hasattr(e,n):delattr(e,n)
            checks=dict(sql_read_only=e.model._causal_archive.db.total_changes==sql,
                cpu_rng=torch.equal(cpu,torch.get_rng_state()),gpu_rng=torch.equal(gpu,torch.cuda.get_rng_state()),
                bank_unchanged=all(torch.equal(t,bank[n]) for n,t in e.model.causal_relations.named_buffers()))
            assert all(checks.values()),checks
            observations.append(dict(cycle=2494+index,choice=record['choice'],source_errors=errors,
                actions=rows,retrospective_actual_response=oracle,checks=checks))
            (args.work/'progress.json').write_text(json.dumps(observations,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
            print(json.dumps(dict(stage='matched successor reads',cycle=2494+index,
                old=rows[j]['mean_value'],new=rows[j]['distribution_value'],checks=checks)),flush=True)
            captured.clear()
            return words,record
        observed._native_ensemble_observer=True
        e._utter=observed
        return loaded

    loader.prepare_continuation,ev.branch_value,mother_checkpoint.save=prepare,branch,save
    try:
        sys.argv=[sys.argv[0],'--checkpoint',str(args.checkpoint),'--work',str(args.work/'replay'),
            '--cycles','4','--native-decision','--native-word-execution',
            '--replay-prefix',str(args.reference/'events.jsonl')]
        run()
    finally:
        loader.prepare_continuation,ev.branch_value,mother_checkpoint.save=original_prepare,original_branch,original_save
    events=[json.loads(x) for x in (args.work/'replay/events.jsonl').read_text(encoding='utf8').splitlines()]
    pairs=summarize(events)
    for row,pair,next_event in zip(observations,pairs,external[1:]):
        target=pair['next_policy_value'];chosen=row['actions'][row['choice']]
        row.update(actual_next_policy_value=target,actual_next_relation=next_event['causal_use']['relation'][0],
            mean_absolute_error=abs(chosen['mean_value']-target),
            distribution_absolute_error=abs(chosen['distribution_value']-target))
        for name in ('mean_value','distribution_value'):
            baseline=row['actions'][0][name]
            row[name+'_action_contrasts']=[r[name]-baseline for r in row['actions']]
    evaluation=json.loads((args.work/'replay/evaluations.json').read_text(encoding='utf8'))
    expected=json.loads((args.reference/'evaluations.json').read_text(encoding='utf8'))
    source_hash=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    reference_result=json.loads((args.reference/'result.json').read_text(encoding='utf8'))
    checks=dict(three_transitions=len(observations)==3,original_material_exact=evaluation==expected,
        source_unchanged=source_hash==reference_result['initial_source_sha256'],
        private=all(all(r['checks'].values()) for r in observations),
        same_ports=all(max(r['source_errors'])<3e-5 for r in observations),
        native_lift=all(b['max_lift_residual']<3e-5 for r in observations for a in r['actions'] for b in a['branches']))
    out=dict(valid=all(checks.values()),checks=checks,observations=observations,
        source_sha256=source_hash,reference=str(args.reference),
        mean_absolute_error=sum(r['mean_absolute_error'] for r in observations)/3,
        distribution_absolute_error=sum(r['distribution_absolute_error'] for r in observations)/3)
    args.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    assert out['valid'],checks
    print(json.dumps({k:v for k,v in out.items() if k!='observations'}),flush=True)


if __name__=='__main__':main()
