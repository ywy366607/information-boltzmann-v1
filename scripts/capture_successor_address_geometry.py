"""Capture finite geometry at three actual boundaries of the exact old life.

Only the original full learner runs. Saved capsules contain current encoder,
address halfspaces, conditional predictive parameters and observed later labels
for offline scores. No historical optimizer, body, token or RNG graph is copied.
"""
import argparse
import copy
import json
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint','reference','work'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    import torch
    from scripts.ib_local import run_mother_continuation as loader
    from scripts.ib_local import epistemic_value_reference as ev
    from scripts.ib_local import prospective_decision as pd
    from scripts.ib_local import mother_checkpoint
    from scripts.ib_local import successor_address_geometry as geometry
    from scripts.ib_local.successor_distribution import quadrature
    from scripts.ib_local.preview_runtime import preview_scope
    from scripts.ib_local.causal_relation_program import Evidence,Port,revise,compose
    from scripts.audit_native_future_boundary import archived_vectors
    from scripts.audit_native_transition_ensemble import checkpoint_without_observer
    from scripts.run_native_preview_restore_comparison import main as run
    a.work.mkdir(parents=True,exist_ok=False)
    rows=[json.loads(x) for x in (a.reference/'events.jsonl').read_text(encoding='utf8').splitlines()]
    external=[r for r in rows if 'causal_use' in r]
    archive,=list((a.reference/'body').glob('causal_archive_*.sqlite'))
    captured=[];capsules=[];index=[0]
    prepare_original,branch_original,save_original=loader.prepare_continuation,ev.branch_value,mother_checkpoint.save

    def freeze(x):
        if torch.is_tensor(x):return x.detach().cpu().clone()
        if isinstance(x,dict):return {k:freeze(v) for k,v in x.items()}
        if isinstance(x,(tuple,list)):return type(x)(freeze(v) for v in x)
        return x

    def branch(model,state,latent,**kw):
        result=branch_original(model,state,latent,**kw)
        if not pd.at_terminal():captured.append((state,kw,float(result[0])))
        return result

    def save(engine,*args,**kw):
        with checkpoint_without_observer(engine):return save_original(engine,*args,**kw)

    def prepare(*args,**kw):
        loaded=prepare_original(*args,**kw);e=loaded.engine;utter=e._utter
        def observed():
            captured.clear();words,record=utter()
            n=index[0];index[0]+=1
            if n>=3:
                captured.clear();return words,record
            source=e._internal_source[None];actions=e._internal_action
            episode=int(external[n]['causal_program']['committed_episode_identity'].split(':')[1])
            actual=archived_vectors(archive,episode)
            vector=lambda name:torch.as_tensor(actual[name],device=e.device,dtype=source.dtype).reshape(1,-1)
            errors=[float((source-vector('source')).abs().max()),
                float((actions[record['choice']:record['choice']+1]-vector('action')).abs().max())]
            assert max(errors)<3e-5,errors
            cpu,gpu=torch.get_rng_state().clone(),torch.cuda.get_rng_state().clone()
            sql=e.model._causal_archive.db.total_changes
            bank={k:v.clone() for k,v in e.model.causal_relations.named_buffers()}
            cases=[]

            def case(name,state,action,mother,distribution,extra=None):
                raw,info=pd.boundary(e,state,source,action,mother,distribution['mean'])
                packet=geometry.snapshot(e.model,raw,distribution['mean'])
                points,_=quadrature(distribution,e.joint_proposal_budget)
                actual_z=e.model.encoder(vector('internal_next'))
                probes=torch.cat((distribution['mean'],points[0],actual_z))
                check=geometry.native_check(e.model,packet,probes)
                assert check['identities_equal'] and check['max_similarity_error']<3e-5 and check['max_encoder_residual']<3e-5,check
                return freeze(dict(name=name,geometry=packet,
                    distribution={k:distribution[k] for k in ('mean','scale2','df')},
                    actual_latent=actual_z,actual_raw=vector('internal_next'),check=check,
                    budget=e.joint_proposal_budget,extra=extra))

            with torch.no_grad(),preview_scope(e.model):
                for j,action in enumerate(actions):
                    selected=[c for c in captured if torch.equal(c[1]['source'],source) and
                              torch.equal(c[1]['action'],action[None])][:e._program_forecasts['hypotheses'].shape[1]]
                    for h,(state,kw,value) in enumerate(selected):
                        cases.append(case(f'action:{j}:response:{h}',state,action[None],kw['mother'],kw['decision_distribution']))
                j=record['choice'];action=actions[j:j+1];mother=e.model.encoder(vector('mother_next'))
                state=revise(e._program_question_state,Evidence(source,action,mother,Port.MOTHER,
                    False,encoded=True),model=e.model)
                path=compose(e.model,state,source,action,mother=mother,momentum=e._momentum_decision_read[None],
                    ridge=e.relation_ridge,reward_valid=e.reward_label_valid(wood=True))
                keys=('mean','scale2','df','x','y','query','base','noise','leverage')
                extra=dict(programs=[{k:d[k] for k in keys if k in d} for d in path['program_predictions']],
                    weights=state.program_weights)
                cases.append(case('actual_response',state,action,mother,path['decision'],extra))
            checks=dict(cpu_rng=torch.equal(cpu,torch.get_rng_state()),gpu_rng=torch.equal(gpu,torch.cuda.get_rng_state()),
                read_only=e.model._causal_archive.db.total_changes==sql,
                bank_unchanged=all(torch.equal(v,bank[k]) for k,v in e.model.causal_relations.named_buffers()))
            assert all(checks.values()),checks
            capsule=dict(cycle=2494+n,choice=record['choice'],episode=episode,
                actual_relation=external[n+1]['causal_use']['relation'][0],cases=cases,checks=checks,source_errors=errors)
            torch.save(capsule,a.work/f'geometry_{2494+n}.pt');capsules.append(capsule)
            (a.work/'capture_status.json').write_text(json.dumps(dict(status='capturing',cycles=[r['cycle'] for r in capsules],
                checks=[r['checks'] for r in capsules]),indent=2)+'\n',encoding='utf8')
            print(json.dumps(dict(cycle=2494+n,cases=len(cases),checks=checks)),flush=True)
            captured.clear();return words,record
        observed._native_ensemble_observer=True;e._utter=observed
        return loaded

    loader.prepare_continuation,ev.branch_value,mother_checkpoint.save=prepare,branch,save
    try:
        sys.argv=[sys.argv[0],'--checkpoint',str(a.checkpoint),'--work',str(a.work/'replay'),
            '--cycles','4','--native-decision','--native-word-execution','--replay-prefix',str(a.reference/'events.jsonl')]
        run()
    finally:
        loader.prepare_continuation,ev.branch_value,mother_checkpoint.save=prepare_original,branch_original,save_original
    assert len(capsules)==3
    (a.work/'capture_status.json').write_text(json.dumps(dict(status='completed',cycles=[r['cycle'] for r in capsules]),indent=2)+'\n',encoding='utf8')


if __name__=='__main__':main()
