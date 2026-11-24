"""Complete 2481: raw/encoded U, delayed-use release and bounded persistence.

The saved last real response is re-executed at common current coordinates.
This is an interface/cost audit, separate from the prospective life experiment.
"""
import argparse
from dataclasses import replace,asdict
import json
import os
from pathlib import Path
import time

os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    import torch
    from scripts.ib_local.run_mother_continuation import continuation_parser,prepare_continuation
    from scripts.ib_local.run_causal_water_reply import sha256_file
    from scripts.ib_local import relation_retention_use as use,epistemic_value_reference as ev
    from scripts.ib_local.causal_relation_program import Port,revise,imagine
    from scripts.ib_local.shared_unknown_bindings import _read_evidence,_with_constraints,_row,retain_constraints
    from scripts.ib_local.relation_evidence import mechanism_evidence,evidence_key
    from scripts.ib_local.causal_use_value import queries
    from scripts.ib_local.finite_use_value import score_state,expected
    from scripts.ib_local.navigation_momentum_wood import decision_momentum_read
    torch.set_num_threads(1)
    before=sha256_file(args.checkpoint)
    opts=continuation_parser().parse_args(['--checkpoint',str(args.checkpoint),
        '--output',str(args.checkpoint.parent),'--resume-output','--cycles','1',
        '--world-source','text','--autonomous-events','--optimizer-policy','budgeted',
        '--table-storage',str(args.work/'table'),'--causal-storage',str(args.work/'causal')])
    loaded=prepare_continuation(opts,resume_output=True)
    e,m=loaded.engine,loaded.engine.model
    try:
        use.configure(e,'apply')
        with torch.no_grad():
            saved=e.internal_pending['causal_program']
            old=score_state(saved['state'])
            scope=old.binding.identity
            old=replace(old,retention_use=use.load(m,scope))
            incoming=_read_evidence(m._causal_archive,int(saved['episode_identity'].split(':')[1]))[0]
            old_ids=[evidence_key(f) for f in mechanism_evidence(m,old,Port.MOTHER)]
            ctx=ev.from_body(e,old,queries(e,old))
            raw=revise(old,incoming,model=m)
            encoded=ev.revise(m,ctx,ev.encode_fact(m,incoming),ridge=e.relation_ridge)
            actual=ev.from_body(e,raw,queries(e,old))
            assert [f.identity for f in encoded.facts]==[f.identity for f in actual.facts]
            for x,y in zip(encoded.retention_use.moments,actual.retention_use.moments):
                assert x[0]==y[0] and x[3]==y[3]
                assert max(abs(x[j]-y[j]) for j in (1,2))<3e-6
            dsource=e._internal_source.reshape(1,-1).expand(3,-1)
            actions=e._internal_action
            from scripts.ib_local.causal_relation_program import mechanism
            distribution=mechanism(m,old,dsource,actions,ridge=e.relation_ridge)
            start=time.perf_counter()
            prod=expected(e,old,dsource,actions,distribution,queries(e,old))
            ref=ev.expected(m,ctx,m.encoder(dsource),m.encoder(actions),ridge=e.relation_ridge)
            parity=float((prod-ref).abs().max())
            assert parity<3e-6
            # One observed contribution is retained; only the NEXT release consumes it.
            kept=mechanism_evidence(m,raw,Port.MOTHER)
            live=tuple(ev.encode_fact(m,f) for f in kept)
            x=torch.cat([f.source for f in live]);aa=torch.cat([f.action for f in live]);y=torch.cat([f.outcome for f in live])
            from scripts.ib_local.causal_relation_program import relation_features
            features=relation_features(m,x,aa,y)
            identities=[f.identity for f in live]
            geometric=use.select(features,identities,replace(raw.retention_use,mode='observe'),e.relation_ridge)[1]
            weighted=use.select(features,identities,raw.retention_use,e.relation_ridge)[1]
            use.commit(m,old.retention_use,raw.retention_use)
            reread=_with_constraints(m._causal_archive,old,_row(m._causal_archive,scope))
            assert reread.retention_use==raw.retention_use
            baseline=replace(raw.retention_use,mode='observe')
            # An already observed OTHER event supplies the protected incoming;
            # only its ID is relevant to this controlled release comparison.
            other_id=next(i for i in range(2481,2400,-1) if f'episode:{i}' not in identities)
            other=_read_evidence(m._causal_archive,other_id)[0]
            a=retain_constraints(m,kept,(other,),len(old.valid),use=baseline)
            b=retain_constraints(m,kept,(other,),len(old.valid),use=reread.retention_use)
            released_a=sorted(set(identities)-{evidence_key(f) for f in a})
            released_b=sorted(set(identities)-{evidence_key(f) for f in b})
            assert released_a!=released_b
            # Full original imagination, all candidates/ports retained.
            started=time.perf_counter()
            state=replace(old,organization=saved['state'].organization)
            mom=decision_momentum_read(m,e.bfss_momenta,dsource[:1]).expand(3,-1)
            forecast=imagine(m,state,dsource,actions,ridge=e.relation_ridge,
                reward_valid=e.reward_label_valid(wood=True),momentum=mom)
            seconds=time.perf_counter()-started
            assert torch.isfinite(forecast['epistemic_after_value']).all()
            report=dict(valid=True,cycle=loaded.cycle,source_sha256=before,
                schema=e.relation_retention_use_schema,settlement_kind=e.settlement_kind,
                source_unchanged=before==sha256_file(args.checkpoint),optimizer_updates=0,world_calls=0,
                scope=scope,arrived_event=incoming.fact_id,old_supports=old_ids,
                raw_encoded_finite_U_max_error=parity,
                first_uses_old_neutral=True,after_use=asdict(raw.retention_use),
                next_geometric_release=geometric,next_use_release=weighted,
                actual_core_releases=dict(observe=released_a,apply=released_b),
                persisted_then_reread=True,budget=len(old.valid),
                complete_imagination_seconds=seconds,
                contextual_future=forecast['epistemic_after_value'].tolist(),
                audit_total_seconds=time.perf_counter()-start,
                scope_note='saved actual response, common current coordinates; controlled next release, not a natural next outcome')
        args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        print(json.dumps(report,ensure_ascii=False))
    finally:m._causal_archive.close()


if __name__=='__main__':main()
