"""Actual arrived-response usefulness versus geometric relation release.

Re-encode a saved pre-event scope in current coordinates. This is a numerical
reconstruction, not an original action-time forecast or a later-use result.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    import torch
    from scripts.ib_local.run_mother_continuation import continuation_parser,prepare_continuation
    from scripts.ib_local.run_causal_water_reply import sha256_file
    from scripts.ib_local.causal_relation_program import Port,mechanism,log_score,revise,relation_features
    from scripts.ib_local.finite_use_value import score_state
    from scripts.ib_local.relation_evidence import mechanism_evidence,evidence_key
    from scripts.ib_local.shared_unknown_bindings import _read_evidence
    torch.set_num_threads(1)
    before=sha256_file(a.checkpoint)
    opts=continuation_parser().parse_args(['--checkpoint',str(a.checkpoint),'--output',str(a.checkpoint.parent),
        '--resume-output','--cycles','1','--world-source','text','--autonomous-events',
        '--optimizer-policy','budgeted','--table-storage',str(a.work/'table'),
        '--causal-storage',str(a.work/'causal')])
    loaded=prepare_continuation(opts,resume_output=True)
    e,m=loaded.engine,loaded.engine.model
    try:
        with torch.no_grad():
            pending=e.internal_pending['causal_program']
            state=score_state(pending['state'])
            incoming=next(f for f in _read_evidence(m._causal_archive,int(pending['episode_identity'].split(':')[1]))
                          if f.kind==Port.MOTHER)
            old=mechanism_evidence(m,state,Port.MOTHER)
            assert incoming.fact_id not in {f.fact_id for f in old}
            def score(s):
                d=mechanism(m,s,incoming.source,incoming.action,ridge=e.relation_ridge)
                return float(log_score(d,m.encoder(incoming.outcome)).mean())
            factual=score(state)
            rows=[]
            for f in old:
                key=evidence_key(f)
                cut=replace(state,constraints=tuple(x for x in state.constraints if evidence_key(x)!=key),
                            extra=tuple(x for x in state.extra if evidence_key(x)!=key))
                rows.append(dict(fact=key,benefit=factual-score(cut)))
            after=revise(state,incoming,model=m)
            kept=mechanism_evidence(m,after,Port.MOTHER)
            x=relation_features(m,*(m.encoder(torch.cat([getattr(f,n) for f in kept]))
                                     for n in ('source','action','outcome')))
            gram=x@x.T; eye=torch.eye(len(x),device=x.device,dtype=x.dtype)
            h=(gram@torch.linalg.solve(gram+e.relation_ridge*eye,eye)).diag()
            benefits={r['fact']:r['benefit'] for r in rows}
            u=h.new_tensor([benefits.get(evidence_key(f),0.) for f in kept])
            b=(u/(u.abs()+m.memory.read_credit_eps)).tanh()
            weighted=h*b.exp()
            for f,hi,bi,wi in zip(kept,h,b,weighted):
                key=evidence_key(f)
                r=next((r for r in rows if r['fact']==key),None)
                if r is not None:r.update(next_leverage=float(hi),next_bias=float(bi),next_priority=float(wi))
            report=dict(cycle=loaded.cycle,source=str(a.checkpoint),source_sha256=before,
                source_unchanged=before==sha256_file(a.checkpoint),world_calls=0,optimizer_updates=0,
                scope=dict(relation=state.identity,generation=state.generation,
                           unknown=state.binding.identity if state.binding else None),
                arrived_event=incoming.fact_id,factual_score=factual,old_contributions=rows,
                actual_first_supports=[evidence_key(f) for f in kept],
                prospective_next_release=dict(geometric=evidence_key(kept[int(h.argmin())]),
                    use_weighted=evidence_key(kept[int(weighted.argmin())])),
                interpretation='Use of the saved real response in common current coordinates; '
                    'next release is a local admission diagnostic, not a real next event.')
        a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        print(json.dumps(report,ensure_ascii=False))
    finally:m._causal_archive.close()


if __name__=='__main__':main()
