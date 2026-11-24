"""Reproduce first actual pre-draft valuation and inspect one U clock.

Capture the original body call before an external world step. Then compare
observation-only and applied use on that identical state/candidate/hypothesis
bank. No training, world response or real evidence commit is performed.
"""
import argparse
from dataclasses import replace,asdict
import json
import os
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--require-neutral-parity',action='store_true')
    args=p.parse_args()
    import torch
    from scripts.ib_local.run_mother_continuation import continuation_parser,prepare_continuation
    from scripts.ib_local.run_causal_water_reply import sha256_file
    from scripts.ib_local import relation_retention_use as use
    from scripts.ib_local import navigation_causal_program as nav
    from scripts.ib_local.causal_relation_program import imagine,mechanism_hypotheses,Evidence,Port,revise
    from scripts.ib_local.finite_use_value import expected
    from scripts.ib_local.causal_use_value import queries
    from scripts.ib_local.autonomous_cycle import dispatch
    from scripts.ib_local.relation_evidence import mechanism_evidence,evidence_key
    torch.set_num_threads(1)
    before=sha256_file(args.checkpoint)
    opts=continuation_parser().parse_args(['--checkpoint',str(args.checkpoint),
        '--output',str(args.checkpoint.parent),'--resume-output','--cycles','1',
        '--world-source','text','--autonomous-events','--optimizer-policy','budgeted',
        '--table-storage',str(args.work/'table'),'--causal-storage',str(args.work/'causal')])
    loaded=prepare_continuation(opts,resume_output=True)
    e,m=loaded.engine,loaded.engine.model
    captured={}
    class Captured(Exception):pass
    def capture(model,state,source,actions,**kw):
        captured.update(state=state,source=source.clone(),actions=actions.clone(),kw=kw)
        raise Captured()
    try:
        use.configure(e,'apply')
        schedule=[]
        for _ in range(64):
            if e.autonomous_ready:break
            row=dispatch(e)
            assert row['autonomous_event']['kind']!='external'
            schedule.append(row['autonomous_event'])
        else:raise RuntimeError('no normal release in bounded audit')
        m._causal_archive.prepare()
        torch.randn(m.encoder.out_features,32,device=e.device)
        original=nav.imagine
        nav.imagine=capture
        try:
            e._utter()
        except Captured:pass
        finally:nav.imagine=original
        def mode_state(st,mode):
            return replace(st,retention_use=(None if getattr(st,'retention_use',None) is None else
                replace(st.retention_use,mode=mode)),organization=(None if st.organization is None else
                mode_state(st.organization,mode)))
        s,source,actions,kw=(captured[k] for k in ('state','source','actions','kw'))
        result={}
        with torch.no_grad():
            for mode in ('observe','apply'):
                m.relation_retention_use_mode=mode
                state=mode_state(s,mode)
                forecast=imagine(m,state,source,actions,**kw)
                q=queries(e,state)
                value=expected(e,state,source,actions,forecast['path']['mother'],q)
                d=forecast['path']['mother']
                hs=mechanism_hypotheses(d,d['query'],m.causal_relations.directions,m.causal_relations.observation_directions)
                cases=[]
                for b in range(len(actions)):
                    for h in range(hs.shape[1]):
                        y=hs[b:b+1,h]+d['noise'].sqrt()*m.causal_relations.observation_directions[h,-1]
                        evidence=Evidence(source[b:b+1],actions[b:b+1],y,Port.MOTHER,False,encoded=True)
                        after=revise(state,evidence,model=m)
                        cases.append(dict(candidate=b,hypothesis=h,operation=(after.revision_plan or {}).get('operation'),
                            support=[evidence_key(f) for f in mechanism_evidence(m,after,Port.MOTHER)],
                            trace=after.retention_trace,after=asdict(after.retention_use)))
                result[mode]=dict(raw_expected=value.tolist(),
                    old_moments=[] if state.retention_use is None else list(state.retention_use.moments),cases=cases)
        mismatches=[dict(candidate=x['candidate'],hypothesis=x['hypothesis'],
            observe=x['support'],apply=y['support'],operations=[x['operation'],y['operation']])
            for x,y in zip(result['observe']['cases'],result['apply']['cases']) if x['support']!=y['support']]
        successor_mismatches=[dict(candidate=x['candidate'],hypothesis=x['hypothesis'])
            for x,y in zip(result['observe']['cases'],result['apply']['cases'])
            if {k:v for k,v in x['after'].items() if k!='mode'} !=
               {k:v for k,v in y['after'].items() if k!='mode'}]
        out=dict(cycle=loaded.cycle,source_sha256=before,source_unchanged=before==sha256_file(args.checkpoint),
            schema=e.relation_retention_use_schema,settlement_kind=e.settlement_kind,
            state=dict(identity=s.identity,generation=s.generation,binding=None if s.binding is None else s.binding.identity),
            schedule=schedule,world_calls=0,optimizer_updates=0,
            max_expected_difference=max(abs(x-y) for x,y in zip(result['observe']['raw_expected'],result['apply']['raw_expected'])),
            support_mismatches=mismatches,successor_mismatches=successor_mismatches,arms=result,
            source_hashes={str(p):sha256_file(p) for p in (Path(__file__),
                Path('scripts/ib_local/causal_relation_program.py'),
                Path('scripts/ib_local/generative_relations.py'),
                Path('scripts/ib_local/relation_retention_use.py'))})
        out['checks']=dict(neutral_old_use=all(not a['old_moments'] for a in result.values()),
            identical_supports=not mismatches,identical_successor_moments=not successor_mismatches,
            immediate_expected_parity=out['max_expected_difference']<3e-6,
            source_unchanged=out['source_unchanged'])
        args.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        print(json.dumps({k:v for k,v in out.items() if k!='arms'},ensure_ascii=False))
        if args.require_neutral_parity:
            assert all(out['checks'].values()),out['checks']
    finally:m._causal_archive.close()


if __name__=='__main__':main()
