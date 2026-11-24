"""Held-out raw-coordinate reconstruction at an actual persistent boundary.

The actual latent target is supplied only in this retrospective interface
check, to isolate the extra raw coordinates needed by native scope tests.
Production forecasts always obtain that latent from the original world model.
"""
import argparse
import json
from pathlib import Path
from dataclasses import replace
from statistics import mean


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    import torch
    from scripts.ib_local.run_mother_continuation import continuation_parser,prepare_continuation
    from scripts.ib_local.run_causal_water_reply import sha256_file
    from scripts.ib_local.relation_evidence import mechanism_evidence,evidence_key
    from scripts.ib_local.causal_relation_program import Port
    from scripts.ib_local.prospective_decision import boundary,lift_boundary
    torch.set_num_threads(1);digest=sha256_file(a.checkpoint)
    opts=continuation_parser().parse_args(['--checkpoint',str(a.checkpoint),
        '--output',str(a.checkpoint.parent),'--resume-output','--cycles','1','--world-source','text',
        '--autonomous-events','--optimizer-policy','budgeted','--table-storage',str(a.work/'table'),
        '--causal-storage',str(a.work/'causal')])
    loaded=prepare_continuation(opts,resume_output=True);e=loaded.engine;model=e.model
    try:
        state=e.internal_pending['causal_program']['state']
        rows=[]
        with torch.no_grad():
            facts=[f for f in mechanism_evidence(model,state,Port.DECISION)
                   if f.observed and not f.encoded and f.mother is not None]
            for f in facts:
                identity=evidence_key(f)
                kept=tuple(x for x in state.constraints if evidence_key(x)!=identity)
                st=replace(state,constraints=kept,extra=tuple(x for x in state.extra
                    if evidence_key(x)!=identity),evidence_bound=True)
                z=model.encoder(f.outcome)
                base=lift_boundary(model,f.source,z)
                predicted,diag=boundary(e,st,f.source,f.action,model.encoder(f.mother),z)
                native=model.causal_relations.binding_address(model,f.outcome)
                old=model.causal_relations.binding_address(model,base)
                new=model.causal_relations.binding_address(model,predicted)
                rows.append(dict(fact=identity,raw_fit_support=diag['raw_support'],
                    anchored_MSE=float((base-f.outcome).square().mean()),
                    posterior_MSE=float((predicted-f.outcome).square().mean()),
                    anchored_address_match=old[:2]==native[:2],posterior_address_match=new[:2]==native[:2],
                    latent_error=diag['latent_residual']))
        out=dict(valid=bool(rows) and max(x['latent_error'] for x in rows)<3e-5,
            source_cycle=loaded.cycle,source_sha256=digest,source_unchanged=digest==sha256_file(a.checkpoint),
            scope='held-out actual raw successor coordinates, conditional on actual latent target',rows=rows,
            anchored_MSE=mean(x['anchored_MSE'] for x in rows),
            posterior_MSE=mean(x['posterior_MSE'] for x in rows),
            optimizer_updates=0,world_calls=0)
        assert out['valid'] and out['source_unchanged']
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        print(json.dumps(dict(valid=True,queries=len(rows),anchored_MSE=out['anchored_MSE'],posterior_MSE=out['posterior_MSE'])))
    finally:model._causal_archive.close()


if __name__=='__main__':main()
