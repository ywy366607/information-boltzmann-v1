"""Fixed current state; inspect posterior credit under an old-content intervention.

The task value identifies the intervention ONLY in this researcher's read-only
audit. Neither targets nor record labels enter the body's retrieval rule.
"""
import argparse
import json
from pathlib import Path
import time
from types import MethodType

import torch

from scripts.ib_local.audit_small_native_content import measure
from scripts.ib_local.run_causal_water_reply import build
from scripts.ib_local.small_gdn_ruler import fingerprint, sha256


@torch.no_grad()
def posterior_measure(engine,row,alternative):
    model=engine.model
    original=model.predict
    captured=[]
    actual=ord(row['outputs'][0][0]);other=ord(alternative[0])

    def predict(self,*args,**kwargs):
        result=original(*args,**kwargs)
        forecast=self.local_replay.forecast
        captured.append(dict(real=forecast.report(torch.tensor([actual],device=engine.device)),
            alternative=forecast.report(torch.tensor([other],device=engine.device))))
        return result

    # measure() uses predict_observed for prompt ingestion; these are exactly
    # its final factual and replaced-memory first-digit predictions.
    model.predict=MethodType(predict,model)
    try:
        content=measure(engine,row,alternative)
    finally:
        del model.predict
    assert len(captured)==2
    records={(r['layer']-1,r['slot']) for r in content['replaced_records']}
    summaries=[]
    for prediction in captured:
        factual=prediction['real'];substitute=prediction['alternative']
        ids=factual['identities']
        layers=[]
        for layer in range(len(model.base.layers)):
            indices=[i for i,identity in enumerate(ids)
                     if identity is not None and identity[0]==layer]
            needle=[i for i in indices if (ids[i][0],ids[i][1]) in records]
            def total(field,which):return sum(field[i] for i in which)
            prior=total(factual['prior_mass'],indices)
            posterior=total(factual['posterior_mass'],indices)
            prior_needle=total(factual['prior_mass'],needle)
            posterior_needle=total(factual['posterior_mass'],needle)
            other_needle=total(substitute['posterior_mass'],needle)
            layers.append(dict(layer=layer+1,prior_mass=prior,posterior_mass=posterior,
                needle_prior_mass=prior_needle,needle_posterior_mass=posterior_needle,
                needle_responsibility_shift=posterior_needle-prior_needle,
                needle_real_minus_alternative_responsibility=posterior_needle-other_needle))
        summaries.append(dict(layers=layers,report=prediction))
    return dict(**content,factual_responsibility=summaries[0],
                changed_memory_responsibility=summaries[1])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','data','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(1)
    if args.device=='cuda':
        from scripts.ib_local import gdn_fused
        gdn_fused.enable()
    engine,_,_,_=build(args.checkpoint,device=args.device)
    assert engine.model.local_replay.schema in (3,4)
    before=fingerprint(engine.model,engine.table);checksum=sha256(args.checkpoint)
    rows=[json.loads(line) for line in args.data.read_text().splitlines()][:8]
    records=[];start=time.time()
    for row in rows:
        alternative=next(r['outputs'][0] for r in rows if r['outputs'][0][0]!=row['outputs'][0][0])
        records.append(posterior_measure(engine,row,alternative))
        print('RESPONSIBILITY',row['index'],records[-1]['factual_content_nll_benefit'],flush=True)
    result=dict(status='complete',optimizer_updates=0,records=records,
        source_sha256=checksum,seconds=time.time()-start,
        body_unchanged=fingerprint(engine.model,engine.table)==before,
        checkpoint_unchanged=sha256(args.checkpoint)==checksum,
        favorable_nll_cases=sum(r['factual_content_nll_benefit']>0 for r in records),
        favorable_contrast_cases=sum(r['stored_value_contrast']>0 for r in records),
        mean_content_nll_benefit=sum(r['factual_content_nll_benefit'] for r in records)/len(records),
        mean_stored_value_contrast=sum(r['stored_value_contrast'] for r in records)/len(records))
    assert result['body_unchanged'] and result['checkpoint_unchanged']
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')


if __name__=='__main__':main()
