"""Reproduce rejected-batch graph ownership on one real full-body proposal.

Only the batch diagnostic is perturbed, forcing the inherited exact one-byte
path. No optimizer/world calls, no source-body writes, no capability verdict.
"""
import argparse
import json
from pathlib import Path
import time

import torch
from scripts.ib_local.run_mother_continuation import continuation_parser,prepare_continuation
from scripts.ib_local.variable_expression import proposal_score,release_fallback_cache


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--body',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    args=continuation_parser().parse_args(['--checkpoint',str(a.body/'last.pt.gz'),
        '--output',str(a.body),'--cycles','1','--resume-output','--world-source','text',
        '--table-storage',str(a.output/'table'),'--causal-storage',str(a.output/'causal')])
    loaded=prepare_continuation(args,resume_output=True);e=loaded.engine
    def forbidden(*args,**kwargs):raise AssertionError('audit cannot advance world or optimizer')
    e.world.take=forbidden;e.optimizer.step=forbidden
    e._utter()
    proposal=max(e._joint_proposals,key=lambda x:len(x['words']))
    if not proposal['words']:raise ValueError('a real nonempty draft is required')
    original=e.model.predict_observed
    retained=[];mode={'retain':False}
    def rejected(ids,state,lp,**kw):
        out=original(ids,state,lp,**kw)
        if ids.shape[1]>1:
            logits=out[0].clone();logits[0,0,proposal['words'][0]]-=100.
            out=(logits,*out[1:])
            if mode['retain']:retained.append(out)
        return out
    e.model.predict_observed=rejected
    results={};reference=None
    try:
        for name,keep in (('retained_reference',True),('released',False)):
            mode['retain']=keep;retained.clear()
            torch.cuda.empty_cache();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            previous=getattr(e,'_batched_replay_fallbacks',0)
            start=time.perf_counter()
            terms,error=proposal_score(e,proposal)
            grads=torch.autograd.grad(terms.sum(),e.params,allow_unused=True)
            torch.cuda.synchronize()
            part=dict(seconds=time.perf_counter()-start,error=error,
                peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                fallback_count=getattr(e,'_batched_replay_fallbacks',0)-previous,
                terms=terms.detach().cpu().tolist())
            current=[None if g is None else g.detach().cpu() for g in grads]
            if reference is None:reference=current
            else:
                errors=[float((g-r).abs().max()) for g,r in zip(current,reference) if g is not None and r is not None]
                part['gradient_presence_equal']=all((g is None)==(r is None) for g,r in zip(current,reference))
                part['gradient_max_error']=max(errors,default=0.)
            del grads,terms,current
            retained.clear()
            part['allocator_release']=release_fallback_cache(e,previous)
            results[name]=part
            print(name,json.dumps({k:v for k,v in part.items() if k!='terms'}),flush=True)
    finally:del e.model.predict_observed
    result=dict(source_cycle=loaded.cycle,source_sha=loaded.source_sha,
        neural_parameters=sum(p.numel() for p in e.model.parameters()),proposal_bytes=len(proposal['words']),
        scope=__doc__,optimizer_calls=0,world_calls=0,source_written=False,arms=results)
    (a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    e.model._causal_archive.db.close()


if __name__=='__main__':main()
