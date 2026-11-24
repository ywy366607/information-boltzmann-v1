"""Paired output-term accounting in the unchanged complete language graph.

Reuse the ordinary validation forward. Report full predictive gain over its
actual byte-table prior, and the decoder residual's gain with learned table
mixing retained. A residual decoder is not treated as a standalone LM.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from scripts.ib_local.run_language_navigation_v2 import validate


@torch.no_grad()
def language_contribution(model, table, valid):
    position = 0
    raw_decoder = []
    totals = []
    original = model.predict_observed
    had_local = 'predict_observed' in model.__dict__
    handle = model.base.decoder.register_forward_hook(lambda m,a,out:raw_decoder.append(out))
    def observed(ids,state,table_logp,*args,**kwargs):
        nonlocal position
        out = original(ids,state,table_logp,*args,**kwargs)
        if len(raw_decoder) != 1: raise RuntimeError('expected one real decoder per validation chunk')
        neural = raw_decoder.pop()
        n = ids.shape[1]
        target = torch.as_tensor(np.asarray(valid[position+1:position+n+1],dtype=np.int64),
                                 device=ids.device).reshape(-1)
        if len(target) != n: raise RuntimeError('validation target alignment changed')
        terms = [out[0], table_logp, out[0]-neural]
        totals.append(torch.stack([F.cross_entropy(x.flatten(0,1),target,reduction='sum') for x in terms]))
        position += n
        return out
    model.predict_observed = observed
    try:
        full = validate(model,table,valid)
    finally:
        if had_local: model.predict_observed = original
        else: del model.predict_observed
        handle.remove()
    if position != len(valid)-1: raise RuntimeError('not every validation byte was scored')
    joint,prior,without_decoder = (torch.stack(totals).double().sum(0)/position).cpu().tolist()
    if abs(full-joint)>1e-6: raise RuntimeError('accounting differs from original complete prediction')
    return dict(bytes=position,full_nll=full,table_prior_nll=prior,
        learned_table_mix_without_decoder_nll=without_decoder,
        full_gain_over_table=prior-full, decoder_residual_gain=without_decoder-full,
        scope='same observed history and complete state/reads; normalized table prior; no extra model forward')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--body',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    a = parser.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    from scripts.ib_local.run_mother_continuation import continuation_parser,prepare_continuation
    from scripts.ib_local.audit_optimizer_complete_loop import validation_data
    opts = ['--checkpoint',str(a.body/'last.pt.gz'),'--output',str(a.body),
        '--cycles','1','--resume-output','--world-source','text',
        '--table-storage',str(a.output/'table'),'--causal-storage',str(a.output/'causal')]
    loaded = prepare_continuation(continuation_parser().parse_args(opts),resume_output=True)
    e = loaded.engine
    scores = {name:language_contribution(e.model,e.table,data)
              for name,data in validation_data(loaded.config).items()}
    result = dict(cycle=loaded.cycle,source_sha=loaded.source_sha,scores=scores,
                  main_written=False,optimizer_updates=0,world_calls=0)
    (a.output/'language_contribution.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    e.model._causal_archive.db.close()


if __name__=='__main__': main()
