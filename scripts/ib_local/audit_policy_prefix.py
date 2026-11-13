"""Compare one-byte action reads with the same observed batched prefix."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.ib_local import gdn_fused
from scripts.ib_local.run_causal_water_reply import build


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(1);gdn_fused.enable()
    e,_,_,_=build(args.checkpoint)
    start=e.recurrent.detach().clone();pending=e.pending
    cls=type(e.model);original=cls.predict_observed;calls=[];films=[]
    def observed(self,*a,**kw):
        out=original(self,*a,**kw)
        calls.append(dict(h=out[3].detach().clone(),slots=out[4].slots.detach().clone(),
                          weights=out[4].weights.detach().clone(),logits=out[0].detach().clone()))
        return out
    cls.predict_observed=observed
    engine_cls=type(e);old_film=engine_cls._film
    def film(self,*a,**kw):
        first=len(calls);out=old_film(self,*a,**kw)
        films.append(calls[first:]);return out
    engine_cls._film=film
    with torch.no_grad():
        # The normal loop draws SIGReg directions before drafting speech.
        torch.randn(64,32,device=e.device)
        utterance,record=e._utter()
        count=e.response_bytes+len(utterance)
        ids=torch.tensor([pending,*utterance[:-1]]+[0]*(count-len(utterance)),device=e.device)[None]
        probs=np.asarray(record['probs'])
        logp=torch.tensor(np.concatenate((np.log(np.maximum(probs,1e-6)),
            np.full((count-len(utterance),256),np.log(1/256),np.float32))),device=e.device)[None]
        e.model.predict_observed(ids,start,logp)
    batched=calls[-1];sampled=films[1][:len(utterance)]
    target=torch.tensor(list(utterance),device=e.device)
    actual=batched['logits'][0].log_softmax(-1).gather(1,target[:,None]).flatten()
    rows=[]
    for i,s in enumerate(sampled):
        rows.append(dict(byte=i,hidden_error=float((s['h'][0]-batched['h'][i]).abs().max()),
            same_read_slots=torch.equal(s['slots'][0],batched['slots'][i]),
            same_read_set=set(s['slots'][0].tolist())==set(batched['slots'][i].tolist()),
            sampled_slots=s['slots'][0].tolist(),batch_slots=batched['slots'][i].tolist(),
            weights_error=float((s['weights'][0]-batched['weights'][i]).abs().max()),
            logp_error=float(abs(actual[i]-record['byte_logp'][i]))))
    result=dict(choice=record['choice'],utterance_hex=utterance.hex(),rows=rows,
                batch_shape=list(ids.shape))
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
