"""Reconstruct a stopped real prefix, save its last valid state, expose the guard."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.ib_local import gdn_fused
from scripts.ib_local.language_navigation_kanli_v4c import choice_replay_error
from scripts.ib_local.navigation_direct_read import DirectReadGainEngine, configure_direct_read
from scripts.ib_local.navigation_checkpoint import save
from scripts.ib_local.run_causal_water_reply import build
from scripts.ib_local.run_language_navigation_v2 import validate


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    cfg=json.loads((args.run/'config.json').read_text())
    prefix=[json.loads(s) for s in (args.run/'metrics.jsonl').read_text().splitlines()]
    torch.set_num_threads(1);gdn_fused.enable()
    e,raw,offset,_=build(Path(cfg['source']['checkpoint']))
    configure_direct_read(e,cfg['mode'],cfg['observer_mode'])
    v=np.asarray(raw[:2*(cfg['validation_offset']+cfg['validation_bytes']+1)])
    v=(v[v>=offset]-offset)[cfg['validation_offset']:cfg['validation_offset']+cfg['validation_bytes']+1].astype(np.uint8)
    del raw
    initial=validate(e.model,e.table,v)
    assert initial==json.loads((args.run/'initial.json').read_text())['validation_nll']
    diagnostic={};sampled={}
    old_value=DirectReadGainEngine._value
    def sample_input(self,h,films):
        sampled['h']=h.detach().clone()
        return old_value(self,h,films)
    DirectReadGainEngine._value=sample_input
    original=DirectReadGainEngine._policy_logp
    def inspected(self,h,logits,targets,own,record):
        result=original(self,h,logits,targets,own,record)
        replay=result[1]
        diagnostic.clear();diagnostic.update(choice=record['choice'],own=own,
            source_hidden_max_error=float((h[:1]-sampled['h']).abs().max()),
            utterance_targets=targets[0,:own].tolist(),
            choice_error=choice_replay_error(replay['choice'],record['choice_logits'],record['habit_logits']),
            replayed_choice=replay['choice'],sampled_choice=record['choice_logits'])
        if 'byte_logp' in replay:
            actual=replay['byte_logp'];old=actual.new_tensor(record['byte_logp'])
            diagnostic.update(byte_errors=(actual-old).abs().tolist(),
                              replayed_logp=actual.tolist(),sampled_logp=old.tolist())
        return result
    DirectReadGainEngine._policy_logp=inspected
    for i in range(len(prefix)+1):
        if i==len(prefix):
            save(e,args.output,cfg['source']['cycle']+i,cfg)
        try:row=e.step(torch.randn(64,32,device=e.device))
        except RuntimeError as error:
            diagnostic.update(failed_turn=i+1,error=str(error),prefix_exact=True)
            (args.output/'failure.json').write_text(json.dumps(diagnostic,indent=2)+'\n')
            print(json.dumps(diagnostic),flush=True);return
        if i<len(prefix):
            for k in ('choice','utterance_hex','wood_target','choice_probability','direct_gain_counts_next'):
                assert row[k]==prefix[i][k],(i+1,k)
            if (i+1)%cfg.get('validate_every',128)==0:validate(e.model,e.table,v)
            if (i+1)%32==0:print('PREFIX EXACT',i+1,flush=True)
    raise AssertionError('failed turn did not reproduce')


if __name__=='__main__':main()
