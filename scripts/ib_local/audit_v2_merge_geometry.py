"""Read-only paired similarity distributions on next real OWT observation."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from scripts.ib_local import gdn_fused
from scripts.ib_local.gated_deltanet import GatedDeltaNetLM
from scripts.ib_local.language_action import digest
from scripts.ib_local.latent_navigation_v2 import LatentNavigationV2
from scripts.ib_local.retrieved_table_credit import RetrievedCreditTable
from scripts.ib_local.train_gdn_fast_table import FLOOR, TableGDN


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    if args.output.exists(): raise FileExistsError('preserve prior audit')
    torch.set_num_threads(2); gdn_fused.enable()
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    cfg=saved['config']
    base=GatedDeltaNetLM(256,128,4,4,'decay_then_delta',256,4,32.,decay_init='log_timescale').cuda()
    model=LatentNavigationV2(TableGDN(base,True).cuda(),threshold=cfg['threshold'],mode=cfg['memory']).cuda()
    model.load_state_dict(saved['model']); model.eval()
    table=RetrievedCreditTable.from_state_dict(saved['table'])
    root=Path('data/ib_owt_10000'); off=json.loads((root/'tokenizer.json').read_text())['offset']
    full=np.load(root/'train.npy',mmap_mode='r')
    lo,hi=cfg['source_raw_bounds']; raw=np.asarray(full[lo:hi]); source=(raw[raw>=off]-off).astype(np.uint8)
    pos=int(saved['world_cursor']); response=source[pos:pos+512]
    if len(response)!=512: raise EOFError('insufficient real continuation')
    raw=np.concatenate((np.array([saved['pending']],np.uint8),response))
    _,probs,_=table.run_credit(raw,max_window=512)
    ids=torch.tensor(raw[:-1].astype(np.int64),device='cuda')[None]
    logp=torch.tensor(np.log(np.maximum(probs,FLOOR)),device='cuda')[None]
    _,_,_,features,_=model.predict_observed(ids,saved['recurrent'].cuda(),logp)
    src,short,long=model.memory.pairs(features)
    encoder=model.encoder; memory=model.memory
    q=F.normalize(encoder(src),dim=-1); k=F.normalize(encoder(memory.source),dim=-1)
    source_sim,idx=(q@k.T).masked_fill(~memory.occupied,-2).max(-1)
    short_sim=F.cosine_similarity(encoder(short),encoder(memory.successor[idx]),dim=-1)
    long_sim=F.cosine_similarity(encoder(long),encoder(memory.long_successor[idx]),dim=-1)
    joint=torch.minimum(source_sim,torch.minimum(short_sim,long_sim))
    sk=F.normalize(encoder(short),dim=-1) @ F.normalize(encoder(memory.successor),dim=-1).T
    lk=F.normalize(encoder(long),dim=-1) @ F.normalize(encoder(memory.long_successor),dim=-1).T
    all_joint=torch.minimum(q@k.T,torch.minimum(sk,lk)).masked_fill(~memory.occupied,-2).max(-1).values
    def stats(x):
        return {str(q):float(torch.quantile(x,q)) for q in (0.,.1,.25,.5,.75,.9,1.)}
    result=dict(scope='same-body read-only next-window paired similarity; not merge-quality test',
        checkpoint_sha256=digest(args.checkpoint), response_sha256=hashlib.sha256(response.tobytes()).hexdigest(),
        pairs=len(src),occupied=int(memory.occupied.sum()), source=stats(source_sim),
        short_successor=stats(short_sim),long_successor=stats(long_sim),joint_min=stats(joint),
        best_joint=stats(all_joint),
        share_at_threshold={str(t):float((joint>=t).float().mean()) for t in (.98,.95,.9,.85,.8,.7,.5,0.)},
        best_joint_share_at_threshold={str(t):float((all_joint>=t).float().mean()) for t in (.98,.95,.9,.85,.8,.7,.5,0.)})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
