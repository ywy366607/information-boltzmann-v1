"""Offline index audit on already-recorded utterances; no model/world mutation.

Compare the registered first-hit index to a deterministic corpus-stratified
index. Counterfactual queries are replayed from logged bytes, so this is an
index-coverage diagnostic only, not a counterfactual learning experiment.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numba
import numpy as np
from scripts.ib_local.language_navigation_world import _build_index, _hash4, _select


@numba.njit(cache=True)
def stratified_index(data, buckets, positions):
    mask=buckets-1; slots=positions.shape[1]
    for i in range(len(data)-3):
        bucket=int(_hash4(data[i],data[i+1],data[i+2],data[i+3]) & np.uint64(mask))
        stratum=min(slots-1,(i*slots)//len(data))
        if positions[bucket,stratum]<0: positions[bucket,stratum]=i


def coverage(intervals):
    end=-1;total=0
    for lo,hi in sorted(intervals):
        if hi>end: total+=hi-max(lo,end);end=hi
    return total


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('preserve prior audit')
    cfg=json.loads((args.run/'config.json').read_text()); log=(args.run/'metrics.jsonl').read_bytes()
    rows=[json.loads(s) for s in log.splitlines() if s.strip()]
    if not rows:raise ValueError('no observed utterances')
    root=Path('data/ib_owt_10000');off=json.loads((root/'tokenizer.json').read_text())['offset']
    train=np.load(root/'train.npy',mmap_mode='r');lo,hi=cfg['source_raw_bounds'];raw=np.asarray(train[lo:hi])
    data=(raw[raw>=off]-off).astype(np.uint8)
    if hashlib.sha256(data.tobytes()).hexdigest()!=cfg['source_byte_sha256']:raise ValueError('source mismatch')
    indexes=[]
    for fn in (_build_index,stratified_index):
        index=np.full((1<<17,8),-1,np.int32);fn(data,len(index),index);indexes.append(index)
    reports={}
    for name,index in zip(('first_hit','stratified'),indexes):
        cursor=0;intervals=[];matches=0
        for r in rows:
            q=np.frombuffer(bytes.fromhex(r['utterance_hex']),np.uint8)
            n=cfg['response_bytes']+cfg['holdout_bytes']
            start,score=_select(q,data,index,n,cursor)
            if start<0:break
            cursor=start+n;intervals.append((start,cursor));matches+=score>0
        total=sum(hi-lo for lo,hi in intervals);unique=coverage(intervals)
        reports[name]=dict(events=len(intervals),matched=matches,unique_byte_positions=unique,
                           repeated_position_fraction=1-unique/total if total else None,
                           maximum_source_end=max((x[1] for x in intervals),default=0))
    result=dict(scope='same recorded utterances replayed against two indexes; model policy does not adapt',
        logged_cycles=len(rows),log_sha256=hashlib.sha256(log).hexdigest(),indexes=reports)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
