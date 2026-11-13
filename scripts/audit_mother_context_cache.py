"""Compare real-provider cache usage on actual recorded child utterances.

Research-side caregiver conversations only; no body training or changes to
the saved individual. Print accounting, never credentials or request headers.
"""
import argparse
import copy
import gzip
import json
from pathlib import Path
import time

import torch

from scripts.ib_local.mother_context_harness import ContextCachedMother
from scripts.ib_local.mother_reply_world import StrictToddlerMother


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--metrics',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--turns',type=int,default=6)
    p.add_argument('--kinds',nargs='+',choices=('rolling','segmented'),default=['rolling','segmented'])
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    with gzip.open(args.checkpoint,'rb') as f:s=torch.load(f,map_location='cpu',weights_only=False)
    seed=s['world']['mother_transcript'];policy=s['config']['mother_policy'];cycle=s['cycle']
    del s
    rows=[json.loads(x) for x in args.metrics.read_text(encoding='utf-8').splitlines()]
    children=[r['world']['child_text'] or '<quiet_turn/>' for r in rows[:args.turns]]
    old=StrictToddlerMother(policy=policy)
    new=ContextCachedMother(policy=policy)
    histories={name:copy.deepcopy(seed) for name in ('rolling','segmented')}
    results=[]
    for i,child in enumerate(children):
        for name,m in [('rolling',old),('segmented',new)]:
            if name not in args.kinds:continue
            turn=histories[name][-12:]+[('child',child)]
            start=time.perf_counter()
            reply=m.reply(turn)
            if name=='segmented':
                m.commit_turn();u=copy.deepcopy(m.last_telemetry['api_usage'])
                detail=copy.deepcopy(m.last_telemetry)
            else:
                total=dict(getattr(m.delegate,'usage',{}))
                prior=next((a['cumulative'] for a in reversed(results) if a['kind']==name),{})
                detail={}
                u={dst:total.get(src,0)-prior.get(src,0) for src,dst in
                   [('cache_hit','cache_hit_tokens'),('cache_miss','cache_miss_tokens'),('output','completion_tokens')]}
                u['prompt_tokens']=u['cache_hit_tokens']+u['cache_miss_tokens']
            histories[name]=(turn+[('mother',reply)])[-12:]
            row=dict(kind=name,turn=i+1,seconds=time.perf_counter()-start,usage=u,
                     reply=reply,child=child,detail=detail)
            if name=='rolling':row['cumulative']=total
            results.append(row)
            (args.output/'requests.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
            if name=='segmented':
                (args.output/'provider_state.json').write_text(json.dumps(
                    dict(provider=new.state_dict(),transcript=histories[name]),
                    ensure_ascii=False,indent=2),encoding='utf-8')
            print(json.dumps({k:row[k] for k in ('kind','turn','seconds','usage')},ensure_ascii=True),flush=True)
    report=dict(source_cycle=cycle,actual_child_turns=len(children),summary={})
    for name in ('rolling','segmented'):
        rr=[a for a in results if a['kind']==name]
        if not rr:continue
        hit=sum(a['usage']['cache_hit_tokens'] for a in rr)
        miss=sum(a['usage']['cache_miss_tokens'] for a in rr)
        output=sum(a['usage']['completion_tokens'] for a in rr)
        report['summary'][name]=dict(hit=hit,miss=miss,output=output,
            hit_rate=hit/(hit+miss) if hit+miss else None,
            mean_seconds=sum(a['seconds'] for a in rr)/len(rr))
    (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=True),flush=True)


if __name__=='__main__':main()
