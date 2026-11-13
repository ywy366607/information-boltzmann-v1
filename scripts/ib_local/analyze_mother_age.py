"""Audit actual single-slot ages from the logged causal-read features."""
import argparse
import json
import math
from pathlib import Path
import statistics

from scripts.ib_local.analyze_mother_continuation import load


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metrics',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--horizon',type=int,default=512)
    args=p.parse_args()
    if args.output.exists() or args.horizon<=0:
        raise ValueError('new output and positive inherited horizon required')
    selected=[]
    for r in load(args.metrics):
        weights=r['causal_channel_read_weights']
        ones=[i for i,w in enumerate(weights) if len(w)==4 and all(abs(float(x)-1)<1e-6 for x in w)]
        if len(ones)!=1 or any(abs(float(x))>1e-6 for i,w in enumerate(weights) if i!=ones[0] for x in w):
            continue
        i=ones[0]
        # CausalWoodEngine._read_features: log1p(age/memory.long), weighted
        # equally over four channels. In this single-slot/full-channel case
        # only, this is the selected slot's most-recent-write age (rounded).
        age=round(args.horizon*math.expm1(r['causal_features'][12]))
        selected.append(dict(cycle=r['cycle'],slot=r['episodic_read_slots'][i],
            approx_age_event_bytes=age,real_error=r['episodic_used_real_error'][3],
            shuffled_error=r['episodic_used_shuffled_error'][3]))
    n=len(selected)
    report=dict(status='complete',count=n,horizon=args.horizon,
        age_over_horizon=sum(x['approx_age_event_bytes']>args.horizon for x in selected),
        max_approx_age=max((x['approx_age_event_bytes'] for x in selected),default=None),
        mean_real_error=statistics.fmean(x['real_error'] for x in selected) if n else None,
        mean_shuffled_error=statistics.fmean(x['shuffled_error'] for x in selected) if n else None,
        real_wins=sum(x['real_error']<x['shuffled_error'] for x in selected),records=selected,
        scope='selected unique single-slot reads with four channel weights all one; age since last write',
        caveat='age inferred from float32 log feature; not proof of long-term credit or world understanding')
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='records'},ensure_ascii=True,indent=2))


if __name__=='__main__':main()
