"""Reconstruct the recorded actor and separate common value from action contrast.

The subtraction is at one recorded boundary. Habit, immediate use, precision,
available actions and original endpoints are held fixed. It measures the
continuation term's contribution to that choice, without a new world branch.
"""
import argparse
import json
import math
from pathlib import Path


def softmax(x):
    peak=max(x)
    p=[math.exp(value-peak) for value in x]
    return [value/sum(p) for value in p]


def measure(row):
    e=row['causal_use']['epistemic_value']
    habit,offset=row['habit_logits'],row['imagination_offset']
    total,future=e['contextual_total'],e['contextual_future']
    available=[i for i in range(len(habit)) if habit[i]>-1000]
    if len(available)!=2:
        raise ValueError('this exact record audit requires two available actions')
    i,j=available
    scale=(offset[j]-offset[i])/(total[j]-total[i])
    center=sum(total[k] for k in available)/len(available)
    reconstructed_offset=[scale*(total[k]-center) if k in available else 0. for k in range(len(habit))]
    offset_error=max(abs(a-b) for a,b in zip(reconstructed_offset,offset))
    logits=[h+o for h,o in zip(habit,offset)]
    original=softmax(logits)
    common=sum(future[k] for k in available)/len(available)
    stripped=[value-(scale*e['gamma']*(future[k]-common) if k in available else 0.)
              for k,value in enumerate(logits)]
    without=softmax(stripped)
    error=max(abs(a-b) for a,b in zip(original,row['choice_probability']))
    assert error<1e-6 and offset_error<3e-6,(error,offset_error)
    return dict(cycle=row['cycle'],selected=row['choice'],future=future,
        common_future=common,future_action_contrast=future[j]-future[i],
        immediate_action_contrast=e['local_expected'][j]-e['local_expected'][i],
        gamma=e['gamma'],effective_logit_scale=scale,
        actual_probability=original,without_future_probability=without,
        future_probability_TV=sum(abs(a-b) for a,b in zip(original,without))/2,
        probability_reconstruction_error=error,offset_reconstruction_error=offset_error)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--events',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    events=[json.loads(line) for line in args.events.read_text(encoding='utf8').splitlines()]
    rows=[measure(row) for row in events if 'causal_use' in row]
    out=dict(valid=True,source=str(args.events),rows=rows,
        mapping='NativeActorExpectation.actual_centered_logit_difference; actual recorded total and delivered offset',
        scope='same recorded choice; real continuation term removed; no new rollout',
        optimizer_updates=0,world_calls=0)
    args.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(valid=True,cycles=len(rows),TV=[r['future_probability_TV'] for r in rows])))


if __name__=='__main__':
    main()
