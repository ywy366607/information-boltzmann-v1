"""Use actual randomized choices to estimate one-off selection usefulness.

The intervention replaces one draw, preserving the body's beliefs, update
rules and subsequent closed loop. It is not training a new uniform actor.
Rewards include real learning, self-hearing, the world and delayed updates.
One algebraic estimator covers fire selection, draft selection and their
factorial interaction. No extra model/world calls or training signal.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics


CHOICES = dict(silence=0,generate=1,reuse=2)


def propensity(vector, selected, target=None):
    if not isinstance(vector,list) or not vector or any(not math.isfinite(p) or p<0 for p in vector):
        raise ValueError('invalid logged categorical distribution')
    total = sum(vector)
    if abs(total-1)>1e-4 or not 0 <= selected < len(vector) or vector[selected] <= 0:
        raise ValueError('selected action lacks its actual positive propensity')
    p = [x/total for x in vector]
    count = sum(x>0 for x in p)
    q = [1/count if x>0 else 0. for x in p] if target is None else target
    if (len(q)!=len(p) or any(not math.isfinite(x) or x<0 for x in q) or
            abs(sum(q)-1)>1e-4 or any(x==0 and y>0 for x,y in zip(p,q))):
        raise ValueError('target policy lacks logged support')
    q = [x/sum(q) for x in q]
    weight = q[selected]/p[selected]
    tv = .5*sum(abs(x-y) for x,y in zip(p,q))
    return weight,tv,count


def softmax(values):
    top=max(values); weights=[math.exp(x-top) for x in values]; total=sum(weights)
    return [x/total for x in weights]


def effects(rows, *, horizon=1, discount=.8, block=128):
    records, skipped = [], Counter()
    past_sum, past_count = 0.,0
    for index,row in enumerate(rows):
        baseline = past_sum/past_count if past_count else 0.
        reward = row.get('metal_g')
        if isinstance(reward,(int,float)) and math.isfinite(reward):
            past_sum += reward; past_count += 1
        else:
            skipped['nonfinite_reward'] += 1; continue
        if index+horizon>len(rows):
            skipped['future_not_observed'] += 1; continue
        future = rows[index:index+horizon]
        if any(r.get('phase') != row.get('phase') for r in future):
            skipped['external_phase_boundary'] += 1; continue
        try:
            if row.get('mode') == 'silence': raise ValueError('forced action')
            wa,tva,na = propensity(row['choice_probability'],CHOICES[row['choice']])
            draft = row['joint_language_search']
            wj,tvj,nj = propensity(draft['distribution'],draft['selected'])
            if len(draft['candidates']) != len(draft['distribution']): raise ValueError('draft mismatch')
            observed = sum(discount**i*r['metal_g'] for i,r in enumerate(future))
            if not math.isfinite(observed): raise ValueError('nonfinite return')
        except (KeyError,TypeError,ValueError):
            skipped['missing_or_invalid_randomization'] += 1; continue
        centered = observed-baseline*sum(discount**i for i in range(horizon))
        records.append(dict(index=index,return_value=observed,baseline=baseline,
            fire= centered*(1-wa), draft=centered*(1-wj),
            both=centered*(1-wa*wj), interaction=centered*(1-wa)*(1-wj),
            fire_weight=wa,draft_weight=wj,both_weight=wa*wj,
            fire_tv=tva,draft_tv=tvj,fire_support=na,draft_support=nj))
        # The actual full logits are habit + the recorded imagination offset.
        # This target removes only that decision offset, retaining the whole
        # body, learned habit, generated drafts and later learning machinery.
        habit = row.get('habit_logits')
        if isinstance(habit,list) and len(habit)==len(row['choice_probability']):
            try:
                wh,tvh,_=propensity(row['choice_probability'],CHOICES[row['choice']],softmax(habit))
                if 'imagination_offset' in row:
                    reconstructed=softmax([x+y for x,y in zip(habit,row['imagination_offset'])])
                    actual=row['choice_probability']; total=sum(actual)
                    if max(abs(x-y/total) for x,y in zip(reconstructed,actual))>1e-5:
                        raise ValueError('recorded imagination is not the actual full offset')
                records[-1].update(imagination=centered*(1-wh),
                    draft_imagination_interaction=centered*(1-wj)*(1-wh),
                    imagination_weight=wh,imagination_tv=tvh)
            except ValueError:
                skipped['imagination_target_not_identified'] += 1
    if not records: return dict(events=0,skipped=dict(skipped))
    result = dict(events=len(records),horizon=horizon,discount=discount,skipped=dict(skipped),
        mean_actual_return=statistics.fmean(r['return_value'] for r in records),
        estimator='baseline-centered ordinary importance contrast; no clipping or self-normalization',
        scope='one-off draw substitution with unchanged beliefs/learning machinery; actual-state distribution')
    for key in ('fire','draft','both','interaction'):
        values = [r[key] for r in records]
        means = [statistics.fmean(values[i:i+block]) for i in range(0,len(values)-block+1,block)]
        part = dict(mean_lift=statistics.fmean(values),full_blocks=len(means),
            positive_blocks=sum(v>0 for v in means),block_means=means,
            uncertainty='block stability only; no iid significance claim')
        if key != 'interaction':
            weights = [r[key+'_weight'] for r in records]
            part.update(uniform_target_ess=sum(weights)**2/sum(w*w for w in weights),
                        maximum_weight=max(weights))
        if key in ('fire','draft'):
            part.update(expected_maximally_coupled_choice_changes=sum(r[key+'_tv'] for r in records),
                        admissible_counts=dict(Counter(r[key+'_support'] for r in records)))
        result[key]=part
    comparable=[r for r in records if 'imagination' in r]
    for key in ('imagination','draft_imagination_interaction'):
        if not comparable: continue
        values=[r[key] for r in comparable]
        means=[statistics.fmean(values[i:i+block]) for i in range(0,len(values)-block+1,block)]
        part=dict(events=len(comparable),mean_lift=statistics.fmean(values),
                  full_blocks=len(means),positive_blocks=sum(x>0 for x in means),block_means=means,
                  target='same learned habit, current imagination offset removed')
        if key=='imagination':
            weights=[r['imagination_weight'] for r in comparable]
            part.update(target_ess=sum(weights)**2/sum(w*w for w in weights),maximum_weight=max(weights),
                expected_maximally_coupled_choice_changes=sum(r['imagination_tv'] for r in comparable))
        result[key]=part
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metrics',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--discount',type=float,default=.8)
    p.add_argument('--block',type=int,default=128)
    a = p.parse_args()
    rows=[]
    with a.metrics.open(encoding='utf-8') as stream:
        for line in stream:
            try: rows.append(json.loads(line))
            except json.JSONDecodeError: break
    if not 0<=a.discount<=1 or a.block<1: raise ValueError('invalid horizon or block')
    result=dict(schema=1,source=str(a.metrics),logged_events=len(rows),
        one_loop=effects(rows,horizon=1,discount=a.discount,block=a.block),
        two_loops=effects(rows,horizon=2,discount=a.discount,block=a.block),
        optimizer_updates=0,world_calls=0)
    a.report.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:{'events':result[k]['events'],
        'fire_lift':result[k].get('fire',{}).get('mean_lift'),
        'draft_lift':result[k].get('draft',{}).get('mean_lift')}for k in ('one_loop','two_loops')}))


if __name__=='__main__':main()
