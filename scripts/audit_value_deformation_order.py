"""Fixed-menu deformation: pair order, exact log-odds crossings, C curve.

The candidate and range [0,1] are fixed. Actual outcomes enter only this
evaluation; no strength is selected or fed back to the ongoing learner.
"""
import argparse
import json
from pathlib import Path

import torch


def analyze(capture, outcomes):
    shadow = json.loads((capture/'root_shadow.json').read_text(encoding='utf8'))
    p = torch.tensor(shadow['original_probability'],dtype=torch.double)
    habit = torch.tensor(shadow['habit_probability'],dtype=torch.double)
    legal = shadow['legal']
    g = torch.tensor([outcomes[str(i)]['G2'] if i in legal else 0. for i in range(len(p))],dtype=p.dtype)
    d = float(shadow['precision'])*(torch.tensor(shadow['critic_alternative'],dtype=p.dtype)-
                                   torch.tensor(shadow['critic_original'],dtype=p.dtype))
    pairs, crossings = [], []
    for n,i in enumerate(legal):
        for j in legal[n+1:]:
            pair = float((g[i]-g[j])*(d[i]-d[j]))
            pairs.append(dict(actions=[i,j],outcome_difference=float(g[i]-g[j]),
                logit_slope_difference=float(d[i]-d[j]),order_product=pair))
            slope = float(d[i]-d[j])
            if slope:
                alpha = -float((p[i]/p[j]).log())/slope
                if 0 <= alpha <= 1:
                    crossings.append(dict(actions=[i,j],strength=alpha))
    curve = []
    for alpha in [0.,.25,.5,.75,1.]:
        logits = p.clamp_min(1e-300).log()+alpha*d
        mask = torch.tensor([i in legal for i in range(len(p))])
        current = logits.masked_fill(~mask,-torch.inf).softmax(0)
        mean = (current*g).sum()
        drift = (current*(g-mean)*d).sum()
        pair_drift = sum(float(current[i]*current[j])*pair['order_product']
                         for pair in pairs for i,j in [pair['actions']])
        assert abs(float(drift)-pair_drift) < 1e-12
        curve.append(dict(strength=alpha,probability=current.tolist(),
            C=float((current-habit)@g),derivative=float(drift)))
    assert max(abs(a-b) for a,b in zip(curve[-1]['probability'],shadow['alternative_probability'])) < 1e-12
    return dict(unknown=shadow['unknown'],legal=legal,logit_direction=d.tolist(),
        pairs=pairs,comonotone=all(pair['order_product']>=0 for pair in pairs),
        strictly_positive_pairs=sum(pair['order_product']>0 for pair in pairs),
        monotonicity='pairwise covariance sign applies at every strength in [0,1]',
        actor_order_crossings=crossings,curve=curve)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    sources = [
        (Path('E:/mengli-persistent-spectrum-20261122'),Path('F:/mengli-option-calibration-use-20261122/outcomes.json')),
        (Path('E:/mengli-persistent-spectrum-boundary04-inherited-20261122'),
         Path('scratch/cgb_wood_policy_use_16rounds_20261120/boundary04_outcomes.json'))]
    cases = [analyze(capture,json.loads(outcomes.read_text(encoding='utf8'))) for capture,outcomes in sources]
    result = dict(schema=1,family='fixed original menu: logits(alpha)=original_logits+alpha*precision*critic_increment',
        interval=[0,1],cases=cases,training_change='none; evaluation only',
        existing_derivative='GenerationRoutingCredit.softmax_read_direction_deriv',
        new_binding='NativeValueDeformationOrder covariance pairs and exact crossing',
        world_calls_added=0,optimizer_updates_added=0)
    a.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    print(json.dumps(cases,ensure_ascii=False))


if __name__=='__main__':
    main()
