"""Four-cell MOTHER support contrasts in the two saved 2587 settlement charts."""
from dataclasses import replace
from itertools import combinations
import json
from pathlib import Path
from statistics import mean, median

import torch
from scripts.ib_local.causal_relation_program import Port, mechanism, log_score


@torch.no_grad()
def measure(path):
    c = torch.load(path, map_location='cpu', weights_only=False)
    model, state, target = c['model'], c['before'], c['incoming']
    facts = tuple(f for f in state.constraints if f.kind == Port.MOTHER)
    assert len({f.fact_id for f in facts}) == len(facts) == 8
    assert all(f.stamp < target.stamp and f.fact_id != target.fact_id for f in facts)
    assert all(f.observed and not f.encoded and f.kind == Port.MOTHER for f in (*facts, target))
    def score(*remove):
        keep = tuple(f for i, f in enumerate(facts) if i not in remove)
        s = replace(state, constraints=keep, extra=(), organization=None, variable_uses=(), evidence_bound=True)
        d = mechanism(model, s, target.source, target.action, ridge=c['relation_ridge'])
        return float(log_score(d, model.encoder(target.outcome)).mean())
    full = score()
    single = [score(i) for i in range(len(facts))]
    pairs = []
    for i,j in combinations(range(len(facts)), 2):
        neither = score(i,j)
        i_only,j_only = single[j],single[i]
        flip_i = (full-j_only)*(i_only-neither) < 0
        flip_j = (full-i_only)*(j_only-neither) < 0
        pairs.append(dict(i=facts[i].fact_id, j=facts[j].fact_id, both=full, neither=neither,
                          i_only=i_only, j_only=j_only, interaction=full-i_only-j_only+neither,
                          flip_i=flip_i, flip_j=flip_j))
    xs = [p['interaction'] for p in pairs]
    return dict(target=target.fact_id, capsule=str(path), pairs=pairs, positive=sum(x>0 for x in xs),
                negative=sum(x<0 for x in xs), flip_pairs=sum(p['flip_i'] or p['flip_j'] for p in pairs),
                summary=dict(min=min(xs), median=median(xs), mean=mean(xs), max=max(xs)))


if __name__ == '__main__':
    torch.set_num_threads(1)
    result = {arm:measure(Path(f'scratch/cgb_action_cross_score_2587_20261120_{arm}/second_settlement.pt'))
              for arm in ('silence','generate')}
    output = Path('results/published/cgb_evidence_pair_interactions_2587_20261121.json')
    output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf8')
    for arm,r in result.items():
        print(arm,r['positive'],r['negative'],r['flip_pairs'],r['summary'])
        for p in r['pairs']:
            if p['flip_i'] or p['flip_j']:
                print(p); break
