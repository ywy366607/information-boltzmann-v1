"""Read completed native-loop records for computation and value allocation.

Uses only recorded forecasts and policies. Each branch variance describes the
recorded representatives, with their original quadrature masses; it is not a
standard error. Forecast maxima are finite-model envelopes, not counterfactual
world payoffs. No model load, optimizer step, or world interaction is performed.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean


def analyze(path):
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    rows = [r for r in rows if r.get('autonomous_event', {}).get('kind') == 'external']
    decisions, branches = [], []
    for row in rows:
        forecast = row['causal_use']['epistemic_value']
        q = forecast['contextual_total']
        p = row['choice_probability']
        active = [value for prob, value in zip(p, q) if prob > 0]
        policy = sum(prob * value for prob, value in zip(p, q))/sum(p)
        decisions.append(dict(cycle=row['cycle'], total_values=q, probabilities=p,
            policy_forecast=policy, maximum_forecast=max(active),
            model_selection_gap=max(active)-policy,
            value_range=max(active)-min(active), actual_use=row['metal_g']))
        for action, routes in enumerate(forecast.get('successor_routes', [])):
            for response, route in enumerate(routes):
                weights = route['weights']
                values = route['values'][0]
                assert len(weights) == len(values) and all(w >= 0 for w in weights)
                assert abs(sum(weights)-1) < 1e-6
                predicted = sum(w*v for w, v in zip(weights, values))
                assert abs(predicted-route['expected'][0]) < 2e-6
                variance = sum(w*(v-predicted)**2 for w, v in zip(weights, values))
                geom = route['joint']
                tail = sum(g['count']/geom['samples'] for g in geom['strata']
                           if g['stratum'] == 'remaining_mass')
                branches.append(dict(cycle=row['cycle'], action=action, response=response,
                    branches=len(weights), weights=weights,
                    maximum_mass=max(weights), fresh_mass=geom['fresh_mass'],
                    grouped_tail_mass=tail, representative_value_range=max(values)-min(values),
                    representative_weighted_std=math.sqrt(variance),
                    weighted_reconstruction_error=abs(predicted-route['expected'][0])))
    return dict(source=str(path), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        cycles=len(rows), decisions=decisions, branches=branches,
        mean_model_selection_gap=mean(d['model_selection_gap'] for d in decisions),
        mean_total_value_range=mean(d['value_range'] for d in decisions),
        mean_branch_std=mean(b['representative_weighted_std'] for b in branches),
        mean_fresh_mass=mean(b['fresh_mass'] for b in branches),
        mean_grouped_tail_mass=mean(b['grouped_tail_mass'] for b in branches),
        max_weighted_reconstruction_error=max(b['weighted_reconstruction_error'] for b in branches))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--events', type=Path, default=Path(
        'scratch/cgb_joint_successor_2493_20261119/joint/events.jsonl'))
    p.add_argument('--output', type=Path, default=Path(
        'results/published/cgb_preview_efficiency_frontier_2493_20261119.json'))
    args = p.parse_args()
    result = analyze(args.events)
    result['measurement'] = dict(new_updates=0, world_calls=0,
        scope='completed schema5 forecasts and actual coupled-loop policy',
        branch_spread='weighted dispersion across recorded representatives',
        selection_gap='max recorded total forecast minus its actual policy expectation')
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ('decisions', 'branches')},
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
