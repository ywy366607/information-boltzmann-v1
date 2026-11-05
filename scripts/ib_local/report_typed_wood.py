"""Prequential binary-score report for the actually selected real actions."""
import argparse
import json
from pathlib import Path

import numpy as np


def summarize(rows):
    y=np.array([r['wood_target'][2] for r in rows])
    base=np.array([r['wood_base'][2] for r in rows])
    def scores(key):
        p=np.array([r[key] for r in rows])
        brier=float(np.mean((p-y)**2))
        clipped=np.clip(p,1e-7,1-1e-7)
        logloss=float(np.mean(-y*np.log(clipped)-(1-y)*np.log(1-clipped)))
        bins=[]
        for i in range(10):
            mask=(p>=i/10)&((p<(i+1)/10) if i<9 else (p<=1))
            if mask.any():
                bins.append(dict(lo=i/10,hi=(i+1)/10,n=int(mask.sum()),
                    forecast=float(p[mask].mean()),frequency=float(y[mask].mean())))
        ece=sum(b['n']*abs(b['forecast']-b['frequency']) for b in bins)/len(rows)
        return dict(brier=brier,logloss=logloss,ece=ece,bins=bins,mean_probability=float(p.mean()),
                    minimum=float(p.min()),maximum=float(p.max()))
    return dict(n=len(rows),actual_acceptance=float(y.mean()),
        base_brier=float(np.mean((base-y)**2)),model=scores('match_model_probability'),
        used=scores('match_probability'),
        mean_mixture_weight=float(np.mean([r['match_residual_mixture_weight'] for r in rows])),
        mean_match_policy_tv=float(np.mean([r['match_policy_tv'] for r in rows])),
        nonzero_match_policy_cycles=sum(r['match_policy_tv']>0 for r in rows))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    args=parser.parse_args()
    rows=[json.loads(line) for line in (args.root/'metrics.jsonl').read_text().splitlines()]
    late=rows[len(rows)//2:]
    summary=json.loads((args.root/'summary.json').read_text())
    config=json.loads((args.root/'config.json').read_text())
    report=dict(scope='chosen-action prequential calibration diagnostic; no calibration proof or all-candidate claim',
        parameter_count=config['parameter_count'],cycles=len(rows),all=summarize(rows),late=summarize(late),
        by_action={a:summarize([r for r in late if r['choice']==a])
                   for a in ('silence','generate','reuse') if any(r['choice']==a for r in late)},
        validation=[dict(cycle=0,nll=summary['initial_nll'])]+[
            dict(cycle=r['cycle'],nll=r['validation_nll']) for r in rows if 'validation_nll' in r],
        max_replay_error=max(r['policy_replay_max_error'] for r in rows))
    report['acceptance']=dict(
        late_brier_beats_base=report['late']['used']['brier']<report['late']['base_brier'],
        language_within_limit=summary['final_nll']<=summary['initial_nll']+.01,
        valid_probabilities=report['all']['used']['minimum']>=0 and report['all']['used']['maximum']<=1)
    (args.root/'calibration.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps({k:report[k] for k in ('cycles','late','validation','acceptance')},indent=2))


if __name__=='__main__':
    main()
