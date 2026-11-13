"""Read-only same-trajectory algebra audit; never a capability comparison.

Only use records whose entire positive-weight payload can be reconstructed
exactly from preceding completed turns. No future labels enter a read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def audit(rows):
    cache, measured = {}, []
    for i,row in enumerate(rows):
        slots=np.array(row['episodic_read_slots'],dtype=int)
        weights=np.array(row['episodic_read_weights'],dtype=float)
        positive=weights>0
        eligible=positive.any() and all(int(s) in cache for s in slots[positive])
        if eligible:
            label=np.array(row['wood_target'])
            base=np.array(row['wood_base'])
            absolute=sum(float(w)*cache[int(s)]['target'] for s,w in zip(slots,weights) if w>0)
            innovation=sum(float(w)*(cache[int(s)]['target']-cache[int(s)]['base'])
                           for s,w in zip(slots,weights) if w>0)
            measured.append(dict(turn=i+1,
                absolute_error=(absolute-label)**2,
                innovation_error=(base+innovation-label)**2,
                current_error=np.array(row['episodic_real_error']),
                neural_error=np.array(row['episodic_plain_error']),
                base_error=np.array(row['wood_base_error']),
                effective_neighbours=float(1/np.square(weights).sum())))
        slot,choice=row['episode_slot'],row['choice']
        if not row['episode_merged']:
            cache[slot]=dict(target=np.array(row['wood_target']),base=np.array(row['wood_base']),
                             count=1,choice=choice)
        elif slot in cache:
            old=cache[slot]
            if old['choice']!=choice:
                del cache[slot]
            else:
                n=old['count']
                old['target']=(old['target']*n+row['wood_target'])/(n+1)
                old['base']=(old['base']*n+row['wood_base'])/(n+1)
                old['count']+=1
    late=[r for r in measured if r['turn']>len(rows)//2]
    return dict(total_turns=len(rows),reconstructed_turns=len(measured),late_reconstructed_turns=len(late),
        late_mean_errors={name:np.mean([r[name] for r in late],axis=0).tolist() for name in
            ('absolute_error','innovation_error','current_error','neural_error','base_error')}
            if late else {},
        effective_neighbours_mean=float(np.mean([r['effective_neighbours'] for r in late])) if late else None,
        scope='Posthoc algebra on unchanged queries/read weights. Exact old-payload reconstruction only; no policy or new-learner benefit claim.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--metrics',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    rows=[json.loads(l) for l in args.metrics.read_text(encoding='utf-8').splitlines()]
    report=audit(rows)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
