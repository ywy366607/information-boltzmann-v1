"""Audit a completed room run using only recorded decisions and heldout scores."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.ib_local.learning_rooms_world import LearningRoomsWorld, ROOMS, entropy_floors


def noise_opportunity(rows):
    selected=[r for r in rows if 1 in r['candidate_rooms'] and
              any(k in (2,3,4) for k in r['candidate_rooms'])]
    if not selected:
        return dict(n=0)
    return dict(n=len(selected),
        policy_noise_probability=float(np.mean([sum(r['choice_probability'][j]
            for j,k in enumerate(r['candidate_rooms']) if k==1) for r in selected])),
        uniform_candidate_noise_probability=float(np.mean([
            r['candidate_rooms'].count(1)/len(r['candidate_rooms']) for r in selected])),
        actual_noise_fraction=float(np.mean([r['world']['anchor']==1 for r in selected])))


def alignment(audits, key='metal_g'):
    from scipy.stats import spearmanr
    g=np.array([r[key] for r in audits])
    future=np.array([r['audit_balanced_g'] for r in audits])
    material=(np.abs(g)>1e-5)&(np.abs(future)>1e-5)
    return dict(n=len(audits),
        pearson=float(np.corrcoef(g,future)[0,1]) if len(audits)>1 and min(g.std(),future.std())>0 else None,
        spearman=float(spearmanr(g,future).statistic) if len(audits)>1 and min(g.std(),future.std())>0 else None,
        material_n=int(material.sum()),
        material_sign_agreement=float(np.mean(np.sign(g[material])==np.sign(future[material]))) if material.any() else None)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path)
    args=parser.parse_args();root=args.root
    rows=[json.loads(line) for line in (root/'metrics.jsonl').read_text().splitlines()]
    summary=json.loads((root/'summary.json').read_text())
    cfg=json.loads((root/'config.json').read_text())
    assert len(rows)==cfg['cycles']==summary['cycles']
    world=LearningRoomsWorld(11,cfg['switch_turn'],cfg['routing'])
    digest=hashlib.sha256()
    for row in rows:
        response,heldout,info=world.take(bytes.fromhex(row['utterance_hex']),257,257)
        assert row['world']==info
        digest.update(response.tobytes());digest.update(heldout.tobytes())
        predicted=[LearningRoomsWorld.destination(bytes.fromhex(w),info['room_before'],cfg['routing'])
                   for w in row['candidate_words_hex']]
        assert predicted==row['candidate_rooms']
    blocks=[]
    for lo in range(0,len(rows),64):
        part=rows[lo:lo+64]
        blocks.append(dict(start=lo+1,end=lo+len(part),
            visits={name:sum(r['world']['room']==name for r in part) for name in ROOMS},
            choices={k:sum(r['choice']==k for r in part) for k in ('silence','generate','reuse')},
            g={name:float(np.mean([r['metal_g'] for r in part if r['world']['room']==name]))
               for name in ROOMS if any(r['world']['room']==name for r in part)},
            multiroom_opportunities=sum(len(set(r['candidate_rooms']))>1 for r in part),
            offered={name:sum(i in r['candidate_rooms'] for r in part) for i,name in enumerate(ROOMS)},
            noise_when_learning_alternative=noise_opportunity(part),
            heldout_nll=part[-1].get('audit_room_post'),
            witnesses=part[-1].get('witness_alphabets')))
    audits=[r for r in rows if 'audit_balanced_g' in r]
    flip=cfg['switch_turn']
    before=[r for r in audits if r['cycle']<=flip][-1]
    after=[r for r in audits if r['cycle']>flip][0]
    return_visits=[r['cycle'] for r in rows if r['cycle']>flip and r['world']['anchor']==4]
    report=dict(
        transition_and_candidate_replay=True,external_sha256=digest.hexdigest(),
        entropy_floors_nats=dict(zip(ROOMS,entropy_floors())),blocks=blocks,
        noise_opportunities=dict(first_half=noise_opportunity(rows[:flip]),
                                second_half=noise_opportunity(rows[flip:])),
        first_visit={name:next((r['cycle'] for r in rows if r['world']['room']==name),None) for name in ROOMS},
        change=dict(turn=flip+1,prechange_nll=before['audit_room_post'][4],
            new_rule_first_nll_before_learning=after['audit_room_pre'][4],
            first_return=return_visits[0] if return_visits else None,
            returned_within_128=bool(return_visits and return_visits[0]<=flip+128),
            prechange_visits=sum(r['cycle']<=flip and r['world']['anchor']==4 for r in rows),
            postchange_visits=len(return_visits),final_nll=summary['final_nll']['changing']),
        gold_future_alignment=dict(overall=alignment(audits),
            first_half=alignment([r for r in audits if r['cycle']<=flip]),
            second_half=alignment([r for r in audits if r['cycle']>flip])),
        continuation_future_alignment=dict(overall=alignment(audits,'progress'),
            first_half=alignment([r for r in audits if r['cycle']<=flip],'progress'),
            second_half=alignment([r for r in audits if r['cycle']>flip],'progress')),
        note='Future bank is a researcher diagnostic, not reward. Empirical allocation is not causal attribution to the neural F; opportunities are reported separately.')
    (root/'analysis.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    curated=dict(summary=summary,analysis=report,source_hashes=cfg['sources_new'],
                 protocol='docs/information_boltzmann/CGB_LEARNING_ROOMS_20261105.md')
    Path('results/published',root.name+'.json').write_text(
        json.dumps(curated,indent=2),encoding='utf8')
    print(json.dumps({k:report[k] for k in ('first_visit','noise_opportunities','change','gold_future_alignment')},indent=2))


if __name__=='__main__':
    main()
