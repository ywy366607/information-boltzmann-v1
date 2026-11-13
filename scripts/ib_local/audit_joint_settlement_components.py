"""Decompose one replayed joint settlement without changing the source life."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from scripts.ib_local import gdn_fused
from scripts.ib_local import language_navigation_kanli_v4c as v4c
from scripts.ib_local.mother_reply_world import MotherReplyWorld, StrictToddlerMother
from scripts.ib_local.run_causal_water_reply import build, sha256_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('refusing to overwrite a previous audit')
    torch.set_num_threads(1)
    gdn_fused.enable()
    with gzip.open(args.checkpoint, 'rb') as f:
        state = torch.load(f, map_location='cpu', weights_only=False)
    config, world_state = state['config'], state['world']
    engine, _, _, provenance = build(args.checkpoint)
    world_state = {k:(v.cpu().numpy() if torch.is_tensor(v) else v) for k,v in world_state.items()}
    policy = config.get('mother_policy', 'legacy')
    mother = StrictToddlerMother(policy=policy, shared_cache=args.cache)
    engine.world = MotherReplyWorld(engine.world, mother, ambient_policy=world_state['mother_ambient_policy'])
    engine.world.load_state_dict(world_state)
    frozen, targets, scores = [], {}, {}
    old_read, old_take, old_predict = v4c.frozen_table_logp, engine.world.take, engine.model.predict

    def capture_read(table, ids):
        out = old_read(table, ids)
        frozen.append(out[engine.response_bytes:].copy())
        return out

    def capture_take(*a, **kw):
        response, heldout, info = old_take(*a, **kw)
        targets['heldout'] = torch.as_tensor(heldout.astype(np.int64), device=engine.device)
        targets['response'] = response.copy()
        return response, heldout, info

    def capture_predict(ids, recurrent, logp):
        out = old_predict(ids, recurrent, logp)
        if len(frozen) == 2 and 'heldout' in targets and ids.shape[1] == len(targets['heldout']):
            options = [torch.as_tensor(x, device=logp.device)[None] for x in frozen]
            if 'pre_old' not in scores and torch.equal(logp, options[0]):
                phase = 'pre'
            elif 'pre_old' in scores and 'post_new' not in scores and torch.equal(logp, options[1]):
                phase = 'post'
            else:
                return out
            with torch.no_grad():
                for label, table_logp in zip(('old', 'new'), options):
                    logits = out[0] if torch.equal(logp, table_logp) else old_predict(ids, recurrent, table_logp)[0]
                    per = F.cross_entropy(logits.flatten(0,1), targets['heldout'], reduction='none')
                    scores[phase+'_'+label] = per.detach().cpu().numpy()
        return out

    v4c.frozen_table_logp = capture_read
    engine.world.take = capture_take
    engine.model.predict = capture_predict
    try:
        row = engine.step(torch.randn(64, 32, device=engine.device))
    finally:
        v4c.frozen_table_logp = old_read
        engine.world.take = old_take
        engine.model.predict = old_predict
    if set(scores) != {'pre_old','pre_new','post_old','post_new'}:
        raise RuntimeError(f'incomplete score capture: {sorted(scores)}')
    means = {k:float(v.mean()) for k,v in scores.items()}
    if (not np.isclose(means['pre_old'], row['joint_pre_nll'], atol=1e-5) or
            not np.isclose(means['post_new'], row['joint_post_nll'], atol=1e-5)):
        raise RuntimeError('audit cross scores do not reproduce actual settlement')
    table_effect = scores['pre_old']-scores['pre_new']
    neural_effect = scores['pre_new']-scores['post_new']
    top = np.argsort(-np.abs(table_effect))[:10]
    target_bytes = targets['heldout'].cpu().numpy()
    old_target_logp = frozen[0][np.arange(len(target_bytes)), target_bytes]
    new_target_logp = frozen[1][np.arange(len(target_bytes)), target_bytes]
    by_byte = []
    for byte in np.unique(target_bytes):
        mask = target_bytes == byte
        by_byte.append(dict(byte=int(byte), count=int(mask.sum()),
            contribution=float(table_effect[mask].sum()/len(target_bytes)),
            response_count=int((targets['response'] == byte).sum())))
    by_byte.sort(key=lambda x: abs(x['contribution']), reverse=True)
    result = dict(source=provenance, replayed_cycle=provenance['cycle']+1,
        source_sha256=sha256_file(args.checkpoint), choice=row['choice'],
        utterance_hex=row['utterance_hex'], mother_reply=row['world']['mother_reply'],
        reported_joint_pre=row['joint_pre_nll'], reported_joint_post=row['joint_post_nll'],
        reported_gain=row['metal_g'], four_way_nll=means,
        table_first_effect=float(table_effect.mean()),
        neural_after_table_effect=float(neural_effect.mean()),
        table_effect_by_target_byte=by_byte,
        table_target_old_floor_count=int((old_target_logp <= np.log(1.00001e-6)).sum()),
        table_target_new_floor_count=int((new_target_logp <= np.log(1.00001e-6)).sum()),
        top_table_affected_positions=[dict(position=int(i), target=int(targets['heldout'][i]),
            old_nll=float(scores['pre_old'][i]), new_nll=float(scores['pre_new'][i])) for i in top],
        scope='same historical response and held-out continuation; audit-only cross scores')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result,ensure_ascii=True))


if __name__ == '__main__':
    main()
