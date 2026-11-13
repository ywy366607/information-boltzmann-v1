"""Chronological, read-only audit of action-conditioned boundary distributions."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from scripts.ib_local import gdn_fused
from scripts.ib_local.run_causal_water_reply import build


def distribution(zs, za, zy, bank, query, *, action, shuffled=False, topk=8, temp=.2):
    """Old observations only; action displacement and source are distinct."""
    source = F.normalize(zs[query], dim=-1)
    keys = F.normalize(zs[bank], dim=-1)
    scores = source @ keys.T
    if action:
        # A wrong-action control must not borrow an action from the future.
        # Its first row uses itself because no earlier row exists in the split.
        action_ids = torch.cat((query[:1], query[:-1])) if shuffled else query
        delta = F.normalize(za[action_ids]-zs[action_ids], dim=-1)
        old_delta = F.normalize(za[bank]-zs[bank], dim=-1)
        scores = (scores+delta @ old_delta.T)/2
    k = min(topk, len(bank))
    val, ids = scores.topk(k, -1)
    weights = (val/temp).softmax(-1)
    targets = zy[bank[ids]]
    mean = (weights[..., None]*targets).sum(1)
    variance = (weights[..., None]*(targets-mean[:, None]).square()).sum(1).mean(-1)
    return mean, variance, bank[ids]


def metric(y, mean, spread, floor):
    error = (y-mean).square().mean(-1)
    var = spread + floor
    return dict(mse=float(error.mean()), standardized_error=float((error/var).mean()),
        gaussian_nll_per_dimension=float((.5*(math.log(2*math.pi)+var.log()+error/var)).mean()),
        predicted_variance=float(var.mean()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('output must be new')
    torch.set_num_threads(1)
    gdn_fused.enable()
    engine, _, _, source = build(args.checkpoint)
    m, model = engine.model.memory, engine.model
    with torch.no_grad():
        valid = m.occupied & m.internal_valid & (m.count == 1) & (m.internal_stamp == m.last_write)
        slots = torch.nonzero(valid).flatten()
        slots = slots[m.last_write[slots].argsort(stable=True)]
        n = len(slots)
        if n < 40:
            raise RuntimeError('insufficient historical boundary evidence')
        train_end, cal_end = n//2, (3*n)//4
        bank, calibration, test = slots[:train_end], slots[train_end:cal_end], slots[cal_end:]
        zs, za, zy = model.encoder(m.source), model.encoder(m.action), model.encoder(m.internal_next)
        branches = {}
        for name, action, shuffled in [('context',False,False),('action',True,False),('shuffled',True,True)]:
            cm, cv, _ = distribution(zs,za,zy,bank,calibration,action=action,shuffled=shuffled,topk=m.topk)
            tm, tv, used = distribution(zs,za,zy,bank,test,action=action,shuffled=shuffled,topk=m.topk)
            # One common residual floor per branch fitted on earlier calibration.
            floor = float(((zy[calibration]-cm).square().mean(-1)-cv).mean().clamp_min(1e-6))
            branches[name] = dict(calibration_floor=floor, **metric(zy[test],tm,tv,floor),
                strict_earlier=bool((m.last_write[used] < m.last_write[test,None]).all()))
        predicted = model.long_predictor(torch.cat((zs[test],za[test]),-1))
        branches['long_head'] = dict(mse=float((zy[test]-predicted).square().mean()))
        other = torch.cat((test[:1],test[:-1]))
        wrong = model.long_predictor(torch.cat((zs[test],za[other]),-1))
        silent = model.long_predictor(torch.cat((zs[test],zs[test]),-1))
        own_err = (zy[test]-predicted).square().mean(-1)
        wrong_err = (zy[test]-wrong).square().mean(-1)
        silent_err = (zy[test]-silent).square().mean(-1)
        branches['long_head'].update(wrong_action_mse=float(wrong_err.mean()),
            silent_substitute_mse=float(silent_err.mean()),
            own_beats_wrong_fraction=float((own_err < wrong_err).float().mean()),
            action_swap_prediction_l2=float((predicted-wrong).norm(dim=-1).mean()),
            wrong_action_control='previous test record, first item uses itself; no future input')
        report = dict(source=source, valid_pairs=n, historical=len(bank), calibration=len(calibration),
            test=len(test), chronological_split='first 50% historical, next 25% calibration, final 25% test',
            old_event_range=[int(m.last_write[bank[0]]),int(m.last_write[bank[-1]])],
            test_event_range=[int(m.last_write[test[0]]),int(m.last_write[test[-1]])],
            results=branches,
            note=('retrospective frozen encoder/parameters at checkpoint: test records may have influenced those weights; '
                  'the neighbor bank itself is strictly earlier, this is not a prequential parameter-held-out test'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))


if __name__ == '__main__':
    main()
