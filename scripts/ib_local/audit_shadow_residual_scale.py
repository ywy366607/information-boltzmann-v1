"""Retrospective chronological check of shadow candidate residual scale."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from scripts.ib_local import gdn_fused
from scripts.ib_local.run_causal_water_reply import build


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
    model, m = engine.model, engine.model.memory
    with torch.no_grad():
        eligible = m.occupied & m.internal_valid & (m.count == 1) & (m.internal_stamp == m.last_write)
        slots = torch.nonzero(eligible).flatten()
        slots = slots[m.last_write[slots].argsort(stable=True)]
        n = len(slots)
        if n < 40:
            raise ValueError('insufficient chronological evidence')
        bank, cal, test = slots[:n//2], slots[n//2:3*n//4], slots[3*n//4:]
        zs, za, zy = model.encoder(m.source), model.encoder(m.action), model.encoder(m.internal_next)
        pred = model.long_predictor(torch.cat((zs, za), -1))
        def candidates(query):
            score = (F.normalize(zs[query],dim=-1) @ F.normalize(zs[bank],dim=-1).T +
                     F.normalize(za[query]-zs[query],dim=-1) @
                     F.normalize(za[bank]-zs[bank],dim=-1).T)/2
            val, idx = score.topk(2, -1)
            weight = val.softmax(-1)
            selected = bank[idx]
            residual = (weight[...,None]*(zy[selected]-pred[selected])).sum(1)
            direct = (weight[...,None]*zy[selected]).sum(1)
            return residual, direct, selected
        dc, _, _ = candidates(cal)
        dt, direct, used = candidates(test)
        numerator = ((zy[cal]-pred[cal])*dc).sum()
        denom = dc.square().sum().clamp_min(1e-12)
        scale = (numerator/denom).clamp(0, 1)
        mse = lambda x: float((zy[test]-x).square().mean())
        result = dict(source=source, valid=n, bank=len(bank), calibration=len(cal), test=len(test),
            calibration_scale=float(scale), mse=dict(base=mse(pred[test]),
                direct_neighbor=mse(direct), full_residual=mse(pred[test]+dt),
                calibrated_residual=mse(pred[test]+scale*dt)),
            strictly_older=bool((m.last_write[used] < m.last_write[test,None]).all()),
            limitations='retrospective: final weights/encoder have seen these events; long head only, not full S/P+relation prediction')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__ == '__main__':
    main()
