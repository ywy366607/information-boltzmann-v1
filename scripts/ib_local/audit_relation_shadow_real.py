"""Read-only shadow relation update using retained actual boundary evidence."""
import argparse
import json
from pathlib import Path

import torch

from scripts.ib_local import gdn_fused
from scripts.ib_local.language_action import peek
from scripts.ib_local.navigation_relation_wood import local_relation
from scripts.ib_local.run_causal_water_reply import build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('output must be new')
    torch.set_num_threads(1)
    gdn_fused.enable()
    engine, _, _, provenance = build(args.checkpoint)
    m = engine.model.memory
    with torch.no_grad():
        engine.decision.start(engine.recurrent)
        _, h = engine.decision.step(engine.pending, peek(engine.table, engine.pending))
        q = h.reshape(1, -1)
        original, packet = local_relation(engine.model, q, q,
            ridge=engine.relation_ridge, scale=engine.relation_scale)
        choices = packet['slots'][0][packet['valid'][0]]
        if len(choices) < 1:
            raise RuntimeError('no compatible actual relation evidence')
        slot = choices[0].reshape(1)
        stamp = m.last_write[slot].clone()
        valid = torch.ones(1, device=q.device, dtype=torch.bool)
        extra = (m.source[slot], m.action[slot], m.internal_next[slot], valid)
        # Remove the chosen item before adding it back. This matches the exact
        # selected evidence set and is not a forecast or a training run.
        without, receipt = local_relation(engine.model, q, q, exclude=slot,
            receipt=(packet['slots'],packet['stamps'],packet['valid']),
            ridge=engine.relation_ridge, scale=engine.relation_scale)
        restored, shadow = local_relation(engine.model, q, q, exclude=slot,
            receipt=(packet['slots'],packet['stamps'],packet['valid']), extra=extra,
            ridge=engine.relation_ridge, scale=engine.relation_scale)
        report = dict(source=provenance, evidence_slot=int(slot), evidence_stamp=int(stamp),
            previous_support=int(packet['valid'].sum()),
            removed_delta=float((without-original).norm()),
            restored_delta=float((restored-original).norm()),
            extra_coefficient=float(shadow['extra_coefficient'][0,0]),
            scope='fixed current encoder and actual retained evidence; no world/optimizer update')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))


if __name__ == '__main__':
    main()
