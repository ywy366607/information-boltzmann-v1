"""Read-only audit of old, versioned evidence in a real question receipt."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from scripts.ib_local.language_action import peek
from scripts.ib_local.navigation_internal_rollout import continuation
from scripts.ib_local.navigation_persistent_question import retained_receipt
from scripts.ib_local.navigation_relation_wood import local_relation
from scripts.ib_local.run_causal_water_reply import build


@torch.no_grad()
def audit(engine):
    model, m = engine.model, engine.model.memory
    model._question_context = None
    engine.decision.start(engine.recurrent)
    _, h = engine.decision.step(engine.pending, peek(engine.table, engine.pending))
    source = h.reshape(1, -1)
    packet = retained_receipt(model, source)
    if packet is None:
        raise ValueError('no retained question receipt')
    slots, stamps = packet['slots'], packet['stamps']
    valid = (packet['valid'] & m.occupied[slots] & m.internal_valid[slots] &
             (m.count[slots] == 1) & (m.last_write[slots] == stamps) &
             (m.internal_stamp[slots] == stamps))
    choices = torch.nonzero(valid[0]).flatten()
    if len(choices) < 2:
        raise ValueError('audit requires two retained real evidence items')
    # Choose the oldest source; this is an audit intervention, not a policy.
    position = choices[torch.argmin(stamps[0, choices])]
    other = choices[choices != position][0]
    chosen, wrong = slots[0, position], slots[0, other]
    original = (slots, stamps, packet['valid'])
    full, _ = local_relation(model, source, source, receipt=original,
                             ridge=engine.relation_ridge, scale=engine.relation_scale)
    reduced_valid = packet['valid'].clone()
    reduced_valid[0, position] = False
    reduced = (slots, stamps, reduced_valid)
    base, _ = local_relation(model, source, source, receipt=reduced,
                             ridge=engine.relation_ridge, scale=engine.relation_scale)

    def add(target):
        evidence = (m.source[chosen:chosen+1], m.action[chosen:chosen+1],
                    target, torch.ones(1, dtype=torch.bool, device=source.device))
        return local_relation(model, source, source, receipt=reduced, extra=evidence,
                              ridge=engine.relation_ridge, scale=engine.relation_scale)[0]

    restored = add(m.internal_next[chosen:chosen+1])
    mismatched = add(m.internal_next[wrong:wrong+1])
    current = model.encoder(source)
    reward_valid = engine.reward_label_valid(wood=True)
    future, _ = continuation(model, current + model.question_reactivation(full), reward_valid=reward_valid)
    future_wrong, _ = continuation(model, current + model.question_reactivation(mismatched), reward_valid=reward_valid)
    return dict(anchor=packet['anchor'], anchor_age_bytes=int(m.clock-packet['stamp']),
                eligible_relation_count=int(valid.sum()), oldest_slot=int(chosen),
                oldest_age_bytes=int(m.clock-stamps[0, position]), wrong_slot=int(wrong),
                remove_relation_norm=float((full-base).norm()),
                real_restore_error=float((full-restored).norm()),
                wrong_target_relation_difference=float((restored-mismatched).norm()),
                retained_port_difference=float((model.question_reactivation(restored)-
                                                model.question_reactivation(mismatched)).norm()),
                continuation_gain_difference=float((future-future_wrong).abs().max()),
                meaning='historical outcomes used as read-only interventions, not online prospective accuracy')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('refusing to overwrite report')
    torch.set_num_threads(1)
    engine, _, _, provenance = build(args.checkpoint, device='cpu')
    result = dict(source=provenance, audit=audit(engine))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
