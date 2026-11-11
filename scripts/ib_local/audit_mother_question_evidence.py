"""Read-only inspection of the selected question's two real outcome boundaries."""
import argparse
import json
from pathlib import Path

import torch

from scripts.ib_local.language_action import peek
from scripts.ib_local.navigation_persistent_question import (
    actual_boundary_evidence, mother_response_evidence, retained_receipt)
from scripts.ib_local.navigation_internal_rollout import next_boundary_prediction
from scripts.ib_local.navigation_shadow_preview import preview
from scripts.ib_local.run_causal_water_reply import build


@torch.no_grad()
def audit(engine):
    m = engine.model.memory
    engine.model._question_context = None
    engine.decision.start(engine.recurrent)
    _, h = engine.decision.step(engine.pending, peek(engine.table, engine.pending))
    receipt = retained_receipt(engine.model, h[None])
    if receipt is None:
        return {'receipt': None}
    mother = mother_response_evidence(engine, receipt)
    later = actual_boundary_evidence(engine, receipt)
    result = {'anchor': receipt['anchor'], 'stamp': receipt['stamp'],
              'age_bytes': int(m.clock) - receipt['stamp'],
              'mother_available': mother is not None,
              'later_available': later is not None}
    if mother is not None and later is not None:
        result['raw_boundary_l2'] = float((mother[2] - later[2]).norm())
        result['encoded_boundary_l2'] = float(
            (engine.model.encoder(mother[2]) - engine.model.encoder(later[2])).norm())
        source = h[None]
        current = next_boundary_prediction(engine.model, source, source)
        current = current + engine.model.mother_from_boundary(current)
        bound = tuple(receipt[k] for k in ('slots', 'stamps', 'valid'))
        kw = dict(ridge=engine.relation_ridge, scale=engine.relation_scale,
                  samples=engine.shadow_samples, candidate_kind=engine.shadow_candidate_kind,
                  relation_receipt=bound, outcome_kind='mother')
        with_actual = preview(engine.model, source, source, current,
                              engine.reward_label_valid(wood=True), relation_actual=mother, **kw)
        without_actual = preview(engine.model, source, source, current,
                                 engine.reward_label_valid(wood=True), **kw)
        result['mother_preview_valid'] = with_actual['valid'].tolist()
        result['mother_preview_usable'] = with_actual['usable'].tolist()
        result['mother_preview_future_delta'] = with_actual['future_delta'].tolist()
        result['actual_evidence_effect_on_shadow'] = float(
            (with_actual['shadow_state_shift'] - without_actual['shadow_state_shift']).norm())
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('refusing to overwrite existing audit')
    torch.set_num_threads(1)
    engine, _, _, provenance = build(args.checkpoint, device='cpu')
    result = {'source': provenance, 'audit': audit(engine)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=True, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=True))


if __name__ == '__main__':
    main()
