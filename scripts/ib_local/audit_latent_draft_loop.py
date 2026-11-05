"""Short real-stream wiring audit; this is not a capability comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from scripts.ib_local import gdn_fused, language_navigation_kanli_v4i
from scripts.ib_local.language_navigation_kanli_v4k import LatentDraftEngine
from scripts.ib_local.run_language_navigation_kanli_v4 import build
from scripts.ib_local.run_language_navigation_v2 import validate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cycles', type=int, default=8)
    parser.add_argument('--ablate-outcome', action='store_true',
                        help='feed all options the silence F outcome, preserving draft and F training')
    parser.add_argument('--validation-bytes', type=int, default=4096)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    gdn_fused.enable()
    # Isolated factory override: production runners and the base model stay unchanged.
    language_navigation_kanli_v4i.FilmEngine = LatentDraftEngine
    engine, table, valid, cfg = build(equal=True, reply=True, film=True, capture=False,
                                      validation_bytes=args.validation_bytes)
    engine.ablate_action_outcome = args.ablate_outcome
    cfg['stage'] = 'v4k: draft-conditioned latent action wiring audit'
    cfg['action_outcome_ablation'] = args.ablate_outcome
    for source in (Path(__file__), Path('scripts/ib_local/language_navigation_kanli_v4k.py')):
        cfg['sources'][str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
    # The observation is scored before the F update, but after this turn's
    # language-response update. This is not fully prequential for the body.
    original_loss = engine._outcome_loss
    measured = {}

    def audited_loss(source, action, future):
        measured.clear()
        memory, enc, outcome = engine.model.memory, engine.model.encoder, engine.model.kanli.outcome
        with torch.no_grad():
            occupied = torch.nonzero(memory.occupied).flatten()
            spoken = occupied[memory.word_count[occupied] > 0]
            if len(spoken) >= 8:
                zs, za, target = (enc(x[None]) for x in (source, action, future))
                # Only previous slots are eligible; deterministic selection
                # does not perturb the live policy's random number stream.
                slot = spoken[int(memory.clock) % len(spoken)]
                other = enc(memory.action[slot][None])
                score = lambda a: float(F.mse_loss(outcome(torch.cat((zs, a), -1)), target))
                # Match the prior source-state, then require a different stored
                # action; all examples predate this turn. This is observational
                # matching, not a claim the contexts or worlds are identical.
                prior_sources = enc(memory.source[spoken])
                prior_actions = enc(memory.action[spoken])
                similarities = F.cosine_similarity(prior_sources, zs, dim=-1)
                different = (prior_actions - za).norm(dim=-1) > 1e-3
                eligible = torch.nonzero(different).flatten()
                if len(eligible):
                    nearest = eligible[similarities[eligible].argmax()]
                    measured.update(near_action=score(prior_actions[nearest:nearest+1]),
                                    near_cosine=float(similarities[nearest]),
                                    near_action_distance=float((prior_actions[nearest]-za[0]).norm()))
                all_sources = enc(memory.source[occupied])
                sim_all = F.cosine_similarity(all_sources, zs, dim=-1)
                k = min(8, len(occupied))
                nearest_slots = occupied[sim_all.topk(k).indices]
                conditional_mean = enc(memory.long_successor[nearest_slots]).mean(0, keepdim=True)
                measured.update(own=score(za), other=score(other), silent=score(zs),
                                mean=float(F.mse_loss(enc(memory.long_successor[occupied]).mean(0, keepdim=True), target)),
                                near_mean=float(F.mse_loss(conditional_mean, target)),
                                prior_spoken=int(len(spoken)))
        return original_loss(source, action, future)

    engine._outcome_loss = audited_loss
    records = []
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    validation_before = validate(engine.model, table, valid)
    for cycle in range(args.cycles):
        item = engine.step(torch.randn(64, 32, device='cuda'))
        record = engine._pending_record
        records.append(dict(choice=item['choice'], world=item['world'],
                            draft_actions=len(engine._draft_actions),
                            candidate_effect=engine.draft_diagnostics,
                            outcome_probe=dict(measured),
                            values=record['values'], offsets=record['offset'],
                            response_nll=item.get('response_nll'), heldout_pre_nll=item.get('heldout_pre_nll'),
                            heldout_post_nll=item.get('heldout_post_nll'), metal_g=item.get('metal_g'),
                            updates=item.get('updates'),
                            action_l2=item.get('draft_actual_action_l2'),
                            outcome_error=item.get('outcome_error'),
                            policy_error=item.get('policy_replay_max_error')))
        if (cycle + 1) % 32 == 0:
            print(json.dumps(dict(cycle=cycle+1, elapsed_s=round(time.perf_counter()-start, 2),
                                  ablate_outcome=args.ablate_outcome)), flush=True)
    validation_after = validate(engine.model, table, valid)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(config=cfg, cycles=args.cycles, elapsed_s=time.perf_counter()-start,
                                          peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                                          validation_bytes=len(valid)-1,
                                          validation_before=validation_before,
                                          validation_after=validation_after,
                                          records=records), indent=2), encoding='utf-8')
    print(json.dumps(dict(cycles=len(records), choices=[r['choice'] for r in records],
                          draft_actions=[r['draft_actions'] for r in records],
                          output=str(args.output)), ensure_ascii=False))


if __name__ == '__main__':
    main()
