"""Observe the live language loop before each turn's learning, without intervening.

F and the encoder are evaluated before the body's first optimizer update. A
private fast-table copy obtains the actual reply/holdout probabilities, and a
no-grad forward pass produces short and long targets in the current encoder's
coordinates. The live learner proceeds normally afterwards. This is a wiring
and temporal-causality audit, not a capability/convergence experiment.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from scripts.ib_local import gdn_fused, language_navigation_kanli_v4i
from scripts.ib_local.language_navigation_kanli_v4k import LatentDraftEngine
from scripts.ib_local import run_language_navigation_kanli_v4 as runner
from scripts.ib_local.run_language_navigation_v2 import validate


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cycles', type=int, default=48)
    p.add_argument('--live-sigreg', action='store_true')
    p.add_argument('--outcome-horizon', choices=('near', 'long'), default='near')
    p.add_argument('--detach-target', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    gdn_fused.enable()
    if args.live_sigreg:
        from scripts.ib_local.language_navigation_kanli_v4l import LiveSigregBody, LiveTargetEngine
        runner.LatentNavigationV3 = LiveSigregBody
        language_navigation_kanli_v4i.FilmEngine = LiveTargetEngine
    else:
        language_navigation_kanli_v4i.FilmEngine = LatentDraftEngine
    engine, table, valid, cfg = runner.build(equal=True, reply=True, film=True, capture=False,
                                  validation_bytes=4096)
    if args.live_sigreg:
        engine.outcome_horizon = args.outcome_horizon
        engine.detach_outcome_target = args.detach_target
    cfg['stage'] = 'pre-update joint water-fire loop audit'
    cfg.update(live_sigreg=args.live_sigreg, outcome_horizon=args.outcome_horizon if args.live_sigreg else 'long',
               detach_outcome_target=args.detach_target if args.live_sigreg else True)
    for path in (Path(__file__), Path('scripts/ib_local/language_navigation_kanli_v4k.py'),
                 Path('scripts/ib_local/language_navigation_kanli_v4l.py')):
        cfg['sources'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    take = engine.world.take
    probe = {}
    turn = {}

    def take_and_observe(utterance, response_bytes, holdout_bytes):
        response, holdout, info = take(utterance, response_bytes, holdout_bytes)
        cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
        versions = [p._version for p in engine.model.parameters()]
        with torch.no_grad():
            # The live table has heard the utterance, but not the reply yet.
            scratch = copy.deepcopy(table)
            last = utterance[-1] if utterance else turn['pending']
            _, p_reply, _ = type(scratch).run_credit(
                scratch, np.concatenate((np.array([last], np.uint8), response)), max_window=response_bytes)
            _, p_hold, _ = type(scratch).run_credit(
                scratch, np.concatenate((response[-1:], holdout)), max_window=holdout_bytes)
            own_probs = np.asarray(engine._pending_record['probs'], np.float32).reshape(-1, 256)
            raw = np.concatenate((np.array([turn['pending']], np.uint8),
                                  np.frombuffer(utterance, np.uint8), response))
            probs = np.concatenate((own_probs, p_reply), 0)
            ids, targets, logp = engine._inputs(raw, probs)
            logits, state, _, h, _ = engine.model.predict_observed(ids, turn['state'].clone(), logp)
            own = len(utterance)
            ce = F.cross_entropy(logits[0, own:], targets[0, own:], reduction='none')
            hraw = np.concatenate((response[-1:], holdout))
            hi, _, hp = engine._inputs(hraw, p_hold)
            _, _, _, h2, _ = engine.model.predict_observed(hi, state, hp)
            timeline = torch.cat((h, h2), 0)
            source, action = timeline[0], timeline[own]
            enc, outcome, memory = engine.model.encoder, engine.model.kanli.outcome, engine.model.memory
            zs, za = enc(source[None]), enc(action[None])
            prediction = outcome(torch.cat((zs, za), -1))
            occupied = torch.nonzero(memory.occupied).flatten()
            entry = dict(reply_nll=float(ce.mean()), reply_first4=float(ce[:4].mean()),
                         reply_first32=float(ce[:32].mean()), own_bytes=own,
                         draft_action_l2=float((engine._draft_actions[engine._pending_record['choice']]-action).norm()),
                         prior_episodes=int(len(occupied)))
            if len(occupied) >= 8:
                zs_old = enc(memory.source[occupied])
                za_old = enc(memory.action[occupied])
                sim = F.cosine_similarity(zs_old, zs, dim=-1)
                # Replace speech with a genuinely different previous spoken
                # action, rather than accidentally selecting old silence.
                different = ((za_old-za).norm(dim=-1) > 1e-3) & (memory.word_count[occupied] > 0)
                eligible = torch.nonzero(different).flatten()
                nearest = eligible[sim[eligible].argmax()] if len(eligible) else None
                near_slots = occupied[sim.topk(min(8, len(occupied))).indices]
                entry['near_source_cosine'] = float(sim.max())
                if nearest is not None:
                    entry['near_spoken_source_cosine'] = float(sim[nearest])
                for name, idx, store in (('short', min(own+memory.short, len(timeline)-1), memory.successor),
                                         ('long', min(memory.long, len(timeline)-1), memory.long_successor)):
                    target = enc(timeline[idx][None])
                    result = dict(own=float(F.mse_loss(prediction, target)),
                                  near_mean=float(F.mse_loss(enc(store[near_slots]).mean(0, keepdim=True), target)),
                                  silent=float(F.mse_loss(outcome(torch.cat((zs, zs), -1)), target)))
                    if nearest is not None:
                        alt = outcome(torch.cat((zs, za_old[nearest:nearest+1]), -1))
                        result['near_action'] = float(F.mse_loss(alt, target))
                    entry[name] = result
            entry['parameters_unchanged'] = versions == [p._version for p in engine.model.parameters()]
            entry['torch_rng_unchanged'] = bool(torch.equal(cpu_rng, torch.get_rng_state()) and
                                                torch.equal(gpu_rng, torch.cuda.get_rng_state()))
        probe.clear()
        probe.update(entry)
        return response, holdout, info

    engine.world.take = take_and_observe
    records = []
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    validation_before = validate(engine.model, table, valid)
    for index in range(args.cycles):
        turn['state'], turn['pending'] = engine.recurrent.clone(), engine.pending
        item = engine.step(torch.randn(64, 32, device='cuda'))
        records.append(dict(choice=item['choice'], world=item['world'], preupdate=dict(probe),
                            response_nll=item['response_nll'], metal_g=item['metal_g'],
                            sigreg=getattr(engine.model, '_last_sigreg', None),
                            action_probability=item['choice_probability'],
                            effect=dict(engine.draft_diagnostics),
                            replay_error=item['policy_replay_max_error'], updates=item['updates'],
                            episode_merged=item['episode_merged']))
        if (index+1) % 16 == 0:
            print(json.dumps(dict(cycle=index+1, seconds=round(time.perf_counter()-start, 2))), flush=True)
    validation_after = validate(engine.model, table, valid)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(config=cfg, cycles=args.cycles,
        elapsed_s=time.perf_counter()-start, peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        validation_bytes=len(valid)-1, validation_before=validation_before, validation_after=validation_after,
        records=records), indent=2), encoding='utf-8')
    print(json.dumps(dict(cycles=len(records), output=str(args.output))), flush=True)


if __name__ == '__main__':
    main()
