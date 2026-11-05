"""Read-only one-turn world forks inside one continuing real-language body.

At one real decision, replay two already-formed candidate utterances from the
same body, fast table and world RNG. Each branch receives its own world reply;
the live individual only consumes its actually sampled action and reply.
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--cycles', type=int, default=24)
    parser.add_argument('--live-sigreg-near', action='store_true')
    parser.add_argument('--fork-every', type=int, default=1)
    parser.add_argument('--validation-every', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.cycles < 1 or args.fork_every < 1 or args.validation_every < 0:
        raise ValueError('invalid cycle or audit interval')
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    gdn_fused.enable()
    if args.live_sigreg_near:
        from scripts.ib_local.language_navigation_kanli_v4l import LiveSigregBody, LiveTargetEngine
        runner.LatentNavigationV3 = LiveSigregBody
        language_navigation_kanli_v4i.FilmEngine = LiveTargetEngine
    else:
        language_navigation_kanli_v4i.FilmEngine = LatentDraftEngine
    engine, table, valid, cfg = runner.build(equal=True, reply=True, film=True, capture=False,
                                  validation_bytes=4096)
    cfg['stage'] = 'same-state real-world one-turn action forks'
    cfg['live_sigreg_near'] = args.live_sigreg_near
    cfg['fork_every'] = args.fork_every
    cfg['validation_every'] = args.validation_every
    for path in (Path(__file__), Path('scripts/ib_local/language_navigation_kanli_v4k.py'),
                 Path('scripts/ib_local/language_navigation_kanli_v4l.py')):
        cfg['sources'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    original_film, original_take = engine._film, engine.world.take
    candidates, turn, observed = [], {}, {}

    def capture_film(words, scratch):
        result = original_film(words, scratch)
        candidates.append(result[0])
        return result

    def branch(words):
        world = copy.copy(engine.world)
        world.seen = engine.world.seen.copy()
        world.rng = np.random.default_rng()
        world.rng.bit_generator.state = copy.deepcopy(engine.world.rng.bit_generator.state)
        # Shallow copies inherit the live instance's patched `take`; invoke the
        # world class implementation explicitly to avoid recursive probing.
        response, _, info = type(world).take(world, words, engine.response_bytes, engine.holdout_bytes)
        scratch = copy.deepcopy(turn['table'])
        byte, probs = turn['pending'], []
        for nxt in words:
            _, p, _ = type(scratch).run_credit(scratch, np.array([byte, nxt], np.uint8), max_window=1)
            probs.append(p[0]); byte = nxt
        _, p, _ = type(scratch).run_credit(scratch,
            np.concatenate((np.array([byte], np.uint8), response[:33])), max_window=33)
        probs = np.concatenate((np.asarray(probs, np.float32).reshape(-1, 256), p), 0)
        raw = np.concatenate((np.array([turn['pending']], np.uint8), np.frombuffer(words, np.uint8), response[:33]))
        ids, targets, logp = engine._inputs(raw, probs)
        with torch.no_grad():
            logits, _, _, h, _ = engine.model.predict_observed(ids, turn['state'].clone(), logp)
            ce = F.cross_entropy(logits[0, len(words):], targets[0, len(words):], reduction='none')
            zsource = engine.model.encoder(h[0][None])
            zaction = engine.model.encoder(h[len(words)][None])
            # Match the actual near-target index used during training.
            zfuture = engine.model.encoder(h[len(words)+32][None])
            predicted = engine.model.kanli.outcome(torch.cat((zsource, zaction), -1))
        return dict(info=info, response_hash=hashlib.sha256(response.tobytes()).hexdigest(),
                    nll4=float(ce[:4].mean()), nll32=float(ce[:32].mean()),
                    action=zaction[0], future=zfuture[0], predicted=predicted[0])

    def take_and_fork(utterance, response_bytes, holdout_bytes):
        cpu, gpu = torch.get_rng_state(), torch.cuda.get_rng_state()
        before = [p._version for p in engine.model.parameters()]
        # Generate vs reuse if possible; otherwise generate vs silence.
        first = candidates[1]
        second = candidates[2] if len(candidates) == 3 else candidates[0]
        a, b = branch(first), branch(second)
        own = a if utterance == first else b if utterance == second else branch(utterance)
        response, heldout, info = original_take(utterance, response_bytes, holdout_bytes)
        actual = b if utterance == second and utterance != first else a if utterance == first else own
        actual_delta, predicted_delta = a['future']-b['future'], a['predicted']-b['predicted']
        cosine = float(F.cosine_similarity(actual_delta[None], predicted_delta[None]).item())
        observed.clear()
        observed.update(kind=('generate_vs_reuse' if len(candidates)==3 else 'generate_vs_silence'),
            different_words=first != second, world_start_a=a['info']['start'], world_start_b=b['info']['start'],
            match_a=a['info']['match'], match_b=b['info']['match'],
            different_response=a['response_hash'] != b['response_hash'],
            nll4_a=a['nll4'], nll4_b=b['nll4'], nll32_a=a['nll32'], nll32_b=b['nll32'],
            action_delta=float((a['action']-b['action']).norm()),
            future_delta=float(actual_delta.norm()), predicted_delta=float(predicted_delta.norm()),
            delta_cosine=cosine, prediction_delta_error=float((predicted_delta-actual_delta).norm()),
            own_response_reproduced=(actual['response_hash']==hashlib.sha256(response.tobytes()).hexdigest()
                                     and actual['info']['start']==info['start']),
            params_unchanged=before==[p._version for p in engine.model.parameters()],
            torch_rng_unchanged=bool(torch.equal(cpu,torch.get_rng_state()) and
                                     torch.equal(gpu,torch.cuda.get_rng_state())))
        return response, heldout, info

    engine._film = capture_film
    rows, start = [], time.perf_counter()
    validations = []
    if args.validation_every:
        validations.append(dict(cycle=0, nll=validate(engine.model, table, valid)))
    torch.cuda.reset_peak_memory_stats()
    for i in range(args.cycles):
        candidates.clear()
        if i % args.fork_every == 0:
            turn.update(state=engine.recurrent.clone(), pending=engine.pending, table=copy.deepcopy(table))
            engine.world.take = take_and_fork
        else:
            engine.world.take = original_take
            observed.clear()
        event = engine.step(torch.randn(64, 32, device='cuda'))
        turn.pop('table', None)
        rows.append(dict(choice=event['choice'], fork=dict(observed), world=event['world'],
                         response_nll=event['response_nll'], metal_g=event['metal_g'],
                         policy=event['choice_probability'], updates=event['updates'],
                         memory_occupied=int(engine.model.memory.occupied.sum()),
                         memory_episodes=int(engine.model.memory.episodes), sigreg=getattr(engine.model, '_last_sigreg', None),
                         policy_effect=dict(engine.draft_diagnostics)))
        if args.validation_every and ((i+1) % args.validation_every == 0 or i+1 == args.cycles):
            validations.append(dict(cycle=i+1, nll=validate(engine.model, table, valid)))
        if (i+1) % 8 == 0:
            print(json.dumps(dict(cycle=i+1, elapsed_s=round(time.perf_counter()-start, 2))), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(config=cfg, cycles=args.cycles,
        elapsed_s=time.perf_counter()-start, peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        validation_bytes=len(valid)-1, validations=validations, records=rows), indent=2), encoding='utf-8')
    print(json.dumps(dict(cycles=len(rows), output=str(args.output))), flush=True)


if __name__ == '__main__':
    main()
