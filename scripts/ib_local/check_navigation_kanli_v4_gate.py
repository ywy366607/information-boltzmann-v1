"""立道 gate for v4 stage 1, 水火既济 (Claude, 2026-11-04). In-memory individual,
learning on for the warm cycles, checkpoint never written; forward / single
backward measurements afterwards. No comparison arm.

Pass rules fixed before running:
 F1 fire descends, world: understood speech moved the topic at least once and
    the turn after a move was at fresh positions
 F2 fire descends, memory: one experience per cycle; words stored with spoken experiences
 F3 fire descends, model: the action-conditioned outcome model uses the action:
    over the last 64 cycles, error with actions shuffled exceeds the true-action
    error by > 1% (median relative excess)
 W1 water rises: reuse of recalled words was chosen at least once
 W2 water rises: recall changes WHAT is said in >= 1 of 128 seeded samples
    (memory on vs off, same seeds), and changes the choice distribution (KL > 0)
 E  earth: the difference two utterances make to the recurrent state is still
    >= 1% after the reply (median over pairs)
 M  metal: some cycles carry usable progress, and silent cycles inside an
    exhausted topic carry none (the bounded premise is real)
 R  rhythm: silence and generated speech both occur
With --imagine (stage 2, rules fixed before running):
 I1 interoception: the reply surprise felt in exhausted-topic turns differs
    from fresh turns by > 0.1 nats (median)
 I2 metal calibrates wood: over the last 128 cycles the imagined value of the
    chosen option correlates positively with the realized value
 I3 imagination enters the choice: precision > 0 and mean |offset| > 0.05 logits
    over the last 64 cycles
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.ib_local import gdn_fused
from scripts.ib_local.check_navigation_turns_v3_gate import clone_table, memory_off, run_bytes, stats
from scripts.ib_local.run_language_navigation_kanli_v4 import build


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cycles', type=int, default=256)
    p.add_argument('--imagine', action='store_true')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('keep earlier gate records')
    torch.set_num_threads(2)
    gdn_fused.enable()
    engine, table, valid, cfg = build(imagine=args.imagine)
    model, memory, world = engine.model, engine.model.memory, engine.world
    records = [engine.step(torch.randn(64, 32, device='cuda')) for _ in range(args.cycles)]
    links = {}
    moved = [i for i, r in enumerate(records) if r['world']['source'] == 'matched']
    links['F1_world'] = dict(moves=len(moved), moved_turn_fresh=[records[i]['fresh'] for i in moved][:20],
                             passed=bool(moved and any(records[i]['fresh'] for i in moved)))
    spoken = [r for r in records if r['spoke']]
    links['F2_memory'] = dict(episodes=int(memory.episodes), cycles=args.cycles,
                              words_stored=int((memory.word_count > 0).sum()), spoken=len(spoken),
                              passed=bool(int(memory.episodes) == args.cycles and int((memory.word_count > 0).sum()) > 0))
    tail = records[-64:]
    excess = [(r['outcome_error_shuffled']-r['outcome_error'])/r['outcome_error'] for r in tail]
    links['F3_outcome_model'] = dict(relative_excess_shuffled=stats(excess),
                                     error=stats([r['outcome_error'] for r in tail]),
                                     error_as_silent=stats([r['outcome_error_as_silent'] for r in tail]),
                                     passed=bool(np.median(excess) > .01))
    choices = [r['choice'] for r in records]
    links['W1_reuse'] = dict(counts={c: choices.count(c) for c in ('silence', 'generate', 'reuse')},
                             reuse_available=float(np.mean([r['choice_probability'][2] > 1e-6 for r in records])),
                             passed=choices.count('reuse') >= 1)
    # W2: memory on/off, same seeds, same table clone.
    differ, kls = 0, []
    original = engine.table
    with torch.no_grad():
        for s in range(128):
            outs = []
            for disabled in (False, True):
                engine.table = clone_table(original)
                saved = memory_off(memory) if disabled else None
                torch.cuda.manual_seed(7000+s); torch.manual_seed(7000+s)
                utt, rec = engine._utter()
                if saved is not None:
                    memory.occupied.copy_(saved)
                outs.append((utt, rec['choice_logits']))
            differ += outs[0][0] != outs[1][0]
            if s == 0:
                a = torch.tensor(outs[0][1]).softmax(-1); b = torch.tensor(outs[1][1]).softmax(-1)
                kls.append(float((a*(a.clamp_min(1e-12)/b.clamp_min(1e-12)).log()).sum()))
    engine.table = original
    links['W2_recall_changes_words'] = dict(changed=f'{differ}/128', choice_kl=kls[0],
                                            passed=bool(differ >= 1 and kls[0] > 0))
    ref = world.data[world.anchor:world.anchor+481].copy()
    pool = [bytes.fromhex(r['utterance_hex']) for r in spoken if r['self_bytes'] >= 2][:17]
    ratios = []
    with torch.no_grad():
        for a, b in zip(pool[:-1], pool[1:]):
            if a == b:
                continue
            ends, mids = [], []
            for u in (a, b):
                raw = np.concatenate(([engine.pending], np.frombuffer(u, np.uint8), ref[:480-len(u)])).astype(np.uint8)
                ids, targets, logp = run_bytes(engine, raw, engine.recurrent)
                _, s_mid, _ = model.predict(ids[:, :1+len(u)], engine.recurrent, logp[:, :1+len(u)])
                _, s_end, _ = model.predict(ids, engine.recurrent, logp)
                mids.append(s_mid); ends.append(s_end)
            ratios.append(float((ends[0]-ends[1]).norm()/(mids[0]-mids[1]).norm().clamp_min(1e-12)))
    links['E_intent_held'] = dict(retention_ratio=stats(ratios), passed=bool(ratios and np.median(ratios) >= .01))
    silent_stale = [r['usable_signal'] for r in records if r['choice'] == 'silence' and not r['fresh']]
    links['M_progress'] = dict(usable_fraction=float(np.mean([r['usable_signal'] != 0 for r in records])),
                               fresh_fraction=float(np.mean([r['fresh'] for r in records])),
                               silent_stale=len(silent_stale), silent_stale_nonzero=int(np.count_nonzero(silent_stale)),
                               progress_fresh=stats([r['progress'] for r in records if r['fresh']]),
                               passed=bool(any(r['usable_signal'] != 0 for r in records) and silent_stale
                                           and not np.count_nonzero(silent_stale)))
    lengths = [r['self_bytes'] for r in records if r['choice'] == 'generate']
    links['R_rhythm'] = dict(length=stats(lengths), passed=bool(choices.count('silence') and lengths))
    if args.imagine:
        stale = [r['response_nll'] for r in records if not r['fresh']]
        fresh = [r['response_nll'] for r in records if r['fresh']]
        gap = float(np.median(fresh)-np.median(stale)) if stale and fresh else 0.
        links['I1_interoception'] = dict(reply_nll_fresh=stats(fresh), reply_nll_stale=stats(stale), gap=gap,
                                         passed=bool(abs(gap) > .1))
        last = records[-128:]
        imagined = [r['imagined_value'][('silence', 'generate', 'reuse').index(r['choice'])] for r in last]
        realized = [r['target_value'] for r in last]
        corr = float(np.corrcoef(imagined, realized)[0, 1]) if np.std(imagined) > 0 and np.std(realized) > 0 else 0.
        links['I2_calibration'] = dict(corr=corr, critic_rms=records[-1]['critic_rms'], passed=bool(corr > 0))
        offsets = [np.abs(np.array(r['imagination_offset'])).mean() for r in records[-64:]]
        links['I3_imagination_in_choice'] = dict(precision=records[-1]['precision'], offset=stats(offsets),
                                                 passed=bool(records[-1]['precision'] > 0 and np.mean(offsets) > .05))
    report = dict(scope='in-memory individual, learning on for warm cycles; checkpoint untouched',
                  warm_cycles=args.cycles, config=cfg, links=links,
                  passed=all(v['passed'] for v in links.values()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    for k, v in links.items():
        print(k, 'PASS' if v['passed'] else 'FAIL', json.dumps({a: b for a, b in v.items() if a != 'passed'})[:300])
    print('GATE', 'PASSED' if report['passed'] else 'NOT PASSED')


if __name__ == '__main__':
    main()
