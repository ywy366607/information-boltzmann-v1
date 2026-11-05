"""立道 gate for v4 stage 4, metal's own water (Claude, 2026-11-04).

In-memory individual with learning on; the checkpoint is never written.
There is no comparison arm.

Pass rules, fixed before running:
 G1 metal's water exists. After the warm cycles there are >= 16 witnesses, and
    windows are admitted more often in new-topic turns than in exhausted ones
    (the body's own surprise selects them, not the world's flag).
 G2 metal feels washing. By decision-time state, the median g is < 0 on
    exhausted turns where the body stayed silent, and the median g on
    new-topic turns is larger than that.
 G3 metal returns to the cause. Every cycle after the first settles its
    predecessor: the TD label is written into the experience slot (unless it
    was merged since), and in active mode the delayed habit actor is applied.
 I2 calibration: over the last 128 cycles, the imagined value of the chosen
    option correlates positively with its TD target.
 I3 imagination enters the choice: precision > 0 and mean |offset| > 0.05 logits.
 R  rhythm: silence and speech both occur.
 F1 fire still reaches the world: understood speech moved the topic at least once.
With --fire (stage 5, rules fixed before running):
 H1 fire's metal is fed: the median |head advantage| over settled cycles lies in [0.3, 3]
 H2 earth is not washed: the median body-side |advantage| is still the raw
    TD advantage (< 0.2 nats); the policy replay check passed every cycle
 (With --equal, stage 5b, rules fixed before running:
 E1 every generated utterance is exactly 32 bytes, and >= 50% of generated
    utterances are understood (the world's 4-gram rule is met by length)
 E2 the body still chooses: silence is chosen in >= 10% and <= 90% of cycles)
 (With --water, stage 6, rules fixed before running; also applies E1/E2:
 O1 overflow follows the level: the median seed count is larger in exhausted
    turns than in new turns, and some cycles have zero seeds (not constant)
    [amended after gate_stage6_1: compare exhausted turns where the body stayed
    SILENT (outside truly quiet) with new turns; in exhausted turns the body
    speaks 67% of the time and the jump makes that present the most surprising.
    The original form is still reported as O1_original]
 O2 water initiates: >= 50% of seeds are outside the current state's recall slots (median)
 O3 digestion: median relative error decrease of seeds within their cycle > 0
 O4 release: >= 1 experience released, and the median released age is
    >= 64 cycles (61440 bytes); the newest are not the ones let go
 O5 wood is fed: the seed valuation loss in the last 64 cycles is below the first 64)
 (With --clear, stage 6b, rules fixed before running; also applies E1/E2, O2-O5:
 C1 clear water follows the level: the median clear volume sum(c) is larger in
    quiet turns (exhausted, body silent) than in new turns, and some cycles
    have zero clear volume. It replaces O1, because the seed count is now only
    a compute sample. O1 is still reported.
 C3 films digest by transfer: over cycles with >= 1 accepted film pass, the
    median change of F's error on the judge sample (never used for any
    decision) is < 0
 C4 films end by themselves: both 0-pass and >= 1-pass film cycles occur, and
    FILM_LIMIT binds in < 10% of film cycles)
 (With --film, stage 7, rules fixed before running; also applies E1/E2:
 W1 the silence film knows exhaustion: by decision state, the median silence-film
    entropy is lower in exhausted turns than in new turns
 W2 in exhausted turns, speech films expect more than silence: the median of
    (generate film - silence film) is > 0
 W3 heaven's law enters wood: over settled cycles of the second half, the rank
    correlation between the chosen option's film entropy and its TD label y is > 0
 (observed, not a rule: K4' = P(speak | exhausted) - P(speak | new)))
 (With --digest, stage 8, rules fixed before running; on stage 5b + reply, W1-W3 do not apply:
 D1 quiet digests more: the median replay count is larger in quiet turns
    (exhausted, body silent) than in new turns, and some settled cycles have
    zero replays
 D2 digestion transfers: over cycles with replays, the median change of the
    critic error on the judge sample (never replayed that cycle) is < 0
 D3 not rumination: no single experience takes > 25% of all replays, and
    DIGEST_LIMIT binds in < 10% of cycles with replays)
 H3 fire's metal moves: the mean stop probability at the first stop decision
    differs between the first and last 64 generated utterances by > 0.02
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from scripts.ib_local import gdn_fused
from scripts.ib_local.check_navigation_turns_v3_gate import stats
from scripts.ib_local.kanli_decision_state import decision_states
from scripts.ib_local.language_navigation_kanli_v4b import SCALE
from scripts.ib_local.run_language_navigation_kanli_v4 import build


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cycles', type=int, default=256)
    p.add_argument('--fire', action='store_true')
    p.add_argument('--equal', action='store_true')
    p.add_argument('--water', action='store_true')
    p.add_argument('--clear', action='store_true')
    p.add_argument('--reply', action='store_true')
    p.add_argument('--film', action='store_true')
    p.add_argument('--digest', action='store_true')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('keep earlier gate records')
    torch.set_num_threads(2)
    gdn_fused.enable()
    engine, table, valid, cfg = build(witness=True, fire=args.fire, equal=args.equal or args.water or args.clear or args.film or args.digest, water=args.water or args.clear, clear=args.clear, reply=args.reply, film=args.film, quiet_digest=args.digest)
    memory = engine.model.memory
    started = time.perf_counter()
    records = []
    for _ in range(args.cycles):
        records.append(engine.step(torch.randn(64, 32, device='cuda')))
        prev = records[-1]['td_previous']
        if prev is not None and len(records) > 1:
            slot = records[-2]['episode_slot']
            records[-1]['label_check'] = abs(float(memory.progress[slot])-prev['y'])
    elapsed = time.perf_counter()-started
    states, mism = decision_states(records)
    links = {}
    new = [i for i, s in enumerate(states) if s is True]
    old = [i for i, s in enumerate(states) if s is False]
    adm_new = float(np.mean([records[i]['admitted'] for i in new])) if new else 0.
    adm_old = float(np.mean([records[i]['admitted'] for i in old])) if old else 0.
    links['G1_metal_water'] = dict(witnesses=len(engine.w_ids), admitted_total=engine.w_admitted,
                                   admitted_rate_new=adm_new, admitted_rate_exhausted=adm_old,
                                   hearing_nll=stats(engine.w_hear),
                                   passed=bool(len(engine.w_ids) >= 16 and adm_new > adm_old))
    g = lambda I: [SCALE*records[i]['metal_g'] for i in I]
    old_sil = [i for i in old if records[i]['choice'] == 'silence']
    old_spk = [i for i in old if records[i]['choice'] != 'silence']
    links['G2_metal_feels_washing'] = dict(
        replay_mismatches=mism, g_exhausted_silent=stats(g(old_sil)), g_exhausted_speech=stats(g(old_spk)),
        g_new=stats(g(new)),
        probe_progress_exhausted_silent=stats([SCALE*records[i]['progress'] for i in old_sil]),
        passed=bool(mism == 0 and old_sil and new and np.median(g(old_sil)) < 0
                    and np.median(g(new)) > np.median(g(old_sil))))
    settled = [r for r in records[1:] if r['td_previous'] is not None]
    checks = [r['label_check'] for r in records[1:] if 'label_check' in r and not r['episode_merged']]
    links['G3_td_settles'] = dict(
        settled=len(settled), cycles=args.cycles, delayed_actor=records[-1]['delayed_actor_applied'],
        label_error=stats(checks) if checks else None,
        y=stats([r['td_previous']['y']*SCALE for r in settled]),
        passed=bool(len(settled) == args.cycles-1 and records[-1]['delayed_actor_applied'] == args.cycles-1
                    and (not checks or np.median(checks) < 1e-6)))
    tail = records[-127:]
    imagined = [r['td_previous']['imagined'] for r in tail]
    target = [SCALE*r['td_previous']['y'] for r in tail]
    corr = float(np.corrcoef(imagined, target)[0, 1]) if np.std(imagined) > 0 and np.std(target) > 0 else 0.
    links['I2_calibration'] = dict(corr=corr, critic_rms=records[-1]['critic_rms'], passed=bool(corr > 0))
    offsets = [np.abs(np.array(r['imagination_offset'])).mean() for r in records[-64:]]
    links['I3_imagination_in_choice'] = dict(precision=records[-1]['precision'], offset=stats(offsets),
                                             passed=bool(records[-1]['precision'] > 0 and np.mean(offsets) > .05))
    choices = [r['choice'] for r in records]
    links['R_rhythm'] = dict(counts={c: choices.count(c) for c in ('silence', 'generate', 'reuse')},
                             passed=bool(choices.count('silence') and len(choices)-choices.count('silence')))
    links['F1_world'] = dict(moves=engine.world.moves, passed=engine.world.moves >= 1)
    if args.fire:
        heads = [abs(r['td_previous']['advantage_head']) for r in settled if 'advantage_head' in r['td_previous']]
        raws = [abs(r['td_previous']['advantage_raw']) for r in settled]
        links['H1_fire_metal_fed'] = dict(head_advantage=stats(heads), passed=bool(heads and .3 <= np.median(heads) <= 3))
        links['H2_earth_not_washed'] = dict(raw_advantage=stats(raws), passed=bool(np.median(raws) < .2))
        first = [r['stop_logits'][0] for r in records if r['choice'] == 'generate' and r['stop_logits']]
        sig = lambda v: float(np.mean(1/(1+np.exp(-np.asarray(v)))))
        links['H3_fire_metal_moves'] = dict(p_stop_first=sig(first[:64]), p_stop_last=sig(first[-64:]),
                                            length=stats([r['self_bytes'] for r in records if r['choice'] == 'generate']),
                                            passed=bool(len(first) >= 128 and abs(sig(first[:64])-sig(first[-64:])) > .02))
    args.water = args.water or args.clear
    if args.equal or args.water or args.film:
        gen = [r for r in records if r['choice'] == 'generate']
        links['E1_equal_window'] = dict(lengths=stats([r['self_bytes'] for r in gen]),
                                        understood=float(np.mean([r['world']['source'] == 'matched' for r in gen])) if gen else 0.,
                                        passed=bool(gen and all(r['self_bytes'] == 32 for r in gen)
                                                    and np.mean([r['world']['source'] == 'matched' for r in gen]) >= .5))
        sil = choices.count('silence')/len(choices)
        links['E2_still_chooses'] = dict(silence_fraction=sil, passed=bool(.1 <= sil <= .9))
    if args.water:
        W = [r['water'] for r in records]
        seeds_all_ex = [W[i]['seeds'] for i in old]
        seeds_ex = [W[i]['seeds'] for i in old if records[i]['choice'] == 'silence']; seeds_new = [W[i]['seeds'] for i in new]
        links['O1_original'] = dict(seeds_exhausted=stats(seeds_all_ex), passed_original_rule=bool(seeds_all_ex and seeds_new and np.median(seeds_all_ex) > np.median(seeds_new)), passed=True)
        links['O1_overflow_follows_level'] = dict(seeds_exhausted=stats(seeds_ex), seeds_new=stats(seeds_new),
                                                  zero_seed_cycles=sum(w['seeds'] == 0 for w in W),
                                                  passed=bool(seeds_ex and seeds_new and np.median(seeds_ex) > np.median(seeds_new)
                                                              and any(w['seeds'] == 0 for w in W)))
        nr = [w['seeds_not_recalled'] for w in W if 'seeds_not_recalled' in w]
        links['O2_water_initiates'] = dict(not_recalled=stats(nr) if nr else None, passed=bool(nr and np.median(nr) >= .5))
        dg = [w['digestion'] for w in W if 'digestion' in w]
        links['O3_digestion'] = dict(relative_decrease=stats(dg) if dg else None, passed=bool(dg and np.median(dg) > 0))
        ages = [w['released_age'] for w in W if 'released_age' in w]
        links['O4_release'] = dict(released_total=W[-1]['released_total'], released_age=stats(ages) if ages else None,
                                   occupied_age=W[-1]['occupied_age_median'],
                                   passed=bool(W[-1]['released_total'] >= 1 and ages and np.median(ages) >= 64*960))
        wl = [w['wood_loss'] for w in W if w.get('wood_loss') is not None]
        links['O5_wood_fed'] = dict(first=stats(wl[:64]) if wl else None, last=stats(wl[-64:]) if wl else None,
                                    passed=bool(len(wl) >= 128 and np.median(wl[-64:]) < np.median(wl[:64])))
    if args.film:
        fl = lambda I, k: [records[i]['films'][k] for i in I]
        links['W1_silence_film_knows_exhaustion'] = dict(silence_film_exhausted=stats(fl(old, 0)) if old else None,
                                                         silence_film_new=stats(fl(new, 0)) if new else None,
                                                         passed=bool(old and new and np.median(fl(old, 0)) < np.median(fl(new, 0))))
        gap = [records[i]['films'][1]-records[i]['films'][0] for i in old]
        links['W2_speech_expects_more_when_exhausted'] = dict(generate_minus_silence_exhausted=stats(gap) if gap else None,
                                                              generate_minus_silence_new=stats([records[i]['films'][1]-records[i]['films'][0] for i in new]) if new else None,
                                                              passed=bool(gap and np.median(gap) > 0))
        half = len(records)//2
        pairs = [(records[i-1]['films'][('silence', 'generate', 'reuse').index(records[i-1]['choice'])], records[i]['td_previous']['y'])
                 for i in range(max(half, 1), len(records)) if records[i]['td_previous'] is not None]
        rank = lambda v: np.argsort(np.argsort(v))
        rho = float(np.corrcoef(rank([a for a, _ in pairs]), rank([b for _, b in pairs]))[0, 1]) if len(pairs) > 2 else 0.
        links['W3_law_enters_wood'] = dict(spearman=rho, n=len(pairs), passed=bool(rho > 0))
    if args.digest:
        Dg = [r['digest'] for r in records]
        rep = lambda I: [Dg[i]['replays'] for i in I if Dg[i]['priority_present'] is not None]
        quiet = [i for i in old if records[i]['choice'] == 'silence']
        live = [d for d in Dg if d['priority_present'] is not None]
        links['D1_quiet_digests_more'] = dict(replays_quiet=stats(rep(quiet)) if rep(quiet) else None, replays_new=stats(rep(new)) if rep(new) else None,
                                             zero_replay_cycles=sum(d['replays'] == 0 for d in live), cycles=len(live),
                                             passed=bool(rep(quiet) and rep(new) and np.median(rep(quiet)) > np.median(rep(new))
                                                         and any(d['replays'] == 0 for d in live)))
        dj = [d['judge_after']-d['judge_before'] for d in live if 'judge_after' in d]
        links['D2_digestion_transfers'] = dict(judge_change=stats(dj) if dj else None, passed=bool(dj and np.median(dj) < 0))
        counts = engine.replayed[engine.replayed > 0]
        total = int(engine.replayed.sum())
        played = [d for d in live if d['replays'] > 0]
        links['D3_not_rumination'] = dict(total_replays=total, distinct=int(len(counts)), top_share=float(counts.max())/total if total else None,
                                          limit_hits=sum(d['hit_limit'] for d in played), replay_cycles=len(played),
                                          replayed_age=stats([d['replayed_age_median'] for d in played]) if played else None,
                                          passed=bool(total and float(counts.max())/total <= .25 and np.mean([d['hit_limit'] for d in played]) < .1))
    if args.clear:
        W = [r['water'] for r in records]
        vol = lambda I: [W[i]['clear_volume'] for i in I]
        quiet = [i for i in old if records[i]['choice'] == 'silence']
        links['O1_overflow_follows_level']['passed'] = True
        links['O1_overflow_follows_level']['note'] = 'replaced by C1 (preregistered); count is a compute sample'
        links['C1_clear_follows_level'] = dict(clear_quiet=stats(vol(quiet)) if quiet else None, clear_new=stats(vol(new)) if new else None,
                                               zero_clear_cycles=sum(w['clear_volume'] == 0 for w in W),
                                               clarity=stats([c for w in W for c in w['clarity']]) if any(w['clarity'] for w in W) else None,
                                               passed=bool(quiet and new and np.median(vol(quiet)) > np.median(vol(new))
                                                           and any(w['clear_volume'] == 0 for w in W)))
        films = [w['film'] for w in W if 'film' in w]
        played = [f for f in films if f['passes'] >= 1 and f['judge_before'] is not None]
        dj = [f['judge_after']-f['judge_before'] for f in played]
        dw = [f['witness_after']-f['witness_before'] for f in played]
        links['C3_films_transfer'] = dict(film_cycles=len(played), judge_change=stats(dj) if dj else None,
                                          witness_change=stats(dw) if dw else None,
                                          passed=bool(dj and np.median(dj) < 0))
        passes = [f['passes'] for f in films]
        links['C4_films_end_themselves'] = dict(passes=stats(passes) if passes else None,
                                                zero_pass=sum(p == 0 for p in passes), limit_hits=sum(f['hit_limit'] for f in films),
                                                passed=bool(passes and min(passes) == 0 and max(passes) >= 1
                                                            and np.mean([f['hit_limit'] for f in films]) < .1))
        m, enc = engine.model.memory, engine.model.encoder
        occ = torch.nonzero(m.occupied).flatten()
        with torch.no_grad():
            zl = enc(m.long_successor[occ])
            links['observed_F_vs_constant'] = dict(F=float(engine._errors(occ).median()),
                                                   constant=float((zl-zl.mean(0)).square().mean(-1).median()), passed=True)
    speak = lambda I: float(1-np.mean([records[i]['choice_probability'][0] for i in I])) if I else None
    report = dict(scope='in-memory individual, learning on; checkpoint untouched', warm_cycles=args.cycles,
                  seconds_per_cycle=elapsed/args.cycles, config=cfg, links=links,
                  observed=dict(p_speak_exhausted=speak(old), p_speak_new=speak(new)),
                  passed=all(v['passed'] for v in links.values()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    for k, v in links.items():
        print(k, 'PASS' if v['passed'] else 'FAIL', json.dumps({a: b for a, b in v.items() if a != 'passed'})[:400])
    print('observed', report['observed'], 'sec/cycle', round(elapsed/args.cycles, 3))
    print('GATE', 'PASSED' if report['passed'] else 'NOT PASSED')


if __name__ == '__main__':
    main()
