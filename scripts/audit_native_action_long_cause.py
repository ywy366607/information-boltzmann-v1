"""Replay real action -> text navigation -> observed gain -> delayed credit.

Uses completed full-body branches, original corpus and production scalar/world
functions. No model reconstruction, optimization, or external interaction.
"""
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from scripts.ib_local.language_navigation_kanli_v4c import GAMMA
from scripts.ib_local.navigation_gain_units import td_gain_target, wood_choice_logits
from scripts.ib_local.text_navigation import stable_suffix
from scripts.ib_local.world_learning_cycle import settle_sources

ROOT = Path(__file__).resolve().parents[1]


def read_events(folder):
    rows = []
    worlds = {}
    path = folder / 'observations/operations.jsonl.gz'
    with gzip.open(path, 'rt', encoding='utf8') as stream:
        for line in stream:
            r = json.loads(line)
            if r['kind'] == 'world':
                worlds[r['event']] = r['data']
            if r['kind'] == 'event' and r['data']['autonomous_event']['kind'] == 'external':
                rows.append(dict(event=r['event'], update_clock=r['update_clock'], **r['data']))
    return rows, worlds


def main():
    torch.set_num_threads(1)
    base = ROOT / 'scratch/native_identity_preview_bound_2401_20261117'
    meta = json.loads((base / 'baseline/body/last.metadata.json').read_text(encoding='utf8'))
    cfg = meta['config']
    token_root = ROOT / 'data/ib_owt_10000'
    offset = json.loads((token_root / 'tokenizer.json').read_text())['offset']
    train = np.load(token_root / 'train.npy', mmap_mode='r')
    start, stop = cfg['source_raw_bounds']
    parts = []
    for begin in range(start, min(stop, len(train)), 1_000_000):
        raw = train[begin:min(begin + 1_000_000, stop, len(train))]
        parts.append((raw[raw >= offset] - offset).astype(np.uint8))
    parts.append(np.fromfile(cfg['natural_text_corpus']['train_path'], dtype=np.uint8))
    data = np.concatenate(parts)
    del parts, train
    digest = hashlib.sha256(data).hexdigest()
    assert digest == cfg['birth']['source_sha256']
    result = dict(source_sha256=digest, gamma=GAMMA, arms={}, new_updates=0, new_world_calls=0,
                  discount_weights={str(n): GAMMA**n for n in (1, 2, 10, 20, 50)})
    labels = ('silence', 'generate', 'reuse')
    for arm in ('baseline', 'bridge_cut', 'matched_cut', 'order_only'):
        events, worlds = read_events(base / arm)
        checks = []
        for i, e in enumerate(events):
            cursor = e['world_before']['anchor'] + e['world_before']['offset']
            if cursor + cfg['topic_window'] > len(data):
                cursor = 0
            words = bytes.fromhex(e['utterance_hex'])
            pos, match, ties, distance = map(int, stable_suffix(
                np.frombuffer(words, np.uint8), data, cfg['topic_window'], cursor))
            pos = pos if match else cursor
            w = worlds[e['event']]
            delivered = bytes.fromhex(w['response_hex']) + bytes.fromhex(w['holdout_hex'])
            assert pos == e['world']['start'] and match == e['world']['match']
            assert data[pos:pos + len(delivered)].tobytes() == delivered
            engine = SimpleNamespace(gamma=GAMMA, gain_units_schema=1,
                                     precision=lambda: e['precision'])
            g = settle_sources(e['world_learning']['readings'])
            assert g == e['settlement']
            assert abs(e['wood_target'][3] - 10 * g) < 1e-6
            td = e['td_previous']
            target = td_gain_target(engine, td['g'], e['value_now'])
            assert abs(target - td['y']) < 1e-10
            previous = events[i-1] if i else None
            if previous:
                assert td['g'] == previous['settlement']
                assert td['choice'] == labels.index(previous['choice'])
                assert td['imagined'] == previous['imagined_value'][td['choice']]
                two = e['internal_rollout']['settled_two_turn']
                assert abs(two['target'] - (previous['wood_target'][3] + GAMMA * e['wood_target'][3])) < 1e-6
            else:
                two = None  # Initial credit belongs to the checkpoint's predecessor.
            search = e['joint_language_search']
            # Event precision is reported AFTER this event's critic update.
            # Decision precision is the preceding event's value; at the first
            # checkpoint boundary recover its single coefficient from logged
            # probabilities and utilities, recording the fit residual.
            u = np.array(search['predicted_after_value'], dtype=float)
            logp = np.log(search['distribution'])
            uc, pc = u-u.mean(), logp-logp.mean()
            inferred_precision = float(uc@pc/(uc@uc)/10) if uc@uc else 0.
            decision_precision = previous['precision'] if previous else inferred_precision
            engine.precision = lambda: decision_precision
            logits = wood_choice_logits(engine, torch.tensor(search['predicted_after_value']))
            prob = logits.softmax(-1).tolist()
            probability_error = float(np.max(np.abs(np.array(prob) - search['distribution'])))
            assert probability_error < 2e-7, (arm,i,probability_error)
            proposals = []
            for candidate in search['candidates']:
                q = bytes.fromhex(candidate)
                p, s, n, d = map(int, stable_suffix(np.frombuffer(q, np.uint8), data, cfg['topic_window'], cursor))
                proposals.append(dict(bytes=len(q), start=p if s else cursor, match=s,
                                      ties=n, distance=d))
            raw_gain = np.array(e['wood_raw'])[3]
            base_gain = np.array(e['wood_base'])[3]
            confidence = np.array(e['wood_all_confidence'])[:, 3].tolist()
            # The continuation's actual readout is a convex combination of
            # action-category bases whenever all contextual gain confidences
            # are zero. Validate that consequence on the recorded futures.
            bases = np.array(e['wood_all_bases'])[:, 3]
            rollout = e['internal_rollout']
            future = np.array(rollout['predicted_future'])
            support = np.array(rollout['support'], dtype=bool)
            if not any(confidence):
                assert np.all(future[support] >= bases.min() - 1e-6)
                assert np.all(future[support] <= bases.max() + 1e-6)
            assert np.isfinite(e['wood_g_gradient_norm'])
            checks.append(dict(event=e['event'], choice=e['choice'], words=words.hex(),
                source_cursor=cursor, actual_start=pos, silent_start=cursor,
                jump=pos-cursor, match=match, ties=ties, action_bytes=len(words),
                response_nll=e['response_nll'], pre=e['heldout_pre_nll'], post=e['heldout_post_nll'], gain=g,
                previous_credit=dict(g=td['g'], bootstrap=td['y']-td['g'], target=td['y'],
                                     advantage=td['advantage'], choice=td['choice']),
                two_turn=two, candidate_worlds=proposals,
                draft_executed=search['executed'], draft_probability=prob,
                decision_precision=decision_precision, post_learning_precision=e['precision'],
                probability_reconstruction_error=probability_error,
                draft_tv_uniform=float(np.abs(np.array(prob)-1/len(prob)).sum()/2),
                raw_gain_prediction=float(raw_gain), baseline_gain_prediction=float(base_gain),
                gain_target=e['wood_target'][3], gain_confidence=confidence,
                action_category_gain_bases=bases.tolist(),
                predicted_future=future.tolist(), future_support=support.tolist(),
                gain_update_loss=e['wood_g_update_loss'],
                gain_gradient_norm=e['wood_g_gradient_norm'],
                internal_choice_tv=e['internal_rollout']['choice_probability_tv']))
        targets = np.array([c['gain_target'] for c in checks])
        twos = [c['two_turn'] for c in checks if c['two_turn'] is not None]
        result['arms'][arm] = dict(events=checks, summary=dict(
            events=len(checks), matched=sum(c['match']>0 for c in checks),
            distinct_from_silence=sum(c['actual_start']!=c['silent_start'] for c in checks),
            candidate_unique_world_counts=[len(set(p['start'] for p in c['candidate_worlds'])) for c in checks],
            matches=[c['match'] for c in checks], mean_gain=float(np.mean([c['gain'] for c in checks])),
            raw_gain_mse=float(np.mean((np.array([c['raw_gain_prediction'] for c in checks])-targets)**2)),
            baseline_gain_mse=float(np.mean((np.array([c['baseline_gain_prediction'] for c in checks])-targets)**2)),
            two_turn_baseline_mse=float(np.mean([(v['baseline']-v['target'])**2 for v in twos])),
            two_turn_prediction_mse=float(np.mean([(v['prediction']-v['target'])**2 for v in twos])),
            mean_draft_tv_uniform=float(np.mean([c['draft_tv_uniform'] for c in checks])),
            gain_confidence_zero=sum(x==0 for c in checks for x in c['gain_confidence']),
            gain_gradient_positive=sum(c['gain_gradient_norm'] > 0 for c in checks),
            gain_gradient_norm_range=[min(c['gain_gradient_norm'] for c in checks),
                                      max(c['gain_gradient_norm'] for c in checks)],
            mean_future_action_spread=float(np.mean([np.ptp(c['predicted_future']) for c in checks]))))
    result['source_code_sha256'] = {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in (
        'scripts/ib_local/natural_text_world.py','scripts/ib_local/text_navigation.py',
        'scripts/ib_local/world_learning_cycle.py','scripts/ib_local/navigation_gain_units.py',
        'scripts/ib_local/navigation_internal_rollout.py','scripts/ib_local/navigation_joint_cycle.py',
        'scripts/ib_local/language_navigation_kanli_v4p.py')}
    output = ROOT / 'results/published/cgb_native_action_long_cause_2401_20261117.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({k:v['summary'] for k,v in result['arms'].items()}, ensure_ascii=False))


if __name__ == '__main__':
    main()
