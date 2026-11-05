"""Compare completed settlement runs without loading or updating either body."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.ib_local.report_learning_rooms import alignment, noise_opportunity


def mean(values):
    return float(np.mean(values)) if values else None


def room_opportunities(rows, room):
    available = [r for r in rows if room in r['candidate_rooms']
                 and any(k != room for k in r['candidate_rooms'])]
    return dict(
        decisions=len(available),
        policy_probability=mean([sum(p for p, k in zip(r['choice_probability'], r['candidate_rooms'])
                                     if k == room) for r in available]),
        uniform_candidate_probability=mean([r['candidate_rooms'].count(room) / len(r['candidate_rooms'])
                                           for r in available]),
        actual_fraction=mean([r['world']['anchor'] == room for r in available]),
    )


def signal_checks(rows, kind):
    """Check all observable deliveries, including the delayed pending credit."""
    if kind not in ('continuation', 'continuation_joint'):
        return dict(scope='historical witness control; new wiring fields not required')
    for i, row in enumerate(rows):
        value = row['joint_progress'] if kind == 'continuation_joint' else row['progress']
        assert row['settlement_kind'] == kind
        assert row['settlement'] == row['metal_g'] == row['usable_signal'] == value
        assert np.isclose(row['witness_g'], row['witness_pre'] - row['witness_post'], atol=1e-12)
        assert np.isclose(row['wood_target'][3], 10 * value, atol=1e-6, rtol=1e-6)
        assert np.isclose(row['interoception'][1], 100 * value, atol=1e-5, rtol=1e-6)
        td = row['td_previous']
        assert td is not None
        if i:
            assert td['g'] == rows[i - 1]['settlement']
            assert not td.get('objective_boundary', False)
            assert np.isclose(td['y'], td['g'] + .8 * row['value_now'] / 100,
                              atol=1e-10, rtol=1e-10)
        else:
            assert td['objective_boundary'] and td['y'] == td['g']
    return dict(checked_rounds=len(rows), reward_to_wood=True, reward_to_interoception=True,
                previous_reward_to_delayed_td=True,
                migrated_pending_reward_has_no_cross_objective_bootstrap=True)


def describe(root):
    rows = [json.loads(s) for s in (root / 'metrics.jsonl').read_text(encoding='utf8').splitlines()]
    cfg = json.loads((root / 'config.json').read_text(encoding='utf8'))
    summary = json.loads((root / 'summary.json').read_text(encoding='utf8'))
    assert len(rows) == summary['cycles'] == cfg['cycles'] == 512
    assert [r['cycle'] for r in rows] == list(range(1, 513))
    split = cfg['switch_turn']
    audits = [r for r in rows if 'audit_balanced_g' in r]
    late = rows[split:]
    kind = cfg.get('settlement', 'witness')
    returns = [r['cycle'] for r in late if r['world']['room'] == 'changing']
    arrivals = [r['cycle'] for r in late if r['world']['anchor'] == 4
                and r['world']['room_before'] != 4]
    risk = [np.mean(r['audit_room_post']) for r in audits]
    by_half = {}
    for name, part in (('first_half', rows[:split]), ('second_half', late)):
        by_half[name] = dict(
            visits={k: sum(r['world']['room'] == k for r in part)
                    for k in ('familiar', 'noise', 'easy', 'hard', 'changing')},
            familiar_fraction=mean([r['world']['room'] == 'familiar' for r in part]),
            familiar_opportunities=room_opportunities(part, 0),
            noise_opportunities=noise_opportunity(part),
            mean_wood_policy_tv=mean([r['wood_policy_tv'] for r in part]),
            mean_residual_policy_tv=mean([r['residual_policy_tv'] for r in part]),
            mean_precision=mean([r['precision'] for r in part]),
            mean_critic_rms=mean([r['critic_rms'] for r in part if r['critic_rms'] is not None]),
            reward_replay_samples=mean([r['wood_reward_replay_samples'] for r in part
                                        if 'wood_reward_replay_samples' in r]),
            gold_alignment=alignment([r for r in part if 'audit_balanced_g' in r]),
        )
    return dict(
        root=str(root), settlement=kind, config=cfg, summary=summary,
        input_sha256=hashlib.sha256((root / 'metrics.jsonl').read_bytes()).hexdigest(),
        signal_delivery=signal_checks(rows, kind), halves=by_half,
        change=dict(first_visit_after_flip=returns[0] if returns else None,
                    first_arrival_after_flip=arrivals[0] if arrivals else None,
                    visits_after_flip=len(returns),
                    already_there_on_first_postflip_turn=bool(
                        rows[split]['world']['anchor'] == 4
                        and rows[split]['world']['room_before'] == 4)),
        final_mean_future_nll=float(np.mean(list(summary['final_nll'].values()))),
        mean_post_update_audit_nll=float(np.mean(risk)),
        audit_mean_scope='Fixed 40 audit times; descriptive, not an integral over all 512 turns.',
        maximum_absolute_settlement=max(abs(r['metal_g']) for r in rows),
        maximum_absolute_value=max(abs(v) for r in rows for v in r['imagined_value']),
        maximum_critic_rms=max(r['critic_rms'] for r in rows if r['critic_rms'] is not None),
        return_interpretation='First post-flip exposure is not necessarily a navigated return; '
            'also report actual arrivals and whether the body was already there.',
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('control', type=Path)
    parser.add_argument('candidate', type=Path)
    args = parser.parse_args()
    old, new = describe(args.control), describe(args.candidate)
    for key in ('source_sha256', 'routing', 'switch_turn', 'response_bytes', 'holdout_bytes',
                'parameter_count', 'validation_sha256'):
        assert old['config'][key] == new['config'][key], key
    assert old['summary']['initial_nll'] == new['summary']['initial_nll']
    old_return, new_return = (r['change']['first_visit_after_flip'] for r in (old, new))
    correlations = [new['halves'][k]['gold_alignment']['spearman']
                    for k in ('first_half', 'second_half')]
    result = dict(
        control=old, candidate=new,
        preregistered=dict(
            familiar_share_reduced=new['halves']['second_half']['familiar_fraction']
                < old['halves']['second_half']['familiar_fraction'],
            changed_room_first_visit_earlier=bool(new_return and old_return and new_return < old_return),
            settlement_future_spearman_above_point3_both_halves=all(
                r is not None and r > .3 for r in correlations),
        ),
        final_mean_future_nll_delta=new['final_mean_future_nll'] - old['final_mean_future_nll'],
        earlier_actual_arrival_after_flip=bool(
            new['change']['first_arrival_after_flip'] is not None
            and old['change']['first_arrival_after_flip'] is not None
            and new['change']['first_arrival_after_flip'] < old['change']['first_arrival_after_flip']),
        scope='One-seed closed-loop development diagnostic; migration is part of intervention. '
              'Balanced future evaluation never enters reward. No convergence or real-language claim.',
    )
    if new['settlement'] == 'continuation_joint':
        # The old correlation threshold remains visible as historical context,
        # but is not silently substituted for this run's registered endpoint.
        result['previous_protocol_diagnostics'] = result.pop('preregistered')
        result['preregistered'] = dict(
            complete_signal_delivery=new['signal_delivery']['checked_rounds'] == 512,
            final_mean_future_nll_improved=result['final_mean_future_nll_delta'] < 0,
            rule_relearning_final_nll=new['summary']['final_nll']['changing'],
            noise_and_familiar_opportunities_reported=True,
            scope='Section 16.7: final whole-body improvement required; rank correlation '
                  'reported by phase, not forced after progress is exhausted.')
    if new['config'].get('reward_memory','td') == 'observed':
        rows=[json.loads(s) for s in (args.candidate/'metrics.jsonl').read_text(encoding='utf8').splitlines()]
        assert all('observed_memory_audit' in r for r in rows)
        for row in rows:
            assert row['td_previous'] is None or row['td_previous']['memory_label_kind']=='observed_gain'
            audit=row['observed_memory_audit']
            if audit['valid'] and audit['complete_wood_mean']:
                assert audit['mean_difference'] is not None and audit['mean_difference'] < 1e-4
        result['prior_protocol_preregistered']=result.pop('preregistered')
        result['preregistered']=dict(
            complete_signal_delivery=new['signal_delivery']['checked_rounds']==512,
            observed_gain_mean_checked_rounds=sum(r['observed_memory_audit']['mean_difference'] is not None for r in rows),
            final_mean_future_nll_not_worse=new['final_mean_future_nll']<=old['final_mean_future_nll'],
            fixed_audit_mean_nll_improved=new['mean_post_update_audit_nll']<old['mean_post_update_audit_nll'],
            scope='Section16.14: joint-settlement TD-memory control versus observed-memory; '
                  'only one seed, endpoint and fixed40 process audit judged together.')
    output = args.candidate / 'comparison.json'
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf8')
    print(json.dumps(dict(output=str(output), preregistered=result['preregistered'],
                          final_mean_future_nll_delta=result['final_mean_future_nll_delta'],
                          control=old['halves'], candidate=new['halves']), indent=2))


if __name__ == '__main__':
    main()
