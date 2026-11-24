"""Read complete battle artifacts into separate functional/cost accounts."""
from collections import Counter
import json
from pathlib import Path

ROOT = Path('E:/mengli-bounded-reading-20261123')
PROJECT = Path('D:/information-boltzmann-v1')


def main():
    path = PROJECT/'results/published/cgb_bounded_reading_battle_20261123.json'
    battle = json.loads(path.read_text(encoding='utf8'))
    results = {}
    for name, arm in battle['arms'].items():
        rows = [r for line in (ROOT/name/'events.jsonl').read_text(encoding='utf8').splitlines()
                if (r := json.loads(line)).get('autonomous_event', {}).get('kind') == 'external']
        counters = Counter()
        spans = Counter()
        stops = Counter()
        for r in rows:
            counters['bytes'] += r['external_bytes']
            counters['new_position_bytes'] += r['external_bytes']-r['world']['response_previously_seen_bytes']-r['world']['holdout_previously_seen_bytes']
            counters['language_updates'] += r['world_learning']['training_pieces']
            counters['actual_updates'] += r['updates']
            counters['native_plans'] += 1
            counters['native_teachers'] += r['causal_use']['native_amortization']['action_planner']['teacher_count']
            receipt = r.get('bounded_reading_option', {})
            spans[receipt.get('blocks', 1)] += 1
            stops[receipt.get('stop', 'one_block')] += 1
            assert len(r['joint_language_search']['candidates']) == 4
            assert sum(r['world']['text_source_bytes'].values()) == r['external_bytes']
            assert r['external_bytes'] == 480*r['world_learning']['training_pieces']
        summary = arm['summary']
        elapsed = summary['life_seconds']
        profile = json.loads((ROOT/name/'runtime_profile.json').read_text(encoding='utf8'))['functions']
        f = arm['audit']['fresh_8192_nll']['segment_nll']
        c = arm['audit']['classic_8192_nll']['segment_nll']
        before = battle['origin']
        results[name] = dict(counters=counters, duration_distribution=spans,
            stop_reasons=stops, life_seconds=elapsed, bytes_per_second=counters['bytes']/elapsed,
            fresh_positions_per_second=counters['new_position_bytes']/elapsed,
            language_updates_per_second=counters['language_updates']/elapsed,
            planning_seconds=profile['engine._utter']['inclusive_seconds'],
            policy_gradient_seconds=profile['engine._policy_grad']['inclusive_seconds'],
            language_objective_seconds=profile['body.objective']['inclusive_seconds'],
            advance_seconds=profile['engine._advance']['inclusive_seconds'],
            fresh_nll=arm['audit']['fresh_8192_nll']['mean_nll'],
            classic_nll=arm['audit']['classic_8192_nll']['mean_nll'],
            combined_nll=arm['audit']['combined_16384_mean_nll'],
            fresh_improvement=arm['fresh_nll_improvement'],
            classic_improvement=arm['classic_nll_improvement'],
            fresh_segment_improvement=[a-b for a,b in zip(before['fresh_8192_nll']['segment_nll'], f)],
            classic_segment_improvement=[a-b for a,b in zip(before['classic_8192_nll']['segment_nll'], c)],
            old_fact_score=arm['audit']['fresh_64_facts']['mean_score'],
            old_fact_improvement=arm['old_fact_score_improvement'],
            calibrated_gain_mse=summary['pre_outcome_gain_mse'],
            checkpoint_sha256=summary['checkpoint_sha256'],
            restoring_option=arm['audit'].get('reading_option_state'))
    a, b = results['strict_600s'], results['option4_600s']
    comparison = dict(throughput_ratio=b['bytes_per_second']/a['bytes_per_second'],
        new_position_throughput_ratio=b['fresh_positions_per_second']/a['fresh_positions_per_second'],
        language_update_rate_ratio=b['language_updates_per_second']/a['language_updates_per_second'],
        candidate_combined_nll_delta=b['combined_nll']-a['combined_nll'],
        candidate_old_fact_delta=b['old_fact_score']-a['old_fact_score'])
    out = dict(source_sha256=battle['source_sha256'], arms=results, comparison=comparison)
    dest = PROJECT/'results/published/cgb_bounded_reading_functional_accounts_20261123.json'
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
