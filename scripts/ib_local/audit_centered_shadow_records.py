"""Audit already settled mother-world forecasts; never load or update a body."""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'results/cgb_bfss_gdn_reference_20261110'
RUNS = ('joint_question_update_follow_16', 'responsive_mother_admission_8',
        'responsive_mother_v2_admission_8', 'responsive_mother_v3_admission_8')


def main():
    output = []
    for name in RUNS:
        rows = [json.loads(line) for line in (BASE / name / 'metrics.jsonl').read_text(
            encoding='utf-8').splitlines() if line.strip()]
        scores = [r['shadow_preview']['settled_candidate_prediction'] for r in rows
                  if r.get('shadow_preview', {}).get('settled_candidate_prediction') is not None]
        if not scores or any(r['shadow_preview']['settled_candidate_kind'] != 'centered'
                             for r in rows if r.get('shadow_preview', {}).get('settled_candidate_prediction') is not None):
            raise ValueError(f'{name}: missing or non-centered settled predictions')
        keys = ('baseline_mse', 'candidate_mean_mse', 'candidate_weighted_mse',
                'candidate_best_mse', 'candidate_spread')
        result = {'run': name, 'cycles': [rows[0]['cycle'], rows[-1]['cycle']],
                  'n_settled': len(scores)}
        result.update({k: statistics.mean(s[k] for s in scores) for k in keys})
        result['weighted_wins'] = sum(s['candidate_weighted_mse'] < s['baseline_mse'] for s in scores)
        result['max_identity_error'] = max(abs(s['candidate_weighted_mse'] - s['baseline_mse']
                                               - s['candidate_spread']) for s in scores)
        result['bound_receipt_count'] = sum(bool(r['shadow_preview']['bound_receipt_used']) for r in rows)
        result['max_abs_policy_offset'] = max(max(map(abs, r['shadow_preview']['delivered_offset'])) for r in rows)
        # These forecasts were fixed at the previous decision and scored only
        # after the real next gain arrived. An equal pair is a tie, not a win.
        forecasts = [r['shadow_preview']['settled_two_turn'] for r in rows
                     if r['shadow_preview'].get('settled_two_turn') is not None
                     and r['shadow_preview']['settled_two_turn']['eligible']]
        if forecasts:
            result['settled_gain_count'] = len(forecasts)
            result['gain_baseline_mse'] = statistics.mean(
                (f['target'] - f['baseline']) ** 2 for f in forecasts)
            result['gain_shadow_mse'] = statistics.mean(
                (f['target'] - f['prediction']) ** 2 for f in forecasts)
            result['gain_shadow_wins'] = sum(abs(f['target'] - f['prediction']) <
                                             abs(f['target'] - f['baseline']) for f in forecasts)
            result['gain_exact_ties'] = sum(f['baseline'] == f['prediction'] for f in forecasts)
        output.append(result)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
