"""Read existing same-source life events to locate recoverable value loss."""
import argparse
import json
from pathlib import Path
from statistics import mean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    records = [json.loads(line) for line in (args.work/'joint/events.jsonl').read_text(encoding='utf8').splitlines()]
    completed = json.loads((args.work/'joint/result.json').read_text(encoding='utf8'))
    rows = [r for r in records if r['autonomous_event']['kind']=='external' and r['cycle']<=completed['cycle']]
    detail = []
    for row in rows:
        epi = row['causal_use']['epistemic_value']
        p, q = row['choice_probability'], epi['contextual_total']
        available = [i for i, probability in enumerate(p) if probability > 0]
        expected = sum(p[i]*q[i] for i in available)
        receipt = row['causal_use']['native_amortization']['action_planner']
        teachers = receipt['teacher_checks']
        pair = None
        if len(teachers)==2:
            chosen = max(range(2), key=lambda i: teachers[i]['predicted'])
            pair = dict(predicted=[t['predicted'] for t in teachers], native=[t['native'] for t in teachers],
                ranking_agrees=(teachers[0]['predicted']-teachers[1]['predicted'])*
                    (teachers[0]['native']-teachers[1]['native']) > 0,
                native_pair_selection_loss=max(t['native'] for t in teachers)-teachers[chosen]['native'])
        detail.append(dict(cycle=row['cycle'], actual_use=row['metal_g'],
            model_action_total=q, probability=p, model_policy_gap=max(q[i] for i in available)-expected,
            teacher_errors=[t['error'] for t in teachers], successor_condition_pair=pair,
            world_prediction=receipt['world_prediction'], world_target=receipt['world_target']))
    reference = json.loads(Path('results/published/cgb_latent_action_loop_2493_20261120_calibrated.json').read_text(encoding='utf8'))
    best = reference['control']['mean_actual_use']
    actual = mean(r['actual_use'] for r in detail)
    out = dict(scope='committed same-source life prefix, read-only; no new world calls or training',
        cycles=[r['cycle'] for r in detail], mean_use=actual, original_native_reference_use=best,
        measured_reference_gap_fraction=best/actual-1,
        mean_model_policy_gap=mean(r['model_policy_gap'] for r in detail),
        teacher_RMSE=mean(error**2 for r in detail for error in r['teacher_errors'])**.5,
        successor_condition_pairs=sum(r['successor_condition_pair'] is not None for r in detail),
        successor_pairs_ordered_correctly=sum(r['successor_condition_pair']['ranking_agrees']
            for r in detail if r['successor_condition_pair'] is not None), details=detail,
        optimization_order=['continuous real-feedback calibration of successor values',
            'spend the same native teacher budget on uncertainty that changes competing action values',
            'adapt selection only after same-condition value calibration'],
        mathematical_contract='true_policy_loss <= 2*uniform_value_error + model_policy_gap',
        ceiling_provenance='CGB_USE_COST_FRONTIER_20261119.md: 1.311873913 is a fixed-query Student-t payment certificate',
        unit_contract='actual_use in finite-U; Q and teacher values in original 10*finite-U value units; pairs compare successor CONDITIONS')
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps({k:v for k,v in out.items() if k not in ('details','optimization_order')}, ensure_ascii=False))


if __name__=='__main__':
    main()
