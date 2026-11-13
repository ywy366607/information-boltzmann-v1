"""Curate completed same-body direct-attention credit runs; no model updates."""
import argparse
import hashlib
import json
from pathlib import Path


def summarize(report):
    if report['status'] != 'complete' or report['replay_schema'] != 5:
        raise ValueError('a completed schema5 report is required')
    arms = {}
    for name, arm in report['arms'].items():
        steps = arm['steps']
        windows = [w for r in steps for w in r['windows'] if w['supervised']]
        curves = []
        for e in arm['evaluations']:
            value = e['short']
            curves.append(dict(example=e['example'], exact=value['exact'],
                count=len(value['cases']), answer_nll=value['answer_nll'],
                first_digit_nll=sum(c['first_digit_nll'] for c in value['cases'])/len(value['cases']),
                owt_nll=value['owt_nll']))
        arms[name] = dict(optimizer_updates=len(windows), examples=len(steps),
            observed_bytes=sum(r['input_bytes'] for r in steps),
            supervised_bytes=sum(w['supervised'] for w in windows),
            mean_prequential_answer_ce=sum(w['ce']*w['supervised'] for w in windows)/sum(w['supervised'] for w in windows),
            curve=curves, final_layer_reads=windows[-1]['layers'],
            final_old_gradients=windows[-1].get('old_gradients'),
            checkpoint=str(Path(report['common_checkpoint']).parent.parent/name/'last.pt.gz'))
    if set(arms) != {'held', 'full'}:
        raise ValueError('both held and full arms are required')
    held, full = arms['held']['curve'][-1], arms['full']['curve'][-1]
    return dict(status='complete', schema=5, new_lean_coverage=0, promoted=False,
        native_output_projection=report.get('native_output_projection',False),
        neural_parameters=report.get('neural_parameters'),
        native_output_parameters=report.get('native_output_parameters',0),
        native_replay_gradient_is_only_arm_difference=True,
        model_forecast='One shared actual forward; no per-record vocabulary experts or suffix states.',
        meaning_of_held='Recall, query/key learning, actual recurrent S and current gradients remain; only old native replay derivatives are cut.',
        capacity_per_layer=report['capacity_per_layer'], history_span_per_record=report['local_replay_bytes'],
        bptt_bytes=report['bptt_bytes'], generation_bytes=report['generation_bytes'],
        test_count=report['test_count'], source_sha256=report['source_sha256'],
        source_unchanged=report['source_unchanged'], executed_sources_unchanged=report['executed_sources_unchanged'],
        seconds=report['seconds'], peak_cuda_bytes=report['peak_cuda_bytes'],
        numerical_audit={k:report['audit'][k] for k in ('neutral_max','held_full_forward_max','resume_next_update_exact','passed')},
        arms=arms, full_minus_held=dict(exact=full['exact']-held['exact'],
            answer_nll=full['answer_nll']-held['answer_nll'],
            first_digit_nll=full['first_digit_nll']-held['first_digit_nll'],
            owt_nll=full['owt_nll']-held['owt_nll']),
        evidence_scope='Fixed-budget official-generator short-layout learning-path comparison, one continuing source and seed; not a full RULER benchmark or general language-life capability claim.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--held-content', type=Path)
    p.add_argument('--full-content', type=Path)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = summarize(json.loads(args.report.read_text(encoding='utf-8')))
    paths = [args.report]
    for name, path in (('held', args.held_content), ('full', args.full_content)):
        if path is not None:
            raw = json.loads(path.read_text(encoding='utf-8'))
            result.setdefault('content_interventions', {})[name] = {k:v for k,v in raw.items() if k != 'records'}
            paths.append(path)
    result['sources'] = [dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in paths]
    result['analyzer_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result['full_minus_held'], indent=2))


if __name__ == '__main__':
    main()
