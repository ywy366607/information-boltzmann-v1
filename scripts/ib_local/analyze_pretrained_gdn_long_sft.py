"""Summarize actual long-SFT learning, retrieval and admission traffic."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def summarize(report):
    result = dict(kind=report['kind'],status=report['status'],
        task_sha256=report['task_sha256'],window=report['window'],capacity=report['capacity'],
        distinct_train_questions=report['distinct_train_examples_per_arm'],
        epochs=report['epochs'],arms={})
    if 'connection_gate' in report:
        result['connection_gate'] = report['connection_gate']
    if 'credit_gate' in report:
        result['credit_gate'] = report['credit_gate']
    for name in ('error','stop_location','stop_reason'):
        if name in report:
            result[name] = report[name]
    for name,arm in report.get('arms',{}).items():
        records = arm['records']
        updates = [u for r in records for u in r['updates']]
        reads = [v for u in updates for v in u['reads'].values()]
        question = [v for u in updates for v in u['question_record'].values()]
        curve = [{key:point[key] for key in ('examples','updates')} | dict(
            answer_nll=point['validation']['mean_answer_nll'],
            question_swap_gap=point['validation']['mean_question_swap_gap'],
            matching_correct=point['validation']['matching_correct'],
            admissions=point.get('admissions',{})) for point in arm['curve']]
        epochs = {}
        for epoch in sorted({r['epoch'] for r in records}):
            items = [r for r in records if r['epoch']==epoch]
            tokens = sum(r['accounting']['answer_tokens'] for r in items)
            epochs[str(epoch)] = dict(examples=len(items),supervised_answer_tokens=tokens,
                prequential_answer_nll=sum(u['supervised_nll_sum'] for r in items for u in r['updates'])/tokens)
        result['arms'][name] = dict(examples=len(records),updates=len(updates),
            real_tokens=sum(r['accounting']['real_tokens'] for r in records),
            compute_padding_tokens=sum(r['accounting']['padding_tokens'] for r in records),
            supervised_answer_tokens=sum(r['accounting']['answer_tokens'] for r in records),
            source_age_max_tokens=max((a for read in reads for a in read['ages_tokens']),default=0),
            question_records_retrieved=sum(q['retrieved'] for q in question),
            question_records_present_after_update=sum(q['present_after_update'] for q in question),
            question_record_opportunities=len(question),curve=curve,
            epochs=epochs,
            admissions=arm.get('admissions'),checkpoint=arm.get('checkpoint'),
            checkpoint_examples=arm.get('checkpoint_examples'),
            episode_storage_max_bytes=max((u['experience_cpu_bytes'] for u in updates),default=0))
    if report['status']=='complete' and report['kind'] in (
            'real_long_sft_held_old_credit','real_long_sft_native_credit_pair'):
        raw = Path(report['reference_report']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=report['reference_report_sha256']:
            raise ValueError('completed full reference changed')
        reference = json.loads(raw.decode('utf-8-sig'))
        if report['kind']=='real_long_sft_held_old_credit':
            result['arms']['full'] = summarize(reference)['arms']['off']
        result.update(answer_nll_full_minus_held=report['answer_nll_full_minus_held'],
            swap_gap_full_minus_held=report['swap_gap_full_minus_held'],
            peak_cuda_allocated_bytes=report['peak_cuda_allocated_bytes'],
            reference_report=report['reference_report'],reference_report_sha256=report['reference_report_sha256'])
        full,held = result['arms']['full']['curve'],result['arms']['held']['curve']
        if [p['examples'] for p in full]!=[p['examples'] for p in held]:
            raise ValueError('full/held curve budgets differ')
        result['paired_curve'] = [dict(examples=a['examples'],
            answer_nll_full_minus_held=a['answer_nll']-b['answer_nll'],
            swap_gap_full_minus_held=a['question_swap_gap']-b['question_swap_gap'])
            for a,b in zip(full,held)]
        trained = result['paired_curve'][1:]
        result['mean_trained_curve_nll_full_minus_held'] = sum(
            p['answer_nll_full_minus_held'] for p in trained)/len(trained)
        result['trained_curve_points_favoring_full'] = sum(
            p['answer_nll_full_minus_held']<0 for p in trained)
    elif report['status']=='complete':
        result.update(answer_nll_on_minus_off=report['answer_nll_on_minus_off'],
            swap_gap_on_minus_off=report['swap_gap_on_minus_off'],
            peak_cuda_allocated_bytes=report['peak_cuda_allocated_bytes'])
        off,on = [report['arms'][name]['curve'] for name in ('off','on')]
        result['paired_curve'] = [dict(examples=a['examples'],
            answer_nll_on_minus_off=b['validation']['mean_answer_nll']-a['validation']['mean_answer_nll'],
            swap_gap_on_minus_off=b['validation']['mean_question_swap_gap']-a['validation']['mean_question_swap_gap'])
            for a,b in zip(off,on) if a['examples']==b['examples']]
        trained = [p for p in result['paired_curve'] if p['examples']>0]
        result['mean_trained_curve_nll_on_minus_off'] = sum(
            p['answer_nll_on_minus_off'] for p in trained)/len(trained)
        result['trained_curve_points_favoring_on'] = sum(
            p['answer_nll_on_minus_off']<0 for p in trained)
        result['trained_curve_points'] = len(trained)
        result['final_minus_initial_nll_difference'] = (
            result['paired_curve'][-1]['answer_nll_on_minus_off']-
            result['paired_curve'][0]['answer_nll_on_minus_off'])
        a,b = off[-1]['validation']['rows'],on[-1]['validation']['rows']
        if [r['paper_id'] for r in a]!=[r['paper_id'] for r in b]:
            raise ValueError('validation paper order differs')
        result['per_paper_final'] = [dict(paper_id=x['paper_id'],
            mean_answer_nll_on_minus_off=sum(y['matrix'][i][i]['mean']-x['matrix'][i][i]['mean'] for i in range(2))/2,
            swap_gap_on_minus_off=y['swap_gap']-x['swap_gap']) for x,y in zip(a,b)]
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--wait',action='store_true',help='background report collection only; no training')
    args=parser.parse_args()
    started=time.monotonic()
    while True:
        if time.monotonic()-started>10800:
            raise TimeoutError('report did not reach a terminal state within three hours')
        try:
            raw=args.input.read_bytes()
        except (PermissionError, FileNotFoundError):
            # Windows can briefly deny access while the producer atomically
            # replaces its report. Observation must not abort the whole wait.
            if not args.wait:
                raise
            time.sleep(1)
            continue
        report=json.loads(raw.decode('utf-8-sig'))
        if not args.wait or report['status']!='running':
            break
        time.sleep(5)
    result=summarize(report)
    result.update(raw_report=str(args.input),raw_sha256=hashlib.sha256(raw).hexdigest(),
        scope='same-seed finite-budget restricted QASPER excerpts; no official benchmark or convergence claim')
    if args.output:
        args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    else:
        print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
