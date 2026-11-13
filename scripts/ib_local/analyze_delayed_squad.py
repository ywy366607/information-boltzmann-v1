"""Matched real-question results; descriptive evidence, not a life certificate."""
import argparse
import json
from pathlib import Path
import statistics


def summarize(cases):
    return dict(cases=len(cases), **{k:statistics.mean(c[k] for c in cases)
                                   for k in ('exact_match','token_f1','first_answer_token_nll')})


def analyze(directory):
    directory=Path(directory)
    reports={arm:json.loads((directory/arm/'report.json').read_text(encoding='utf-8'))
             for arm in ('raw','cut','mismatch')}
    base=reports['raw']
    keys=('task_sha256','source_sha256','historical_runtime_files','execution_sources','numeric_runtime',
          'frozen_mlp_precision','model_source_sha256','ordinary_teaching')
    for arm,report in reports.items():
        if report['status']!='complete' or report['episodes_completed']!=80:
            raise ValueError(f'{arm}: complete80-case study required')
        if any(report[k]!=base[k] for k in keys):raise ValueError(f'{arm}: provenance mismatch')
        if report.get('frozen_weight_placement','cuda')!=base.get('frozen_weight_placement','cuda'):
            raise ValueError(f'{arm}: frozen weight placement differs')
        if [c['qa_id'] for c in report['cases']] != [c['qa_id'] for c in base['cases']]:
            raise ValueError('unmatched questions')
        for c in report['cases']:
            if (c['stage'].endswith(('retention','transfer')) or c['stage'] in ('A_before_B','B_before_A')) and (c['context_reintroduced'] or c['correction_provided']):
                raise ValueError('retention/transfer answer or passage leaked into live history')
            if c['first_corrective_target_distance'] is not None and c['first_corrective_target_distance']<=128:
                raise ValueError('correction arrived inside the calculation window')
            if c['feedback_start_event']-c['emission_end_event']!=c['feedback_delay_tokens']:
                raise ValueError('claimed delay differs from actually consumed history')
        for before,after in zip(report['cases'],report['cases'][1:]):
            if before['end_event']!=after['origin_events']:
                raise ValueError('case history was skipped or reset')
        if report['end_seen']-report['source_seen'] != report['end_events']-report['source_events']:
            raise ValueError('native seen count differs from actually lived history')
    stages=list(dict.fromkeys(c['stage'] for c in base['cases']))
    summaries={arm:{stage:summarize([c for c in r['cases'] if c['stage']==stage]) for stage in stages}
               for arm,r in reports.items()}
    deltas={control:{stage:{k:summaries['raw'][stage][k]-summaries[control][stage][k]
                                  for k in ('exact_match','token_f1','first_answer_token_nll')}
                         for stage in stages} for control in ('cut','mismatch')}
    delays={}
    for delay in (256,512):
        delays[delay]={arm:summarize([c for c in r['cases']
                                      if c['stage'].endswith(('retention','transfer')) and c['feedback_delay_tokens']==delay])
                       for arm,r in reports.items()}
    trajectories={}
    for arm,r in reports.items():
        by_stage={stage:{c['qa_id']:c for c in r['cases'] if c['stage']==stage} for stage in stages}
        trajectories[arm]={}
        for group,baseline,returned in [('A','A_before_B','A_retention'),('B','B_before_A','B_retention')]:
            cases=[]
            for qa_id,pre in by_stage[baseline].items():
                post=by_stage[returned][qa_id]
                item=dict(qa_id=qa_id,pre_nll=pre['first_answer_token_nll'],post_nll=post['first_answer_token_nll'],
                          nll_forgetting=post['first_answer_token_nll']-pre['first_answer_token_nll'],
                          pre_f1=pre['token_f1'],post_f1=post['token_f1'],
                          f1_forgetting=pre['token_f1']-post['token_f1'])
                if group=='A':
                    item['relearning']=[dict(stage=s,nll=by_stage[s][qa_id]['first_answer_token_nll'],
                        f1=by_stage[s][qa_id]['token_f1'],feedbacks_before=by_stage[s][qa_id]['study_qa_feedbacks_before'],
                        optimizer_updates=by_stage[s][qa_id]['updates']) for s in ('A_relearn1','A_relearn2')]
                cases.append(item)
            trajectories[arm][group]=dict(cases=cases,
                mean_nll_forgetting=statistics.mean(c['nll_forgetting'] for c in cases),
                mean_f1_forgetting=statistics.mean(c['f1_forgetting'] for c in cases),
                nonzero_behavioral_baseline=any(c['pre_f1']>0 for c in cases))
    return dict(status='complete',summaries=summaries,raw_minus_control=deltas,delay_strata=delays,
        matched_question_trajectories=trajectories,
        rules='Positive EM/F1 and negative NLL difference favor raw. Retention/transfer have no fresh passage or labels. '
              'Different question difficulty cannot establish improvement over time; compare matching questions/arms. '
              'This is one short real-data mechanism study, not universal monotonic improvement or convergence.',
        scope='local native-S delayed-feedback update; old capsule replay remains common to all three arms')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);args=p.parse_args()
    result=analyze(args.directory)
    output=args.directory/'analysis.json'
    if output.exists():raise FileExistsError('preserve previous analysis')
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')
