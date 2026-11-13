"""Compact endpoint/curve summary of the matched continuous native-credit pair."""
import argparse
import json
from pathlib import Path


def write_json(path,value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2),encoding='utf-8')
    temporary.replace(path)


def summarize(report):
    if (report['status'] != 'complete' or report['protocol'].get('comparison_kind') != 'credit' or
            set(report['arms']) != {'held','full'}):
        raise ValueError('completed held/full old-native comparison required')
    if not (report['source_checkpoint_unchanged'] and report['executed_code_unchanged'] and
            report['first_training_forward_exact']):
        raise ValueError('source/code/initial-forward integrity check failed')
    result = {k:report[k] for k in ('checkpoint','checkpoint_sha256','source_updates',
        'dataset_revision','source_stream_sha256','executed_stream_sha256','protocol',
        'source_checkpoint_unchanged','executed_code_unchanged','first_training_forward_exact')}
    result.update(status='complete',executed_optimizer_updates=report['executed_optimizer_updates'],arms={})
    for name,arm in report['arms'].items():
        rows = arm['steps']
        if len(rows) != report['protocol']['updates_per_arm'] or arm['status'] != 'complete':
            raise ValueError('arm has an incomplete learning history')
        result['arms'][name] = dict(new_updates=arm['new_updates'],final_updates=rows[-1]['updates'],
            train_tokens=arm['train_tokens'],checkpoint=arm['checkpoint'],phases=arm['phases'],
            stream_nll=sum(r['nll_sum'] for r in rows)/sum(r['target_tokens'] for r in rows),
            evaluations=[dict(step=e['step'],phase=e['phase'],
                nll={k:v['nats_per_token'] for k,v in e['metrics'].items()},
                state_unchanged=e['persistent_state_unchanged']) for e in arm['evaluations']],
            curve=[dict(step=r['step'],phase=r['phase'],ce=r['ce'],sigreg=r['sigreg'],
                old_read_cut_nll_delta=r['diagnostics'].get('old_read_cut_nll_delta')) for r in rows],
            layers={})
        for layer in rows[0]['reads']:
            reads = [r['reads'][layer] for r in rows]
            result['arms'][name]['layers'][str(int(layer)+1)] = dict(
                query_count=sum(r['query_count'] for r in reads),
                unique_replays_per_window_min=min(r['unique_replays'] for r in reads),
                unique_replays_per_window_max=max(r['unique_replays'] for r in reads),
                max_source_age_tokens=max(max(r['ages_tokens'],default=0) for r in reads),
                old_query_participations=sum(sum(age > 128 for age in r['ages_tokens']) for r in reads),
                scope='query participation, not independent finite-benefit observations')
    full,held = result['arms']['full'],result['arms']['held']
    result['full_minus_held'] = dict(stream_nll=full['stream_nll']-held['stream_nll'],
        phases={phase:full['phases'][phase]-held['phases'][phase] for phase in full['phases']},
        evaluations=[dict(step=f['step'],nll={k:f['nll'][k]-h['nll'][k] for k in f['nll']})
                     for f,h in zip(full['evaluations'],held['evaluations'])])
    # Descriptive paired blocks only: consecutive windows are dependent and
    # there is only one source individual, so do not attach an iid p-value.
    result['paired_16_window_blocks'] = [dict(first_step=i+1,last_step=min(i+16,len(full['curve'])),
        full_minus_held=sum(f['ce']-h['ce'] for f,h in zip(full['curve'][i:i+16],held['curve'][i:i+16]))/
            len(full['curve'][i:i+16]))
        for i in range(0,len(full['curve']),16)]
    result['scope'] = 'single-source fixed-budget continuous learning; differences descriptive, not convergence or global necessity'
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    report = summarize(json.loads(args.report.read_text(encoding='utf-8')))
    report['raw_report'] = str(args.report)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    write_json(args.output,report)
    print(json.dumps(report['full_minus_held']),flush=True)


if __name__=='__main__':
    main()
