"""Report delayed teaching, actual continuity and matched future answers."""
import argparse
import json
from pathlib import Path


def summarize(arm):
    cases = arm['cases']
    rows = [r for c in cases for r in c['settlement']['rows']]
    previous = arm['source_seen']
    continuity = True
    for case in cases:
        continuity &= case['incoming_seen'] == previous
        previous = case['outgoing_seen']
    scores = {}
    for name, group in (('all',cases), ('first_half',cases[:len(cases)//2]),
                        ('second_half',cases[len(cases)//2:])):
        scores[name] = dict(correct=sum(c['first_number_exact'] for c in group), count=len(group))
    return dict(scores=scores, optimizer_updates=arm['end_updates']-arm['source_updates'],
        events=arm['events'], phase_counts=arm['phase_counts'],
        contiguous_life=continuity,
        live_cache_preserved=all(w['live_cache_preserved'] for w in arm['windows']),
        max_reconstruction_target_ce_error=max(abs(r['ce']-r['lived_ce']) for r in rows),
        max_reconstruction_logit_error=max(r['replay_logit_max_difference'] for r in rows),
        max_original_operator_logit_error=max(r['inference_logit_max_difference'] for r in rows),
        max_pending_mib=max(c['settlement']['pending_cpu_bytes'] for c in cases)/2**20,
        actual_late_supervised_positions=len(rows),
        feedback_delay_tokens=dict(min=min(r['feedback_delay_tokens'] for r in rows),
                                   max=max(r['feedback_delay_tokens'] for r in rows)),
        retrieved_ages_tokens={layer:dict(min=min(ages), max=max(ages),
            over_128=sum(age>128 for age in ages), selected_positions=len(ages))
            for layer in rows[0]['reads']
            if (ages:=[age for r in rows for age in r['reads'][layer]['source_ages']])})


def parameter_difference(held_path, full_path):
    import torch
    a, b = (torch.load(path, map_location='cpu', weights_only=False, mmap=True)['learner']
            for path in (held_path,full_path))
    result = {}
    for layer in a['layers']:
        names = [name for name in a['native'] if name.startswith(f'layer{layer}.')]
        count = changed = 0
        square = 0.
        for name in names:
            x, y = a['native'][name], b['native'][name]
            count += x.numel()
            changed += int((x!=y).sum())
            square += float((x.float()-y.float()).square().sum())
        result[str(layer+1)] = dict(native_coordinates=count, changed_fp_coordinates=changed,
                                   fraction=changed/count, difference_norm=square**.5)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--endpoint-difference',action='store_true')
    args = parser.parse_args()
    raw = json.loads(args.report.read_text())
    if raw['status'] != 'complete':
        raise ValueError('incomplete run cannot pass admission or paired reporting')
    result = {name:raw[name] for name in ('source','source_sha256','source_files',
        'source_unchanged','source_files_unchanged','task_seed','official_revision','seconds')}
    result.update(status='complete',admission_only=raw['admission_only'],
        answer_contract=raw['answer_contract'],new_lean_coverage=0,
        arms={name:summarize(arm) for name,arm in raw['arms'].items()},
        interpretation='Finite-budget route validation on previously examined official training tasks.')
    if not all(a['contiguous_life'] and a['live_cache_preserved'] for a in result['arms'].values()):
        raise AssertionError('actual life continuity failed')
    if raw['admission_only']:
        result['gradient_admission']=raw['gradient_admission']
        result['disk_pending_recovery_equal']=raw.get('disk_pending_recovery_equal',False)
        if not result['disk_pending_recovery_equal']:
            raise AssertionError('pending disk recovery did not pass')
    else:
        held, full = (raw['arms'][name]['cases'] for name in ('held','full'))
        if len(held)!=len(full) or len(full)!=raw['examples']:
            raise AssertionError('incomplete matched sequence')
        for a,b in zip(held,full):
            if (a['case'],a['references']) != (b['case'],b['references']):
                raise AssertionError('world cases differ')
        result['paired']=dict(full_only=sum(b['first_number_exact'] and not a['first_number_exact']
                                            for a,b in zip(held,full)),
            held_only=sum(a['first_number_exact'] and not b['first_number_exact'] for a,b in zip(held,full)),
            both_correct=sum(a['first_number_exact'] and b['first_number_exact'] for a,b in zip(held,full)),
            both_wrong=sum(not a['first_number_exact'] and not b['first_number_exact'] for a,b in zip(held,full)),
            differing_predictions=sum(a['prediction']!=b['prediction'] for a,b in zip(held,full)))
        if args.endpoint_difference:
            result['native_endpoint_difference']=parameter_difference(
                args.report.parent/'held_latest.pt', args.report.parent/'full_latest.pt')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
