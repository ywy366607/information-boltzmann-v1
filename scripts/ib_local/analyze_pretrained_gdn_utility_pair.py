"""Summarize completed finite-utility retention pairs without Torch/GPU work."""
import argparse
import json
from pathlib import Path


def trace_written_reuse(steps, endpoints, layer=None):
    written, live_first, survival, returns = {}, set(), {}, []
    for row in steps:
        for event in row['admissions']:
            if layer is not None and event['layer']!=layer:
                continue
            for read in event['reads']:
                source = written.get(read['identity'])
                if row['phase']=='X_revisit' and source is not None and source['phase']=='X_continue':
                    returns.append(dict(step=row['step'],written_step=source['step'],
                        elapsed_since_write_tokens=(row['step']-source['step']-1)*128,**read))
            live_first.discard(event['victim'])
            written[event['new_identity']]=dict(step=row['step'],phase=row['phase'])
            if row['phase']=='X_continue':
                live_first.add(event['new_identity'])
        if row['step'] in endpoints:
            survival[str(row['step'])]=len(live_first)
    beneficial=[r for r in returns if r['removal_benefit']>1e-5]
    return dict(first_X_records_still_alive=survival,first_X_reads_during_revisit=len(returns),
        beneficial_first_X_reads_during_revisit=len(beneficial),
        distinct_beneficial_first_X_identities=len({r['identity'] for r in beneficial}),
        maximum_beneficial_source_age_tokens=max((r['age_tokens'] for r in beneficial),default=0),
        maximum_beneficial_elapsed_since_write_tokens=max((r['elapsed_since_write_tokens'] for r in beneficial),default=0),
        records=returns)


def layer_metrics(steps, endpoints, layer):
    old,positive,repeats,rescued=[],[],[],[]
    last={}
    for row in steps:
        for event in row['admissions']:
            if event['layer']!=layer:
                continue
            for read in event['reads']:
                if read['age_tokens']<=128:
                    continue
                item=dict(step=row['step'],phase=row['phase'],**read)
                old.append(item)
                if read['removal_benefit']>1e-5:
                    positive.append(item)
                    if read['identity'] in last:
                        repeats.append(dict(item,gap_tokens=(row['step']-last[read['identity']])*128))
                    last[read['identity']]=row['step']
        rescued.extend(dict(step=row['step'],**read) for read in row['later_rescued_reads']
            if int(read['identity'].split(':')[0])==layer and read['age_tokens']>128
            and read.get('removal_benefit',0)>1e-5 and read['original_evidence']>0)
    return dict(layer_index=layer,human_layer_number=layer+1,old_read_events=len(old),
        beneficial_old_reads=len(positive),harmful_old_reads=sum(r['removal_benefit'] < -1e-5 for r in old),
        distinct_beneficial_old_identities=len(last),repeated_beneficial_old_reads=len(repeats),
        repeated_beneficial_gaps_over_window=sum(r['gap_tokens']>128 for r in repeats),
        max_beneficial_reuse_gap_tokens=max((r['gap_tokens'] for r in repeats),default=0),
        max_beneficial_read_age_tokens=max((r['age_tokens'] for r in positive),default=0),
        beneficial_old_protection_increases=sum(r['preference_after']>r['preference_before']+1e-7 for r in positive),
        beneficial_but_negative_dose_credit=sum(r['signed_read_parameter_credit'] < -1e-7 for r in positive),
        mean_absolute_read_parameter_credit=sum(abs(r['signed_read_parameter_credit']) for r in old)/max(1,len(old)),
        confirmed_beneficial_rescued_reuses=len(rescued),confirmed_rescued_records=rescued,
        first_X_retention_and_reuse=trace_written_reuse(steps,endpoints,layer))


def analyze(report):
    placement=set(report['arms'])=={'input','output'}
    if report['status']!='complete' or (not placement and set(report['arms'])!={'retention','utility'}):
        raise ValueError('a complete utility or input/output placement pair is required')
    old_name,new_name=('input','output') if placement else ('retention','utility')
    difference_name=f'{new_name}_minus_{old_name}'
    for arm,data in report['arms'].items():
        for row in data['steps']:
            for event in row['admissions']:
                if any('removal_benefit' not in r for r in event['reads']):
                    raise ValueError('missing same-record finite utility measurement')
            for credit in row['credit'].values():
                measured=credit['retention_utility']
                if (arm=='retention' or placement) and (measured['records']!=0 or not measured['diagnostic_only']):
                    raise ValueError('control applied finite utility to learning')
    traces={arm:trace_written_reuse(data['steps'],report['protocol']['endpoints'])
            for arm,data in report['arms'].items()}
    layers=sorted({e['layer'] for d in report['arms'].values() for r in d['steps'] for e in r['admissions']})
    by_layer={str(layer):{arm:layer_metrics(data['steps'],report['protocol']['endpoints'],layer)
                         for arm,data in report['arms'].items()} for layer in layers}
    old,new=report['arms'][old_name],report['arms'][new_name]
    phases={name:{old_name:old['phases'][name],new_name:new['phases'][name],
            difference_name:new['phases'][name]-old['phases'][name]} for name in old['phases']}
    evaluations=[]
    for left,right in zip(old['evaluations'],new['evaluations']):
        if left['step']!=right['step']:
            raise ValueError('evaluation endpoints differ')
        evaluations.append(dict(step=left['step'],phase=left['phase'],**{
            old_name:{k:v['nats_per_token'] for k,v in left['metrics'].items()},
            new_name:{k:v['nats_per_token'] for k,v in right['metrics'].items()},
            difference_name:{k:right['metrics'][k]['nats_per_token']-v['nats_per_token']
                             for k,v in left['metrics'].items()}}))
    all_stream={arm:sum(r['nll_sum'] for r in data['steps'])/
                   sum(r['target_tokens'] for r in data['steps']) for arm,data in report['arms'].items()}
    result=dict(status='complete',source_checkpoint=report['checkpoint'],
        source_updates=report['source_updates'],protocol=report['protocol'],
        first_training_forward_exact=report.get('first_training_forward_exact'),
        source_checkpoint_unchanged=report['source_checkpoint_unchanged'],
        executed_code_unchanged=report['executed_code_unchanged'],
        executed_optimizer_updates=report['executed_optimizer_updates'],
        sidecars_preserve_complete_state=all(e['persistent_state_unchanged'] for d in report['arms'].values() for e in d['evaluations']),
        phases=phases,all_stream=dict(all_stream,**{difference_name:all_stream[new_name]-all_stream[old_name]}),
        evaluations=evaluations,reuse={arm:d['reuse'] for arm,d in report['arms'].items()},
        first_X_retention_and_reuse=traces,
        by_layer=by_layer,
        scope='fixed real-stream finite budget; finite utility is current full-objective ablation, not all-future value or convergence')
    if placement:
        result['placement_migration_preserves_existing_state']=report['placement_migration_preserves_existing_state']
        result['initial_output_minus_input']=report['initial_output_minus_input']
        for entry in evaluations:
            entry['output_minus_input_relative_to_initial']={
                name:value-report['initial_output_minus_input'][name]
                for name,value in entry[difference_name].items()}
        result['scope']='same saved body input/output placement pair; initial forward difference intentional; fixed real-stream budget, no convergence claim'
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--extend',action='store_true',help='add fields only; all existing aggregate values must remain identical')
    args=parser.parse_args()
    if args.output.exists() and not args.extend:
        raise FileExistsError('preserve existing analysis')
    report=json.loads(args.input.read_text(encoding='utf-8'))
    result=analyze(report)
    if args.output.exists():
        old=json.loads(args.output.read_text(encoding='utf-8'))
        if any(key not in result or value!=result[key] for key,value in old.items()):
            raise ValueError('extension would change prior evidence')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    brief={k:result[k] for k in ('phases','all_stream')}
    brief['first_X_retention_and_reuse']={arm:{k:v for k,v in values.items() if k!='records'}
        for arm,values in result['first_X_retention_and_reuse'].items()}
    print(json.dumps(brief,ensure_ascii=False))


if __name__=='__main__':
    main()
