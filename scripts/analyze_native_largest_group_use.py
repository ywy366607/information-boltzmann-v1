"""Compare the large-component interventions in complete normal learning.

Read traces in a stream. Exact fact/read pairing requires equal scope identity
and clock. Actual prediction errors are paired only for the same real target,
not different imagined targets or unrelated latent representations.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path

import numpy as np

from scripts.analyze_native_path_composition import load
from scripts.analyze_native_relation_mode_use import compare_choices, external
from scripts.measure_native_relation_percolation import build, read_frames
from scripts.run_native_relation_mode_use import write


def inspect_observations(directory):
    mechanisms = defaultdict(list)
    scored, event_boundaries, updates, designs = [], [], [], []
    with gzip.open(directory/'operations.jsonl.gz', 'rt', encoding='utf8') as f, \
            (directory/'tensors.bin').open('rb') as tensors:
        for line in f:
            row = json.loads(line)
            kind, data = row['kind'], row['data']
            if kind == 'event' and data['autonomous_event']['kind'] == 'external':
                event_boundaries.append(row['event'])
            if kind == 'all_parameter_update_participation':
                updates.append(dict(clock=row['update_clock'], parameters=[
                    {k: p[k] for k in ('name', 'gradient_norm', 'sampled_applied_step_rms')}
                    for p in data['parameters']]))
            if kind != 'relation_signal':
                continue
            if data['operation']=='_bound_evidence_design':
                designs.append(dict(event=row['event'],cycle=row['external_cycle'],
                    update_clock=row['update_clock'],port=data['kind'],transport=data['transport'],
                    fact_ids=[e['fact_id'] for e in data['facts']],
                    observed=[e['observed'] for e in data['facts']]))
            if data['operation'] == 'mechanism':
                key = (row['event'], row['update_clock'], data['prediction']['mean']['tensor_ref']['sha256'])
                mechanisms[key].append(dict(relation=[data['state']['identity'],data['state']['generation']],
                    caller=data['caller'], port=data['kind'], transport=data['mother'] is not None,
                    mean=data['prediction']['mean'], call_id=data['call_id']))
            if data['operation'] == 'log_score' and data['caller']['function'] == '_outcome_loss':
                key = (row['event'], row['update_clock'], data['prediction']['mean']['tensor_ref']['sha256'])
                matching = mechanisms.get(key, [])
                # A scalar score that could identify multiple different original
                # calls is retained as ambiguous rather than arbitrarily assigned.
                if len(matching) != 1:
                    continue
                mechanism = matching[0]
                target = load(tensors, data['target'])
                mean = load(tensors, data['prediction']['mean'])
                scored.append(dict(event=row['event'], cycle=row['external_cycle'],
                    update_clock=row['update_clock'], relation=mechanism['relation'],
                    port=mechanism['port'], transport=mechanism['transport'],
                    actual_target_sha256=data['target']['tensor_ref']['sha256'],
                    mse=float(np.square(mean-target).mean()),
                    native_log_score=float(load(tensors, data['score']).mean())))
    final = None
    for path in sorted(directory.glob('physical_*.npz')):
        with np.load(path) as data:
            for i, event in enumerate(data['event']):
                if event == event_boundaries[-1] and data['kind'][i] == 'event_after':
                    final = {k: data[k][i].astype(float) for k in ('node_energy','theta','omega')}
    assert final is not None
    return dict(scored=scored, updates=updates, final_physical=final,designs=designs)


def compare_reads(a, b):
    def keyed(rows):
        result, counts = {}, Counter()
        for r in rows:
            base = (r['external_cycle'], r['update_clock'], tuple(r['relation']))
            counts[base] += 1
            result[base+(counts[base],)] = r
        return result
    left, right = keyed(a), keyed(b)
    paired = sorted(left.keys() & right.keys())
    return dict(baseline_reads=len(a), intervened_reads=len(b), exact_scope_clock_pairs=len(paired),
        changed_fact_lists=sum(left[k]['facts'] != right[k]['facts'] for k in paired),
        changed_fact_multisets=sum(Counter(left[k]['facts']) != Counter(right[k]['facts']) for k in paired),
        order_only_fact_changes=sum(left[k]['facts'] != right[k]['facts'] and
            Counter(left[k]['facts']) == Counter(right[k]['facts']) for k in paired),
        changed_unknown_lists=sum(left[k]['unknowns'] != right[k]['unknowns'] for k in paired),
        unpaired_baseline=len(left.keys()-right.keys()), unpaired_intervened=len(right.keys()-left.keys()),
        examples=[dict(cycle=k[0], update_clock=k[1], relation=k[2],
            baseline_facts=left[k]['facts'], intervened_facts=right[k]['facts'])
            for k in paired if left[k]['facts'] != right[k]['facts']][:8])


def compare_scored(a, b):
    def keyed(rows):
        result, count = {}, Counter()
        for r in rows:
            key=(r['cycle'],r['update_clock'],tuple(r['relation']),r['port'],r['transport'],r['actual_target_sha256'])
            count[key]+=1
            result[key+(count[key],)]=r
        return result
    left,right=keyed(a),keyed(b)
    return [dict(cycle=k[0],update_clock=k[1],relation=k[2],port=k[3],transport=k[4],
                 actual_target_sha256=k[5],baseline_mse=left[k]['mse'],
                 intervened_mse=right[k]['mse'],baseline_log_score=left[k]['native_log_score'],
                 intervened_log_score=right[k]['native_log_score']) for k in left.keys() & right.keys()]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('work',type=Path)
    p.add_argument('output',type=Path)
    args=p.parse_args()
    summary=json.loads((args.work/'summary.json').read_text(encoding='utf8'))
    plan=json.loads((args.work/'plan.json').read_text(encoding='utf8'))
    assert plan['checkpoint']==summary['plan']['checkpoint']
    baseline_path=(Path(plan['baseline_reused_from']) if plan.get('baseline_reused_from')
                   else args.work/'baseline')
    baseline=json.loads((baseline_path/'result.json').read_text(encoding='utf8'))
    bo=inspect_observations(baseline_path/'observations')
    results={}
    names=['bridge_cut','matched_cut']
    if (args.work/'order_only/result.json').exists():
        names.append('order_only')
    for name in names:
        result=json.loads((args.work/name/'result.json').read_text(encoding='utf8'))
        current=inspect_observations(args.work/name/'observations')
        choices=compare_choices(baseline,result)
        physical={}
        for k,x in bo['final_physical'].items():
            y=current['final_physical'][k]; d=y-x
            physical[k]=dict(relative_l2=float(np.linalg.norm(d)/np.linalg.norm(x)),
                max_abs=float(np.abs(d).max()),changed_layers=int(np.any(d!=0,axis=-1).sum()),
                per_layer_l2=np.linalg.norm(d,axis=-1).tolist())
        updates=[]
        for a,b in zip(bo['updates'],current['updates']):
            if a['clock']!=b['clock']:
                break
            ap={r['name']:r for r in a['parameters']}; bp={r['name']:r for r in b['parameters']}
            shared=ap.keys()&bp.keys()
            updates.append(dict(clock=a['clock'],baseline_participants=len(ap),intervened_participants=len(bp),
                changed_gradient_norms=sum(ap[n]['gradient_norm']!=bp[n]['gradient_norm'] for n in shared),
                changed_sampled_steps=sum(ap[n]['sampled_applied_step_rms']!=bp[n]['sampled_applied_step_rms'] for n in shared)))
        curves=[]
        for f in read_frames(args.work/name/'observations'):
            graph=build(f)
            curves.append(dict(cycle=f['external_cycle'],largest=len(graph['largest']),
                total=graph['archive_count'],edges=len(graph['witnesses'])))
        results[name]=dict(seconds=result['seconds'],optimizer_calls=result['actual_optimizer_calls'],
            whole_loop=choices,actual_reads=compare_reads(baseline['read_calls'],result['read_calls']),
            read_view_source='registered loaded_state constraints; any downstream order control is logged separately',
            native_encoded_fact_rows=current['designs'],
            removed=len(result['withdrawn_references']),restored=len(result['natively_restored_references']),
            original_group_reads=result['identified_group_read_calls'],
            R_curve=curves,final_R=result['final_R'],
            same_real_target_scores=compare_scored(bo['scored'],current['scored']),
            paired_real_target_rule='same actual target bytes, port, scope, transport and update clock; imagined scores excluded',
            updates=updates,final_all_layer_physical=physical,
            endpoint_hash_equal={k:baseline['after'][k]==v for k,v in result['after'].items()},
            observer_rng_equal=result['observer_rng_equal'],provider_calls=result['provider_calls'])
        if name=='order_only':
            assert not result['withdrawn_references']
            assert result['before']==baseline['before']
            results[name]['fact_order_interventions']=result['fact_order_interventions']
            cut=json.loads((args.work/'bridge_cut/result.json').read_text(encoding='utf8'))
            results[name]['versus_largest_cut']=compare_choices(cut,result)
    report=dict(status='completed',plan=plan,source_summary=str(args.work/'summary.json'),
        main_unchanged=summary['main_checkpoint_and_archive_unchanged'],
        complete_starts_equal=summary['complete_starts_equal'],arms=results)
    audit=args.work/'first_fact_order_audit.json'
    if audit.exists():
        report['first_fact_order_audit']=json.loads(audit.read_text(encoding='utf8'))
    order_status=args.work/'order_control_status.json'
    if 'order_only' in results:
        status=(dict(status=summary['status'],
                     main_checkpoint_and_archive_unchanged=summary['main_checkpoint_and_archive_unchanged'],
                     whole_carried_order_equivalence=True)
                if summary.get('whole_carried_order_equivalence') else
                json.loads(order_status.read_text(encoding='utf8')))
        assert status['status']=='completed' and status['main_checkpoint_and_archive_unchanged']
        report['order_control_verification']=status
    write(args.output,report)
    for name,result in results.items():
        print(json.dumps(dict(arm=name,removed=result['removed'],restored=result['restored'],
            final_R=result['final_R'],reads=result['actual_reads'],
            changes={k:result['whole_loop'][k] for k in ('changed_sampled_choices','changed_world_landings','changed_utterances','max_choice_probability_change')},
            energy_relative_l2=result['final_all_layer_physical']['node_energy']['relative_l2']),ensure_ascii=False))


if __name__=='__main__':
    main()
