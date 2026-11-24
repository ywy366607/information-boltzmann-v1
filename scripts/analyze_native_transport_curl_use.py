"""Actual matched-input mode use through autonomy, learning and physical state."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from scripts.measure_native_inner_loop import analyze, load, sha
from scripts.analyze_native_inner_loop_use import ports


def initial_path(directory):
    obs = directory/'observations'
    rows = [json.loads(line) for line in gzip.open(obs/'operations.jsonl.gz', 'rt', encoding='utf8')]
    calls = {r['data']['call_id']: r for r in rows if r['kind'] == 'relation_signal'}
    root = next(r['data']['call_id'] for r in calls.values() if r['event'] == 1 and
                r['data']['operation'] == 'compose' and r['data']['call_id'] == 22)
    mechanism = next(r['data'] for r in calls.values() if r['data']['parent_id'] == root and
                     r['data']['operation'] == 'mechanism' and r['data'].get('mother') is not None)
    executed = next(r['data'] for r in calls.values() if r['data']['parent_id'] == root and r['data']['operation'] == 'execute_programs')
    with (obs/'tensors.bin').open('rb') as stream:
        values = {key: load(stream, mechanism['prediction'][key]) for key in ('mean', 'scale2', 'y')}
        values['coordinated_mean'] = load(stream, executed['coordination']['mean'])
        values['coordinated_scale2'] = load(stream, executed['coordination']['scale2'])
        values['program_weights'] = load(stream, executed['weights'])
        values['direct_mean'] = load(stream, executed['predictions'][1]['mean'])
        values['direct_scale2'] = load(stream, executed['predictions'][1]['scale2'])
        weights = values['program_weights'].astype(float)
        precision_t = weights[0]/values['scale2'].astype(float)
        precision_d = weights[1]/values['direct_scale2'].astype(float)
        values['transport_precision_share'] = precision_t/(precision_t+precision_d)
    reuses = []
    for row in calls.values():
        d = row['data']
        if d['operation'] == 'mechanism' and row['event'] > 1 and (d['state']['identity'], d['state']['generation']) == (213, 4):
            reuses.append(dict(event=row['event'], update_clock=row['update_clock'], call_id=d['call_id'],
                              kind=d['kind'], transport=d['mother'] is not None))
    return values, reuses


def compare(a, b, sig_a, sig_b, pa, pb):
    external = []
    for x, y in zip(a['actions'], b['actions']):
        external.append(dict(event_a=x['event'], event_b=y['event'],
            choice_changed=x['choice'] != y['choice'], utterance_changed=x['utterance_hex'] != y['utterance_hex'],
            anchor_changed=x['world_anchor'] != y['world_anchor'],
            probability_max_abs=float(np.max(np.abs(np.array(x['probability'])-y['probability']))),
            heldout_pre_nll_delta=y['heldout_pre_nll']-x['heldout_pre_nll'],
            program_loss_delta=y['causal_program_loss']-x['causal_program_loss']))
    delta = sig_b[f'X_o_{b["phase"]["samples"][-1]["sample"]}']-sig_a[f'X_o_{a["phase"]["samples"][-1]["sample"]}']
    updates = []
    for x, y in zip(pa['parameter_updates'], pb['parameter_updates']):
        ma = {v['name']: v for v in x['parameters']}; mb = {v['name']: v for v in y['parameters']}
        differences = []
        for name in ma.keys() & mb.keys():
            if ma[name] != mb[name]:
                differences.append(dict(name=name, gradient_norm_a=ma[name]['gradient_norm'], gradient_norm_b=mb[name]['gradient_norm'],
                    gradient_norm_difference=mb[name]['gradient_norm']-ma[name]['gradient_norm'],
                    step_rms_a=ma[name]['sampled_applied_step_rms'], step_rms_b=mb[name]['sampled_applied_step_rms']))
        if differences:
            updates.append(dict(update_clock=x['update_clock'], changed_count=len(differences),
                top=sorted(differences, key=lambda d: abs(d['gradient_norm_difference']), reverse=True)[:8]))
    return dict(external=external, changed_choices=sum(r['choice_changed'] for r in external),
                changed_utterances=sum(r['utterance_changed'] for r in external),
                changed_anchors=sum(r['anchor_changed'] for r in external),
                max_external_probability_change=max(r['probability_max_abs'] for r in external),
                final_X_o_difference_norm=float(np.linalg.norm(delta)),
                final_X_o_difference_by_layer=np.linalg.norm(delta, axis=1).tolist(), updates=updates)


def main():
    p = argparse.ArgumentParser(); p.add_argument('work', type=Path); p.add_argument('output', type=Path)
    args = p.parse_args()
    names = ('baseline', 'bridge_cut', 'matched_cut')
    reports, signals, port, first, reuses, runtime = {}, {}, {}, {}, {}, {}
    for name in names:
        reports[name], signals[name] = analyze(args.work/name)
        port[name] = ports(args.work/name)
        first[name], reuses[name] = initial_path(args.work/name)
        runtime[name] = json.loads((args.work/name/'result.json').read_text(encoding='utf8'))
    target = runtime['bridge_cut']['transport_curl_interventions'][0]
    control = runtime['matched_cut']['transport_curl_interventions'][0]
    assert target['relative_motion_curl_remaining'] < 1e-4
    assert control['relative_motion_curl_change'] < 1e-4
    assert abs(target['input_delta_norm']-control['input_delta_norm']) < 1e-6
    assert np.linalg.norm(np.array(target['mean_delta'])-control['mean_delta']) < 1e-6
    pairs = [('baseline', 'bridge_cut'), ('baseline', 'matched_cut'), ('bridge_cut', 'matched_cut')]
    comparison = {}; initial = {}
    for a, b in pairs:
        key = f'{a}_to_{b}'
        comparison[key] = compare(reports[a], reports[b], signals[a], signals[b], port[a], port[b])
        initial[key] = {k: dict(norm=float(np.linalg.norm(first[b][k]-first[a][k])),
                                     max_abs=float(np.abs(first[b][k]-first[a][k]).max()),
                                     reference_norm=float(np.linalg.norm(first[a][k]))) for k in first[a]}
        initial[key]['prior_delta'] = port[b]['previews'][0]['continuation_prior_gap']-port[a]['previews'][0]['continuation_prior_gap']
        initial[key]['internal_probability_delta'] = port[b]['previews'][0]['probability'][1]-port[a]['previews'][0]['probability'][1]
        initial[key]['feature_gap_delta'] = port[b]['previews'][0]['content_feature_gap']-port[a]['previews'][0]['content_feature_gap']
        fa, fb = first[a], first[b]
        # Exact finite-change decomposition of the body's actual precision
        # consensus, using its two predicted means and per-coordinate scales.
        trust = (fb['transport_precision_share']-fa['transport_precision_share'])*(fa['mean']-fa['direct_mean'])
        means = (fb['transport_precision_share']*(fb['mean']-fa['mean']) +
                 (1-fb['transport_precision_share'])*(fb['direct_mean']-fa['direct_mean']))
        observed = fb['coordinated_mean']-fa['coordinated_mean']
        initial[key]['consensus_decomposition'] = dict(trust_change_norm=float(np.linalg.norm(trust)),
            means_change_norm=float(np.linalg.norm(means)), observed_change_norm=float(np.linalg.norm(observed)),
            reconstruction_residual=float(np.linalg.norm(observed-trust-means)))
    arrays = {f'{name}_{key}': value for name in names for key, value in signals[name].items()}
    arrays.update({f'{name}_first_{key}': value for name in names for key, value in first[name].items()})
    np.savez_compressed(args.output.with_suffix('.npz'), **arrays)
    result = dict(status='completed', work=str(args.work), arms={name:dict(
        summary=reports[name]['summary'], sources=reports[name]['sources'], first_choice=port[name]['previews'][0],
        mode_receipt=runtime[name]['transport_curl_interventions'][0],
        later_same_relation_calls=reuses[name], actions=reports[name]['actions'], phase=reports[name]['phase'],
        learning_updates=reports[name]['learning_updates']) for name in names},
        comparisons=comparison, first_path_comparisons=initial,
        arrays=dict(path=str(args.output.with_suffix('.npz')), sha256=sha(args.output.with_suffix('.npz'))),
        main_source_hashes_preserved=json.loads((args.work/'summary.json').read_text(encoding='utf8'))['main_source_hashes_preserved'])
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(dict(first_path=initial, comparisons={k:{f:v for f,v in d.items() if f not in ('external','updates','final_X_o_difference_by_layer')} for k,d in comparison.items()},
        first_update={k:d['updates'][:1] for k,d in comparison.items()}), ensure_ascii=False))


if __name__ == '__main__':
    main()
