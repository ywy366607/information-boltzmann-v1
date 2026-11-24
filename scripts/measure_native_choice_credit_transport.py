"""Join actual preview, binary choice, delayed credit and full-loop readouts.

No model is loaded and no trajectory is replayed. Inputs are the admitted
complete recordings. The linear selector score is reconstructed from its
recorded features/probabilities and checked against runtime eligibility norms.
"""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np

from scripts.measure_native_inner_loop import ROOT, load, sha


def read_arm(directory):
    obs = directory/'observations'
    rows = [json.loads(line) for line in gzip.open(obs/'operations.jsonl.gz', 'rt', encoding='utf8')]
    calls = {r['data']['call_id']: r for r in rows if r['kind'] == 'relation_signal'}
    events = {r['event']: r for r in rows if r['kind'] == 'event'}
    updates = [r for r in rows if r['kind'] == 'all_parameter_update_participation']
    external = [r for r in events.values() if r['data']['autonomous_event']['kind'] == 'external']
    previews, boundaries, certificates = [], [], []
    accumulated = np.zeros(388, dtype=np.float32)
    arrays = {}
    since_boundary = []
    norm_errors = []
    with (obs/'tensors.bin').open('rb') as stream:
        for r in rows:
            d = r['data']
            if r['kind'] == 'inner_loop_preview_ports':
                event = r['event']
                f = load(stream, d['selector_features']).astype(np.float32)
                prior = load(stream, d['selector_prior'])
                p = np.array(d['admission']['probability'], dtype=np.float32)
                s = int(d['admission']['selected'] == 'internal')
                assert f.shape == (2, 388) and p.shape == (2,)
                # Keep both FP32 coefficients, including their rounding.
                c = -p.copy(); c[s] += np.float32(1)
                score = c @ f
                accumulated += score
                actual = events[event]['data']['autonomous_event']
                assert actual['admission'] == d['admission']
                norm_error = abs(float(np.linalg.norm(accumulated))-actual['eligibility_norm'])
                norm_errors.append(norm_error)
                assert norm_error < 2e-5, (directory, event, norm_error)
                roots = sorted([v for v in calls.values() if v['event'] == event
                    and v['data']['operation'] == 'compose'
                    and v['data']['caller']['function'] == '_sample_internal_choice'],
                    key=lambda v: v['data']['call_id'])
                assert len(roots) == 2
                children = []
                for root in roots:
                    assert root['update_clock'] == r['update_clock']
                    modes = [v for v in calls.values() if v['data']['operation'] == 'mechanism'
                             and v['data']['parent_id'] == root['data']['call_id']]
                    assert len(modes) == 3
                    children.append({('mother' if v['data']['kind'] == 'mother' else
                        'transport' if v['data']['mother'] is not None else 'direct'):v['data']['call_id']
                        for v in modes})
                target = next(x for x in external if x['event'] > event)
                settled = next((x for x in external if x['event'] > target['event']), None)
                applied = [] if settled is None else [v for v in updates if v['event'] == settled['event']
                    and any(p['name'] == 'binding_choice.weight' for p in v['data']['parameters'])]
                assert len(applied) == int(settled is not None)
                clock = applied[0]['update_clock'] if applied else 0
                if settled is not None:
                    certificates.append([event, r['update_clock'], roots[0]['update_clock'],
                        roots[1]['update_clock'], target['event'], settled['event'], clock,
                        roots[0]['data']['call_id'], roots[1]['data']['call_id'],
                        children[0]['transport'], children[1]['transport']])
                delta = f[1].astype(float)-f[0]
                log_odds = float(np.log(float(p[1])/float(p[0])))
                item = dict(event=event, update_clock=r['update_clock'],
                    relation=[roots[0]['data']['state']['identity'], roots[0]['data']['state']['generation']],
                    compose_calls=[v['data']['call_id'] for v in roots], child_calls=children,
                    selected=s, probability=p.tolist(), prior_gap=float(prior[1]-prior[0]),
                    total_log_odds=log_odds, learned_head_log_odds=log_odds-float(prior[1]-prior[0]),
                    feature_gap_by_block=[float(np.linalg.norm(delta[a:b])) for a,b in
                                          ((0,128),(128,256),(256,384),(384,388))],
                    score_norm=float(np.linalg.norm(score)), content_score_norm=float(np.linalg.norm(score[:384])),
                    accumulated_norm=float(np.linalg.norm(accumulated)), runtime_norm=actual['eligibility_norm'],
                    norm_reconstruction_error=norm_error, boundary_event=target['event'],
                    settled_at_event=None if settled is None else settled['event'], applied_at_update=clock or None)
                if d['hold_preview']:
                    assert np.array_equal(f[0,:384], f[1,:384])
                previews.append(item); since_boundary.append(event)
                arrays[f'features_{event}'] = f
                arrays[f'score_{event}'] = score
            elif r['kind'] == 'event' and d['autonomous_event']['kind'] == 'external':
                receipt = d['generative_relation']['autonomous']
                assert len(since_boundary) == receipt['choices']
                event = r['event']
                future = next((v for v in external if v['event'] > event), None)
                factor = None if future is None else future['data']['td_previous']['advantage']
                arrays[f'boundary_eligibility_{event}'] = accumulated.copy()
                if factor is not None:
                    arrays[f'boundary_actor_contribution_{event}'] = -factor*accumulated.astype(float)
                boundaries.append(dict(event=event, preview_events=since_boundary,
                    content_eligibility_norm=float(np.linalg.norm(accumulated[:384])),
                    tag_eligibility_norm=float(np.linalg.norm(accumulated[384:])),
                    total_eligibility_norm=float(np.linalg.norm(accumulated)),
                    eligibility_energy_by_block=[float(np.square(accumulated[a:b].astype(float)).sum())
                        for a,b in ((0,128),(128,256),(256,384),(384,388))],
                    content_energy_fraction=float(np.square(accumulated[:384].astype(float)).sum()/
                        max(np.square(accumulated.astype(float)).sum(),1e-30)),
                    later_factor=factor, later_event=None if future is None else future['event'],
                    before_clip_contribution_norm=None if factor is None else
                        abs(factor)*float(np.linalg.norm(accumulated)),
                    source_settlement=d['settlement']))
                accumulated = np.zeros(388, dtype=np.float32); since_boundary = []
    structural = []
    for r in rows:
        if r['kind'] == 'structural_relation_dependencies':
            edges = r['data']['edges']
            structural.append(dict(event=r['event'], update_clock=r['update_clock'],
                count=len(edges), identity_version_digest=hashlib.sha256(
                    json.dumps(edges, separators=(',',':')).encode()).hexdigest()))
    return dict(previews=previews, boundaries=boundaries, structural_updates=structural,
        maximum_eligibility_norm_error=max(norm_errors),
        source=dict(path=str(obs/'operations.jsonl.gz'), sha256=sha(obs/'operations.jsonl.gz'))), arrays, certificates


def compare(a, b, aa, ab):
    assert [p['event'] for p in a['previews']] == [p['event'] for p in b['previews']]
    previews = []
    for x,y in zip(a['previews'], b['previews']):
        e=x['event']; f0=aa[f'features_{e}'].astype(float); f1=ab[f'features_{e}'].astype(float)
        p0=x['probability'][1];p1=y['probability'][1];s=x['selected']
        assert s == y['selected']
        # Exact real binary-score finite difference. Original two-coefficient
        # FP32 reconstruction is stored separately to expose rounding.
        probability_term=(p0-p1)*(f0[1]-f0[0])
        content_term=(s-p1)*((f1[1]-f1[0])-(f0[1]-f0[0]))
        raw=ab[f'score_{e}'].astype(float)-aa[f'score_{e}']
        previews.append(dict(event=e, probability_delta=p1-p0,
            prior_log_odds_delta=y['prior_gap']-x['prior_gap'],
            learned_head_log_odds_delta=y['learned_head_log_odds']-x['learned_head_log_odds'],
            score_delta_norm=float(np.linalg.norm(raw)),
            probability_term_norm=float(np.linalg.norm(probability_term)),
            content_term_norm=float(np.linalg.norm(content_term)),
            fp32_decomposition_residual=float(np.linalg.norm(raw-probability_term-content_term))))
    boundaries=[]
    for x,y in zip(a['boundaries'],b['boundaries']):
        assert x['event'] == y['event']
        e=x['event']; diff=ab[f'boundary_eligibility_{e}'].astype(float)-aa[f'boundary_eligibility_{e}']
        key=f'boundary_actor_contribution_{e}'
        boundaries.append(dict(event=e, eligibility_delta_norm=float(np.linalg.norm(diff)),
            content_delta_norm=float(np.linalg.norm(diff[:384])), tag_delta_norm=float(np.linalg.norm(diff[384:])),
            factor_a=x['later_factor'], factor_b=y['later_factor'],
            contribution_delta_norm=None if key not in aa else float(np.linalg.norm(ab[key]-aa[key]))))
    return dict(previews=previews,boundaries=boundaries,
        structural_updates_equal=a['structural_updates']==b['structural_updates'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    groups={'inner':('scratch/native_inner_loop_use_2401_20261117',
                     ('baseline','bridge_cut','matched_cut','order_only')),
            'curl':('scratch/native_transport_curl_use_2401_20261117',
                    ('baseline','bridge_cut','matched_cut'))}
    report={}; all_arrays={}; witnesses=[]
    for group,(folder,names) in groups.items():
        arms={}; arrays={}
        for name in names:
            arms[name],arrays[name],cert=read_arm(ROOT/folder/name)
            witnesses.extend(cert)
            all_arrays.update({f'{group}_{name}_{k}':v for k,v in arrays[name].items()})
        comparisons={name:compare(arms['baseline'],arms[name],arrays['baseline'],arrays[name]) for name in names[1:]}
        if group=='curl':comparisons['target_to_control']=compare(arms['bridge_cut'],arms['matched_cut'],arrays['bridge_cut'],arrays['matched_cut'])
        report[group]=dict(arms=arms,comparisons=comparisons)
    proof=ROOT/'lean/CGBAxioms/reviews/NativeChoiceCreditActual2401.lean'
    proof.write_text('''import Mathlib.Tactic
namespace CGB.NativeChoiceCreditActual2401
def witnesses : List (List Nat) := DATA
def valid (r : List Nat) : Bool :=
  r.length == 11 && r[1]! == r[2]! && r[2]! == r[3]! &&
  r[0]! < r[4]! && r[4]! < r[5]! && r[1]! < r[6]! &&
  r[7]! < r[9]! && r[9]! < r[8]! && r[8]! < r[10]!
example : witnesses.all valid = true := by native_decide
example : witnesses.length = COUNT := by native_decide
end CGB.NativeChoiceCreditActual2401
'''.replace('DATA',str(witnesses)).replace('COUNT',str(len(witnesses))),encoding='utf8')
    logs=[]
    proofs=[ROOT/'lean/CGBAxioms/reviews/NativeChoiceCreditTransport.lean',proof]
    for path in proofs:
        done=subprocess.run(['lake','env','lean',str(path.relative_to(ROOT/'lean/CGBAxioms'))],
            cwd=ROOT/'lean/CGBAxioms',capture_output=True,text=True,encoding='utf8')
        logs.append(str(path)+'\n'+done.stdout+done.stderr)
        args.output.with_suffix('.lean.log').write_text('\n'.join(logs),encoding='utf8')
        if done.returncode:raise RuntimeError(logs[-1])
    output=args.output.with_suffix('.npz');np.savez_compressed(output,**all_arrays)
    manifests=['cgb_native_inner_loop_use_analysis_2401_20261117.json',
        'cgb_native_inner_hodge_transport_2401_20261117.json',
        'cgb_native_transport_curl_use_2401_20261117.json',
        'cgb_native_paper_phase_dirac_2401_20261117.json',
        'cgb_native_relation_percolation_2401_20261117.json']
    inner=json.loads((ROOT/'results/published'/manifests[0]).read_text(encoding='utf8'))
    # Keep native carriers distinct, joined by actual event/call/update IDs.
    # The selector does not acquire fabricated simplices or angle signals.
    atlas={}
    for name, arm in report['inner']['arms'].items():
        prior=inner['arms'][name]
        pair_by_event={v['event']:v for v in prior['preview_pairs']}
        choice_paths=[]
        for preview in arm['previews']:
            p=pair_by_event[preview['event']]
            assert p['compose_calls']==preview['compose_calls']
            assert p['update_clock']==preview['update_clock'] and p['relation']==preview['relation']
            choice_paths.append(dict(event=preview['event'],relation=preview['relation'],
                compose_calls=preview['compose_calls'],child_calls=preview['child_calls'],
                physical_X_o_change_by_layer=p['X_o_change_by_layer'],
                physical_P_read_change_by_layer=p['momentum_change_by_layer'],
                relation_prediction_changes=p['predictions'],
                source_to_choice=preview,
                settlement=next(v for v in arm['boundaries'] if v['event']==preview['boundary_event']),
                delayed_dynamics_update=next((v for v in prior['learning_updates']
                    if v['update_clock']==preview['applied_at_update']),None)))
        atlas[name]=dict(choice_paths=choice_paths,committed_phase_trajectory=prior['committed_event_changes'],
            structural_updates=arm['structural_updates'],
            measured_update_difference=None if name=='baseline' else inner['comparisons'][name]['update_differences'],
            first_event_hodge_dirac_source=manifests[1])
    result=dict(status='completed',main_body_cycle=2401,new_training_updates=0,new_world_calls=0,
        groups=report,actual_witnesses=len(witnesses),
        full_loop_atlas=atlas,
        proofs=[dict(path=str(x),sha256=sha(x)) for x in proofs],
        integration_sources=[dict(path=str(ROOT/'results/published'/x),sha256=sha(ROOT/'results/published'/x)) for x in manifests],
        arrays=dict(path=str(output),sha256=sha(output)),
        source_files={x:sha(ROOT/x) for x in ('scripts/ib_local/autonomous_cycle.py',
            'scripts/ib_local/navigation_generative_cycle.py','scripts/ib_local/world_learning_cycle.py',
            'scripts/ib_local/language_navigation_v2.py')})
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({g:{k:dict(first_preview=v['previews'][0],first_boundary=v['boundaries'][0],
        structure_equal=v['structural_updates_equal']) for k,v in r['comparisons'].items()} for g,r in report.items()},ensure_ascii=False))


if __name__=='__main__':main()
