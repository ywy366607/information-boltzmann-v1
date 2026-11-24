"""Summarize completed native supply arms by separate body functions."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import mean


def load(path):
    return json.loads(path.read_text(encoding='utf8'))


def lines(path):
    return [json.loads(x) for x in path.read_text(encoding='utf8').splitlines() if x.strip()]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda:stream.read(1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def arm(root):
    result = load(root/'run/result.json')
    assert result['status'] == 'completed' and result['provider_calls'] == 0
    events = lines(root/'run/events.jsonl')
    external = [r for r in events if r['autonomous_event']['kind'] == 'external']
    evaluation = load(root/'run/evaluations.json')
    first, last = evaluation[0], evaluation[-1]
    before = {(q['relation'], q['query']):q for q in first['queries']}
    after = {(q['relation'], q['query']):q for q in last['queries']}
    assert before.keys() == after.keys() and len(before) == 64
    old_delta = [after[k]['score']-before[k]['score'] for k in before]
    readings = load(root/'independent_scores.json')['readings']
    final_text = readings[str(result['cycles'])]
    initial_text = readings['0']
    objectives = load(root/'supply_runtime.json')
    training = [r for r in objectives['objectives'] if r['grad_enabled']]
    finite = [r['causal_use']['finite_U'] for r in external]
    assert all(f['verified'] for f in finite)
    actions = load(root/'observations.json')
    timing = lines(root/'dispatch_timing.jsonl')
    supports = Counter(s for r in external for call in r['read_use'] for s in call['support_ids'])
    phases = Counter(r['autonomous_event'].get('phase', r['autonomous_event']['kind']) for r in events)
    checkpoint = Path(result['checkpoint'])
    assert sha(checkpoint) == result['checkpoint_sha256']
    supply_progress = None
    if training:
        n = min(4, len(training))
        supply_progress = dict(training_objectives=len(training),
            first_mean={k:mean(r[k] for r in training[:n]) for k in ('q', 'k', 'k_exc', 'k_tr', 'loss')},
            last_mean={k:mean(r[k] for r in training[-n:]) for k in ('q', 'k', 'k_exc', 'k_tr', 'loss')},
            active=sum(r['loss'] > 0 for r in training),
            first_current_supply_gradient_norm=next(r['first_current_supply_gradient_norm']
                for r in training if 'first_current_supply_gradient_norm' in r))
        assert all(r['current_graph_live'] for r in training)
    return dict(root=str(root), completion=result,
        old_fact_prediction=dict(count=64, initial_score=first['mean_score'],
            final_score=last['mean_score'], score_change=last['mean_score']-first['mean_score'],
            initial_mse=first['mean_mse'], final_mse=last['mean_mse'],
            mse_drop=first['mean_mse']-last['mean_mse'],
            improved=sum(d > 0 for d in old_delta), worsened=sum(d < 0 for d in old_delta),
            trajectory=[{k:r[k] for k in ('cycle', 'mean_score', 'mean_mse')} for r in evaluation]),
        language_prediction=dict(initial_nats_per_byte=initial_text['mean_nll'],
            final_nats_per_byte=final_text['mean_nll'],
            improvement=initial_text['mean_nll']-final_text['mean_nll'],
            segments_improvement=[a-b for a,b in zip(initial_text['segment_nll'], final_text['segment_nll'])]),
        lived_outcome_prediction=dict(mean_replayed_outcome_error=mean(r['outcome_error'] for r in external),
            errors=[r['outcome_error'] for r in external],
            mother_prequential=[r['causal_program']['mother_prequential'] for r in external],
            mean_mother_prequential_mse=mean(r['causal_program']['mother_prequential']['mse']
                                           for r in external),
            mean_mother_prequential_log_score=mean(r['causal_program']['mother_prequential']['log_score']
                                                 for r in external),
            decision_prequential=[r['causal_program']['decision_prequential'] for r in external],
            mean_relation_prediction_tv=mean(r['relation_wood']['choice_probability_tv_without_relation']
                                            for r in external)),
        experience_use=dict(support_calls=sum(len(r['read_use']) for r in external),
            distinct_supports=len(supports), support_frequency=dict(supports),
            observed_episode=[f['observed_episode'] for f in finite],
            natural_revisits=sum(bool(r['causal_program']['roles']['resumed_unknown']) for r in external),
            retention=[r['causal_program'].get('retention_organization') for r in external]),
        actions={k:v['action'] for k,v in actions.items() if 'action' in v},
        schedule=dict(phases=phases, internal_ticks=[r['internal_ticks'] for r in external]),
        cost=dict(dispatch_seconds=sum(r['dispatch_seconds'] for r in timing),
            completed_external_rounds=len(external), optimizer_updates=result['updates'],
            external_bytes=result['external_bytes'],
            world_activation_peak_bytes=max(r['world_activation_bytes'] for r in external),
            checkpoint_bytes=checkpoint.stat().st_size),
        contracts=dict(finite_U_verified=len(finite),
            max_policy_replay_error=max(r['policy_replay_max_error'] for r in external)),
        supply=supply_progress)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    status = load(a.work/'pipeline_status.json')
    assert status['status'] == 'completed' and status['completed'] == ['original', 'supply']
    protocol = load(a.work/'protocol.json')
    arms = dict(original=arm(a.work/'original'), supply=arm(a.candidate))
    assert sha(Path(protocol['source'])) == protocol['source_sha256']
    first = arms['original']['actions']['1']
    second = arms['supply']['actions']['1']
    assert first == second, 'pre-training action must match exactly'
    originals = arms['original']['actions']
    candidates = arms['supply']['actions']
    differences = []
    for key, left in originals.items():
        right = candidates[key]
        differences.append(dict(round=int(key), option_changed=left['option'] != right['option'],
            utterance_changed=left['utterance_hex'] != right['utterance_hex'],
            production_probability_tv=.5*sum(abs(x-y) for x,y in zip(left['production'],right['production'])),
            source_changed=left['source_sha256'] != right['source_sha256']))
    def path_rows(root):
        rows = [r for r in lines(root/'run/events.jsonl')
                if r['autonomous_event']['kind'] == 'external']
        return [dict(binding=r['causal_program']['roles']['binding_signature'],
            reads=[{k:x[k] for k in ('target_fact_id', 'support_ids', 'relation')}
                   for x in r['read_use']],
            retention={k:r['causal_program']['retention_organization']['trace'][k]
                       for k in ('old_ids', 'retained', 'released')}) for r in rows]
    original_paths = path_rows(a.work/'original')
    supply_paths = path_rows(a.candidate)
    path_match = [x == y for x,y in zip(original_paths, supply_paths)]
    value = dict(schema=1, protocol=protocol, arms=arms, action_differences=differences,
        functional_differences=dict(
            old_fact_score_change=arms['supply']['old_fact_prediction']['score_change']-
                arms['original']['old_fact_prediction']['score_change'],
            language_prediction_improvement=arms['supply']['language_prediction']['improvement']-
                arms['original']['language_prediction']['improvement'],
            dispatch_seconds=arms['supply']['cost']['dispatch_seconds']-arms['original']['cost']['dispatch_seconds']),
        source_sha256=protocol['source_sha256'], initial_action_exact=True,
        binding_read_and_retention_exact=path_match,
        question_premise_signatures_equal=[x['binding'] == y['binding']
                                           for x,y in zip(original_paths, supply_paths)],
        actual_read_paths_equal=[x['reads'] == y['reads']
                                for x,y in zip(original_paths, supply_paths)],
        retention_identities_equal=[x['retention'] == y['retention']
                                    for x,y in zip(original_paths, supply_paths)],
        measurement_units='old facts: per-latent-coordinate log score/MSE; language: nats/byte; time: seconds',
        functional_aggregation='separate readings for separate functions')
    port_names = ('2805', 'original_2813', 'supply_2813')
    port_files = [Path('results/published')/f'cgb_native_supply_ports_{name}_20261122.json'
                  for name in port_names]
    if all(p.exists() for p in port_files):
        value['history_supply_endpoints'] = {
            name:{k:load(file)['roles']['history'][k]
                  for k in ('q', 'k', 'k_exc', 'k_tr', 'loss', 'samples')}
            for name,file in zip(port_names, port_files)}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps(value['functional_differences'],ensure_ascii=False))


if __name__ == '__main__':
    main()
