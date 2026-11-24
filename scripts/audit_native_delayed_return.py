"""Bind delayed selector credit to completed paired dispatches and checkpoint tensors.

Read only trusted local archives. Time is cumulative dispatch time, evaluated
at recorded dispatch boundaries. The historical scalar projection holds each
logged menu fixed and shifts the internal selector logit by a common amount.
"""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

import torch

from scripts.delayed_return_jet import DatedEligibility


def sources_for_returns(timing, rows):
    groups, sources = [], []
    elapsed = 0.
    group = dict(sources=None, arrival=None, delivery=None)
    for tick, row in zip(timing, rows, strict=True):
        event = row['autonomous_event']
        if tick['event'] != event or tick['before_external'] != len(groups):
            raise ValueError('unmatched dispatch/event record')
        elapsed += tick['dispatch_seconds']
        kind, phase = event['kind'], event.get('phase')
        if kind in ('internal', 'release'):
            p_release, p_internal = event['admission']['probability']
            internal = kind == 'internal'
            sources.append(dict(event=event['event'], time=elapsed, kind=kind,
                within_learning=event.get('within_external', False),
                score=(1-p_internal) if internal else -p_internal,
                logp=math.log(p_internal if internal else p_release),
                prefix_norm=event['eligibility_norm'], prefix_logp=event['eligible_logp'],
                internal_relation=(event['internal']['relation'], event['internal']['version'])
                    if internal else None))
            assert event['choices'] == len(sources)
            assert math.isclose(sum(s['logp'] for s in sources), event['eligible_logp'], abs_tol=2e-6)
        elif phase == 'world_arrived':
            group.update(sources=sources, arrival=(event['event'], elapsed))
            sources = []
        elif phase == 'continuation_updated':
            # _advance receives the previous actor on the last text piece.
            group['delivery'] = (event['event'], elapsed)
        elif kind == 'external':
            assert group['arrival'] and group['delivery']
            receipt = row['generative_relation']['autonomous']
            if groups:  # First release predates this trace.
                carried = group['sources']
                assert carried and carried[-1]['kind'] == 'release'
                assert len(carried) == receipt['choices']
                assert sum(s['kind'] == 'internal' for s in carried) == receipt['internal_steps']
                assert carried[-1]['prefix_logp'] == receipt['sampled_logp']
                prev = groups[-1]['row']
                td = row['td_previous']
                assert td['g'] == prev['metal_g'] and td['choice'] == prev['causal_use']['choice']
                assert row['delayed_actor_applied'] == prev['delayed_actor_applied'] + 1
            group.update(row=row, end=(event['event'], elapsed))
            groups.append(group)
            group = dict(sources=None, arrival=None, delivery=None)
    assert group['arrival'] is None and groups
    return groups, sources


def project(sources, endpoint, factor):
    """Evaluate the fixed-menu score channel from two moments and explicit sources."""
    clock, count = endpoint
    z = torch.zeros((), dtype=torch.double)
    state = DatedEligibility(z, z)
    last = 0.
    for s in sources:
        state.advance(s['time']-last, source=z.new_tensor(s['score']))
        last = s['time']
    e, age = state.read(z.new_tensor(factor), count-last)
    parts = [dict(source_event=s['event'], kind=s['kind'],
                  within_learning=s['within_learning'], internal_relation=s['internal_relation'],
                  dispatches=clock-s['event'], seconds=count-s['time'],
                  credit=factor*s['score']) for s in sources]
    error = max(abs(float(e)-math.fsum(p['credit'] for p in parts)),
                abs(float(age)-math.fsum(p['credit']*p['seconds'] for p in parts)))
    assert error < 1e-12
    return dict(endpoint_event=clock, eligibility=float(e), age_moment=float(age),
        source_count=len(sources), joint_rank=min(2, len({p['seconds'] for p in parts})),
        last_stamp_error=abs(float(age)-float(e)*parts[-1]['seconds']) if parts else None,
        reconstruction_error=error, sources=parts)


def checkpoint_tail(state, final, sources):
    """Recover exactly what was carried, retaining the two ownership stages."""
    row = final['row']
    eng = state['engine']
    assert state['cycle'] == row['cycle'] and eng['autonomous_event_clock'] == final['end'][0]
    assert eng['prev']['g'] == row['metal_g'] and eng['prev']['choice'] == row['causal_use']['choice']
    assert eng['autonomous_choices'] == len(sources) == 2
    assert eng['autonomous_internal_steps'] == sum(s['kind'] == 'internal' for s in sources)
    assert eng['autonomous_logp'] == sources[-1]['prefix_logp']
    nxt, = eng['autonomous_eligibility']
    action, = eng['autonomous_action_eligibility']
    owner = 'binding_choice.weight'
    assert state['model'][owner].shape == nxt.shape == action.shape
    assert math.isclose(float(nxt.norm()), sources[-1]['prefix_norm'], abs_tol=2e-6)
    assert math.isclose(float(action.norm()), final['sources'][-1]['prefix_norm'], abs_tol=2e-6)

    # The saved action actor also includes the same head's question-policy score.
    choice, = eng['_binding_choices']
    features = choice['features'].double()
    probs = features.new_tensor(row['generative_relation']['query_probabilities'])
    assert math.isclose(math.log(float(probs[choice['selected']])), choice['sampled_logp'], abs_tol=2e-6)
    q_score = features[choice['selected']] - probs @ features
    index, = [i for i, g in enumerate(eng['prev']['actor']) if g is not None and g.shape == action.shape]
    actor = eng['prev']['actor'][index].double()
    reconstructed = action.double() + q_score.reshape_as(action)
    torch.testing.assert_close(actor, reconstructed, atol=2e-6, rtol=2e-6)

    # A = age_last * E + (t_last - t_first) * g_first.  The second term's
    # norm is identifiable even though the archive lost g_first's direction.
    age_gap = sources[1]['time']-sources[0]['time']
    return dict(parameter=owner, parameter_coordinates=nxt.numel(),
        action_selector_norm=float(action.norm()), action_question_norm=float(q_score.norm()),
        action_actor_norm=float(actor.norm()),
        action_actor_reconstruction_max_error=float((actor-reconstructed).abs().max()),
        next_selector_norm=float(nxt.norm()), next_selector_first_norm=sources[0]['prefix_norm'],
        next_selector_ready=eng['autonomous_ready'], next_selector_source_gap_seconds=age_gap,
        next_selector_last_stamp_error_norm=age_gap*sources[0]['prefix_norm'])


def audit(arm):
    def jsonl(path):
        return [json.loads(line) for line in path.read_text(encoding='utf8').splitlines()]

    root = arm/'run'
    result = json.loads((root/'result.json').read_text(encoding='utf8'))
    timing, events = jsonl(arm/'dispatch_timing.jsonl'), jsonl(root/'events.jsonl')
    assert result['status'] == 'completed' and result['provider_calls'] == 0
    groups, sources = sources_for_returns(timing, events)
    assert groups[-1]['row']['cycle'] == result['cycle']
    returns = []
    for before, after in zip(groups, groups[1:]):
        row, future = before['row'], after['row']
        td = future['td_previous']
        factor = td.get('advantage_head', td['advantage'])
        returns.append(dict(action_cycle=row['cycle'], action=row['choice'],
            action_question=row['causal_use']['unknown'], action_relation=row['causal_use']['relation'],
            future_cycle=future['cycle'], future_question=future['causal_use']['unknown'],
            future_relation=future['causal_use']['relation'], future_actual_value=future['value_now'],
            td_target=td['y'], factor=factor, arrival_event=after['arrival'][0],
            arrival_seconds=after['arrival'][1]-before['arrival'][1],
            return_read=project(before['sources'], after['delivery'], factor)
                if returns else None))
    final = groups[-1]
    body = root/'body'
    metadata = json.loads((body/'last.metadata.json').read_text(encoding='utf8'))
    checkpoint = body/'last.pt.gz'
    sha = hashlib.sha256()
    with checkpoint.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            sha.update(chunk)
    digest = sha.hexdigest()
    assert metadata['checkpoint_sha256'] == digest
    with gzip.open(checkpoint, 'rb') as stream:
        state = torch.load(stream, map_location='cpu', weights_only=False)
    from types import SimpleNamespace
    from scripts.ib_local.navigation_gain_units import td_gain_target
    engine = SimpleNamespace(gamma=state['engine']['gamma'],
                             gain_units_schema=state['engine'].get('gain_units_schema', 0))
    value_slope = td_gain_target(engine, 0., 1.)
    for entry, before, after in zip(returns, groups, groups[1:]):
        td = after['row']['td_previous']
        compatible = not td.get('objective_boundary', False)
        expected = td_gain_target(engine, before['row']['metal_g'], entry['future_actual_value'], compatible)
        assert math.isclose(entry['td_target'], expected, abs_tol=1e-8)
        entry['td_value_slope'] = value_slope if compatible else 0.
    return dict(cycle=result['cycle'], checkpoint_sha256=digest, dispatches=len(timing),
        rounds=len(groups), dispatch_seconds=final['end'][1],
        inherited_actor_return=dict(future_cycle=groups[0]['row']['cycle'],
            future_event=groups[0]['delivery'][0]),
        returned=returns, pending=dict(action_cycle=result['cycle'],
            action_question=final['row']['causal_use']['unknown'],
            action_relation=final['row']['causal_use']['relation'],
            action_selector=project(final['sources'], final['end'], 1.),
            next_selector=project(sources, final['end'], 1.),
            **checkpoint_tail(state, final, sources)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--comparison', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    report = dict(schema=1, world_calls_added=0, optimizer_updates_added=0,
        clock='cumulative non-overlapping dispatch seconds, observed at dispatch end',
        projection='score derivative in a common internal-logit direction, fixed logged menus; credit signed as ascent',
        pending='two separate owners: 2609 prior actor and selector for the next action; no assigned future factor',
        arms={arm:audit(a.comparison/arm) for arm in ('original','refined')})
    a.output.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n', encoding='utf8')


if __name__ == '__main__':
    main()
