import copy

import pytest

from scripts.ib_local.mengli_harness.mengli_world_client import (
    DshMother, branch_for_output, migration_turns,
)
from scripts.ib_local.mother_context_harness import ContextMotherWorld
from test_mother_reply_world import Ambient


class Bridge:
    def __init__(self):
        self.calls = []
        self.pending = {}
        self.results = {}
        self.generated = 0

    def __call__(self, operation, payload):
        self.calls.append((operation, copy.deepcopy(payload)))
        branch = payload['branch_id']
        if operation == 'restore':
            ref = payload['checkpoint_ref'] or dict(session_id=branch, turn_end_seq=0,
                                                    assistant_seq=None, summary_version=0)
            return dict(checkpoint_ref=copy.deepcopy(ref))
        if operation == 'prepare':
            key = (branch, payload['step_id'])
            if key not in self.pending:
                self.generated += 1
                ref = copy.deepcopy(payload['checkpoint_ref'])
                ref.update(turn_end_seq=ref['turn_end_seq'] + 3,
                           assistant_seq=ref['turn_end_seq'] + 2)
                raw = '  A real transport reply.\n'
                self.pending[key] = dict(step_id=payload['step_id'], raw=raw,
                    delivered=raw.strip(), result_ref=ref,
                    telemetry=dict(source='api', api_usage=dict(prompt_tokens=200,
                        cache_hit_tokens=0, cache_miss_tokens=200, completion_tokens=8)))
            return copy.deepcopy(self.pending[key])
        if operation == 'commit':
            key = (branch, payload['step_id'])
            ready = self.pending[key]
            self.results[key] = dict(checkpoint_ref=ready['result_ref'], telemetry=ready['telemetry'])
            return copy.deepcopy(self.results[key])
        raise AssertionError(operation)


def make(bridge, branch='research-a', **kwargs):
    return DshMother(branch_id=branch, transport=bridge, **kwargs)


def test_world_protocol_commits_once_and_only_sends_the_new_child_turn():
    bridge = Bridge()
    mother = make(bridge)
    world = ContextMotherWorld(Ambient(), mother, history=2)
    response, _, info = world.take(b'hello', 64, 5)
    assert bytes(response).startswith(b'A real transport reply.\n')
    assert [op for op, _ in bridge.calls] == ['restore', 'prepare', 'commit']
    assert bridge.calls[1][1]['transcript'] == 'hello'
    assert mother.turn_count == 1 and mother.pending is None
    assert info['mother_harness']['api_usage']['cache_hit_tokens'] == 0
    before = len(bridge.calls)
    mother.commit_turn()
    assert len(bridge.calls) == before and mother.turn_count == 1


def test_ambient_failure_leaves_pending_and_retry_reuses_the_reply():
    class Broken(Ambient):
        def take(self, *args):
            raise RuntimeError('ambient failed')
    bridge = Bridge()
    mother = make(bridge)
    world = ContextMotherWorld(Broken(), mother)
    with pytest.raises(RuntimeError, match='ambient failed'):
        world.take(b'hello', 64, 5)
    assert mother.checkpoint_ref['turn_end_seq'] == 0
    assert mother.pending is not None and world.transcript == []
    with pytest.raises(RuntimeError, match='undelivered'):
        mother.state_dict()
    with pytest.raises(RuntimeError, match='not committed'):
        mother.reply([('child', 'different')])
    world.base = Ambient()
    world.take(b'hello', 64, 5)
    assert bridge.generated == 1 and world.base.clock == 1
    assert mother.turn_count == 1


def test_failed_prepare_does_not_advance_the_real_world():
    bridge = Bridge()
    def transport(operation, payload):
        if operation == 'prepare':
            raise RuntimeError('service failed')
        return bridge(operation, payload)
    world = ContextMotherWorld(Ambient(), make(transport))
    with pytest.raises(RuntimeError, match='service failed'):
        world.take(b'hello', 64, 5)
    assert world.base.clock == 0 and world.transcript == []
    assert world.mother.turn_count == 0


def test_disk_state_keeps_opaque_session_boundary_and_nondefault_world_history():
    bridge = Bridge()
    world = ContextMotherWorld(Ambient(), make(bridge), history=12)
    world.take(b'hello', 64, 5)
    world.mother.checkpoint_ref['result_seq'] = 9
    state = copy.deepcopy(world.state_dict())
    restored = ContextMotherWorld(Ambient(), make(bridge), history=state['mother_history'])
    restored.load_state_dict(state)
    assert restored.state_dict() == state
    restored.take(b'next', 64, 5)
    restore_request = [p for op, p in bridge.calls if op == 'restore'][-1]
    assert restore_request['checkpoint_ref']['result_seq'] == 9
    assert restored.base.clock == 2 and restored.mother.turn_count == 2
    assert restored.mother.last_pair[0] == ['child', 'next']


def test_migration_uses_only_checkpoint_raw_context_and_does_not_feed_summary():
    bridge = Bridge()
    mother = make(bridge)
    old = dict(policy='responsive_v3', context_harness=dict(turn_count=20, context=[
        dict(role='user', content='actual child'),
        dict(role='assistant', content='  raw reply\n', delivered='raw reply')]))
    mother.load_state_dict(old)
    mother.inherit_world_history([('child', 'actual child'), ('mother', 'raw reply')])
    mother.reply([('child', 'actual child'), ('mother', 'raw reply'), ('child', 'new question')])
    migration = bridge.calls[0][1]['migration']
    assert migration['turns'] == [dict(input='actual child', raw='  raw reply\n',
        delivered='raw reply', provenance='checkpoint_raw_context')]
    assert bridge.calls[1][1]['transcript'] == 'new question'
    assert mother.turn_count == 20


def test_legacy_delivered_history_is_labelled_and_future_state_cleared():
    bridge = Bridge()
    mother = make(bridge)
    world = ContextMotherWorld(Ambient(), mother)
    world.take(b'future', 64, 5)
    mother.load_state_dict(dict(policy='responsive_v3'))
    mother.inherit_world_history([('child', 'past'), ('mother', 'delivered only')])
    assert mother.checkpoint_ref is None and mother.turn_count == 0
    mother.reply([('child', 'past'), ('mother', 'delivered only'), ('child', 'next')])
    turns = bridge.calls[-2][1]['migration']['turns']
    assert turns[0]['input'] == 'past'
    assert turns[0]['provenance'] == 'checkpoint_delivered_only'
    assert 'future' not in str(turns)


def test_two_branches_restore_the_same_prefix_with_different_identities():
    bridge = Bridge()
    a = make(bridge, 'a')
    w = ContextMotherWorld(Ambient(), a)
    w.take(b'first', 64, 5)
    saved = a.state_dict()
    b = make(bridge, 'b')
    b.load_state_dict(saved)
    b.reply(w.transcript + [('child', 'second')])
    restore = [p for op, p in bridge.calls if op == 'restore'][-1]
    assert restore['branch_id'] == 'b'
    assert restore['checkpoint_ref'] == saved['dsh_harness']['checkpoint_ref']
    assert b.branch_id != a.branch_id
    assert a.turn_count == 1 and b.turn_count == 1


def test_wrong_world_pair_is_rejected_before_an_api_request():
    bridge = Bridge()
    mother = make(bridge)
    world = ContextMotherWorld(Ambient(), mother)
    world.take(b'actual', 64, 5)
    before = len(bridge.calls)
    with pytest.raises(ValueError, match='does not match'):
        mother.reply([('child', 'other'), ('mother', 'wrong reply'), ('child', 'new')])
    assert len(bridge.calls) == before


def test_provider_does_not_accept_non_loopback_or_credential_urls():
    for endpoint in ['https://api.example.com', 'http://user:pass@127.0.0.1:8900',
                     'http://127.0.0.1:8900/arbitrary']:
        with pytest.raises(ValueError, match='loopback'):
            DshMother(branch_id='a', endpoint_url=endpoint)


def test_incomplete_import_is_not_accepted_as_a_committed_turn():
    with pytest.raises(ValueError, match='complete pairs'):
        migration_turns([dict(role='user', content='unanswered')])


def test_checkpoint_keeps_billing_separate_from_the_session_prefix():
    bridge = Bridge()
    mother = make(bridge)
    meter = dict(api_calls=2, api_successes=2, api_failures=0, local_cache_hits=1,
                 shared_cache_hits=3, prompt_tokens=400, completion_tokens=16,
                 cache_hit_tokens=100, cache_miss_tokens=300)
    mother.load_state_dict(dict(policy='responsive_v3', context_harness=dict(meter=meter)))
    saved = mother.state_dict()
    restored = make(bridge, 'restored')
    restored.load_state_dict(saved)
    restored.reply([('child', 'actual words')])
    request = bridge.calls[0][1]
    assert request['usage_totals']['api_calls'] == 2
    assert request['usage_totals']['reply_usage']['total_tokens'] == 416
    assert restored.usage_totals == saved['dsh_harness']['usage_totals']


def test_output_identity_is_stable_and_separate_across_arms(tmp_path):
    assert branch_for_output(tmp_path/'raw') == branch_for_output(tmp_path/'raw')
    assert branch_for_output(tmp_path/'raw') != branch_for_output(tmp_path/'cut')
