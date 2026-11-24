import copy
import gzip
import io
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from scripts.ib_local.mengli_harness.body_worker import BodyOwner, ContinuationBody, serve
from scripts.ib_local.mengli_harness.family_provider import FamilyCaregiver, delivery_receipt
from scripts.ib_local.mengli_harness.mengli_world_client import DshMother
from scripts.ib_local.mother_context_harness import ContextMotherWorld
from test_dsh_mother_provider import Bridge
from test_mother_reply_world import Ambient


class ContextBridge(Bridge):
    def __call__(self, operation, payload):
        if operation == 'context':
            self.calls.append((operation, copy.deepcopy(payload)))
            ref = dict(payload['checkpoint_ref'])
            ref['turn_end_seq'] += 1
            return dict(checkpoint_ref=ref)
        return super().__call__(operation, payload)


def family_world(bridge=None, capacity=8):
    bridge = bridge or ContextBridge()
    provider = FamilyCaregiver(DshMother(branch_id='family-a', transport=bridge))
    return ContextMotherWorld(Ambient(), provider), bridge


def turn(world, cycle, caregiver='parent', text='actual parent words', plan=None, capacity=8):
    world.mother.begin_turn(caregiver=caregiver, text=text, plan=plan,
                            cycle=cycle, response_bytes=capacity)
    return world.take(b'pre-input child', capacity, 5)


def test_parent_port_records_full_input_and_exact_truncated_delivery_without_api():
    world, bridge = family_world()
    response, heldout, info = turn(world, 11, text='  abc梦漓xyz  ', capacity=5)
    assert bytes(response) == b'abc\xe6\xa2'
    assert bytes(heldout) == b'yyyyy'
    assert [op for op, _ in bridge.calls] == ['restore', 'context']
    assert world.mother.observations == [] and bridge.generated == 0
    assert bridge.calls[-1][1]['observations'][0]['caregiver_text'] == '  abc梦漓xyz  '
    event = world.mother.last_event
    assert event['caregiver'] == 'parent'
    assert event['caregiver_text'] == '  abc梦漓xyz  '
    assert event['child_utterance'] == 'pre-input child'
    assert event['delivered']['hex'] == bytes(response).hex()
    assert event['delivered']['truncated'] and not event['delivered']['utf8_complete']
    assert event['delivered']['candidate_bytes'] == len('abc梦漓xyz\n'.encode())
    assert info['mother_harness']['source'] == 'parent'
    assert world.base.clock == 1


def test_parent_and_plan_sync_before_checkpoint_then_resume_mother_without_duplicate_context():
    world, bridge = family_world()
    plan = dict(version=4, text='Follow the child’s interest.')
    turn(world, 11, plan=plan)
    saved = copy.deepcopy(world.state_dict())
    assert [op for op, _ in bridge.calls] == ['restore', 'context']
    assert saved['mother_provider_state']['family_caregiver']['observations'] == []
    assert bridge.generated == 0
    restored, _ = family_world(bridge)
    restored.load_state_dict(saved)
    assert restored.state_dict() == saved
    turn(restored, 12, caregiver='mother', plan=plan)
    assert [op for op, _ in bridge.calls] == ['restore', 'context', 'restore', 'prepare', 'commit']
    context = bridge.calls[1][1]
    assert context['plan'] == plan
    assert context['observations'][0]['caregiver'] == 'parent'
    assert context['observations'][0]['event_id'] == 'family-a:cycle:11'
    assert bridge.calls[3][1]['checkpoint_ref'] == dict(context['checkpoint_ref'], turn_end_seq=1)
    assert restored.mother.observations == []
    assert restored.mother.synced_plan_version == 4
    turn(restored, 13, caregiver='mother', plan=plan)
    assert [op for op, _ in bridge.calls].count('context') == 1


def test_legacy_dsh_state_remains_flat_and_full_automatic_state_survives_parent():
    bridge = ContextBridge()
    automatic = DshMother(branch_id='family-a', transport=bridge)
    old = ContextMotherWorld(Ambient(), automatic)
    old.take(b'original', 16, 5)
    old_state = copy.deepcopy(old.state_dict())
    world, _ = family_world(bridge)
    world.load_state_dict(old_state)
    automatic_state = copy.deepcopy(automatic.state_dict())
    turn(world, 2)
    family_state = world.mother.state_dict()
    expected = copy.deepcopy(automatic_state)
    expected['dsh_harness']['checkpoint_ref']['turn_end_seq'] += 1
    expected['dsh_harness']['last_pair'] = [
        ['child', 'pre-input child'], ['mother', 'actual parent words']]
    assert {k: family_state[k] for k in automatic_state} == expected
    assert world.mother.observations == []
    restored, _ = family_world(bridge)
    restored.load_state_dict(copy.deepcopy(world.state_dict()))
    assert restored.state_dict() == world.state_dict()
    before = len(bridge.calls)
    turn(restored, 3, caregiver='mother')
    assert [op for op, _ in bridge.calls[before:]] == ['restore', 'prepare', 'commit']


def test_long_parent_takeover_keeps_checkpoint_history_bounded():
    world, bridge = family_world()
    sizes = []
    plan = dict(version=1, text='Keep the same external care arrangement.')
    for cycle in range(1000, 1128):
        turn(world, cycle, text='same fixed-size parent input', plan=plan)
        state = world.state_dict()
        assert state['mother_provider_state']['family_caregiver']['observations'] == []
        assert world.mother.observations == []
        sizes.append(len(json.dumps(state, sort_keys=True)))
    # After the existing finite world transcript fills, only fixed counters
    # gain digits; no additional history accumulates in later body saves.
    full_history_sizes = sizes[world.history - 1:]
    assert max(full_history_sizes) - min(full_history_sizes) <= 16
    contexts = [payload for op, payload in bridge.calls if op == 'context']
    assert len(contexts) == 128 and all(len(p['observations']) == 1 for p in contexts)
    assert sum('plan' in p for p in contexts) == 1
    assert bridge.generated == 0
    assert not any(op in ('prepare', 'commit') for op, _ in bridge.calls)
    assert world.mother.checkpoint_ref['turn_end_seq'] == 128
    assert world.mother.last_pair == [
        ['child', 'pre-input child'], ['mother', 'same fixed-size parent input']]


@pytest.mark.parametrize('next_caregiver', ['mother', 'parent'])
def test_legacy_pending_parent_observations_flush_on_first_sync(monkeypatch, next_caregiver):
    old_world, bridge = family_world()
    # Reproduce the old provider's deferred queue, without changing its schema.
    monkeypatch.setattr(old_world.mother, '_sync_context', lambda: None)
    turn(old_world, 10, text='older parent input')
    turn(old_world, 11, text='newer parent input')
    saved = copy.deepcopy(old_world.state_dict())
    assert len(saved['mother_provider_state']['family_caregiver']['observations']) == 2
    assert bridge.calls == []
    restored, _ = family_world(bridge)
    restored.load_state_dict(saved)
    turn(restored, 12, caregiver=next_caregiver)
    contexts = [payload for op, payload in bridge.calls if op == 'context']
    assert len(contexts) == 1
    observations = contexts[0]['observations']
    assert [item['cycle'] for item in observations[:2]] == [10, 11]
    assert len(observations) == (2 if next_caregiver == 'mother' else 3)
    assert restored.mother.observations == []
    assert restored.state_dict()['mother_provider_state']['family_caregiver']['observations'] == []
    assert bridge.generated == (1 if next_caregiver == 'mother' else 0)


@pytest.mark.parametrize('turn_input', [dict(caregiver='other'), dict(caregiver='parent'),
    dict(caregiver='parent', text='  '), dict(caregiver='mother', plan={'version': True, 'text': 'x'})])
def test_invalid_input_rejected_before_world_advances(turn_input):
    world, bridge = family_world()
    with pytest.raises(ValueError):
        world.mother.begin_turn(cycle=1, response_bytes=8, **turn_input)
    assert world.base.clock == 0 and bridge.calls == []


def test_plan_version_is_immutable_and_monotone():
    world, _ = family_world()
    turn(world, 1, plan=dict(version=2, text='a'))
    for plan in (dict(version=1, text='a'), dict(version=2, text='b')):
        with pytest.raises(ValueError, match='plan'):
            turn(world, 2, plan=plan)
    assert world.base.clock == 1


def test_failed_ambient_does_not_commit_parent_or_allow_partial_checkpoint():
    class Broken(Ambient):
        def take(self, *args):
            raise RuntimeError('ambient failed')
    world, bridge = family_world()
    world.base = Broken()
    with pytest.raises(RuntimeError, match='ambient failed'):
        turn(world, 1)
    assert not world.mother.observations and world.mother.last_event is None
    with pytest.raises(RuntimeError, match='incomplete'):
        world.state_dict()
    assert bridge.calls == []


def make_body(tmp_path, *, fail_save=False):
    world, bridge = family_world()
    calls, snapshots = [], []

    class Optimizer:
        def step(self):
            calls.append('update')

    class Engine:
        response_bytes, device = 8, 'cpu'
        optimizer = Optimizer()

        def step(self, directions):
            calls.append('utter')
            response, _, info = self.world.take(b'before', self.response_bytes, 5)
            calls.append('learn')
            self.optimizer.step()
            self.optimizer.step()
            return dict(utterance_hex=b'before'.hex(), choice='generate', world=info)

    def save(engine, output, cycle, config):
        calls.append('save')
        if fail_save:
            raise OSError('save failed')
        snapshots.append(copy.deepcopy(world.state_dict()))

    def export(engine, **kwargs):
        calls.append('export')
        return dict(cycle=kwargs['cycle'])

    engine = Engine()
    engine.world = world
    body = ContinuationBody()
    body.loaded = SimpleNamespace(engine=engine, config={}, cycle=10, output=tmp_path,
        life_card=dict(path=str(tmp_path/'card.json')), life_branch='family-a', life_card_data=None)
    body.runner = SimpleNamespace(torch=SimpleNamespace(randn=lambda *args, **kwargs: None),
                                  save=save, export_live=export)
    body.started, body.poisoned = time.perf_counter(), False
    return body, calls, snapshots, bridge


def test_body_step_preserves_order_and_acknowledges_only_full_save_and_export(tmp_path):
    body, calls, snapshots, bridge = make_body(tmp_path)
    engine = body.loaded.engine
    result = body.step(dict(caregiver='parent', text='hello parent', plan=dict(version=1, text='care')))
    assert body.loaded.engine is engine
    assert calls == ['utter', 'learn', 'update', 'update', 'save', 'export']
    assert result['cycle'] == 11 and result['updates'] == 2
    assert result['utterance'] == dict(text='before', hex=b'before'.hex(), spoke=True,
                                        timing='before_caregiver_input')
    assert result['caregiver_text'] == 'hello parent'
    assert result['delivered']['hex'] == b'hello pa'.hex()
    assert result['plan_version'] == 1
    assert snapshots[0]['mother_provider_state']['family_caregiver']['observations'] == []
    assert [op for op, _ in bridge.calls] == ['restore', 'context']
    assert bridge.generated == 0
    body.step(dict(caregiver='parent', text='next'))
    assert body.loaded.cycle == 12 and engine.world.base.clock == 2
    assert len((tmp_path/'metrics.jsonl').read_text().splitlines()) == 2


def test_save_failure_never_acknowledges_or_repeats_partial_cycle(tmp_path):
    body, calls, _, _ = make_body(tmp_path, fail_save=True)
    with pytest.raises(OSError, match='save failed'):
        body.step(dict(caregiver='parent', text='hello'))
    assert calls[-1] == 'save' and 'export' not in calls
    with pytest.raises(RuntimeError, match='latest committed'):
        body.step(dict(caregiver='parent', text='retry'))
    assert body.loaded.cycle == 10


def test_protocol_is_serial_single_load_and_stdout_contains_only_json(capsys):
    loads = []

    class Session:
        cycle = 3509

        def status(self):
            return dict(cycle=self.cycle)

        def step(self, request):
            self.cycle += 1
            print('step diagnostic')
            return self.status()

    def load(request):
        loads.append(request)
        print('restore diagnostic')
        return Session()

    requests = [dict(id=0, command='step'), dict(id=1, command='load', checkpoint='x', output='y'),
                dict(id=2, command='load', checkpoint='other', output='other'),
                dict(id=3, command='step', caregiver='parent', text='input'),
                dict(id=4, command='close')]
    output = io.StringIO()
    serve(io.StringIO('\n'.join(json.dumps(r) for r in requests)), output,
          owner=BodyOwner(session_factory=load))
    rows = [json.loads(line) for line in output.getvalue().splitlines()]
    assert [r['id'] for r in rows] == list(range(5))
    assert [r['ok'] for r in rows] == [False, True, False, True, True]
    assert rows[0]['error']['type'] == 'RuntimeError'
    assert rows[3]['result']['cycle'] == 3510 and len(loads) == 1
    captured = capsys.readouterr()
    assert captured.out == '' and 'restore diagnostic' in captured.err


def test_startup_and_close_do_not_import_torch_or_runner():
    root = Path(__file__).resolve().parents[1]
    worker = root/'scripts/ib_local/mengli_harness/body_worker.py'
    wire = json.dumps(dict(id=1, command='close')) + '\n'
    program = (
        f"import sys,io,runpy; module=runpy.run_path({str(worker)!r}); "
        "assert 'torch' not in sys.modules; "
        f"module['serve'](io.StringIO({wire!r})); "
        "assert 'torch' not in sys.modules; "
        "assert 'scripts.ib_local.run_mother_continuation' not in sys.modules"
    )
    completed = subprocess.run([sys.executable, '-c', program], cwd=root,
                               capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)['result']['closed'] is True


def test_shared_runner_restore_reuses_full_build_and_resumes_without_initial_save(tmp_path, monkeypatch):
    from scripts.ib_local import run_mother_continuation as runner
    from scripts.ib_local.mengli_harness.mengli_world_client import DEFAULT_SETTINGS

    world, _ = family_world()
    turn(world, 3509, plan=dict(version=3, text='retain this plan'))
    state = world.state_dict()
    config = dict(mother_mode='real', mother_policy='responsive_v3',
                  mother_context_harness=dict(schema=2, kind='dsh', branch_id='family-a',
                      endpoint_url='http://127.0.0.1:8900', settings=dict(DEFAULT_SETTINGS)))
    snapshot = dict(config=config, world=state, cycle=3509)
    source = tmp_path/'source.pt.gz'
    with gzip.open(source, 'wb') as stream:
        stream.write(b'offline fixture, not a real body')
    source_bytes = source.read_bytes()
    builds, saves, exports = [], [], []
    sentinel = object()

    class DirectReadGainEngine:
        def __init__(self):
            self.world = Ambient()
            self.model = SimpleNamespace(named_parameters=lambda: [])
            self.optimizer = SimpleNamespace(param_groups=[])
            self.recurrent = sentinel

    def build(checkpoint):
        engine = DirectReadGainEngine()
        builds.append((checkpoint, engine))
        return engine, None, None, 'offline source'

    def save(engine, output, cycle, config):
        saves.append(cycle)
        with gzip.open(output/'last.pt.gz', 'wb') as stream:
            stream.write(b'offline latest fixture')

    monkeypatch.setattr(runner.torch, 'load', lambda *args, **kwargs: copy.deepcopy(snapshot))
    monkeypatch.setattr(runner.gdn_fused, 'enable', lambda: False)
    monkeypatch.setattr(runner, 'build', build)
    monkeypatch.setattr(runner, 'save', save)
    monkeypatch.setattr(runner, 'export_live', lambda *args, **kwargs: exports.append(kwargs) or {'cycle': 3509})
    output = tmp_path/'new-family'
    args = runner.continuation_parser().parse_args([
        '--checkpoint', str(source), '--output', str(output), '--cycles', '1'])
    loaded = runner.prepare_continuation(args, provider_factory=FamilyCaregiver, resume_output=True)
    assert saves == [3509] and len(exports) == 1 and len(builds) == 1
    assert loaded.engine is builds[0][1] and loaded.engine.recurrent is sentinel
    assert loaded.engine.world.mother.state_dict() == state['mother_provider_state']
    assert source.read_bytes() == source_bytes
    snapshot['config'] = loaded.config
    args.checkpoint = output/'last.pt.gz'
    resumed = runner.prepare_continuation(args, resume_output=True)
    assert isinstance(resumed.engine.world.mother, FamilyCaregiver)
    assert resumed.engine.world.mother.state_dict() == state['mother_provider_state']
    assert saves == [3509] and len(exports) == 1
    args.checkpoint = source
    with pytest.raises(ValueError, match='own latest'):
        runner.prepare_continuation(args, resume_output=True)
    assert source.read_bytes() == source_bytes


def test_delivery_receipt_short_input_keeps_newline_and_no_false_truncation():
    receipt = delivery_receipt(' hi ', 8)
    assert receipt['hex'] == b'hi\n'.hex() and not receipt['truncated']
    assert receipt['byte_end'] == 3 and receipt['candidate_bytes'] == 3
