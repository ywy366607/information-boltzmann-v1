"""Small CPU fixtures exercise the display interface, not body capabilities."""
import ast
import copy
import gzip
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.ib_local import life_card


@pytest.fixture
def display_root(tmp_path, monkeypatch):
    monkeypatch.setattr(life_card, 'ROOT', tmp_path)
    monkeypatch.setattr(life_card, 'LIFE_ROOT', tmp_path / 'data/harness/life')
    return life_card.LIFE_ROOT


@pytest.fixture
def snapshot():
    return dict(
        cycle=3509,
        config=dict(mother_context_harness=dict(kind='dsh', branch_id='actual-dsh-branch')),
        model={
            'memory.clock': torch.tensor(100),
            'memory.occupied': torch.tensor([True, False, True, False]),
            'memory.episodes': torch.tensor(7),
            'memory.last_write': torch.tensor([20, 0, 90, 0]),
        },
        world=dict(mother_provider_state=dict(dsh_harness=dict(
            checkpoint_ref=dict(session_id='mother-session-not-individual', event_id='event-3'),
            turn_count=2))))


def make_card(snapshot, **kwargs):
    return life_card.from_state_dict(
        snapshot['model'], config=snapshot['config'], checkpoint='fixture.pt.gz',
        source_sha='a' * 64, cycle=snapshot['cycle'],
        mother_state=snapshot['world']['mother_provider_state'], **kwargs)


def save_fixture(snapshot, tmp_path):
    checkpoint = tmp_path / 'fixture.pt.gz'
    with gzip.open(checkpoint, 'wb') as stream:
        torch.save(snapshot, stream)
    return checkpoint


def test_fixed_schema_values_and_scopes(snapshot):
    row = dict(cycle=3509, updates=4, cumulative_optimizer_updates=999,
               bfss_dynamics=dict(state_change=.408542, momentum_norms=[1., 2., 3., 4.]))
    card = make_card(snapshot, row=row)
    assert set(card) == {'schema', 'identity', 'branch_id', 'source', 'age', 'memory',
                         'latest', 'mother', 'exported_at', 'explanation'}
    assert card['schema'] == 1
    assert card['identity'] == dict(name='梦漓', individual_id='mengli-' + 'a' * 16)
    assert card['branch_id'] == 'actual-dsh-branch'
    assert card['source']['cycle'] == 3509
    assert card['source']['sha256'] == 'a' * 64
    assert card['age'] == dict(cycles=3509, byte_events=100, born_at=None, active_seconds=None)
    assert card['memory'] == dict(occupied=2, capacity=4, episodes_committed=7,
                                  oldest_since_write_events=80)
    assert card['latest'] == dict(updates=4, state_change=.408542, momentum_norms=[1., 2., 3., 4.])
    assert card['mother'] == dict(
        checkpoint_ref=snapshot['world']['mother_provider_state']['dsh_harness']['checkpoint_ref'],
        turns=2, turn_scope='since_dsh_migration')
    assert 'self-hearing' in card['explanation']['byte_events']
    assert 'merges' in card['explanation']['episodes_committed']
    assert 'not lifetime' in card['explanation']['updates']


def test_initial_row_and_unknown_clock_are_null(snapshot):
    del snapshot['model']['memory.clock']
    card = make_card(snapshot)
    assert card['age']['byte_events'] is None
    assert card['memory']['oldest_since_write_events'] is None
    assert card['latest'] == dict(updates=None, state_change=None, momentum_norms=None)
    empty = life_card.from_state_dict({}, config={}, checkpoint='empty.pt',
                                      source_sha='0' * 64, cycle=0)
    assert empty['memory'] == dict(occupied=None, capacity=None, episodes_committed=None,
                                   oldest_since_write_events=None)
    assert empty['mother']['turns'] is None


def test_real_zero_values_and_empty_memory(snapshot):
    snapshot['model']['memory.clock'].zero_()
    snapshot['model']['memory.episodes'].zero_()
    snapshot['model']['memory.occupied'].zero_()
    snapshot['world']['mother_provider_state']['dsh_harness']['turn_count'] = 0
    card = make_card(snapshot, row=dict(updates=0, bfss_dynamics=dict(
        state_change=0., momentum_norms=[0., 0., 0., 0.])))
    assert card['age']['byte_events'] == 0
    assert card['memory'] == dict(occupied=0, capacity=4, episodes_committed=0,
                                  oldest_since_write_events=None)
    assert card['latest'] == dict(updates=0, state_change=0., momentum_norms=[0.] * 4)
    assert card['mother']['turns'] == 0


def test_identity_inheritance_is_separate_from_branches(snapshot):
    config = snapshot['config']
    original = copy.deepcopy(config)
    identity = life_card.identity_for(config, 'f' * 64)
    assert identity['name'] == '梦漓'
    assert config == original
    assert life_card.identity_for({}, 'f' * 64, '另一个名字')['name'] == '另一个名字'
    config['life_identity'] = dict(identity, custom=dict(label='display only'))
    inherited = life_card.identity_for(config, 'b' * 64, 'ignored override')
    assert inherited == config['life_identity']
    inherited['custom']['label'] = 'changed'
    assert config['life_identity']['custom']['label'] == 'display only'
    config['mother_context_harness']['branch_id'] = 'new-research-arm'
    card = make_card(snapshot)
    assert card['identity']['individual_id'] == identity['individual_id']
    assert card['branch_id'] == 'new-research-arm'
    assert card['identity']['individual_id'] != card['mother']['checkpoint_ref']['session_id']


def test_branch_paths_inherit_only_for_same_owner(display_root):
    first = life_card.card_config_for({}, 'branch-a')
    assert Path(first['path']).parent == display_root
    assert Path(first['path']).name != 'current.json'
    config = dict(life_card=first)
    before = copy.deepcopy(config)
    assert life_card.card_config_for(config, 'branch-a') == first
    second = life_card.card_config_for(config, 'branch-b')
    assert second['path'] != first['path']
    assert second['owner_branch_id'] == 'branch-b'
    explicit = life_card.card_config_for(config, 'branch-b', 'data/harness/life/current.json')
    assert Path(explicit['path']) == display_root / 'current.json'
    assert config == before


def test_non_dsh_output_branches_are_independent(display_root):
    a = life_card.branch_id_for({}, display_root / 'arm-a/last.pt.gz')
    b = life_card.branch_id_for({}, display_root / 'arm-b/last.pt.gz')
    assert a != b
    assert a == life_card.branch_id_for({}, display_root / 'arm-a/last.pt.gz')


@pytest.mark.parametrize('output', ['outside.json', 'data/harness/life/../outside.json'])
def test_output_is_confined_to_life_directory(display_root, output):
    with pytest.raises(ValueError, match='data/harness/life'):
        life_card.write_card(dict(schema=1), output)
    assert not display_root.exists()


def test_inputs_are_unchanged_and_output_does_not_alias(snapshot):
    before = copy.deepcopy(snapshot)
    row = dict(updates=4, bfss_dynamics=dict(momentum_norms=[1., 2., 3., 4.]))
    old_row = copy.deepcopy(row)
    card = make_card(snapshot, row=row)
    for name, tensor in before['model'].items():
        assert torch.equal(tensor, snapshot['model'][name])
    assert before['config'] == snapshot['config']
    assert before['world'] == snapshot['world']
    assert row == old_row
    card['latest']['momentum_norms'][0] = 99
    card['mother']['checkpoint_ref']['event_id'] = 'changed'
    assert row == old_row
    assert before['world'] == snapshot['world']


def test_atomic_json_replace(display_root, monkeypatch):
    output = display_root / 'current.json'
    life_card.write_card(dict(name='old'), output)
    original_replace = life_card.os.replace
    replacements = []

    def observed_replace(source, destination):
        assert output.read_text(encoding='utf-8').strip() == '{\n  "name": "old"\n}'
        assert Path(source).parent == output.parent
        assert json.loads(Path(source).read_text(encoding='utf-8')) == dict(name='梦漓', zero=0)
        replacements.append(destination)
        original_replace(source, destination)

    monkeypatch.setattr(life_card.os, 'replace', observed_replace)
    life_card.write_card(dict(name='梦漓', zero=0), output)
    assert replacements == [output]
    assert json.loads(output.read_text(encoding='utf-8')) == dict(name='梦漓', zero=0)
    assert list(display_root.iterdir()) == [output]


def test_rejected_json_keeps_existing_card(display_root):
    output = display_root / 'current.json'
    life_card.write_card(dict(zero=0), output)
    before = output.read_bytes()
    with pytest.raises(ValueError):
        life_card.write_card(dict(bad=float('nan')), output)
    assert output.read_bytes() == before
    assert list(display_root.iterdir()) == [output]


@pytest.mark.parametrize('matching', [True, False])
def test_checkpoint_cpu_export_binds_exact_cycle(snapshot, display_root, tmp_path, monkeypatch, matching):
    checkpoint = save_fixture(snapshot, tmp_path)
    before = checkpoint.read_bytes()
    metrics = tmp_path / 'metrics.jsonl'
    rows = [dict(cycle=3508, updates=99)]
    if matching:
        rows.append(dict(cycle=3509, updates=4, bfss_dynamics=dict(
            state_change=.408542, momentum_norms=[1., 2., 3., 4.])))
    rows.append(dict(cycle=3510, updates=100))
    metrics.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
    original_load = torch.load
    calls = []

    def cpu_load(*args, **kwargs):
        assert kwargs['map_location'] == 'cpu'
        calls.append(1)
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, 'load', cpu_load)
    card = life_card.export_checkpoint(checkpoint, metrics=metrics,
                                       output=display_root / 'current.json', name='梦漓')
    assert calls == [1]
    assert card['latest']['updates'] == (4 if matching else None)
    assert card['latest']['state_change'] == (.408542 if matching else None)
    assert card['source']['sha256'] == life_card.sha256_file(checkpoint)
    assert card['source']['checkpoint'] == str(checkpoint.resolve())
    assert checkpoint.read_bytes() == before
    assert json.loads((display_root / 'current.json').read_text(encoding='utf-8')) == card


def test_mismatched_row_is_not_used(snapshot):
    card = make_card(snapshot, row=dict(cycle=3508, updates=99))
    assert card['latest'] == dict(updates=None, state_change=None, momentum_norms=None)


def test_runtime_reads_existing_state_without_loading(snapshot, display_root, tmp_path, monkeypatch):
    checkpoint = save_fixture(snapshot, tmp_path)
    state_calls = []

    def existing_state():
        state_calls.append(1)
        return snapshot['model']

    def forbidden_load(*args, **kwargs):
        pytest.fail('live export must not reload a checkpoint')

    monkeypatch.setattr(torch, 'load', forbidden_load)
    engine = SimpleNamespace(model=SimpleNamespace(state_dict=existing_state),
        world=SimpleNamespace(mother=SimpleNamespace(
            state_dict=lambda: snapshot['world']['mother_provider_state'])))
    # Runtime row.cycle is populated only after the export hook.
    row = dict(updates=4, bfss_dynamics=dict(state_change=.5, momentum_norms=[1.] * 4))
    card = life_card.export_live(engine, checkpoint=checkpoint, config=snapshot['config'],
                                 cycle=3509, output=display_root / 'current.json', row=row)
    assert state_calls == [1]
    assert card['latest']['updates'] == 4
    assert card['source']['cycle'] == 3509
    assert 'cycle' not in row


def test_cli_with_small_fixture(snapshot, display_root, tmp_path, monkeypatch, capsys):
    checkpoint = save_fixture(snapshot, tmp_path)
    monkeypatch.setattr(sys, 'argv', ['life_card', '--checkpoint', str(checkpoint),
        '--output', 'data/harness/life/current.json', '--name', '梦漓'])
    life_card.main()
    card = json.loads(capsys.readouterr().out)
    assert card['identity']['name'] == '梦漓'
    assert card['latest']['updates'] is None
    assert (display_root / 'current.json').exists()


def test_runner_exports_after_both_saves_and_sets_metadata_first():
    runner = Path(__file__).resolve().parents[1] / 'scripts/ib_local/run_mother_continuation.py'
    tree = ast.parse(runner.read_text(encoding='utf-8'))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name)]
    saves = sorted(node.lineno for node in calls if node.func.id == 'save')
    exports = sorted(node.lineno for node in calls if node.func.id == 'export_live')
    identities = [node.lineno for node in calls if node.func.id == 'identity_for']
    assert len(saves) == len(exports) == 2
    assert identities[0] < saves[0] < exports[0] < saves[1] < exports[1]
    export_calls = sorted((node for node in calls if node.func.id == 'export_live'),
                          key=lambda node: node.lineno)
    first_keywords = {kw.arg: kw.value for kw in export_calls[0].keywords}
    next_keywords = {kw.arg: kw.value for kw in export_calls[1].keywords}
    assert 'row' not in first_keywords
    assert next_keywords['cycle'].id == 'cycle'
    assert next_keywords['row'].id == 'row'
