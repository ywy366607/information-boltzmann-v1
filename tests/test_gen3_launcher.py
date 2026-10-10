"""Launcher selection and import isolation; no model, CUDA or training runs."""
import importlib
import json
from pathlib import Path
import sys

import pytest

from scripts.ib import launch_gen3_training as entry


def checkpoint(root, directory, *, protocol=None, steps=5000):
    directory = root / directory
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'last.pt'
    path.write_bytes(b'placeholder; tests never load a model')
    config = {'training_groups': steps, 'constructor': {}}
    if protocol:
        config['constructor']['hopf_recomposition'] = True
        config['hopf_recomposition'] = {'protocol': protocol}
    (directory / 'config.json').write_text(json.dumps(config), encoding='utf-8')
    return path


def parse(*arguments):
    return entry.parser().parse_args(arguments)


def test_import_keeps_original_argv_and_no_training(tmp_path, monkeypatch):
    before = ['caller.py', '--unrelated', 'argument']
    monkeypatch.setattr(sys, 'argv', before)
    monkeypatch.chdir(tmp_path)
    importlib.reload(entry)
    assert sys.argv is before
    assert list(tmp_path.iterdir()) == []


def test_gen2_adoption_preserves_source_and_new_physical_output(tmp_path):
    source = checkpoint(tmp_path, entry.GEN2_CHECKPOINT.parent)
    original = (source.parent / 'config.json').read_bytes()
    arguments = entry.build_trainer_arguments(parse('--check-resume-only'), root=tmp_path)
    assert Path(arguments[arguments.index('--output') + 1]) == tmp_path / entry.DEFAULT_OUTPUT
    assert arguments[arguments.index('--resume') + 1] == str(source)
    assert '--adopt-hopf-branch' in arguments and '--check-resume-only' in arguments
    assert arguments[arguments.index('--steps') + 1] == '5000'
    assert '--allow-entrypoint-change' not in arguments
    assert (source.parent / 'config.json').read_bytes() == original
    assert not (tmp_path / entry.DEFAULT_OUTPUT).exists()


def test_own_checkpoint_selected_for_strict_continuation(tmp_path):
    source = checkpoint(tmp_path, entry.DEFAULT_OUTPUT, protocol=entry.JUNCTION_PROTOCOL)
    arguments = entry.build_trainer_arguments(parse(), root=tmp_path)
    assert arguments[arguments.index('--resume') + 1] == str(source)
    assert '--adopt-hopf-branch' not in arguments


def test_explicit_resume_output_and_inherited_steps(tmp_path):
    source = checkpoint(tmp_path, Path('archive/gen2'), steps=8000)
    args = parse('--output', 'results/custom-gen3', '--resume', str(source), '--steps', '8000')
    arguments = entry.build_trainer_arguments(args, root=tmp_path)
    assert arguments[arguments.index('--steps') + 1] == '8000'
    with pytest.raises(ValueError, match='inherited budget'):
        entry.build_trainer_arguments(parse('--resume', str(source)), root=tmp_path)
    assert not (tmp_path / 'results/custom-gen3').exists()


def test_old_config_without_complete_checkpoint_is_preserved(tmp_path):
    directory = tmp_path / entry.DEFAULT_OUTPUT
    directory.mkdir(parents=True)
    config = directory / 'config.json'
    original = '{"hopf_recomposition":{"protocol":"legacy_feature_split"}}'
    config.write_text(original, encoding='utf-8')
    checkpoint(tmp_path, entry.GEN2_CHECKPOINT.parent)
    with pytest.raises(ValueError, match='older or unrecognized protocol'):
        entry.build_trainer_arguments(parse(), root=tmp_path)
    assert config.read_text(encoding='utf-8') == original


def test_old_source_and_overwrite_of_existing_individual_are_rejected(tmp_path):
    old = checkpoint(tmp_path, Path('archive/old'), protocol='legacy_feature_split')
    with pytest.raises(ValueError, match='old readout branch'):
        entry.build_trainer_arguments(parse('--resume', str(old)), root=tmp_path)
    checkpoint(tmp_path, entry.DEFAULT_OUTPUT, protocol=entry.JUNCTION_PROTOCOL)
    other = checkpoint(tmp_path, entry.GEN2_CHECKPOINT.parent)
    with pytest.raises(ValueError, match='already owns another'):
        entry.build_trainer_arguments(parse('--resume', str(other)), root=tmp_path)


def test_main_delegates_check_flag_and_restores_process_context(tmp_path, monkeypatch):
    source = checkpoint(tmp_path, entry.GEN2_CHECKPOINT.parent)
    monkeypatch.setattr(entry, 'ROOT', tmp_path)
    # This CPU-only stub replaces the production trainer, which is never called.
    from scripts.ib import train_medium_active_stream as trainer
    observed = []
    def inspect():
        observed.append((sys.argv.copy(), Path.cwd()))
    monkeypatch.setattr(trainer, 'main', inspect)
    old_argv, old_cwd = sys.argv, Path.cwd()
    entry.main(['--resume', str(source), '--check-resume-only'])
    assert '--check-resume-only' in observed[0][0]
    assert observed[0][1] == tmp_path
    assert sys.argv is old_argv and Path.cwd() == old_cwd
