"""CPU-only checks of real-stream bookkeeping and strict continuation interfaces."""
from dataclasses import asdict, fields
import copy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[1]/'scripts/ib/train_fly_pipeline_stream.py'
spec = importlib.util.spec_from_file_location('fly_pipeline_stream_cli_tested', SCRIPT)
cli = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = cli
spec.loader.exec_module(cli)


def test_calibration_is_first_target_window_and_resume_starts_at_32():
    train = np.arange(1000, dtype=np.int64)
    ledger = cli.StreamLedger()
    values = ledger.targets(train, train)
    np.testing.assert_array_equal(values, np.arange(32))
    ledger.commit(values, [2.]*32)
    restored = cli.StreamLedger(**json.loads(json.dumps(asdict(ledger))))
    restored.validate()
    np.testing.assert_array_equal(restored.targets(train, train), np.arange(32, 64))
    assert restored.train_updates == 1 and restored.events == 32


def test_all_registered_updates_and_active_phases_are_counted_without_reset():
    train = np.arange(96007, dtype=np.int64) % cli.VOCAB
    validation = np.arange(1547, dtype=np.int64)
    ledger = cli.StreamLedger(train_cursor=7, val_cursor=11)
    reports = []
    train_ranges = []
    while not ledger.complete:
        phase = ledger.phase
        cursor = ledger.train_cursor
        values = ledger.targets(train, validation)
        report = ledger.commit(values, [3.]*32)
        ledger.validate(7, 11)
        if phase == 'train':
            train_ranges.append(cursor)
        if report:
            reports.append(report)
    assert ledger.train_updates == 3000 and ledger.eval_updates == 72
    assert ledger.events == 98304 and ledger.val_cursor == 1547
    assert train_ranges == list(range(7, 96007, 32))
    assert ledger.a_tokens == train[7:135].tolist()
    assert ledger.a_first_event_interval == [0, 128]
    assert [r['train_updates'] for r in reports] == [500, 1000, 1500, 2000, 2500, 3000]
    assert reports[0]['A2_event_start'] - 128 == 16128
    assert reports[-1]['A2_event_start'] - 128 == 98048
    assert reports[-1]['event_end'] == 98304


@pytest.mark.parametrize('phase,windows', [('B', 3), ('replay', 10)])
def test_partial_active_phase_resume_continues_exact_next_targets(phase, windows):
    train = np.arange(96000, dtype=np.int64) % cli.VOCAB
    val = np.arange(2000, dtype=np.int64)
    ledger = cli.StreamLedger()
    for _ in range(500+windows):
        ledger.commit(ledger.targets(train, val), [2.]*32)
    assert ledger.phase == phase
    resumed = cli.StreamLedger(**json.loads(json.dumps(asdict(ledger))))
    resumed.validate()
    assert resumed.events == ledger.events
    np.testing.assert_array_equal(resumed.targets(train, val), ledger.targets(train, val))
    expected = val[ledger.val_cursor:ledger.val_cursor+32] if phase == 'B' else train[64:96]
    np.testing.assert_array_equal(resumed.targets(train, val), expected)


def test_skipping_due_eval_or_changing_cursors_is_rejected():
    ledger = cli.StreamLedger(train_cursor=16000, train_updates=500,
        a_tokens=[1]*128, a_scores=[2.]*128)
    with pytest.raises(ValueError, match='must not be skipped'):
        ledger.validate()
    ledger.train_updates = 499
    with pytest.raises(ValueError, match='train cursor'):
        ledger.validate()


def test_npy_and_old_extended_uint16_binary_reference_are_identical(tmp_path):
    values = np.array([1, 1, 4, 8], dtype=np.uint16)
    np.save(tmp_path/'train.npy', values)
    values.tofile(tmp_path/'train.bin')
    np.testing.assert_array_equal(cli.load_tokens(tmp_path/'train.npy'), cli.load_tokens(tmp_path/'train.bin'))
    nll, metadata = cli.fixed_reference(tmp_path/'train.npy')
    assert metadata['tokens'] == 4 and metadata['smoothing'] == 'add one'
    assert not metadata['validation_labels_used_for_prior']
    assert nll[1] == pytest.approx(-np.log(3/(cli.VOCAB+4)))


def test_matched_initialization_imports_only_token_table_and_training_prior(tmp_path):
    from safetensors.torch import save_file
    from test_fly_bptt_learning import make_model
    model = make_model(tmp_path, decoder_bias=True)
    table = torch.arange(27, dtype=torch.float32).reshape(9, 3) * .01
    path = tmp_path / 'token_table.safetensors'
    save_file({'wte.weight': table}, str(path))
    prior = np.log(np.arange(1, 10, dtype=np.float32))
    cli.initialize_pretrained_head(model, path, prior, .1)
    torch.testing.assert_close(model.embedding.weight, table, atol=0, rtol=0)
    torch.testing.assert_close(model.decoder.weight, table, atol=0, rtol=0)
    assert model.embedding.weight.data_ptr() != model.decoder.weight.data_ptr()
    torch.testing.assert_close(model.decoder.bias, -torch.from_numpy(prior), atol=0, rtol=0)
    torch.testing.assert_close(model.read_norm.weight, torch.full((3,), .1), atol=0, rtol=0)
    assert model.latent_predictor is None


def test_journal_refuses_implicit_rollback_or_duplicate_targets(tmp_path):
    path = tmp_path/'scores.jsonl'
    cli.append_json(path, {'event_interval': [0, 32]})
    cli.verify_score_journal(path, 32)
    with pytest.raises(ValueError, match='diverge'):
        cli.verify_score_journal(path, 0)
    cli.append_json(path, {'event_interval': [0, 32]})
    with pytest.raises(ValueError, match='duplicates'):
        cli.verify_score_journal(path, 64)


def test_strict_full_pipeline_restore_retains_weights_physics_and_adam(tmp_path):
    from test_fly_bptt_learning import make_model, physical
    from information_boltzmann.core.fly_pipeline import FlyPipelineLearner
    torch.manual_seed(15)
    model = make_model(tmp_path)
    state = physical(model)
    state.ring = tuple(torch.full_like(state.h, .2) for _ in range(4))
    first = FlyPipelineLearner(model, state)
    first.observe([1, 2, 3, 4, 5, 6, 7, 8])
    saved = copy.deepcopy({'format': cli.FORMAT, 'model': dict(model.named_parameters()),
        'learner': first.state_dict(), 'optimizer_layout': cli.optimizer_layout(first),
        'rng_cpu': torch.get_rng_state(), 'rng_cuda': []})
    restored_model = make_model(tmp_path)
    restored = FlyPipelineLearner(restored_model, physical(restored_model))
    cli.restore_learner(saved, restored)
    a, _ = first.observe([8, 7, 6, 5, 4, 3, 2, 1])
    b, _ = restored.observe([8, 7, 6, 5, 4, 3, 2, 1])
    assert a == b
    assert first.events == restored.events == 16
    for item in fields(first.state):
        x, y = getattr(first.state, item.name), getattr(restored.state, item.name)
        for left, right in zip(x if item.name == 'ring' else (x,), y if item.name == 'ring' else (y,)):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, dict(restored_model.named_parameters())[name], atol=0, rtol=0)
    saved['format'] = 'fly-pipeline-best-weights-v1'
    with pytest.raises(ValueError, match='full'):
        cli.restore_learner(saved, restored)


def test_missing_physical_state_and_changed_protocol_cannot_fallback(tmp_path):
    from test_fly_bptt_learning import make_model, physical
    from information_boltzmann.core.fly_pipeline import FlyPipelineLearner
    model = make_model(tmp_path)
    learner = FlyPipelineLearner(model, physical(model))
    saved = {'format': cli.FORMAT, 'model': dict(model.named_parameters()),
        'learner': learner.state_dict(), 'optimizer_layout': cli.optimizer_layout(learner)}
    saved['learner']['prediction_protocol'] = 'wrong'
    with pytest.raises(ValueError, match='prediction_protocol'):
        cli.restore_learner(saved, learner)
    physical_saved = learner.state.state_dict()
    physical_saved.pop('h_mean')
    with pytest.raises(ValueError, match='no fallback'):
        cli.physical_from_saved(physical_saved, model, 'cpu')


def test_disk_gate_preserves_existing_continuation(tmp_path, monkeypatch):
    path = tmp_path/'last.pt'
    path.write_bytes(b'prior continuation')
    monkeypatch.setattr(cli.shutil, 'disk_usage', lambda p: type('Disk', (), {'free': 0})())
    with pytest.raises(OSError, match='Insufficient'):
        cli.atomic_checkpoint(path, {'weight': torch.ones(4)})
    assert path.read_bytes() == b'prior continuation'
    assert not (tmp_path/'last.pt.tmp').exists()


def test_endpoint_blocks_keep_partial_weights_and_exclude_active_eval_gaps(tmp_path):
    journal = tmp_path/'scores.jsonl'
    event = 0
    for update in range(1, 3001):
        cli.append_json(journal, {'phase': 'train', 'train_updates': update,
            'event_interval': [event, event+32], 'scores': [.5]*32,
            'fixed_unigram_scores': [1.]*32})
        event += 32
        if update % 500 == 0:
            for _ in range(8):
                cli.append_json(journal, {'phase': 'B', 'train_updates': update,
                    'event_interval': [event, event+32], 'scores': [2.]*32,
                    'fixed_unigram_scores': [1.]*32})
                event += 32
            event += 128
    summary = cli.terminal_risk_summary(journal)
    primary = summary['primary_256_token_interval']
    assert summary['primary_tokens'] == 32000
    assert primary['actual_blocks'] == 126 and primary['partial_block_tokens'] == [128, 128]
    assert summary['primary_model_minus_reference'] == -.5
    assert primary['paired_percentile_95_interval'] == [-.5, -.5]
    assert summary['budget_risk_advantage_supported']
    assert summary['final_B_direction_check']['model_minus_reference'] == 1.
    assert 'no bootstrap' in summary['final_B_direction_check']['scope']
