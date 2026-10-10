"""File/continuation contracts; these tests do not train a capability task."""
import json
import os

import pytest
import torch

from information_boltzmann.runtime import checkpoint_receipt as checkpoint
from scripts.ib.train_medium_active_stream import (
    prepare_hopf_continuation, validate_junction_pending_gradients,
)


def payload(step=4, cursor=129):
    return {'model': {'weight': torch.tensor([1., 2.])},
            'optimizer': {'state': {7: {'exp_avg': torch.tensor([.3])}}},
            'belief': {'field': torch.tensor([.4]), 'elapsed': torch.tensor([2.])},
            'learner': {'events': 160, 'optimizer_updates': 5, 'pending': 0,
                        'carry_token': 12, 'prior_sample': torch.tensor([.2])},
            'step': step, 'cursor': cursor, 'val_cursor': 32, 'next_eval': 160,
            'cpu_rng': torch.get_rng_state(), 'cuda_rng': None}


def test_successful_save_receipt_matches_actual_complete_payload(tmp_path):
    before = payload()
    rng = torch.get_rng_state().clone()
    destination = tmp_path / 'last.pt'
    receipt = checkpoint.save_complete_checkpoint(destination, before, attempt_id='run-a')
    restored = torch.load(destination, weights_only=False)
    sidecar = json.loads((tmp_path / 'last.pt.receipt.json').read_text())
    assert restored['checkpoint_receipt']['checkpoint_id'] == receipt['checkpoint_id']
    assert receipt == sidecar
    assert receipt['bytes'] == destination.stat().st_size
    assert receipt['position']['fresh_training_tokens'] == 128
    checkpoint.require_saved_position(receipt, checkpoint.checkpoint_position(restored))
    assert torch.equal(restored['belief']['field'], before['belief']['field'])
    assert torch.equal(restored['optimizer']['state'][7]['exp_avg'], before['optimizer']['state'][7]['exp_avg'])
    assert torch.equal(torch.get_rng_state(), rng)
    assert 'checkpoint_receipt' not in before


def test_checkpoint_replace_lock_exhaustion_preserves_old_file_and_cannot_complete(tmp_path, monkeypatch):
    destination = tmp_path / 'last.pt'
    original = payload()
    receipt = checkpoint.save_complete_checkpoint(destination, original, attempt_id='run-a')
    before = destination.read_bytes()
    calls = []
    def denied(*args):
        calls.append(args)
        raise PermissionError('reader holds replace lock')
    monkeypatch.setattr(checkpoint.os, 'replace', denied)
    with pytest.raises(checkpoint.CheckpointWriteError, match='checkpoint replacement failed'):
        checkpoint.save_complete_checkpoint(destination, payload(step=5, cursor=161),
            attempt_id='run-a', replace_attempts=3, retry_delay=0)
    assert len(calls) == 3
    assert destination.read_bytes() == before
    assert list(tmp_path.glob('last.pt.*.tmp'))
    with pytest.raises(checkpoint.CheckpointWriteError, match='trails live state'):
        checkpoint.require_saved_position(receipt, checkpoint.checkpoint_position(payload(step=5, cursor=161)))


def test_transient_replace_lock_retries_then_produces_receipt(tmp_path, monkeypatch):
    original_replace = os.replace
    calls = []
    def replace(source, destination):
        calls.append(str(destination))
        if len(calls) < 3:
            raise PermissionError('transient lock')
        return original_replace(source, destination)
    monkeypatch.setattr(checkpoint.os, 'replace', replace)
    receipt = checkpoint.save_complete_checkpoint(tmp_path / 'last.pt', payload(),
        attempt_id='run-b', replace_attempts=3, retry_delay=0)
    assert receipt['attempt_id'] == 'run-b'
    assert len(calls) == 4  # three checkpoint attempts plus one sidecar replace


def test_receipt_failure_reports_save_failure_even_when_checkpoint_was_replaced(tmp_path, monkeypatch):
    original_replace = os.replace
    def replace(source, destination):
        if str(destination).endswith('receipt.json'):
            raise PermissionError('receipt path locked')
        return original_replace(source, destination)
    monkeypatch.setattr(checkpoint.os, 'replace', replace)
    with pytest.raises(checkpoint.CheckpointWriteError, match='receipt replacement failed'):
        checkpoint.save_complete_checkpoint(tmp_path / 'last.pt', payload(),
            attempt_id='run-c', replace_attempts=2, retry_delay=0)
    assert (tmp_path / 'last.pt').exists()
    with pytest.raises(checkpoint.CheckpointWriteError, match='requires a committed'):
        checkpoint.require_saved_position(None, checkpoint.checkpoint_position(payload()))


def test_resume_journal_keeps_ahead_logs_and_identifies_replayed_intervals(tmp_path):
    rows = [
        {'step': 4, 'fresh_training_tokens': 128},
        {'step': 5, 'fresh_training_tokens': 160},
        {'step': 5, 'fresh_training_tokens': 160},
        {'step': 5, 'fresh_training_tokens': 160, 'attempt_id': 'old',
         'source_sha256': 'trainhash', 'source_start': 129, 'source_end': 161},
        {'step': 5, 'fresh_training_tokens': 160, 'attempt_id': 'retry',
         'source_sha256': 'trainhash', 'source_start': 129, 'source_end': 161},
    ]
    metrics = tmp_path / 'metrics.jsonl'
    metrics.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    before = metrics.read_bytes()
    (tmp_path / 'progress.json').write_text(json.dumps({'status': 'completed', 'fresh_training_tokens': 160}))
    position = checkpoint.checkpoint_position(payload())
    journal = checkpoint.append_attempt_journal(tmp_path, attempt_id='new', position=position,
        source_checkpoint=tmp_path / 'last.pt', source_receipt={'checkpoint_id': 'source-a'})
    gap = journal['recovery']
    assert gap['uncheckpointed_fresh_suffix'] == 32
    detail = gap['files']['metrics.jsonl']
    assert detail['rows_ahead_of_checkpoint'] == 4
    assert detail['legacy_rows_without_attempt'] == 3
    assert detail['repeated_record_positions'] == 1
    assert detail['replayed_source_interval_rows'] == 1
    assert journal['parent_checkpoint_id'] == 'source-a'
    assert metrics.read_bytes() == before
    assert json.loads((tmp_path / 'resume_journal.jsonl').read_text()) == journal


def test_committed_log_rows_excludes_failed_attempt_suffix_and_deduplicates(tmp_path):
    older = checkpoint.checkpoint_position(payload())
    final = checkpoint.checkpoint_position(payload(step=5, cursor=161))
    journals = [
        {'attempt_id': 'old', 'parent_attempt_id': None, 'source_checkpoint': None,
         'restored_position': {**older, 'fresh_training_tokens': 0}},
        {'attempt_id': 'new', 'parent_attempt_id': 'old', 'source_checkpoint': 'last.pt',
         'restored_position': older},
    ]
    (tmp_path / 'resume_journal.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in journals))
    rows = [
        {'attempt_id': 'old', 'step': 4, 'fresh_training_tokens': 128,
         'source_sha256': 'train', 'source_start': 97, 'source_end': 129},
        {'attempt_id': 'old', 'step': 5, 'fresh_training_tokens': 160,
         'source_sha256': 'train', 'source_start': 129, 'source_end': 161, 'nll': 999.},
        {'attempt_id': 'new', 'step': 5, 'fresh_training_tokens': 160,
         'source_sha256': 'train', 'source_start': 129, 'source_end': 161, 'nll': 7.},
        {'attempt_id': 'new', 'step': 5, 'fresh_training_tokens': 160,
         'source_sha256': 'train', 'source_start': 129, 'source_end': 161, 'nll': 7.},
    ]
    (tmp_path / 'metrics.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    report = checkpoint.committed_log_rows(tmp_path, 'metrics.jsonl',
        {'checkpoint_id': 'committed', 'attempt_id': 'new', 'position': final})
    assert report['selected_rows'] == 2
    assert report['excluded_raw_rows'] == 1
    assert report['deduplicated_source_rows'] == 1
    assert report['rows'][-1]['nll'] == 7.
    assert report['unattributed_legacy_rows'] == 0


def old_config():
    return {'constructor': {'shape': [8, 8, 8], 'channels': 768},
            'parameters': 100, 'trainable_parameters': 80, 'active_graph_parameters': 50,
            'lr': .0002, 'evaluation': {'validate_every_tokens': 5000},
            'source_hashes': {'information_boltzmann/core/plastic_medium.py': 'old',
                              'information_boltzmann/runtime/lifelong_evaluation.py': 'stable'}}


def junction_config():
    config = old_config()
    config['constructor']['hopf_recomposition'] = True
    for name in ('parameters', 'trainable_parameters', 'active_graph_parameters'):
        config[name] += 36
    config['hopf_recomposition'] = {'protocol': 'capacity_paid_persistent_flux_junction_v1'}
    config['source_hashes']['information_boltzmann/core/plastic_medium.py'] = 'new'
    config['source_hashes']['information_boltzmann/core/medium_junction.py'] = 'junction'
    config['source_hashes']['information_boltzmann/runtime/checkpoint_receipt.py'] = 'receipt'
    return config


def migrate(before, after):
    return prepare_hopf_continuation(before, after, source_checkpoint='source/last.pt',
        parameter_names=['medium.hopf_pathway.gate.bias', 'medium.hopf_pathway.gate.weight'],
        state_names=['medium.hopf_pathway._extra_state', 'medium.hopf_pathway.route_mask'], added_count=36)


def test_junction_config_adoption_lists_exact_parameters_and_preserves_cadence():
    before, after = old_config(), junction_config()
    recorded = migrate(before, after)
    assert before == after
    assert recorded['added_parameter_count'] == 36
    assert recorded['added_parameter_names'] == ['medium.hopf_pathway.gate.bias', 'medium.hopf_pathway.gate.weight']
    assert recorded['added_state_dict_names'] == ['medium.hopf_pathway._extra_state', 'medium.hopf_pathway.route_mask']
    assert recorded['before']['information_boltzmann/core/plastic_medium.py'] == 'old'
    assert after['evaluation']['validate_every_tokens'] == 5000


def test_junction_config_adoption_rejects_unrelated_sources_or_extra_parameters():
    changed = junction_config()
    changed['source_hashes']['information_boltzmann/runtime/lifelong_evaluation.py'] = 'changed'
    with pytest.raises(ValueError, match='Unaudited unrelated junction'):
        migrate(old_config(), changed)
    extra = junction_config()
    extra['parameters'] += 1
    with pytest.raises(ValueError, match='declared parameters'):
        migrate(old_config(), extra)


def test_junction_source_compatibility_requires_exact_reviewed_hash_pair():
    from pathlib import Path
    audit = json.loads((Path(__file__).resolve().parents[1] /
        'results/published/medium_gen3_source_continuation_audit_20261009.json').read_text())
    before, after = old_config(), junction_config()
    for record in audit['records']:
        before['source_hashes'][record['path']] = record['before_sha256']
        after['source_hashes'][record['path']] = record['after_sha256']
    recorded = migrate(before, after)
    assert before == after
    assert len(recorded['compatibility_source_evidence']) == 3
    assert 'conditional VJP' in recorded['solver_migration']
    altered_before, altered_after = old_config(), junction_config()
    record = audit['records'][0]
    altered_before['source_hashes'][record['path']] = record['before_sha256']
    altered_after['source_hashes'][record['path']] = 'unreviewed-change'
    with pytest.raises(ValueError, match='exact reviewed diff'):
        migrate(altered_before, altered_after)


def test_junction_config_adoption_does_not_authorize_learning_or_evaluation_change():
    before, after = old_config(), junction_config()
    after['lr'] = .0001
    migrate(before, after)
    # The caller's mandatory equality check still rejects unrelated config fields.
    assert before != after
    assert before['lr'] == .0002


def test_junction_gradient_adoption_preserves_old_entries_and_only_omits_new_gate():
    class Model:
        def named_parameters(self):
            return iter([('source.weight', None), ('decoder.bias', None),
                         ('medium.hopf_pathway.gate.weight', None),
                         ('medium.hopf_pathway.gate.bias', None)])
    good = {'pending': 0, 'pending_gradients': {'source.weight': None,
                                              'decoder.bias': torch.zeros(2)}}
    validate_junction_pending_gradients(Model(), good)
    with pytest.raises(ValueError, match='preserve every inherited'):
        validate_junction_pending_gradients(Model(), {'pending': 0,
            'pending_gradients': {'source.weight': None}})
    with pytest.raises(ValueError, match='completed credit boundary'):
        validate_junction_pending_gradients(Model(), {**good, 'pending': 1})
