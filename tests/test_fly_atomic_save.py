"""A resource failure must not replace an individual's last complete state."""
from pathlib import Path

import pytest
import torch

from scripts.ib import train_fly_bptt_stream as trainer


def test_memory_failure_retries_with_clean_storage_and_preserves_old_until_commit(tmp_path, monkeypatch):
    destination = tmp_path / 'last.pt'
    real_save = torch.save
    real_save({'old': torch.tensor([7.])}, destination)
    original = destination.read_bytes()
    payload = {'brain': torch.tensor([1., 2.]), 'pending': {'reply': 3}}
    calls, cleared = [], []
    monkeypatch.setattr(trainer, '_CHECKPOINT_DIR', None)
    monkeypatch.setattr(trainer.time, 'sleep', lambda _: None)
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: True)
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: cleared.append(True))

    def fail_once(value, path):
        calls.append(Path(path))
        assert destination.read_bytes() == original
        if len(calls) == 1:
            Path(path).write_bytes(b'incomplete')
            raise MemoryError('staging buffer exhausted')
        real_save(value, path)

    monkeypatch.setattr(torch, 'save', fail_once)
    trainer.atomic_save(destination, payload)
    assert len(calls) == len(cleared) == 2
    assert not calls[0].exists() and not calls[1].exists()
    saved = torch.load(destination, weights_only=False)
    torch.testing.assert_close(saved['brain'], payload['brain'])
    assert saved['pending'] == payload['pending']


def test_exhausted_retries_preserve_complete_previous_checkpoint(tmp_path, monkeypatch):
    destination = tmp_path / 'last.pt'
    destination.write_bytes(b'previous complete state')
    monkeypatch.setattr(trainer, '_CHECKPOINT_DIR', None)
    monkeypatch.setattr(trainer.time, 'sleep', lambda _: None)
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: False)

    def fail(value, path):
        Path(path).write_bytes(b'partial')
        raise MemoryError('no temporary memory')

    monkeypatch.setattr(torch, 'save', fail)
    with pytest.raises(MemoryError, match='atomic_save failed after retries'):
        trainer.atomic_save(destination, {'brain': torch.ones(1)})
    assert destination.read_bytes() == b'previous complete state'
    assert not list(tmp_path.glob('*.tmp*'))
