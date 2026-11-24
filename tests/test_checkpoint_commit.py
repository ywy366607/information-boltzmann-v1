from pathlib import Path

import pytest

from scripts.ib_local.checkpoint_commit import publish_checkpoint


def test_transient_lock_preserves_previous_until_atomic_publication(tmp_path,monkeypatch):
    temporary,target=tmp_path/'last.tmp',tmp_path/'last.pt'
    temporary.write_bytes(b'new complete event');target.write_bytes(b'previous complete event')
    original=Path.replace
    attempts=[]
    def replace(self,destination):
        attempts.append(1)
        if len(attempts)<3:
            assert target.read_bytes()==b'previous complete event'
            raise PermissionError('transient Windows lock')
        return original(self,destination)
    monkeypatch.setattr(Path,'replace',replace)
    monkeypatch.setattr('scripts.ib_local.checkpoint_commit.time.sleep',lambda _:None)
    publish_checkpoint(temporary,target)
    assert target.read_bytes()==b'new complete event' and len(attempts)==3


def test_persistent_lock_keeps_both_recoverable_files(tmp_path,monkeypatch):
    temporary,target=tmp_path/'last.tmp',tmp_path/'last.pt'
    temporary.write_bytes(b'new');target.write_bytes(b'old')
    def fail(*args):raise PermissionError('persistent lock')
    monkeypatch.setattr(Path,'replace',fail)
    monkeypatch.setattr('scripts.ib_local.checkpoint_commit.time.sleep',lambda _:None)
    with pytest.raises(PermissionError):publish_checkpoint(temporary,target)
    assert temporary.read_bytes()==b'new' and target.read_bytes()==b'old'
