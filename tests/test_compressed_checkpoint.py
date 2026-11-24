import gzip
import io

import pytest
import torch

from scripts.ib_local import compressed_checkpoint as checkpoint


@pytest.mark.parametrize('available,cap,buffered', [(10**10, 1024**2, True),
    (0, 1024**2, False), (10**10, 32, False)])
def test_same_serialized_tensors_and_state_with_bounded_or_streaming_reads(tmp_path, monkeypatch, available, cap, buffered):
    path = tmp_path/'body.pt.gz'
    expected = dict(weight=torch.arange(100).reshape(10, 10), optimizer={'step': 17},
                    rng=torch.get_rng_state(), pending=['old question', 4.5])
    with gzip.open(path, 'wb') as stream:
        torch.save(expected, stream)
    monkeypatch.setattr(checkpoint, 'available_memory', lambda: available)
    with checkpoint.checkpoint_stream(path, max_buffer_bytes=cap) as stream:
        assert isinstance(stream, io.BytesIO) == buffered
        actual = torch.load(stream, weights_only=False)
    assert actual['optimizer'] == expected['optimizer'] and actual['pending'] == expected['pending']
    assert torch.equal(actual['weight'], expected['weight']) and torch.equal(actual['rng'], expected['rng'])


def test_consumer_failure_closes_buffer(tmp_path, monkeypatch):
    path = tmp_path/'body.pt.gz'
    with gzip.open(path, 'wb') as stream:
        stream.write(b'complete payload')
    monkeypatch.setattr(checkpoint, 'available_memory', lambda: 10**10)
    with pytest.raises(RuntimeError), checkpoint.checkpoint_stream(path) as stream:
        raise RuntimeError('consumer failed')
    assert stream.closed


def test_disk_spool_is_seekable_with_low_ram_and_is_removed(tmp_path, monkeypatch):
    path = tmp_path/'body.pt.gz'
    spool = tmp_path/'staging'
    expected = {'weight': torch.arange(1000), 'pending': ['same world', 7]}
    with gzip.open(path, 'wb') as stream:
        torch.save(expected, stream)
    source = path.read_bytes()
    monkeypatch.setattr(checkpoint, 'available_memory', lambda: 0)
    with checkpoint.checkpoint_stream(path, spool_directory=spool) as stream:
        assert not isinstance(stream, (io.BytesIO, gzip.GzipFile))
        actual = torch.load(stream, weights_only=False)
    assert stream.closed and list(spool.iterdir()) == []
    assert actual['pending'] == expected['pending']
    assert torch.equal(actual['weight'], expected['weight'])
    assert path.read_bytes() == source


def test_spool_consumer_failure_closes_file(tmp_path, monkeypatch):
    path = tmp_path/'body.pt.gz'
    with gzip.open(path, 'wb') as stream:
        stream.write(b'bounded temporary file')
    monkeypatch.setattr(checkpoint, 'available_memory', lambda: 0)
    spool = tmp_path/'spool'
    with pytest.raises(RuntimeError), checkpoint.checkpoint_stream(path, spool_directory=spool) as stream:
        raise RuntimeError('consumer failed')
    assert stream.closed and list(spool.iterdir()) == []


def test_low_disk_space_uses_original_gzip_stream(tmp_path, monkeypatch):
    from types import SimpleNamespace
    path = tmp_path/'body.pt.gz'
    with gzip.open(path, 'wb') as stream:
        stream.write(b'payload')
    monkeypatch.setattr(checkpoint, 'available_memory', lambda: 0)
    monkeypatch.setattr(checkpoint.shutil, 'disk_usage', lambda path: SimpleNamespace(free=0))
    with checkpoint.checkpoint_stream(path, spool_directory=tmp_path/'spool') as stream:
        assert isinstance(stream, gzip.GzipFile)
        assert stream.read() == b'payload'
