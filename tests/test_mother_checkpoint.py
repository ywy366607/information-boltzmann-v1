import gzip
import io

import torch
import pytest

from scripts.ib_local.mother_checkpoint import ChunkedWriter


def test_growth_metadata_is_hash_bound_and_does_not_duplicate_model(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from scripts.ib_local import mother_checkpoint,navigation_checkpoint
    from scripts.ib_local.causal_archive import sha256
    import json
    engine=SimpleNamespace(model=SimpleNamespace(),world=SimpleNamespace(
        checkpoint_state_dict=lambda:dict(mother_transcript=[('child','a'),('mother','b')],
                                         seen=torch.ones(10,dtype=torch.bool))))
    def write(*args): (tmp_path/'last.pt.gz').write_bytes(b'complete-checkpoint')
    monkeypatch.setattr(navigation_checkpoint,'save',write)
    config=dict(body_growth=dict(schema=1))
    mother_checkpoint.save(engine,tmp_path,5356,config)
    result=json.loads((tmp_path/'last.metadata.json').read_text(encoding='utf-8'))
    assert result['checkpoint_sha256']==sha256(tmp_path/'last.pt.gz')
    assert result['cycle']==5356 and 'seen' not in result['mother_world']


def test_chunked_torch_checkpoint_roundtrip():
    class TrackingStream:
        def __init__(self, stream):
            self.stream = stream
            self.largest = 0

        def write(self, data):
            self.largest = max(self.largest, len(data))
            return self.stream.write(data)

        def flush(self):
            self.stream.flush()

        def tell(self):
            return self.stream.tell()

    raw = io.BytesIO()
    expected = {'table': torch.arange(300_000, dtype=torch.float32), 'cycle': 1894}
    with gzip.GzipFile(fileobj=raw, mode='wb') as compressed:
        tracked = TrackingStream(compressed)
        torch.save(expected, ChunkedWriter(tracked, chunk_size=8192))
        assert tracked.largest <= 8192
    raw.seek(0)
    with gzip.GzipFile(fileobj=raw, mode='rb') as compressed:
        actual = torch.load(compressed, weights_only=False)
    assert actual['cycle'] == expected['cycle']
    assert torch.equal(actual['table'], expected['table'])


def test_native_metadata_is_hash_bound_without_a_growth_ancestor(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from scripts.ib_local import mother_checkpoint,navigation_checkpoint
    import json
    engine=SimpleNamespace(model=SimpleNamespace(),world=SimpleNamespace(
        checkpoint_state_dict=lambda:dict(seen=torch.zeros(10,dtype=torch.bool))))
    monkeypatch.setattr(navigation_checkpoint,'save',
        lambda *args:(tmp_path/'last.pt.gz').write_bytes(b'native-complete-body'))
    mother_checkpoint.save(engine,tmp_path,0,dict(native_body=dict(schema=1)))
    result=json.loads((tmp_path/'last.metadata.json').read_text(encoding='utf-8'))
    assert result['cycle']==0 and result['config']['native_body']['schema']==1
    assert 'body_growth' not in result['config'] and result['mother_world']=={}


def test_streamed_hash_covers_actual_gzip_and_needs_no_file_reread(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from scripts.ib_local import mother_checkpoint, navigation_checkpoint, run_causal_water_reply
    import hashlib
    import json
    expected = {'state':torch.arange(80_000), 'pending':torch.randn(1024)}
    engine = SimpleNamespace(model=SimpleNamespace(), world=SimpleNamespace(checkpoint_state_dict=lambda:{}))
    def write(*args):
        with mother_checkpoint.legacy.gzip.open(tmp_path/'last.tmp.gz', 'wb', compresslevel=0) as stream:
            torch.save(expected, stream)
        (tmp_path/'last.tmp.gz').replace(tmp_path/'last.pt.gz')
    monkeypatch.setattr(navigation_checkpoint, 'save', write)
    def forbidden(*args):
        raise AssertionError('streamed save must not re-read the file for SHA256')
    monkeypatch.setattr(run_causal_water_reply, 'sha256_file', forbidden)
    mother_checkpoint.save(engine, tmp_path, 278, dict(native_body=dict(schema=1), checkpoint_compression_level=0))
    metadata = json.loads((tmp_path/'last.metadata.json').read_text())
    assert metadata['checkpoint_sha256'] == hashlib.sha256((tmp_path/'last.pt.gz').read_bytes()).hexdigest()
    with gzip.open(tmp_path/'last.pt.gz', 'rb') as stream:
        actual = torch.load(stream, weights_only=True)
    for name in expected:
        assert torch.equal(actual[name], expected[name])


def test_serialization_failure_keeps_previous_complete_file_and_restores_gzip(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import pytest
    from scripts.ib_local import mother_checkpoint, navigation_checkpoint
    target = tmp_path/'last.pt.gz'
    target.write_bytes(b'previous-complete-body')
    old_open = mother_checkpoint.legacy.gzip.open
    engine = SimpleNamespace(model=SimpleNamespace())
    def fail(*args):
        with mother_checkpoint.legacy.gzip.open(tmp_path/'last.tmp.gz', 'wb') as stream:
            stream.write(b'partial serialization')
            raise RuntimeError('injected serializer failure')
    monkeypatch.setattr(navigation_checkpoint, 'save', fail)
    with pytest.raises(RuntimeError, match='serializer failure'):
        mother_checkpoint.save(engine, tmp_path, 278, dict(native_body={}))
    assert target.read_bytes() == b'previous-complete-body'
    assert not (tmp_path/'last.metadata.json').exists()
    assert mother_checkpoint.legacy.gzip.open is old_open


@pytest.mark.parametrize('workers', [1, 4])
def test_isal_save_restores_with_standard_reader_and_hashes_complete_footer(tmp_path, monkeypatch, workers):
    pytest.importorskip('isal.igzip_threaded')
    from types import SimpleNamespace
    from scripts.ib_local import mother_checkpoint, navigation_checkpoint
    from scripts.ib_local.compressed_checkpoint import checkpoint_stream
    import hashlib
    import json
    expected = dict(table=torch.arange(300_000), moments=torch.randn(1024), cycle=2815)
    engine = SimpleNamespace(model=SimpleNamespace(), world=SimpleNamespace(checkpoint_state_dict=lambda:{}))
    config = dict(native_body=dict(schema=1))
    mother_checkpoint.configure_compression(config, backend='isal', workers=workers)
    inherited = dict(config)
    mother_checkpoint.configure_compression(inherited)
    assert inherited == config
    def write(*args):
        with mother_checkpoint.legacy.gzip.open(tmp_path/'last.tmp.gz', 'wb', compresslevel=1) as stream:
            torch.save(expected, stream)
            assert stream.tell() > 0
        (tmp_path/'last.tmp.gz').replace(tmp_path/'last.pt.gz')
    monkeypatch.setattr(navigation_checkpoint, 'save', write)
    mother_checkpoint.save(engine, tmp_path, 2815, config)
    output = tmp_path/'last.pt.gz'
    assert json.loads((tmp_path/'last.metadata.json').read_text())['checkpoint_sha256'] == hashlib.sha256(output.read_bytes()).hexdigest()
    # Ordinary old gzip, plus the existing bounded seekable-spool reader.
    with gzip.open(output, 'rb') as stream:
        actual = torch.load(stream, weights_only=True)
    with checkpoint_stream(output, spool_directory=tmp_path) as stream:
        staged = torch.load(stream, weights_only=True)
    for key in ('table', 'moments'):
        assert torch.equal(expected[key], actual[key]) and torch.equal(expected[key], staged[key])


def test_isal_write_failure_does_not_publish_or_leave_background_threads(tmp_path, monkeypatch):
    pytest.importorskip('isal.igzip_threaded')
    from types import SimpleNamespace
    from scripts.ib_local import mother_checkpoint, navigation_checkpoint
    import threading
    previous = {thread.ident for thread in threading.enumerate()}
    (tmp_path/'last.pt.gz').write_bytes(b'previous complete body')
    engine = SimpleNamespace(model=SimpleNamespace())
    config = {}
    mother_checkpoint.configure_compression(config, backend='isal', workers=2)
    def fail_write(self, data):
        raise OSError('injected destination failure')
    monkeypatch.setattr(mother_checkpoint._DigestWriter, 'write', fail_write)
    def write(*args):
        with mother_checkpoint.legacy.gzip.open(tmp_path/'last.tmp.gz', 'wb') as stream:
            torch.save(torch.arange(300_000), stream)
        pytest.fail('failed compressed destination must not commit')
    monkeypatch.setattr(navigation_checkpoint, 'save', write)
    with pytest.raises(OSError, match='injected destination failure'):
        mother_checkpoint.save(engine, tmp_path, 2815, config)
    assert (tmp_path/'last.pt.gz').read_bytes() == b'previous complete body'
    assert {thread.ident for thread in threading.enumerate()} == previous
