"""Bounded-write serialization for the continuous mother individual."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

from scripts.ib_local import navigation_checkpoint
from scripts.ib_local import run_navigation_residual_wood as legacy


class ChunkedWriter:
    """Limit each compressor input without copying the underlying tensor buffer."""
    def __init__(self, stream, chunk_size=1024 * 1024):
        self.stream = stream
        self.chunk_size = chunk_size
        self.position = 0

    def write(self, data):
        view = memoryview(data).cast('B')
        for start in range(0, len(view), self.chunk_size):
            self.stream.write(view[start:start + self.chunk_size])
        self.position += len(view)
        return len(view)

    def flush(self):
        self.stream.flush()

    def tell(self):
        # Streaming compressors need not support seek/tell themselves.
        return self.position


def configure_compression(config, *, backend=None, workers=None):
    """Explicit saved execution policy; the checkpoint remains standard gzip."""
    old = config.get('checkpoint_compressor', {})
    backend = backend or old.get('backend', 'gzip')
    workers = workers if workers is not None else (old.get('workers', 1)
        if backend == old.get('backend', backend) else 1)
    if backend not in ('gzip', 'isal') or workers not in (1, 2, 4):
        raise ValueError('checkpoint compressor requires gzip/isal and 1, 2, or 4 workers')
    if backend == 'gzip' and workers != 1:
        raise ValueError('standard gzip uses one compression worker')
    if backend == 'isal':
        from isal import igzip_threaded  # optional runtime dependency; fail before saving
    config['checkpoint_compressor'] = dict(schema=1, backend=backend, workers=workers,
        block_bytes=1024**2, isal_level=1, format='single standard gzip stream')


def save(engine, output, cycle, config, *, streamed_hash=True):
    """Keep the existing atomic checkpoint and carried-state contract."""
    if getattr(engine, '_rhythm_runtime', None) is not None:
        raise RuntimeError('finish the active external transaction before committing its checkpoint')
    if getattr(engine, '_reading_option_session', None) is not None:
        raise RuntimeError('finish the active reading option before committing its checkpoint')
    archive = getattr(engine.model, '_causal_archive', None)
    previous_archive = config.get('causal_archive')
    if archive is not None:
        config['causal_archive'] = archive.checkpoint(output)
    original_open = legacy.gzip.open
    written = {}

    def bounded_open(filename, mode='rb', *args, **kwargs):
        if mode=='wb' and config.get('checkpoint_compression_level') is not None:
            kwargs['compresslevel']=config['checkpoint_compression_level']
        if mode != 'wb':
            return original_open(filename, mode, *args, **kwargs)
        compression = config.get('checkpoint_compressor', {})
        if not streamed_hash and compression.get('backend', 'gzip') == 'gzip':
            return _WriterContext(original_open(filename, mode, *args, **kwargs))
        stream = _HashedWriterContext(filename, *args, compression=compression, **kwargs)
        if streamed_hash:
            written[Path(filename).resolve()] = stream
        return stream

    try:
        legacy.gzip.open = bounded_open
        navigation_checkpoint.save(engine, output, cycle, config)
    except BaseException:
        if previous_archive is None:
            config.pop('causal_archive', None)
        else:
            config['causal_archive'] = previous_archive
        raise
    finally:
        legacy.gzip.open = original_open
    if config.get('body_growth') is not None or config.get('native_body') is not None:
        # A hash-bound sidecar lets restore read configuration and caregiver
        # state without loading the entire 0.8GB body twice. The body checkpoint
        # remains authoritative; a missing/stale sidecar uses the old path.
        from scripts.ib_local.run_causal_water_reply import sha256_file
        world=engine.world.checkpoint_state_dict()
        streamed = written.get((Path(output)/'last.tmp.gz').resolve())
        digest = streamed.digest if streamed is not None else sha256_file(Path(output)/'last.pt.gz')
        mother_world_preview = {}
        for k, v in world.items():
            if not k.startswith('mother_'):
                continue
            if k == 'mother_unified_port' and isinstance(v, dict) and isinstance(v.get('gaze_dynamic_state'), dict):
                v_copy = dict(v)
                g_copy = dict(v['gaze_dynamic_state'])
                g_copy.pop('optimizer_state_b64', None)
                v_copy['gaze_dynamic_state'] = g_copy
                mother_world_preview[k] = v_copy
            else:
                mother_world_preview[k] = v
        metadata=dict(schema=1,checkpoint_sha256=digest,
            cycle=cycle,config=config,
            mother_world=mother_world_preview)
        temporary=Path(output)/'last.metadata.tmp.json'
        temporary.write_text(json.dumps(metadata,ensure_ascii=False)+'\n',encoding='utf-8')
        from scripts.ib_local.checkpoint_commit import publish_checkpoint
        publish_checkpoint(temporary,Path(output)/'last.metadata.json')
    # Only remove the replaced rolling snapshot in THIS output directory,
    # after the body checkpoint commits. Ancestor/other branches stay intact.
    if archive is not None and previous_archive is not None:
        previous = Path(previous_archive['snapshot']).resolve()
        current = Path(config['causal_archive']['snapshot']).resolve()
        if (previous != current and previous.parent == Path(output).resolve()
                and previous.name == 'causal_archive_'+previous_archive['sha256']+'.sqlite'):
            previous.unlink(missing_ok=True)
    return digest if config.get('body_growth') is not None or config.get('native_body') is not None else None


class _WriterContext:
    def __init__(self, stream):
        self.stream = stream

    def __enter__(self):
        return ChunkedWriter(self.stream.__enter__())

    def __exit__(self, *args):
        return self.stream.__exit__(*args)


class _DigestWriter:
    """Hash the actual compressed bytes sent to disk, including gzip footer."""
    def __init__(self, stream):
        self.stream, self.hash = stream, hashlib.sha256()
        self.name = stream.name

    def write(self, data):
        count = self.stream.write(data)
        if count != len(data):
            raise OSError('incomplete checkpoint write')
        self.hash.update(data)
        return count

    def flush(self):
        return self.stream.flush()

    def tell(self):
        return self.stream.tell()


class _HashedWriterContext:
    def __init__(self, filename, *args, compression=None, **kwargs):
        self.filename, self.args, self.kwargs = filename, args, kwargs
        self.compression = compression or {}

    def __enter__(self):
        self.raw = open(self.filename, 'wb')
        self.hashed = _DigestWriter(self.raw)
        try:
            if self.compression.get('backend', 'gzip') == 'isal':
                from isal import igzip_threaded
                self.compressed = igzip_threaded.open(self.hashed, 'wb',
                    compresslevel=self.compression['isal_level'],
                    threads=self.compression['workers'], block_size=self.compression['block_bytes'])
            else:
                self.compressed = gzip.GzipFile(filename=self.filename, mode='wb',
                    fileobj=self.hashed, *self.args, **self.kwargs)
        except BaseException:
            self.raw.close()
            raise
        return ChunkedWriter(self.compressed)

    def __exit__(self, *args):
        try:
            self.compressed.close()
            self.digest = self.hashed.hash.hexdigest()
        finally:
            self.raw.close()
