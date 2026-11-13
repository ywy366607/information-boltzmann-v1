"""Bounded-write serialization for the continuous mother individual."""
from __future__ import annotations

import gzip
from pathlib import Path

from scripts.ib_local import navigation_checkpoint
from scripts.ib_local import run_navigation_residual_wood as legacy


class ChunkedWriter:
    """Limit each compressor input without copying the underlying tensor buffer."""
    def __init__(self, stream, chunk_size=1024 * 1024):
        self.stream = stream
        self.chunk_size = chunk_size

    def write(self, data):
        view = memoryview(data).cast('B')
        for start in range(0, len(view), self.chunk_size):
            self.stream.write(view[start:start + self.chunk_size])
        return len(view)

    def flush(self):
        self.stream.flush()

    def tell(self):
        return self.stream.tell()


def save(engine, output, cycle, config):
    """Keep the existing atomic checkpoint and carried-state contract."""
    archive = getattr(engine.model, '_causal_archive', None)
    previous_archive = config.get('causal_archive')
    if archive is not None:
        config['causal_archive'] = archive.checkpoint(output)
    original_open = legacy.gzip.open

    def bounded_open(filename, mode='rb', *args, **kwargs):
        stream = original_open(filename, mode, *args, **kwargs)
        if mode != 'wb':
            return stream
        return _WriterContext(stream)

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
    # Only remove the replaced rolling snapshot in THIS output directory,
    # after the body checkpoint commits. Ancestor/other branches stay intact.
    if archive is not None and previous_archive is not None:
        previous = Path(previous_archive['snapshot']).resolve()
        current = Path(config['causal_archive']['snapshot']).resolve()
        if (previous != current and previous.parent == Path(output).resolve()
                and previous.name == 'causal_archive_'+previous_archive['sha256']+'.sqlite'):
            previous.unlink(missing_ok=True)


class _WriterContext:
    def __init__(self, stream):
        self.stream = stream

    def __enter__(self):
        return ChunkedWriter(self.stream.__enter__())

    def __exit__(self, *args):
        return self.stream.__exit__(*args)
