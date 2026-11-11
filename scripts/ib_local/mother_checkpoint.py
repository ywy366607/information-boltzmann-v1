"""Bounded-write serialization for the continuous mother individual."""
from __future__ import annotations

import gzip

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
    original_open = legacy.gzip.open

    def bounded_open(filename, mode='rb', *args, **kwargs):
        stream = original_open(filename, mode, *args, **kwargs)
        if mode != 'wb':
            return stream
        return _WriterContext(stream)

    try:
        legacy.gzip.open = bounded_open
        navigation_checkpoint.save(engine, output, cycle, config)
    finally:
        legacy.gzip.open = original_open


class _WriterContext:
    def __init__(self, stream):
        self.stream = stream

    def __enter__(self):
        return ChunkedWriter(self.stream.__enter__())

    def __exit__(self, *args):
        return self.stream.__exit__(*args)
