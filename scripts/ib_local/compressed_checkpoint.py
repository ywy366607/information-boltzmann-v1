"""Bounded single-inflation reads of existing atomic gzip checkpoints."""
from __future__ import annotations

from contextlib import contextmanager
import ctypes
import gzip
import io
import os
from pathlib import Path
import struct
import shutil
import tempfile


def available_memory():
    if os.name == 'nt':
        class Status(ctypes.Structure):
            _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong),
                        *[(name, ctypes.c_ulonglong) for name in
                          ('total', 'available', 'page', 'free_page', 'virtual', 'free_virtual', 'extended')]]
        state = Status()
        state.length = ctypes.sizeof(state)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):
            return 0
        return state.available
    try:
        return os.sysconf('SC_AVPHYS_PAGES') * os.sysconf('SC_PAGE_SIZE')
    except (OSError, ValueError, AttributeError):
        return 0


@contextmanager
def checkpoint_stream(path, *, max_buffer_bytes=1024**3, spool_directory=None):
    """Use RAM for one inflation when it and the decoded tensors fit comfortably.

    torch.load's zip reader seeks backwards. Seeking a gzip stream repeatedly
    re-inflates its prefix. A bounded forward read creates the same byte stream
    once. An explicit disk spool provides the same seeking stream with bounded
    host RAM. Larger checkpoints and unavailable staging retain gzip streaming.
    The gzip footer is only an admission hint; actual reads are bounded too.
    """
    path = Path(path)
    with path.open('rb') as raw:
        raw.seek(-4, 2)
        expected = struct.unpack('<I', raw.read(4))[0]
    admitted = 0 < expected <= max_buffer_bytes and available_memory() >= 3*expected + 512*1024**2
    directory = spool_directory or os.environ.get('CGB_CHECKPOINT_SPOOL')
    buffer = None
    if admitted:
        buffer = io.BytesIO()
    elif directory is not None and 0 < expected <= max_buffer_bytes:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(directory).free >= expected+256*1024**2:
            # TemporaryFile owns its unique path and releases it on close.
            # No checkpoint or output file is replaced by the working stream.
            buffer = tempfile.TemporaryFile(mode='w+b', dir=directory)
    if buffer is not None:
        try:
            with gzip.open(path, 'rb') as compressed:
                while chunk := compressed.read(min(8*1024**2, max_buffer_bytes+1-buffer.tell())):
                    buffer.write(chunk)
                    if buffer.tell() > max_buffer_bytes:
                        break
            if buffer.tell() <= max_buffer_bytes:
                buffer.seek(0)
                yield buffer
                return
        finally:
            buffer.close()
    with gzip.open(path, 'rb') as compressed:
        yield compressed
