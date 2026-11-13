"""Read-only CPU mapping of trusted project checkpoints on this Torch build.

Windows torch.load(mmap=True) creates a private copy-on-write file mapping.
This adapter supplies a genuinely read-only buffer to the SAME Torch zip
deserializer. Returned tensors are immutable snapshot inputs: clone/copy into
live state before modifying. Byte swapping and non-CPU restore are disallowed.
Uses a private Torch API; covered by alias/value/lifetime tests, opt-in only.
"""
import mmap
import pickle
import sys
import warnings

import torch


def load_readonly_checkpoint(path):
    if sys.byteorder != 'little':
        raise RuntimeError('read-only mapping admits little-endian host only')
    with open(path, 'rb') as handle:
        mapping = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='The given buffer is not writable.*', category=UserWarning)
        buffer = torch.frombuffer(mapping, dtype=torch.uint8)
    with torch.serialization._open_zipfile_reader(str(path)) as archive:
        if not archive.has_record('byteorder') or archive.get_record('byteorder') != b'little':
            raise ValueError('read-only checkpoint requires an explicit matching byte order')
        # Tensor storage retains the original buffer owner after this returns.
        # No copy-on-write mapping or mutable shared mapping of the source.
        return torch.serialization._load(archive, 'cpu', pickle,
                                          overall_storage=buffer.untyped_storage())
