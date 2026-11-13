"""Decode a bounded corpus slice without expanding it to one giant int64 array."""
import numpy as np


def decoded_world_bytes(raw, offset, chunk_size=1 << 20):
    """Match ``(raw[raw >= offset] - offset).astype(uint8)`` in source order.

    The output is bounded by the raw slice length. Only one bounded temporary
    integer selection and one mask are alive, even for a memory-mapped corpus.
    """
    if raw.ndim != 1 or chunk_size < 1:
        raise ValueError('a one-dimensional source and positive chunk size are required')
    output = np.empty(len(raw), dtype=np.uint8)
    written = 0
    for start in range(0, len(raw), chunk_size):
        part = raw[start:start + chunk_size]
        values = part[part >= offset]
        values -= offset
        end = written + len(values)
        output[written:end] = values
        written = end
    return output[:written]
