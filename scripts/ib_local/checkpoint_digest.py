"""Bounded-buffer canonical value digests for full continuation-state audits.

Hashes ALL declared state, not just losses. Shared tensor views reuse their
already computed digest within one frozen walk. No second complete terminal
checkpoint is required solely for equality testing. Not a learning mechanism.
"""
import dataclasses
import hashlib
import struct

import numpy as np
import torch


def state_digest(value):
    memo = {}
    stats = dict(tensor_occurrences=0, unique_tensor_views=0,
                 tensor_bytes_hashed=0, largest_tensor_buffer_bytes=0)

    def visit(v):
        h = hashlib.sha256()
        if isinstance(v, torch.Tensor):
            stats['tensor_occurrences'] += 1
            key = (str(v.device), v.untyped_storage().data_ptr(), v.storage_offset(),
                   str(v.dtype), tuple(v.shape), tuple(v.stride()))
            if key in memo:
                return memo[key]
            h.update(repr(('tensor', str(v.dtype), str(v.device), tuple(v.shape), tuple(v.stride()))).encode())
            data = v.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
            h.update(memoryview(data.numpy()))
            stats['unique_tensor_views'] += 1
            stats['tensor_bytes_hashed'] += data.numel()
            stats['largest_tensor_buffer_bytes'] = max(stats['largest_tensor_buffer_bytes'], data.numel())
            memo[key] = h.digest()
            return memo[key]
        if isinstance(v, np.ndarray):
            if v.dtype.hasobject:
                raise TypeError('object arrays require an explicit audit schema')
            h.update(repr(('numpy', v.dtype.str, v.shape, v.strides)).encode())
            h.update(memoryview(np.ascontiguousarray(v)).cast('B'))
        elif isinstance(v, dict):
            h.update(b'dict')
            for k in sorted(v, key=lambda x: (type(x).__qualname__, repr(x))):
                h.update(visit(k)); h.update(visit(v[k]))
        elif isinstance(v, (list, tuple)):
            h.update(type(v).__name__.encode())
            for item in v:
                h.update(visit(item))
        elif isinstance(v, slice):
            h.update(b'slice')
            h.update(visit((v.start, v.stop, v.step)))
        elif isinstance(v, (set, frozenset)):
            h.update(type(v).__name__.encode())
            for item in sorted(visit(x) for x in v):
                h.update(item)
        elif isinstance(v, np.generic):
            h.update(('numpy_scalar:'+v.dtype.str).encode()); h.update(v.tobytes())
        elif isinstance(v, (torch.dtype, torch.device)):
            h.update((type(v).__name__+':'+str(v)).encode())
        elif dataclasses.is_dataclass(v):
            h.update((type(v).__module__+'.'+type(v).__qualname__).encode())
            for f in dataclasses.fields(v):
                h.update(visit(f.name)); h.update(visit(getattr(v, f.name)))
        elif v is None:
            h.update(b'None')
        elif isinstance(v, bool):
            h.update(b'True' if v else b'False')
        elif isinstance(v, int):
            h.update(('int:'+str(v)).encode())
        elif isinstance(v, float):
            h.update(b'float:'); h.update(struct.pack('!d', v))
        elif isinstance(v, (bytes, str)):
            h.update(type(v).__name__.encode()); h.update(v.encode() if isinstance(v, str) else v)
        elif isinstance(v, type):
            h.update(('type:'+v.__module__+'.'+v.__qualname__).encode())
        elif hasattr(v, '__dict__'):
            h.update((type(v).__module__+'.'+type(v).__qualname__).encode()); h.update(visit(vars(v)))
        else:
            raise TypeError(f'unhandled full-state audit field: {type(v)}')
        return h.digest()

    return dict(sha256=visit(value).hex(), **stats)
