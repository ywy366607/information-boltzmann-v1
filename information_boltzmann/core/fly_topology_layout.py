"""Stable incoming edge layouts with bounded temporary memory and a topology-only disk cache."""
import hashlib
import json
import os
from pathlib import Path
import uuid

import numpy as np


def incoming_arrays(pre, post, splits, n_neurons, *, chunk_edges=131072, cache_dir=None):
    """Return int32 (original edge index, source, row offsets) per delay tier.

    Sort each small block stably, then scatter its row runs to their final global positions using row counts. Earlier
    blocks precede later blocks within every row, exactly matching a full stable argsort. No full-size int64
    permutation or gathered source array is needed. Cached outputs use file-backed arrays; weights never enter it.
    """
    if chunk_edges < 1 or n_neurons < 1:
        raise ValueError('Chunk size and neuron count must be positive')
    if pre.ndim != 1 or post.ndim != 1 or len(pre) != len(post):
        raise ValueError('Source and destination arrays must be matching vectors')
    if max(len(pre), n_neurons) > np.iinfo(np.int32).max:
        raise ValueError('Incoming layouts require int32-compatible edge and neuron counts')
    if not splits or splits[0] != 0 or splits[-1] != len(pre) or any(a > b for a, b in zip(splits, splits[1:])):
        raise ValueError('Delay splits must cover the topology in order')
    folder = None
    if cache_dir is not None:
        digest = hashlib.blake2b(digest_size=20)
        digest.update(json.dumps({'version': 1, 'splits': list(splits), 'neurons': n_neurons,
                                  'pre_dtype': str(pre.dtype), 'post_dtype': str(post.dtype)}).encode())
        for values in (pre, post):
            for left in range(0, len(values), chunk_edges):
                digest.update(np.ascontiguousarray(values[left:left + chunk_edges]).tobytes())
        folder = Path(cache_dir) / digest.hexdigest()
        if (folder / 'complete.json').exists():
            return _read_cached(folder, splits, n_neurons)
        folder.mkdir(parents=True, exist_ok=True)

    result = []
    for tier, (start, end) in enumerate(zip(splits, splits[1:])):
        counts = np.zeros(n_neurons, dtype=np.int64)
        for left in range(start, end, chunk_edges):
            destinations = post[left:min(left + chunk_edges, end)]
            if destinations.min() < 0 or destinations.max() >= n_neurons:
                raise ValueError('Destination outside neuron range')
            counts += np.bincount(destinations, minlength=n_neurons)
        offsets = np.empty(n_neurons + 1, dtype=np.int32)
        offsets[0] = 0
        np.cumsum(counts, out=offsets[1:])
        temp_paths = []
        arrays = []
        for name in ('order', 'source'):
            if folder is None:
                array = np.empty(end - start, dtype=np.int32)
            else:
                path = folder / f'{tier}_{name}.{uuid.uuid4().hex}.tmp.npy'
                array = np.lib.format.open_memmap(path, mode='w+', dtype=np.int32, shape=(end - start,))
                temp_paths.append((path, folder / f'{tier}_{name}.npy'))
            arrays.append(array)
        order, source = arrays
        written = np.zeros(n_neurons, dtype=np.int32)
        for left in range(start, end, chunk_edges):
            right = min(left + chunk_edges, end)
            local = np.argsort(post[left:right], kind='stable')
            destinations = post[left:right][local]
            block_counts = np.bincount(destinations, minlength=n_neurons).astype(np.int32)
            block_start = np.empty(n_neurons, dtype=np.int32)
            block_start[0] = 0
            np.cumsum(block_counts[:-1], out=block_start[1:])
            position = (offsets[destinations] + written[destinations]
                        + np.arange(right - left, dtype=np.int32) - block_start[destinations])
            original = local.astype(np.int32) + left
            order[position] = original
            source[position] = pre[original]
            written += block_counts
        if folder is None:
            result.append((order, source, offsets))
        else:
            for array in arrays:
                array.flush()
            # Windows cannot replace an open memory mapping. Close every local reference first.
            for array in arrays:
                array._mmap.close()
            for temporary, target in temp_paths:
                os.replace(temporary, target)
            temporary = folder / f'{tier}_offsets.{uuid.uuid4().hex}.tmp.npy'
            np.save(temporary, offsets)
            os.replace(temporary, folder / f'{tier}_offsets.npy')
    if folder is not None:
        temporary = folder / f'complete.{uuid.uuid4().hex}.tmp.json'
        temporary.write_text(json.dumps({'version': 1, 'neurons': n_neurons, 'splits': list(splits)}), encoding='utf-8')
        os.replace(temporary, folder / 'complete.json')
        return _read_cached(folder, splits, n_neurons)
    return tuple(result)


def _read_cached(folder, splits, n_neurons):
    result = []
    for tier, (start, end) in enumerate(zip(splits, splits[1:])):
        arrays = tuple(np.load(folder / f'{tier}_{name}.npy', mmap_mode='c')
                       for name in ('order', 'source', 'offsets'))
        shapes = ((end - start,), (end - start,), (n_neurons + 1,))
        if any(a.dtype != np.int32 or a.shape != shape for a, shape in zip(arrays, shapes)):
            raise ValueError('Invalid cached incoming layout')
        if arrays[2][0] != 0 or arrays[2][-1] != end - start:
            raise ValueError('Invalid cached incoming row offsets')
        result.append(arrays)
    return tuple(result)
