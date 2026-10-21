"""Atomic, non-pickle persistent archives for the CGB local medium."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .medium import Medium


def save_medium(path, medium, provenance=None):
    path = Path(path)
    arrays = {}

    def encode(value):
        if isinstance(value, np.ndarray):
            key = f'array_{len(arrays)}'
            arrays[key] = value
            return {'array_key': key}
        if isinstance(value, dict):
            return {key: encode(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [encode(item) for item in value]
        if isinstance(value, np.generic):
            return value.item()
        return value

    metadata = encode({'state': medium.state_dict(), 'provenance': provenance or {}})
    arrays['metadata'] = np.array(json.dumps(metadata, allow_nan=False))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    try:
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        load_medium(temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_medium(path):
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata']))

        def decode(value):
            if isinstance(value, dict):
                if set(value) == {'array_key'}:
                    return archive[value['array_key']].copy()
                return {key: decode(item) for key, item in value.items()}
            if isinstance(value, list):
                return [decode(item) for item in value]
            return value

        content = decode(metadata)
    return Medium.from_state_dict(content['state']), content['provenance']
