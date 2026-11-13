"""Exact weight conversion without mapping source storage."""
import json
import struct

import numpy as np
import pytest
import torch

from scripts.ib_local.pretrained_gdn_stream_load import stream_parameters


def checkpoint(path, tensors):
    header, parts, offset = {}, [], 0
    for name, array in tensors.items():
        array = np.asarray(array, dtype='<f4')
        raw = array.tobytes()
        header[name] = dict(dtype='F32', shape=list(array.shape), data_offsets=[offset, offset+len(raw)])
        parts.append(raw)
        offset += len(raw)
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack('<Q', len(encoded))+encoded+b''.join(parts))


def test_stream_load_exact_fp16_and_time_parameter_fp32(tmp_path):
    with torch.device('meta'):
        model = torch.nn.Linear(3, 2)
        model.register_parameter('dt_bias', torch.nn.Parameter(torch.zeros(2)))
    tensors = dict(weight=np.arange(6).reshape(2, 3)/7, bias=[.1, -.1], dt_bias=[.00012345, 1.234567])
    path = tmp_path/'weights.safetensors'
    checkpoint(path, {**tensors, 'old.D':[0, 0]})
    result = stream_parameters(model, path, device='cpu', legacy_zero_keys={'old.D'})
    for name, value in tensors.items():
        dtype = torch.float32 if name == 'dt_bias' else torch.float16
        assert torch.equal(getattr(model, name), torch.tensor(np.array(value, dtype=np.float32)).to(dtype))
    assert result['parameter_elements'] == 10


def test_stream_loader_rejects_learned_obsolete_parameter(tmp_path):
    model = torch.nn.Linear(1, 1, bias=False)
    path = tmp_path/'weights.safetensors'
    checkpoint(path, {'weight':[[1]], 'old.D':[1]})
    with pytest.raises(ValueError, match='nonzero'):
        stream_parameters(model, path, device='cpu', legacy_zero_keys={'old.D'})
