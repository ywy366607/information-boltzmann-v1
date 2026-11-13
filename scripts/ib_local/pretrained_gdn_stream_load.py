"""Bounded non-mmap loading for the existing all-F32 pure-GDN safetensor file.

Windows torch storage slicing can access-violate in safe_open/get_slice. Read
one declared tensor with ordinary binary IO and copy it directly to its final
device. Architecture, float16 weights and original F32 A_log/dt_bias match the
existing loader. Only the exact24 obsolete zero D vectors may be omitted.
"""
import json
from pathlib import Path
import struct

import numpy as np
import torch


def stream_parameters(model, path, *, device='cuda', dtype=torch.float16, legacy_zero_keys=()):
    expected = dict(model.named_parameters())
    if list(model.named_buffers()):
        raise ValueError('streaming port requires a parameter-only source model')
    with Path(path).open('rb') as source:
        header_length = struct.unpack('<Q', source.read(8))[0]
        header = json.loads(source.read(header_length))
        header.pop('__metadata__', None)
        if set(header) != set(expected) | set(legacy_zero_keys):
            raise ValueError('checkpoint keys do not exactly match native parameters and admitted zero D')
        start = 8+header_length
        for name, info in header.items():
            if info['dtype'] != 'F32':
                raise ValueError('this exact source loader requires F32 safetensors')
            lo, hi = info['data_offsets']
            if hi-lo != int(np.prod(info['shape']))*4:
                raise ValueError('tensor byte count/shape mismatch')
            source.seek(start+lo)
            raw = source.read(hi-lo)
            if len(raw) != hi-lo:
                raise ValueError('truncated checkpoint tensor')
            array = np.frombuffer(raw, dtype='<f4').reshape(info['shape'])
            if name in legacy_zero_keys:
                if np.count_nonzero(array):
                    raise ValueError('obsolete D is learned/nonzero')
                continue
            old = expected[name]
            if tuple(old.shape) != tuple(info['shape']):
                raise ValueError('native parameter shape mismatch')
            target_dtype = torch.float32 if name.endswith(('A_log', 'dt_bias')) else dtype
            value = torch.from_numpy(array.copy()).to(device=device, dtype=target_dtype)
            module, _, key = name.rpartition('.')
            owner = model.get_submodule(module) if module else model
            setattr(owner, key, torch.nn.Parameter(value, requires_grad=old.requires_grad))
            del raw, array, value
    if any(p.is_meta for p in model.parameters()):
        raise AssertionError('unloaded meta parameter')
    return dict(parameters=len(expected), omitted_zero_keys=sorted(legacy_zero_keys),
                parameter_elements=sum(p.numel() for p in model.parameters()),
                device=str(device), loading='ordinary_binary_per_tensor_no_mmap')


def load_streamed_pure_gdn(path, *, device='cuda', dtype=torch.float16):
    from fla.models.gated_deltanet import GatedDeltaNetConfig, GatedDeltaNetForCausalLM
    path = Path(path)
    config = GatedDeltaNetConfig.from_pretrained(path, local_files_only=True)
    if config.attn is not None:
        raise ValueError('pure fixed-state GDN only')
    with torch.device('meta'):
        model = GatedDeltaNetForCausalLM(config)
    legacy = {f'model.layers.{i}.attn.D' for i in range(config.num_hidden_layers)}
    audit = stream_parameters(model, path/'model.safetensors', device=device, dtype=dtype,
                              legacy_zero_keys=legacy)
    model.eval()
    return model, audit
