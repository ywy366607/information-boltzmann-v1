"""Optional row-int8 storage for already frozen MLPs; input credit stays live.

Weights are unpacked for one linear operation and discarded, including during
backward. Active GDN ports are preserved. Optional read-only disk mapping moves
already packed fixed coefficients out of CUDA; it changes placement, not values.
"""
import torch
from torch import nn
from torch.nn import functional as F


def unpack(qweight, scale, dtype, device=None):
    device = qweight.device if device is None else device
    return (qweight.to(device=device, dtype=dtype)*scale.to(device=device, dtype=dtype)).contiguous()


def map_frozen_linear(packed, directory):
    """Read-only content-addressed coefficients; never map mutable life state."""
    import hashlib
    import mmap
    from pathlib import Path
    import warnings
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    def mapped(tensor):
        value = tensor.detach().cpu().contiguous()
        data = memoryview(value.numpy()).cast('B')
        checksum = hashlib.sha256(data).hexdigest()
        path = directory/(checksum+'.bin')
        try:
            with path.open('xb') as handle:
                handle.write(data)
        except FileExistsError:
            # A damaged/partial cache is not silently accepted or overwritten.
            check = hashlib.sha256()
            with path.open('rb') as handle:
                for chunk in iter(lambda: handle.read(65536), b''):
                    check.update(chunk)
            if path.stat().st_size != len(data) or check.hexdigest() != checksum:
                raise ValueError('frozen mapped coefficient cache differs')
        with path.open('rb') as handle:
            mapping = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='The given buffer is not writable.*', category=UserWarning)
            return torch.frombuffer(mapping, dtype=value.dtype).reshape(value.shape)

    if not isinstance(packed, FrozenInt8Linear) or packed.bias is not None:
        raise ValueError('only bias-free fixed packed projections may be mapped')
    packed.qweight = mapped(packed.qweight)
    packed.scale = mapped(packed.scale)
    return packed


class _FrozenLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, qweight, scale, bias):
        ctx.save_for_backward(qweight, scale)
        ctx.input_dtype = x.dtype
        ctx.input_device = x.device
        return F.linear(x, unpack(qweight, scale, x.dtype, x.device), bias)

    @staticmethod
    def backward(ctx, gradient):
        qweight, scale = ctx.saved_tensors
        weight = unpack(qweight, scale, ctx.input_dtype, ctx.input_device)
        input_gradient = F.linear(gradient.to(ctx.input_dtype), weight.t())
        return input_gradient, None, None, None


class FrozenInt8Linear(nn.Module):
    def __init__(self, linear):
        super().__init__()
        if not isinstance(linear, nn.Linear) or linear.weight.is_meta:
            raise TypeError('fully loaded Linear required')
        if linear.weight.requires_grad or (linear.bias is not None and linear.bias.requires_grad):
            raise ValueError('only already frozen weights may be packed')
        self.in_features, self.out_features = linear.in_features, linear.out_features
        with torch.no_grad():
            largest = linear.weight.detach().float().abs().amax(dim=1, keepdim=True)
            scale = torch.where(largest == 0, torch.ones_like(largest), largest/127)
            qweight = (linear.weight.detach().float()/scale).round().clamp_(-127, 127).to(torch.int8)
        self.register_buffer('qweight', qweight)
        self.register_buffer('scale', scale)
        self.register_buffer('bias', None if linear.bias is None else linear.bias.detach().clone())
        self.train(linear.training)

    def forward(self, x):
        return _FrozenLinear.apply(x, self.qweight, self.scale, self.bias)


def pack_frozen_mlps(model):
    """Keep original MLP/SwiGLU graph, use its unfused down-projection branch.

    The original fused SwiGLU+linear asks directly for a dense weight, which
    would keep the unpacked weight alive for backward. Its existing unfused
    branch calls the packed Linear and preserves gradients into earlier layers.
    """
    modules, old_bytes, new_bytes = [], 0, 0
    for index, block in enumerate(model.model.layers):
        mlp = block.mlp
        names = ('gate_proj', 'up_proj', 'down_proj')
        # Validate the whole MLP before modifying any of its projections.
        for name in names:
            item = getattr(mlp, name)
            if not isinstance(item, nn.Linear) or any(p.requires_grad for p in item.parameters()):
                raise ValueError(f'layer{index}.{name} is not an already frozen Linear')
        mlp.fuse_swiglu = False
        for name in names:
            old = getattr(mlp, name)
            old_bytes += sum(p.numel()*p.element_size() for p in old.parameters())
            packed = FrozenInt8Linear(old)
            new_bytes += sum(b.numel()*b.element_size() for b in packed.buffers())
            setattr(mlp, name, packed)
            modules.append(f'model.layers.{index}.mlp.{name}')
            del old
    return dict(codec='symmetric_row_int8_scale_fp32_v1', modules=modules,
                old_frozen_weight_bytes=old_bytes, packed_frozen_weight_bytes=new_bytes,
                saved_weight_bytes=old_bytes-new_bytes,
                gdn_parameters_changed=False, extra_trainable_parameters=0,
                precision_change=True, original_swiglu_unfused_branch=True)


def stream_frozen_mlp_parameters(model, path, *, device='cuda', dtype=torch.float16, legacy_zero_keys=(),
                                 pack_fixed_gdn=False, native_layers=(1,12), mapped_directory=None):
    """Load directly into packed storage, avoiding persistent FP16 allocator holes."""
    import json
    from pathlib import Path
    import struct
    import re
    import numpy as np
    expected=dict(model.named_parameters())
    modules, old_bytes, new_bytes=[],0,0
    if list(model.named_buffers()):raise ValueError('parameter-only meta model required')
    with Path(path).open('rb') as source:
        length=struct.unpack('<Q',source.read(8))[0]
        header=json.loads(source.read(length));header.pop('__metadata__',None)
        if set(header)!=set(expected)|set(legacy_zero_keys):raise ValueError('source keys differ')
        start=8+length
        for name,info in header.items():
            if info['dtype']!='F32':raise ValueError('all-F32 source required')
            lo,hi=info['data_offsets']
            if hi-lo!=int(np.prod(info['shape']))*4:raise ValueError('source shape/bytes differ')
            source.seek(start+lo);raw=source.read(hi-lo)
            if len(raw)!=hi-lo:raise ValueError('truncated source')
            array=np.frombuffer(raw,dtype='<f4').reshape(info['shape'])
            if name in legacy_zero_keys:
                if np.count_nonzero(array):raise ValueError('obsolete D is learned/nonzero')
                del raw,array
                continue
            old=expected[name]
            if tuple(old.shape)!=tuple(info['shape']):raise ValueError('native source shape differs')
            module,_,key=name.rpartition('.')
            owner=model.get_submodule(module) if module else model
            tensor=torch.from_numpy(array.copy())
            is_mlp='.mlp.' in name or name.startswith('mlp.')
            match=re.search(r'(?:^|\.)layers\.(\d+)\.attn\.(q_proj|k_proj|v_proj|g_proj|o_proj)\.weight$',name)
            is_fixed_gdn=bool(pack_fixed_gdn and match and int(match.group(1)) not in native_layers)
            if is_mlp or is_fixed_gdn:
                if key!='weight' or not isinstance(owner,nn.Linear) or owner.bias is not None:
                    raise ValueError('only known bias-free frozen projections admitted')
                # Match the original FP16 loader's coefficients before row quantization.
                owner.weight=nn.Parameter(tensor.to(dtype=dtype),requires_grad=False)
                packed=FrozenInt8Linear(owner)
                packed=(packed.to(device) if mapped_directory is None else
                        map_frozen_linear(packed, mapped_directory))
                parent,_,projection=module.rpartition('.')
                parent_module=model.get_submodule(parent) if parent else model
                if is_mlp and projection not in ('gate_proj','up_proj','down_proj'):
                    raise ValueError('unrecognized MLP projection')
                if is_mlp:parent_module.fuse_swiglu=False
                old_bytes+=old.numel()*torch.empty((),dtype=dtype).element_size()
                new_bytes+=sum(b.numel()*b.element_size() for b in packed.buffers())
                setattr(parent_module,projection,packed);modules.append(module)
                del packed
            else:
                target=torch.float32 if name.endswith(('A_log','dt_bias')) else dtype
                setattr(owner,key,nn.Parameter(tensor.to(device=device,dtype=target),requires_grad=old.requires_grad))
            del tensor,raw,array,owner
    if any(p.is_meta for p in model.parameters()) or any(b.is_meta for b in model.buffers()):
        raise AssertionError('unloaded meta storage')
    return dict(parameters=len(expected),parameter_elements=sum(p.numel() for p in expected.values()),
                loading='ordinary_binary_direct_packed_frozen_mlp',device=str(device),
                omitted_zero_keys=sorted(legacy_zero_keys),
                frozen_mlp=dict(codec='symmetric_row_int8_scale_fp32_v1',modules=modules,
                    old_frozen_weight_bytes=old_bytes,packed_frozen_weight_bytes=new_bytes,
                    saved_weight_bytes=old_bytes-new_bytes,gdn_parameters_changed=pack_fixed_gdn,
                    native_gdn_parameters_changed=False, preserved_native_layers=list(native_layers),
                    scope='all_mlp_and_fixed_gdn_qkvg_o' if pack_fixed_gdn else 'all_mlp_only',
                    embedding_and_lm_head_unmodified=True,gdn_a_b_gates_unmodified=True,
                    storage_placement='cuda' if mapped_directory is None else 'readonly_mapped_cpu',
                    extra_trainable_parameters=0,precision_change=True,original_swiglu_unfused_branch=True))


def load_streamed_int8_mlp(path, *, device='cuda', dtype=torch.float16, pack_fixed_gdn=False,
                         mapped_directory=None):
    """Pure-GDN original architecture; only explicitly frozen MLP storage differs."""
    from pathlib import Path
    from fla.models.gated_deltanet import GatedDeltaNetConfig,GatedDeltaNetForCausalLM
    path=Path(path)
    config=GatedDeltaNetConfig.from_pretrained(path,local_files_only=True)
    if config.attn is not None:raise ValueError('pure fixed-state GDN only')
    with torch.device('meta'):model=GatedDeltaNetForCausalLM(config)
    legacy={f'model.layers.{i}.attn.D' for i in range(config.num_hidden_layers)}
    audit=stream_frozen_mlp_parameters(model,path/'model.safetensors',device=device,dtype=dtype,
                                     legacy_zero_keys=legacy,pack_fixed_gdn=pack_fixed_gdn,
                                     mapped_directory=mapped_directory)
    model.eval()
    return model,audit


def load_streamed_int8_frozen(path, *, device='cuda', dtype=torch.float16, mapped_directory=None):
    return load_streamed_int8_mlp(path,device=device,dtype=dtype,pack_fixed_gdn=True,
                                mapped_directory=mapped_directory)
