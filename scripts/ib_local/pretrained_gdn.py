"""Ports for a local, pretrained pure FLA Gated DeltaNet.

The original model owns embeddings, projections, gates, MLPs and delta kernels.
Only bounded cache compatibility and a neutral recall inlet are supplied here.
This module does not replace the continuing navigation individual or assert that
its weights/experience can be transferred between incompatible feature spaces.
"""
from __future__ import annotations

from contextlib import contextmanager
import inspect
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def new_recurrent_cache(seen_tokens: int = 0):
    """Bridge the FLA 0.2.2 cache metadata to Transformers 4.49 or 4.57.

    FLA overrides all actual state operations. New HF versions additionally
    require an empty ``layers`` list in the otherwise unused parent cache.
    No attention KV cache is introduced.
    """
    from fla.models.utils import Cache
    from transformers.cache_utils import Cache as ParentCache

    cache = Cache.__new__(Cache)
    if 'layers' in inspect.signature(ParentCache.__init__).parameters:
        ParentCache.__init__(cache, layers=[])
    else:
        ParentCache.__init__(cache)
    cache.states = []
    cache._seen_tokens = seen_tokens
    return cache


def cache_nbytes(cache) -> int:
    """Unique tensor storage, excluding the bounded Python metadata."""
    if cache is None:
        return 0
    seen = set()
    total = 0
    for state in cache.states:
        for value in state.values():
            values = value if isinstance(value, (tuple, list)) else (value,)
            for tensor in values:
                if isinstance(tensor, torch.Tensor):
                    key = (tensor.device, tensor.untyped_storage().data_ptr())
                    if key not in seen:
                        total += tensor.untyped_storage().nbytes()
                        seen.add(key)
    return total


def clone_cache(cache, *, detach=True):
    """Copy the fixed recurrent matrices and convolution buffers for a branch."""
    result = new_recurrent_cache(0 if cache is None else cache._seen_tokens)
    if cache is None:
        return result

    def copy(value):
        if isinstance(value, torch.Tensor):
            return (value.detach() if detach else value).clone()
        if isinstance(value, tuple):
            return tuple(copy(item) for item in value)
        if isinstance(value, list):
            return [copy(item) for item in value]
        return value

    result.states = [{key: copy(value) for key, value in state.items()}
                     for state in cache.states]
    return result


def streaming_short_convolution(module, x, residual=None, mask=None, cache=None,
                               output_final_state=False, cu_seqlens=None, **kwargs):
    """Apply the existing causal depthwise convolution with its actual prefix.

    FLA 0.2.2's multi-token fallback overwrites an incoming prefix before its
    convolution. Prepending the last W-1 raw projections fixes that continuation
    boundary. Kernels, learned weights, activation and causal receptive field
    are unchanged. State is returned separately, never overwriting a branch's
    supplied prefix; no past graph is retained by this helper.
    """
    if cu_seqlens is not None:
        raise ValueError('This continuous-individual port does not pack sequences')
    if mask is not None:
        x = x * mask.unsqueeze(-1)
    width = module.kernel_size[0]
    current = x.transpose(1, 2)
    if cache is None:
        prefix = current.new_zeros(current.shape[0], current.shape[1], width-1)
        previous = current.new_zeros(current.shape[0], current.shape[1], width)
    else:
        prefix = cache[..., -(width-1):] if width > 1 else cache[..., :0]
        previous = cache
    extended = torch.cat((prefix, current), dim=-1)
    out = F.conv1d(extended, module.weight, module.bias, groups=current.shape[1])
    if module.activation in ('silu', 'swish'):
        out = F.silu(out)
    elif module.activation is not None:
        raise ValueError(f'Unsupported pretrained activation {module.activation}')
    out = out.transpose(1, 2)
    if residual is not None:
        out = out + residual
    state = torch.cat((previous, current), dim=-1)[..., -width:].contiguous()
    return out, state if output_final_state or cache is not None else None


def enable_streaming_convolution(model):
    """Install an explicit, reversible compatibility path, preserving parameters."""
    import types
    count = 0
    for block in model.model.layers:
        for name in ('q_conv1d', 'k_conv1d', 'v_conv1d'):
            conv = getattr(block.attn, name)
            if not hasattr(conv, '_cgb_original_forward'):
                conv._cgb_original_forward = conv.forward
                conv.forward = types.MethodType(streaming_short_convolution, conv)
                count += 1
    return count


def load_pure_gdn(path: str | Path, *, device='cuda', dtype=torch.float16):
    """Load existing weights strictly; importing FLA is deliberately optional."""
    from fla.models.gated_deltanet import GatedDeltaNetConfig, GatedDeltaNetForCausalLM
    config = GatedDeltaNetConfig.from_pretrained(path, local_files_only=True)
    if config.attn is not None:
        raise ValueError('Only a pure, fixed-recurrent-state checkpoint is admitted')
    # device_map loads safetensors through a meta storage slice and faults in
    # UntypedStorage.__getitem__. Read the tiny F32 time/gain values first, close
    # that map, then load the module on CPU. A second map of the 1.6GB file
    # does not fit the current commit limit once the weights are resident.
    from safetensors import safe_open
    expected_legacy = {f'model.layers.{i}.attn.D' for i in range(config.num_hidden_layers)}
    weight_file = Path(path)/'model.safetensors'
    with safe_open(str(weight_file), framework='pt', device='cpu') as source:
        present = set(source.keys())
        if expected_legacy - present:
            raise ValueError('legacy D tensors missing from the published checkpoint')
        legacy = [source.get_slice(name)[:].contiguous().clone() for name in expected_legacy]
        if not all(torch.count_nonzero(value) == 0 for value in legacy):
            raise ValueError('Legacy D values are learned/nonzero; exact source implementation required')
        fp32_names = [name for name in present if name.endswith(('A_log', 'dt_bias'))]
        saved_fp32 = {name: source.get_slice(name)[:].contiguous().clone() for name in fp32_names}
    model, loading = GatedDeltaNetForCausalLM.from_pretrained(
        path, config=config, torch_dtype=dtype, low_cpu_mem_usage=False,
        local_files_only=True, output_loading_info=True)
    unexpected = set(loading.get('unexpected_keys', []))
    if unexpected == expected_legacy:
        loading['ignored_legacy_zero_tensors'] = sorted(unexpected)
        loading['unexpected_keys'] = []
    for name in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'):
        if loading.get(name):
            raise ValueError(f'Pretrained checkpoint mismatch: {name}={loading[name]}')
    model.to(device)
    model.eval()
    # Preserve tiny time/gain parameters directly from their F32 source, rather
    # than first rounding them to half and casting the rounded values back.
    with torch.no_grad():
        parameters = dict(model.named_parameters())
        missing = [name for name in saved_fp32 if name not in parameters]
        if missing:
            raise ValueError(f'F32 source tensors are not model parameters: {missing}')
        for name, value in saved_fp32.items():
            parameters[name].data = value.to(device=device, dtype=torch.float32)
    return model, loading


class RecallInlet(nn.Module):
    """A zero-initialized map from an existing latent recall into one body layer."""
    def __init__(self, width: int, latent_dim: int = 64, *, device=None):
        super().__init__()
        self.projection = nn.Linear(latent_dim, width, bias=False, device=device,
                                    dtype=torch.float32)
        nn.init.zeros_(self.projection.weight)

    @contextmanager
    def attach(self, model, recalled: torch.Tensor, *, layer=1):
        def inject(module, positional, keyword):
            if positional:
                hidden = positional[0]
                residual = self.projection(recalled.float()).to(hidden.dtype)
                return (hidden + residual, *positional[1:]), keyword
            keyword = dict(keyword)
            hidden = keyword['hidden_states']
            keyword['hidden_states'] = hidden + self.projection(recalled.float()).to(hidden.dtype)
            return positional, keyword

        handle = model.model.layers[layer].register_forward_pre_hook(inject, with_kwargs=True)
        try:
            yield
        finally:
            handle.remove()
