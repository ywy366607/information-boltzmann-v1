"""Forward-AD compatible native GDN recurrence for numerical admission only."""
import torch
import torch.nn.functional as F
from types import MethodType


def recurrent_reference(q, k, v, g, beta, *, initial_state=None,
                        output_final_state=False, use_qk_l2norm_in_kernel=False,
                        scale=None, cu_seqlens=None, head_first=False, **kwargs):
    if cu_seqlens is not None or head_first:
        raise ValueError('reference admits one ordinary token-major stream')
    if use_qk_l2norm_in_kernel:
        q = q.float() * (q.float().square().sum(-1,keepdim=True)+1e-6).rsqrt()
        k = k.float() * (k.float().square().sum(-1,keepdim=True)+1e-6).rsqrt()
    q, k, value = q.float(), k.float(), v.float()
    scale = q.shape[-1]**-.5 if scale is None else scale
    state = (q.new_zeros(q.shape[0],q.shape[2],q.shape[3],value.shape[-1])
             if initial_state is None else initial_state.float())
    outputs = []
    for t in range(q.shape[1]):
        state = state * g[:,t].float().exp()[...,None,None]
        residual = value[:,t] - (state*k[:,t,...,None]).sum(-2)
        state = state + k[:,t,...,None] * (beta[:,t].float()[...,None]*residual)[...,None,:]
        outputs.append((state*q[:,t,...,None]).sum(-2)*scale)
    return torch.stack(outputs,dim=1).to(v.dtype), state if output_final_state else None


def install_reference(model):
    """Replace unsupported fused calls in this process for an audit only."""
    import fla.layers.gated_deltanet as layer
    from scripts.ib_local.frozen_mlp_int8 import FrozenInt8Linear, unpack
    layer.chunk_gated_delta_rule = recurrent_reference
    layer.fused_recurrent_gated_delta_rule = recurrent_reference
    patched = {}
    def norm(self,x,residual=None,prenorm=False,residual_in_fp32=False):
        combined = x if residual is None else (x+residual)
        y = combined.float() * (combined.float().square().mean(-1,keepdim=True)+self.eps).rsqrt()
        if self.weight is not None:
            y = y*self.weight.float()
        if self.bias is not None:
            y = y+self.bias.float()
        y = y.to(x.dtype)
        return (y, combined.float() if residual_in_fp32 else combined) if prenorm else y
    def gated(self,x,g,residual=None,prenorm=False,residual_in_fp32=False):
        if residual is not None or prenorm:
            raise ValueError('unsupported gated normalization configuration')
        y=x.float()*(x.float().square().mean(-1,keepdim=True)+self.eps).rsqrt()
        if self.weight is not None:
            y=y*self.weight.float()
        gate = torch.sigmoid(g.float()) if self.activation=='sigmoid' else F.silu(g.float())
        return (y*gate).to(x.dtype)
    def linear(self,x):
        return F.linear(x,unpack(self.qweight,self.scale,x.dtype,x.device),self.bias)
    def mlp(self,x,**kwargs):
        return self.down_proj(F.silu(self.gate_proj(x))*self.up_proj(x))
    for module in model.modules():
        cls=type(module).__name__
        replacement = (gated if cls=='FusedRMSNormGated' else
                       norm if cls=='RMSNorm' and type(module).__module__.startswith('fla.') else
                       linear if isinstance(module,FrozenInt8Linear) else
                       mlp if cls=='GatedMLP' and type(module).__module__.startswith('fla.') else None)
        if replacement is not None:
            module.forward=MethodType(replacement,module)
            patched[cls]=patched.get(cls,0)+1
    return patched
