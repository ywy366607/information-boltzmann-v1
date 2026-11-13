from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from scripts.ib_local.frozen_mlp_int8 import FrozenInt8Linear, pack_frozen_mlps, unpack, stream_frozen_mlp_parameters


def test_mapped_coefficients_preserve_values_input_credit_and_cache_bytes(tmp_path):
    from scripts.ib_local.frozen_mlp_int8 import map_frozen_linear
    torch.manual_seed(17)
    packed=FrozenInt8Linear(nn.Linear(13,17,bias=False).requires_grad_(False))
    q,s=packed.qweight.clone(),packed.scale.clone()
    mapped=map_frozen_linear(packed,tmp_path)
    before={p.name:p.read_bytes() for p in tmp_path.iterdir()}
    x=torch.randn(3,13,requires_grad=True)
    reference=x.detach().clone().requires_grad_(True)
    expected=F.linear(reference,unpack(q,s,x.dtype))
    actual=mapped(x)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    actual.square().sum().backward();expected.square().sum().backward()
    torch.testing.assert_close(x.grad,reference.grad,rtol=0,atol=0)
    assert torch.equal(q,mapped.qweight) and torch.equal(s,mapped.scale)
    assert before=={p.name:p.read_bytes() for p in tmp_path.iterdir()}
    map_frozen_linear(mapped,tmp_path)  # Reuse verifies existing contents.
    corrupt=tmp_path/'corrupt';corrupt.mkdir()
    for name,data in before.items():(corrupt/name).write_bytes(data)
    path=next(corrupt.iterdir());path.write_bytes(b'damaged cache')
    fresh=FrozenInt8Linear(nn.Linear(13,17,bias=False).requires_grad_(False))
    fresh.qweight=q;fresh.scale=s
    with pytest.raises(ValueError,match='cache differs'):map_frozen_linear(fresh,corrupt)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('bias', [False, True])
def test_input_credit_matches_same_unpacked_frozen_matrix(dtype, bias):
    torch.manual_seed(11)
    original=nn.Linear(13,17,bias=bias).requires_grad_(False)
    packed=FrozenInt8Linear(original)
    x=torch.randn(2,3,13,dtype=dtype,requires_grad=True)
    other=x.detach().clone().requires_grad_(True)
    expected=F.linear(other,unpack(packed.qweight,packed.scale,dtype),
                      None if packed.bias is None else packed.bias.to(dtype))
    if packed.bias is not None:packed.bias=packed.bias.to(dtype)
    actual=packed(x)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    upstream=torch.randn_like(actual)
    actual.backward(upstream);expected.backward(upstream)
    torch.testing.assert_close(x.grad,other.grad,rtol=1e-6,atol=1e-7)
    assert list(packed.parameters())==[]


def test_zero_rows_remain_zero_and_no_dense_weight_is_saved_for_backward():
    linear=nn.Linear(128,256,bias=False).requires_grad_(False)
    with torch.no_grad():linear.weight[0].zero_()
    packed=FrozenInt8Linear(linear)
    x=torch.randn(2,128,requires_grad=True)
    saved=[]
    def hold(tensor):
        saved.append((tensor.dtype,tuple(tensor.shape)))
        return tensor
    with torch.autograd.graph.saved_tensors_hooks(hold,lambda x:x):
        output=packed(x)
        assert torch.count_nonzero(output[:,0])==0
        output.sum().backward()
    assert saved==[(torch.int8,(256,128)),(torch.float32,(256,1))]
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert sum(t.numel()*t.element_size() for t in packed.buffers()) < linear.weight.numel()*2


def test_trainable_weights_are_rejected_instead_of_silently_frozen():
    with pytest.raises(ValueError,match='already frozen'):
        FrozenInt8Linear(nn.Linear(2,2))


def test_packing_does_not_touch_native_gdn_or_block_cross_layer_credit():
    class MLP(nn.Module):
        def __init__(self):
            super().__init__();self.fuse_swiglu=True
            self.gate_proj=nn.Linear(8,16,bias=False).requires_grad_(False)
            self.up_proj=nn.Linear(8,16,bias=False).requires_grad_(False)
            self.down_proj=nn.Linear(16,8,bias=False).requires_grad_(False)
        def forward(self,x):return self.down_proj(F.silu(self.gate_proj(x))*self.up_proj(x))
    block=nn.Module();block.attn=nn.Linear(8,8);block.mlp=MLP()
    model=SimpleNamespace(model=SimpleNamespace(layers=[block]))
    before=block.attn.weight.detach().clone()
    info=pack_frozen_mlps(model)
    x=torch.randn(3,8)
    block.mlp(block.attn(x)).square().sum().backward()
    assert block.attn.weight.grad is not None and block.attn.weight.grad.norm()>0
    assert torch.equal(before,block.attn.weight)
    assert info['saved_weight_bytes']>0 and not block.mlp.fuse_swiglu


def test_direct_stream_load_keeps_native_coefficients_and_packed_input_gradient(tmp_path):
    import json,struct
    class Model(nn.Module):
        def __init__(self):
            super().__init__();self.attn=nn.Linear(16,16,bias=False);self.mlp=nn.Module()
            for name,a,b in [('gate_proj',16,32),('up_proj',16,32),('down_proj',32,16)]:
                setattr(self.mlp,name,nn.Linear(a,b,bias=False))
    original=Model()
    header,blobs,cursor={},[],0
    for name,p in original.named_parameters():
        value=p.detach().float().numpy().tobytes()
        header[name]=dict(dtype='F32',shape=list(p.shape),data_offsets=[cursor,cursor+len(value)])
        blobs.append(value);cursor+=len(value)
    data=json.dumps(header).encode();path=tmp_path/'weights.safetensors'
    path.write_bytes(struct.pack('<Q',len(data))+data+b''.join(blobs))
    with torch.device('meta'):target=Model()
    report=stream_frozen_mlp_parameters(target,path,device='cpu')
    assert torch.equal(original.attn.weight.half(),target.attn.weight)
    assert report['frozen_mlp']['saved_weight_bytes']>0
    assert not target.mlp.fuse_swiglu and not any(p.is_meta for p in target.parameters())
    assert isinstance(target.mlp.gate_proj,FrozenInt8Linear)
    x=torch.randn(2,16,dtype=torch.float16,requires_grad=True)
    target.mlp.down_proj(F.silu(target.mlp.gate_proj(x))*target.mlp.up_proj(x)).float().sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_fixed_gdn_packing_preserves_both_native_ports_and_all_gate_coefficients(tmp_path):
    import json,struct
    class Model(nn.Module):
        def __init__(self):
            super().__init__();self.layers=nn.ModuleList([nn.Module(),nn.Module()])
            for layer in self.layers:
                layer.attn=nn.Module()
                for name in ('q_proj','k_proj','v_proj','g_proj','o_proj','a_proj','b_proj'):
                    setattr(layer.attn,name,nn.Linear(16,16,bias=False))
    original=Model();header={};blobs=[];cursor=0
    for name,p in original.named_parameters():
        value=p.detach().float().numpy().tobytes()
        header[name]=dict(dtype='F32',shape=list(p.shape),data_offsets=[cursor,cursor+len(value)])
        blobs.append(value);cursor+=len(value)
    data=json.dumps(header).encode();path=tmp_path/'weights.safetensors'
    path.write_bytes(struct.pack('<Q',len(data))+data+b''.join(blobs))
    with torch.device('meta'):target=Model()
    report=stream_frozen_mlp_parameters(target,path,device='cpu',pack_fixed_gdn=True)
    assert isinstance(target.layers[0].attn.q_proj,FrozenInt8Linear)
    assert isinstance(target.layers[1].attn.q_proj,nn.Linear)
    for i in (0,1):
        assert torch.equal(original.layers[i].attn.a_proj.weight.half(),target.layers[i].attn.a_proj.weight)
        assert torch.equal(original.layers[i].attn.b_proj.weight.half(),target.layers[i].attn.b_proj.weight)
    assert torch.equal(original.layers[1].attn.q_proj.weight.half(),target.layers[1].attn.q_proj.weight)
    assert report['frozen_mlp']['native_gdn_parameters_changed'] is False
