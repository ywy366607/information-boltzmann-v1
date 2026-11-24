"""One fused FP32 cell for the bounded, one-byte past/present exchange.

Only no-grad decoding uses this kernel. The differentiable causal chunk scan
and all parameter/eligibility gradients keep the existing training path.
There is no change to any gate, slot, null candidate, or shared operator.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _cell(H, M, OLD, KEYS, BIAS, VALID, ENABLED, HN, QW, VW, KW, OW,
          MN, GW, UW, DW, FW, BW, GATEW, GATEB, READW, READB, NULLW, NULLB,
          OUT, NEXT, D: tl.constexpr, R: tl.constexpr, K: tl.constexpr,
          N: tl.constexpr, MLPSIZE: tl.constexpr,
          H_EPS: tl.constexpr, M_EPS: tl.constexpr):
    d = tl.arange(0, D)
    r = tl.arange(0, R)
    k = tl.arange(0, K)
    n = tl.arange(0, N)
    hidden = tl.load(H+d)
    hn = hidden*tl.rsqrt(tl.sum(hidden*hidden, 0)/D+H_EPS)*tl.load(HN+d)
    q = tl.sum(tl.load(QW+r[:, None]*D+d[None, :])*hn[None, :], 1)
    write = libdevice.tanh(tl.sum(tl.load(VW+r[:, None]*D+d[None, :])*hn[None, :], 1))
    incoming = tl.load(M+k[:, None]*R+r[None, :])
    normed = incoming*tl.rsqrt(tl.sum(incoming*incoming, 1)/R+M_EPS)[:, None]*tl.load(MN+r)[None, :]
    shared_key = tl.sum(normed[:, None, :]*tl.load(
        KW+r[None, :, None]*D+r[None, None, :]), 2)
    norm_key = shared_key/tl.maximum(tl.sqrt(tl.sum(shared_key*shared_key, 1)), 1e-12)[:, None]
    norm_q = q/tl.maximum(tl.sqrt(tl.sum(q*q, 0)), 1e-12)
    valid = tl.load(VALID+k)
    enabled = tl.load(ENABLED)
    score = tl.sum(norm_q[None, :]*(tl.load(KEYS+k[:, None]*R+r[None, :])+norm_key), 1)/tl.sqrt(float(R))
    score = tl.where(valid, score+tl.load(BIAS+k), -1e4)
    empty = tl.sum(q*tl.load(NULLW+r), 0)+tl.load(NULLB)
    maximum = tl.maximum(tl.max(score, 0), empty)
    mass = tl.exp(score-maximum)
    weight = mass/(tl.sum(mass, 0)+tl.exp(empty-maximum))*valid*enabled
    gate = tl.sum(normed[:, None, :]*tl.load(GW+n[None, :, None]*D+r[None, None, :]), 2)
    up = tl.sum(normed[:, None, :]*tl.load(UW+n[None, :, None]*D+r[None, None, :]), 2)
    shared_mid = gate*tl.sigmoid(gate)*up
    shared = tl.sum(shared_mid[:, None, :]*tl.load(DW+r[None, :, None]*MLPSIZE+n[None, None, :]), 2)
    private = tl.sum(normed[:, None, :]*tl.load(FW+n[None, :, None]*R+r[None, None, :]), 2)
    private = private*tl.sigmoid(private)
    private = tl.sum(private[:, None, :]*tl.load(BW+r[None, :, None]*N+n[None, None, :]), 2)
    proposal = libdevice.tanh(incoming+shared+private+weight[:, None]*write[None, :])
    gate_q = tl.sum(q*tl.load(GATEW+r), 0)+tl.load(GATEB)
    beta = tl.sigmoid(gate_q+tl.sum(normed*tl.load(GATEW+R+r)[None, :], 1))*valid*enabled
    previous = tl.load(OLD+k[:, None]*R+r[None, :])
    workspace = (1-beta[:, None])*previous+beta[:, None]*proposal
    tl.store(NEXT+k[:, None]*R+r[None, :], workspace)
    recalled = tl.sum(weight[:, None]*workspace, 0)
    read = tl.sigmoid(tl.sum(q*tl.load(READW+r), 0)+tl.load(READB))
    delta = tl.sum(tl.load(OW+d[:, None]*D+r[None, :])*recalled[None, :], 1)*read
    tl.store(OUT+d, hidden+delta)


def exchange(field, layer, hidden, incoming, previous, keys, bias):
    r, k, d = field.rank, field.slots, hidden.shape[-1]
    if (hidden.shape[:2] != (1, 1) or hidden.dtype != torch.float32 or
            r not in (8, 16, 32) or k not in (1, 2, 4, 8) or d not in (32, 64, 128, 256) or
            layer.mlp is None or layer.mlp['gate'].weight.shape[0] < 2*r):
        return None
    output, work = torch.empty_like(hidden), torch.empty_like(incoming)
    _cell[(1,)](hidden, incoming, previous, keys, bias, field.valid, field.enabled,
        layer.norm.weight, layer.q_proj.weight, layer.v_proj.weight, layer.k_proj.weight,
        layer.o_proj.weight, field.memory_norm.weight,
        layer.mlp['gate'].weight, layer.mlp['up'].weight, layer.mlp['down'].weight,
        field.memory_ffn[0].weight, field.memory_ffn[2].weight,
        field.memory_gate.weight, field.memory_gate.bias, field.read_gate.weight,
        field.read_gate.bias, field.null.weight, field.null.bias, output, work,
        d, r, k, 2*r, layer.mlp['gate'].weight.shape[0],
        torch.finfo(hidden.dtype).eps if layer.norm.eps is None else layer.norm.eps,
        torch.finfo(hidden.dtype).eps if field.memory_norm.eps is None else field.memory_norm.eps,
        num_warps=8, enable_fp_fusion=False)
    return output, work, {}
