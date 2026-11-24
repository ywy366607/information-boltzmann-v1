"""Five-kernel execution of an original one-byte GDN layer, inference only.

The live layer owns every parameter. Packed projections are refreshed into
stable graph buffers at each draft boundary; they are never checkpoint state.
Sequence training and all derivatives continue through gdn_fused.forward.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _project(X, N, W, Bias, Y, D: tl.constexpr, O: tl.constexpr,
             EPS: tl.constexpr, R: tl.constexpr):
    row = tl.program_id(0) * R + tl.arange(0, R)
    col = tl.arange(0, D)
    x = tl.load(X + col)
    # Cast the normalized intermediate to FP32, as in nn.RMSNorm.
    x = (x * tl.rsqrt(tl.sum(x*x, 0) / D + EPS)) * tl.load(N + col)
    w = tl.load(W + row[:, None]*D + col[None, :], row[:, None] < O, 0.)
    y = tl.sum(w*x[None, :], 1) + tl.load(Bias + row, row < O, 0.)
    tl.store(Y + row, y, row < O)


@triton.jit
def _recur(P, CW, S0, S1, O, A, DT, D: tl.constexpr, H: tl.constexpr,
           K: tl.constexpr, C: tl.constexpr, BUDGET: tl.constexpr,
           LOG_DECAY: tl.constexpr):
    h = tl.program_id(0)
    i = tl.arange(0, K)
    index = h*K+i
    q = tl.load(P+index)
    k = tl.load(P+D+index)
    v = tl.load(P+2*D+index)
    if C > 0:
        # The source keeps RAW projection history, oldest first.
        qsum = tl.full((K,), 0., tl.float32)
        ksum = tl.full((K,), 0., tl.float32)
        vsum = tl.full((K,), 0., tl.float32)
        for t in tl.static_range(C-1):
            qsum += tl.load(S0+H*K*K+index*(C-1)+t)*tl.load(CW+index*C+t)
            ksum += tl.load(S0+H*K*K+(D+index)*(C-1)+t)*tl.load(CW+(D+index)*C+t)
            vsum += tl.load(S0+H*K*K+(2*D+index)*(C-1)+t)*tl.load(CW+(2*D+index)*C+t)
        qsum += q*tl.load(CW+index*C+C-1)
        ksum += k*tl.load(CW+(D+index)*C+C-1)
        vsum += v*tl.load(CW+(2*D+index)*C+C-1)
        for t in tl.static_range(C-2):
            for channel in tl.static_range(3):
                j = channel*D+index
                tl.store(S1+H*K*K+j*(C-1)+t, tl.load(S0+H*K*K+j*(C-1)+t+1))
        for channel in tl.static_range(3):
            j = channel*D+index
            tl.store(S1+H*K*K+j*(C-1)+C-2, tl.load(P+j))
        q = qsum*tl.sigmoid(qsum)
        k = ksum*tl.sigmoid(ksum)
        v = vsum*tl.sigmoid(vsum)
    q = q / tl.maximum(tl.sqrt(tl.sum(q*q, 0)), 1.e-12)
    k = k / tl.maximum(tl.sqrt(tl.sum(k*k, 0)), 1.e-12)
    ar = tl.load(P+4*D+h)
    if LOG_DECAY:
        z = ar+tl.load(DT+h)
        softplus = tl.where(z > 20., z, tl.log(1.+tl.exp(z)))
        alpha = tl.exp(-tl.exp(tl.load(A+h))*softplus)
    else:
        alpha = tl.sigmoid(ar)
    beta = tl.sigmoid(tl.load(P+4*D+H+h))
    grid = i[:, None]*K+i[None, :]
    s = tl.load(S0+h*K*K+grid)
    residual = v-alpha*tl.sum(s*k[:, None], 0)
    nxt = alpha*s+beta*k[:, None]*residual[None, :]
    if BUDGET > 0:
        mag = tl.abs(nxt)
        den = tl.maximum(tl.maximum(tl.sum(mag, 1)[:, None], tl.sum(mag, 0)[None, :]), BUDGET)
        nxt = nxt*(BUDGET/den)
    tl.store(S1+h*K*K+grid, nxt)
    value = tl.sum(nxt*q[:, None], 0)
    g = tl.load(P+3*D+index)
    tl.store(O+index, value*(g*tl.sigmoid(g)))


@triton.jit
def _residual(X, W, V, Y, D: tl.constexpr, I: tl.constexpr,
              PAD: tl.constexpr, R: tl.constexpr):
    row = tl.program_id(0)*R+tl.arange(0, R)
    col = tl.arange(0, PAD)
    v = tl.load(V+col, col < I, 0.)
    w = tl.load(W+row[:, None]*I+col[None, :], (row[:, None] < D)&(col[None, :] < I), 0.)
    out = tl.sum(w*v[None, :], 1)+tl.load(X+row, row < D, 0.)
    tl.store(Y+row, out, row < D)


@triton.jit
def _mlp(X, N, W, Y, D: tl.constexpr, M: tl.constexpr,
          EPS: tl.constexpr, R: tl.constexpr):
    row = tl.program_id(0)*R+tl.arange(0, R)
    col = tl.arange(0, D)
    x = tl.load(X+col)
    x = (x*tl.rsqrt(tl.sum(x*x, 0)/D+EPS))*tl.load(N+col)
    gate_w = tl.load(W+row[:, None]*D+col[None, :], row[:, None] < M, 0.)
    up_w = tl.load(W+(M+row[:, None])*D+col[None, :], row[:, None] < M, 0.)
    gate = tl.sum(gate_w*x[None, :], 1)
    up = tl.sum(up_w*x[None, :], 1)
    tl.store(Y+row, (gate*tl.sigmoid(gate))*up, row < M)


def supported(layer):
    """This kernel implements the actual decay/delta body, including admission."""
    return (layer.recurrence == 'decay_then_delta' and not layer.share_size and
            layer.d & (layer.d-1) == 0 and layer.dk & (layer.dk-1) == 0 and
            layer.conv_kernel != 1 and layer.norm.weight.is_cuda and
            layer.norm.weight.dtype == torch.float32)


@torch.no_grad()
def make_bundle(layer):
    if not supported(layer):
        return None
    names = ('q_proj', 'k_proj', 'v_proj', 'g_proj', 'alpha_proj', 'beta_proj')
    weights = [getattr(layer, n).weight for n in names]
    total = sum(w.shape[0] for w in weights)
    weight_buf = torch.empty((total, layer.d), device=weights[0].device, dtype=weights[0].dtype)
    bias_buf = torch.zeros(total, device=weights[0].device, dtype=weights[0].dtype)
    weight_views, bias_views = [], []
    offset = 0
    for n, w in zip(names, weights):
        end = offset + w.shape[0]
        weight_views.append((n, weight_buf[offset:end]))
        bias_views.append((n, bias_buf[offset:end]))
        offset = end
    bundle = dict(weight=weight_buf, bias=bias_buf, _weight_views=tuple(weight_views),
                  _bias_views=tuple(bias_views), _had_bias=set())
    if layer.mlp is not None:
        m = layer.mlp['gate'].out_features
        mlp_buf = torch.empty((2*m, layer.d), device=weights[0].device, dtype=weights[0].dtype)
        bundle['mlp'] = mlp_buf
        bundle['_mlp_views'] = (mlp_buf[:m], mlp_buf[m:])
    refresh(layer, bundle)
    return bundle


@torch.no_grad()
def refresh_many(pairs):
    """Copy all layer projection weights into bundled buffers via foreach copy."""
    dst, src = [], []
    for layer, bundle in pairs:
        had_bias = bundle.setdefault('_had_bias', set())
        weight_views = bundle.get('_weight_views')
        bias_views = bundle.get('_bias_views')
        if weight_views is not None and bias_views is not None:
            for (name, w_dst), (_, b_dst) in zip(weight_views, bias_views):
                projection = getattr(layer, name)
                dst.append(w_dst)
                src.append(projection.weight)
                if projection.bias is not None:
                    dst.append(b_dst)
                    src.append(projection.bias)
                    had_bias.add(name)
                elif name in had_bias:
                    b_dst.zero_()
                    had_bias.remove(name)
        else:
            offset = 0
            for name in ('q_proj', 'k_proj', 'v_proj', 'g_proj', 'alpha_proj', 'beta_proj'):
                projection = getattr(layer, name)
                end = offset + projection.out_features
                dst.append(bundle['weight'][offset:end])
                src.append(projection.weight)
                if projection.bias is not None:
                    dst.append(bundle['bias'][offset:end])
                    src.append(projection.bias)
                else:
                    bundle['bias'][offset:end].zero_()
                offset = end
        if layer.mlp is not None:
            mlp_views = bundle.get('_mlp_views')
            if mlp_views is not None:
                dst.append(mlp_views[0])
                src.append(layer.mlp['gate'].weight)
                dst.append(mlp_views[1])
                src.append(layer.mlp['up'].weight)
            else:
                m = layer.mlp['gate'].out_features
                dst.append(bundle['mlp'][:m])
                src.append(layer.mlp['gate'].weight)
                dst.append(bundle['mlp'][m:])
                src.append(layer.mlp['up'].weight)
    if dst:
        torch._foreach_copy_(dst, src)


@torch.no_grad()
def refresh(layer, bundle):
    refresh_many(((layer, bundle),))


def step(layer, x, state, bundle):
    """Original single-event map; FP32 reductions have their own rounding."""
    d, h, k = layer.d, layer.heads, layer.dk
    projected = x.new_empty(4*d+2*h)
    value = x.new_empty(d)
    nxt = torch.empty_like(state)
    after = torch.empty_like(x)
    eps = torch.finfo(x.dtype).eps if layer.norm.eps is None else layer.norm.eps
    _project[(triton.cdiv(len(projected), 8),)](
        x, layer.norm.weight, bundle['weight'], bundle['bias'], projected,
        d, len(projected), eps, 8, enable_fp_fusion=False)
    _recur[(h,)](projected, layer.qkv_conv.weight if layer.conv_kernel else projected,
        state, nxt, value, getattr(layer, 'A_log', projected), getattr(layer, 'dt_bias', projected),
        d, h, k, layer.conv_kernel, layer.content_budget, layer.decay_init == 'log_timescale',
        enable_fp_fusion=False)
    _residual[(triton.cdiv(d, 8),)](x, layer.o_proj.weight, value, after, d, d, d, 8,
                                    enable_fp_fusion=False)
    if layer.mlp is None:
        return after, nxt, {}
    m = layer.mlp['gate'].out_features
    hidden, result = x.new_empty(m), torch.empty_like(x)
    eps = torch.finfo(x.dtype).eps if layer.mlp_norm.eps is None else layer.mlp_norm.eps
    _mlp[(triton.cdiv(m, 8),)](after, layer.mlp_norm.weight, bundle['mlp'], hidden, d, m, eps, 8,
                               enable_fp_fusion=False)
    _residual[(triton.cdiv(d, 8),)](after, layer.mlp['down'].weight, hidden, result,
                                    d, m, triton.next_power_of_2(m), 8, enable_fp_fusion=False)
    return result, nxt, {}
