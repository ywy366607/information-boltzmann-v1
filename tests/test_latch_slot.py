"""Latch slot (scripts/ib_local/latch_slot.py) against an event-by-event reference."""

import math

import pytest
import torch
import torch.nn.functional as F

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="fused kernel needs CUDA")


def _model():
    from scripts.ib_local.gated_deltanet import GatedDeltaNetLM
    torch.manual_seed(0)
    return GatedDeltaNetLM(256, 128, 2, 4, "feedback_delta", 256, 4, 32.0,
                           decay_init="log_timescale", feedback_init="threshold_spread").cuda()


def _reference(layer, x, state):
    """The slot rule written out with the kernel's arithmetic, one event at a time."""
    B, T, d = x.shape
    x_norm = layer.norm(x)
    q, k, v = layer.q_proj(x_norm), layer.k_proj(x_norm), layer.v_proj(x_norm)
    history = state[:, layer.matrix_size:].reshape(B, 3 * d, layer.conv_kernel - 1)
    mixed, _ = layer._mix_qkv(torch.cat((q, k, v), -1).transpose(1, 2), history)
    q, k, v = [p.reshape(B, T, layer.heads, layer.dk) for p in mixed.transpose(1, 2).split(d, -1)]
    q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
    alpha = layer.retention(x_norm)
    beta = torch.sigmoid(layer.beta_proj(x_norm))
    eta = layer.feedback_logit.sigmoid()[None, :, None]
    gain = F.softplus(layer.feedback_gain_raw)[None, :, None]
    u = F.normalize(layer.slot_key, dim=-1)[None]
    rho = (layer.slot_rate_logit.sigmoid() * layer.slot_mask)[None].expand(B, -1)
    S = state[:, :layer.matrix_size].reshape(B, layer.heads, layer.dk, layer.dv)
    b = layer.content_budget

    def event(S, kt, vt, at, bt):  # at, bt: [B, H]
        r = torch.einsum("bhkv,bhk->bhv", S, kt)
        z = gain * r + vt
        e = vt + eta * (z / (1 + z.abs()) - vt) - at[..., None] * r
        P = at[..., None, None] * S + bt[..., None, None] * kt[..., None] * e[..., None, :]
        den = torch.maximum(torch.maximum(P.abs().sum(-1, keepdim=True), P.abs().sum(-2, keepdim=True)),
                            torch.tensor(b, device=P.device))
        return P * (b / den)

    outs = []
    for t in range(T):
        S = event(S, k[:, t], v[:, t], alpha[:, t], beta[:, t])
        S = event(S, u.expand(B, -1, -1), torch.zeros_like(v[:, t]), torch.ones_like(rho), rho)
        outs.append(torch.einsum("bhkv,bhk->bhv", S, q[:, t]))
    return torch.stack(outs, 1).reshape(B, T, d), S


@cuda
def test_slot_off_is_exact_noop():
    from scripts.ib_local import gdn_fused
    from scripts.ib_local.latch_slot import add_latch_slots
    gdn_fused.enable()
    base = _model()
    x = torch.randn(1, 64, 128, device="cuda")
    state = base.initial_state(1, device="cuda")[0] + 0.01 * torch.randn(1, base.layers[0].state_size, device="cuda")
    layer = base.layers[0]
    plain, plain_state, _ = layer(x, state)
    add_latch_slots(base)
    with torch.no_grad():
        layer.slot_mask.zero_()
    slotted, slot_state, _ = layer(x, state)
    assert torch.allclose(plain, slotted, atol=1e-6)
    assert torch.allclose(plain_state, slot_state, atol=1e-6)


@cuda
def test_slot_matches_reference_values_and_gradients():
    from scripts.ib_local import gdn_fused
    from scripts.ib_local.latch_slot import add_latch_slots
    gdn_fused.enable()
    base = add_latch_slots(_model())
    layer = base.layers[0]
    with torch.no_grad():
        layer.slot_key.add_(0.3 * torch.randn_like(layer.slot_key))
    x = torch.randn(1, 48, 128, device="cuda")
    state = base.initial_state(1, device="cuda")[0] + 0.05 * torch.randn(1, layer.state_size, device="cuda")
    result, next_state, _ = layer(x, state)
    out_fused = result[0]
    ref_out, ref_S = _reference(layer, x, state)
    fused_S = next_state[:, :layer.matrix_size].reshape_as(ref_S)
    assert torch.allclose(fused_S, ref_S, atol=2e-5)
    weights = torch.randn_like(out_fused)
    params = [layer.slot_key, layer.slot_rate_logit, layer.feedback_logit, layer.feedback_gain_raw]
    fused_grads = torch.autograd.grad((out_fused * weights).sum() + fused_S.square().sum(), params)
    ref_result = x + layer.o_proj(ref_out * F.silu(layer.g_proj(layer.norm(x))))
    if layer.mlp is not None:
        n = layer.mlp_norm(ref_result)
        ref_result = ref_result + layer.mlp["down"](F.silu(layer.mlp["gate"](n)) * layer.mlp["up"](n))
    assert torch.allclose(ref_result[0], out_fused, atol=2e-5)
    ref_grads = torch.autograd.grad((ref_result[0] * weights).sum() + ref_S.square().sum(), params)
    for f, r in zip(fused_grads, ref_grads):
        assert torch.allclose(f, r, atol=1e-4, rtol=1e-3), (f, r)
    assert fused_grads[0][layer.slot_mask == 0].abs().max() == 0  # no slot, no gradient
    assert fused_grads[1][layer.slot_mask > 0].abs().max() > 0


@cuda
def test_slot_latches_without_input():
    """With the write gate closed (beta = 0 everywhere) a latching head keeps its slot content."""
    from scripts.ib_local import gdn_fused
    from scripts.ib_local.latch_slot import add_latch_slots
    gdn_fused.enable()
    base = add_latch_slots(_model())
    layer = base.layers[0]
    with torch.no_grad():
        layer.beta_proj.weight.zero_()
        layer.beta_proj.bias.fill_(-30.0)
    x = torch.randn(1, 512, 128, device="cuda")
    state = base.initial_state(1, device="cuda")[0].clone()
    S = state[:, :layer.matrix_size].view(1, layer.heads, layer.dk, layer.dv)
    S[:, :, 0, :] = 0.5  # content along u = e0 in every head
    with torch.no_grad():
        _, after, _ = layer(x, state)
    read = after[:, :layer.matrix_size].view(1, layer.heads, layer.dk, layer.dv)[:, :, 0, :].abs().mean(-1)[0]
    eta = layer.feedback_logit.sigmoid()
    gain = F.softplus(layer.feedback_gain_raw)
    for h in range(layer.heads):
        if layer.slot_mask[h] > 0 and eta[h] * gain[h] > 1.5:
            assert read[h] > 0.1, (h, read)
