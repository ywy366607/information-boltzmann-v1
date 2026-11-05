"""Latch slot: input-independent renewal for the latching heads (2026-11-04).

Lean: `lean/CGBAxioms/CGBAxioms/GenerationLatchSlot.lean`. In `feedback_delta` the renewal of a
head and the text it admits share the event's write gate beta and key k: a head that closes its
gate to text also switches its renewal off (`coupled_small_gate_forgets`), and a head that latches
admits text above a floor (`coupled_disturbance_floor`). This is what E2 measured: the trained
latch holds for 32k events only if the text written into its direction is scaled to 0.1-0.3.
In cortex the persistent activity renews itself whatever the input, and the input is gated.

The slot gives each latching head (threshold_spread birth ratio > 1) a learned direction u and a
renewal rate rho of its own. After every ordinary event the head runs one renewal event
(key u, value 0, retention 1, gate rho), with the head's own eta and gain:
    r = S^T u,   S += rho u (eta softsign(g r) - r)^T
so along u the content is renewed by fb(1 - rho, rho eta, g), which latches iff eta g > 1
(times the event's decay alpha: `slot_supercritical_iff`) whatever the write gate does.
The interleaved event goes through the same fused kernel (`gdn_fused.FusedDelta`), so the
gradients for u, rho, eta and g are exact; outputs are read after the renewal.
Nothing names the long state: what is written along u, and when the gate opens, are learned.
Other heads are unchanged (their renewal gate is 0, an exact no-op).
"""

from __future__ import annotations

import math
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.ib_local import gdn_fused
from scripts.ib_local.soc import birth_ratios


def slot_forward(self, x, state):
    """`gdn_fused.fused_forward` with one renewal event after each input event."""
    B, T, d = x.shape
    x_norm = self.norm(x)
    q = self.q_proj(x_norm).view(B, T, self.heads, self.dk)
    k = self.k_proj(x_norm).view(B, T, self.heads, self.dk)
    v = self.v_proj(x_norm).view(B, T, self.heads, self.dv)
    S = state[:, :self.matrix_size].reshape(B, self.heads, self.dk, self.dv)
    history = state[:, self.matrix_size:].reshape(B, 3 * d, self.conv_kernel - 1)
    projected = torch.cat((q.reshape(B, T, d), k.reshape(B, T, d), v.reshape(B, T, d)),
                          dim=-1).transpose(1, 2)
    mixed, next_history = self._mix_qkv(projected, history)
    q, k, v = [part.reshape(B, T, self.heads, self.dk)
               for part in mixed.transpose(1, 2).split(d, dim=-1)]
    g = self.g_proj(x_norm)
    alpha = self.retention(x_norm)
    beta = torch.sigmoid(self.beta_proj(x_norm))
    q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
    strength = self.feedback_logit.sigmoid()
    feedback_gain = F.softplus(self.feedback_gain_raw)
    u = F.normalize(self.slot_key, dim=-1)[None, None].expand(B, T, self.heads, self.dk)
    rho = (self.slot_rate_logit.sigmoid() * self.slot_mask)[None, None].expand(B, T, self.heads)

    def weave(event, renewal):  # [B, T, ...] x2 -> [B, 2T, ...], input event first
        return torch.stack((event, renewal), dim=2).flatten(1, 2)

    out, S, esq, corr = gdn_fused.FusedDelta.apply(
        weave(q, q).transpose(1, 2), weave(k, u).transpose(1, 2),
        weave(v, torch.zeros_like(v)).transpose(1, 2),
        weave(alpha, torch.ones_like(alpha)).transpose(1, 2), weave(beta, rho).transpose(1, 2),
        strength, feedback_gain, S, self.content_budget, True)
    out = out[:, :, 1::2].transpose(1, 2).reshape(B, T, d)
    result = x + self.o_proj(out * F.silu(g))
    if self.mlp is not None:
        normalized = self.mlp_norm(result)
        result = result + self.mlp["down"](
            F.silu(self.mlp["gate"](normalized)) * self.mlp["up"](normalized))
    diagnostics = {
        "alpha_mean": alpha.detach().mean(),
        "beta_mean": beta.detach().mean(),
        "error_norm": (esq[..., 0::2].mean() / self.dv).sqrt(),
        "state_norm": S.detach().square().mean().sqrt(),
        "feedback_strength_mean": strength.detach().mean(),
        "feedback_gain_mean": feedback_gain.detach().mean(),
        "slot_rate_mean": rho[0, 0].detach().sum() / self.slot_mask.sum().clamp_min(1),
    }
    if self.content_budget:
        diagnostics.update(
            budget_correction_rms=(corr.sum() / (B * self.matrix_size)).sqrt(),
            content_mass_max=torch.maximum(S.detach().abs().sum(-1).max(),
                                           S.detach().abs().sum(-2).max()))
    S = torch.cat([S.reshape(B, self.matrix_size), next_history.reshape(B, self.history_size)], dim=-1)
    return result, S, diagnostics


def add_latch_slots(base: nn.Module, rate_init: float = 0.5):
    """Give every latching head of every feedback_delta layer a slot, in place.

    Deterministic initialization (no RNG draw): u = first basis vector, rho = rate_init, so the
    shared weights match a run without slots."""
    for layer in base.layers:
        if layer.recurrence != "feedback_delta" or not layer.conv_kernel or layer.share_size:
            raise ValueError("slots need feedback_delta with the packed conv state")
        device = layer.feedback_logit.device
        mask = torch.tensor([float(r > 1) for r in birth_ratios(layer.heads)], device=device)
        layer.register_buffer("slot_mask", mask)
        key = torch.zeros(layer.heads, layer.dk, device=device)
        key[:, 0] = 1.0
        layer.slot_key = nn.Parameter(key)
        layer.slot_rate_logit = nn.Parameter(
            torch.full((layer.heads,), math.log(rate_init / (1 - rate_init)), device=device))
        layer.forward = types.MethodType(slot_forward, layer)
    base.latch_slots = True
    return base


@torch.no_grad()
def slot_reads(base: nn.Module, state: torch.Tensor) -> torch.Tensor:
    """|S^T u| per layer and latching head, [L, H] (0 for heads without a slot)."""
    out = torch.zeros(base.num_layers, base.heads, device=state.device)
    for index, layer in enumerate(base.layers):
        B = state.shape[1]
        S = state[index, :, :layer.matrix_size].reshape(B, layer.heads, layer.dk, layer.dv)
        u = F.normalize(layer.slot_key, dim=-1)
        out[index] = torch.einsum("bhkv,hk->bhv", S, u).abs().mean(dim=(0, 2)) * layer.slot_mask
    return out
