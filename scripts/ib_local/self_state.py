"""Self-state memory: a second S per layer that never decays and is written through its own gate.

User's design, 2026-11-03: what has to be held (where am I, what am I doing, what I decided)
goes into another S matrix with no decay; erasing and writing follow the ordinary delta rule.
No attractor is needed to hold it, because nothing leaks in silicon; what the latch lacked
(results/cgb_mode_task_20261103/RESULT.md, E2) was protection from being overwritten by the
text, and here that is the write gate beta_s, born nearly closed.

Per layer, alongside the unchanged GDN layer and reading the same input x:
    q, k, v = own projections of RMSNorm(x), heads_s heads of width dk; q, k unit
    beta_s = sigmoid(w . x + b), b = -5 at birth (writes about 0.7% per event)
    S_s' = S_s + beta_s k (v - S_s^T k)^T          (delta rule, retention exactly 1)
    out += o_s(S_s'^T q)
What is written, where, and when the gate opens are learned; nothing names the mode marker.
Oracle variant (user, same day): the gate is not learned but given: beta_s = 1 at the mode
marker's letter byte (the byte after "<mode:"), 0 everywhere else, so only the task-relevant
event writes; what is written and how it is read are still learned.  It separates "is a held
bit used when the right thing is written" from "can the gate be learned".
Latch variant (user, 2026-11-04): the self S also gets attractor correction, the
feedback_delta renewal with its own eta (born 0.8) and gain (born 8). With retention 1 the
latch strength is kappa = eta * g = 6.4 whatever beta is, and a write that leaks through the
gate is pulled back while it stays below the tolerance of `latch_tolerance`
(lean GenerationHoldCorrect); a strong cue still flips it. The self S does not model text, so
a high eta costs no text ability.
The state is packed after the layer's own state, so checkpoints and the graph trainer keep
working with a longer per-layer state.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.ib_local import gdn_fused


class SelfStateLayer(nn.Module):
    def __init__(self, inner: nn.Module, heads: int = 1, dk: int = 32, gate_bias: float = -5.0,
                 oracle: bool = False, latch: bool = False):
        super().__init__()
        self.inner, self.self_heads, self.self_dk, self.oracle = inner, heads, dk, oracle
        self.latch = latch
        if latch:
            self.self_eta_logit = nn.Parameter(torch.full((heads,), math.log(0.8 / 0.2)))
            self.self_gain_raw = nn.Parameter(torch.full((heads,), math.log(math.expm1(8.0))))
        self.write_mask = None  # [B, T] set per call by the caller in oracle mode
        d = inner.d
        self.self_norm = nn.RMSNorm(d)
        self.self_qkv = nn.Linear(d, 3 * heads * dk, bias=False)
        self.self_beta = nn.Linear(d, heads, bias=True)
        nn.init.constant_(self.self_beta.bias, gate_bias)
        self.self_o = nn.Linear(heads * dk, d, bias=False)
        nn.init.normal_(self.self_o.weight, std=0.02 / math.sqrt(2 * 4))
        self.self_size = heads * dk * dk
        self.state_size = inner.state_size + self.self_size
        for name in ("d", "heads", "dk", "dv", "matrix_size", "recurrence", "conv_kernel"):
            setattr(self, name, getattr(inner, name))

    def forward(self, x, state):
        B, T, _ = x.shape
        inner_state = state[:, :self.inner.state_size]
        S = state[:, self.inner.state_size:].reshape(B, self.self_heads, self.self_dk, self.self_dk)
        result, inner_next, diagnostics = self.inner(x, inner_state)
        q, k, v = self.self_qkv(self.self_norm(x)).view(B, T, 3, self.self_heads, self.self_dk).unbind(2)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        if self.oracle:
            beta = self.write_mask[..., None].to(x.dtype).expand(B, T, self.self_heads)
        else:
            beta = torch.sigmoid(self.self_beta(x))                   # [B, T, H]
        ones = torch.ones_like(beta)
        if self.latch:
            eta, gain = self.self_eta_logit.sigmoid(), F.softplus(self.self_gain_raw)
        else:
            eta = gain = torch.zeros(self.self_heads, device=x.device, dtype=x.dtype)
        read, S, _, _ = gdn_fused.FusedDelta.apply(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            ones.transpose(1, 2), beta.transpose(1, 2), eta, gain, S, 0.0, self.latch)
        result = result + self.self_o(read.transpose(1, 2).reshape(B, T, -1))
        diagnostics["self_beta_mean"] = beta.detach().mean()
        diagnostics["self_beta_max"] = beta.detach().max()
        return result, torch.cat([inner_next, S.reshape(B, self.self_size)], dim=-1), diagnostics


def add_self_state(base: nn.Module, heads: int = 1, dk: int = 32, gate_bias: float = -5.0,
                   oracle: bool = False, latch: bool = False):
    """Wrap every layer of a GatedDeltaNetLM in place (conv_kernel > 0, packed state)."""
    if not base.conv_kernel:
        raise ValueError("self-state packing needs the packed (conv_kernel > 0) state layout")
    base.layers = nn.ModuleList(SelfStateLayer(layer, heads, dk, gate_bias, oracle, latch)
                                for layer in base.layers)
    base.self_oracle = oracle
    return base


MARKER_PREFIX = b"<mode:"


def marker_letter_mask(ids: torch.Tensor) -> torch.Tensor:
    """True at bytes preceded by "<mode:" inside this window (a marker split across a
    window boundary is missed; at most one in 512 bytes, rarely)."""
    mask = torch.ones_like(ids, dtype=torch.bool)
    n = len(MARKER_PREFIX)
    for back, byte in enumerate(reversed(MARKER_PREFIX), start=1):
        shifted = torch.roll(ids, back, dims=-1)
        mask &= shifted == byte
    mask[..., :n] = False
    return mask
