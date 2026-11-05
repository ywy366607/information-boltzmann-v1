"""Self-organized criticality for feedback_delta heads, per role (2026-11-04).

User's goal: no single mechanism should take everything; the parameters that decide
forgetting, holding and being overwritten should sit at the edges where a rich repertoire
exists, and get there by themselves. The user's correction: tune the full set, not only the
decay side: the text write into the held direction (lambda = 1 - eta, the E2 knob) is a
control parameter too.

Theory (lean GenerationHoldCorrect). Along the key it reads, one renewal is, per value
component, r' = a r + beta (1 - eta) v + beta eta softsign(g r + v), a = alpha (1 - beta). Two
dimensionless balances decide the regime:
* timescale edge: the loop gain G = a + beta eta g softsign'(g r + v); G < 1 forgets, G = 1
  is the transition with diverging timescales (`rates_vanish_at_transition`,
  `long_timescale_band`);
* input edge: the text disturbance per event D = beta (1 - eta) |v| against the latch's
  tolerance T = (sqrt(beta eta) - sqrt((1 - a)/g))^2 (`latch_tolerance`, kappa > 1).
  u = D / T > 1: text overwrites the held state; u < 1 by far: the latch is deaf (a trap,
  `trap_insensitive`); u = 1: ordinary text cannot move it, a strong cue can.

Pushing every head to one edge would remove the fast heads, so each head keeps the role its
birth gave it (threshold_spread birth ratios 0, 0.5, 1.22, 3 for heads 0-3) and only the
regulated quantity moves to that role's edge; what each head stores is left to the gradient:
* fast head (birth ratio 0): not regulated;
* integrator (birth ratio < 1): log g += rate (1 - mean G), toward the timescale edge;
* latch (birth ratio >= 1): logit eta += rate log(mean D / (u* mean T)), toward the input
  edge, and log g += rate when kappa < 1.2, so that a basin exists at all. u* = 1 puts the
  mean disturbance at the tolerance, which `latch_corrects` requires of every event, so a
  fluctuating text stream is then overwritten about half the time (v2_soc_s11); u* < 1 leaves
  a margin, calibrated from E2 (calibrate_input_edge.py).
All statistics are the head's own (local, physiological). r is read from the state at the
start of the window (S_0^T k_t), a mean-field estimate; the rule acts every few updates.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def birth_ratios(heads: int) -> list[float]:
    """threshold_spread birth ratios (gated_deltanet.GatedDeltaNetLayer)."""
    ratio = [0.0] * heads
    for i in range(1, heads):
        ratio[i] = 0.5 * 6.0 ** ((i - 1) / max(heads - 2, 1))
    return ratio


@torch.no_grad()
def head_balances(model, ids, state):
    """Per feedback layer and head over one window: mean loop gain G, mean disturbance D,
    mean tolerance T and kappa, each [L, H] (nan for layers without feedback)."""
    base = model.base
    h = base.source.embedding(ids)
    shape = (base.num_layers, base.heads)
    out = {name: torch.full(shape, float("nan"), device=ids.device) for name in ("G", "D", "T", "kappa")}
    for index, layer in enumerate(base.layers):
        inner = getattr(layer, "inner", layer)
        if getattr(inner, "recurrence", None) == "feedback_delta":
            B, T, d = h.shape
            x_norm = inner.norm(h)
            q, k, v = inner.q_proj(x_norm), inner.k_proj(x_norm), inner.v_proj(x_norm)
            st = state[index]
            history = st[:, inner.matrix_size:inner.matrix_size + inner.history_size].reshape(
                B, 3 * d, inner.conv_kernel - 1)
            mixed, _ = inner._mix_qkv(torch.cat((q, k, v), -1).transpose(1, 2), history)
            _, k, v = [p.reshape(B, T, inner.heads, inner.dk) for p in mixed.transpose(1, 2).split(d, -1)]
            k = F.normalize(k, dim=-1)
            S0 = st[:, :inner.matrix_size].reshape(B, inner.heads, inner.dk, inner.dv)
            r = torch.einsum("bhkv,bthk->bthv", S0, k)
            alpha = inner.retention(x_norm)[..., None]
            beta = torch.sigmoid(inner.beta_proj(x_norm))[..., None]
            eta = inner.feedback_logit.sigmoid()[None, None, :, None]
            gain = F.softplus(inner.feedback_gain_raw)[None, None, :, None]
            a = alpha * (1 - beta)
            c = beta * eta
            G = a + c * gain / (1.0 + (gain * r + v).abs()) ** 2
            D = beta * (1 - eta) * v.abs()
            tol = (c.sqrt() - ((1 - a) / gain).sqrt()).clamp_min(0) ** 2
            kappa = c * gain / (1 - a).clamp_min(1e-6)
            out["G"][index] = G.mean(dim=(0, 1, 3))
            out["D"][index] = D.mean(dim=(0, 1, 3))
            out["T"][index] = tol.expand_as(D).mean(dim=(0, 1, 3))
            out["kappa"][index] = kappa.mean(dim=(0, 1, 3))
        h, _, _ = layer(h, state[index])
    return out


@torch.no_grad()
def soc_update(model, ids, state, rate: float = 0.05, input_target: float = 1.0):
    """One homeostatic step per role; returns the measured balances."""
    bal = head_balances(model, ids, state)
    for index, layer in enumerate(model.base.layers):
        inner = getattr(layer, "inner", layer)
        if getattr(inner, "recurrence", None) != "feedback_delta":
            continue
        roles = birth_ratios(inner.heads)
        g = F.softplus(inner.feedback_gain_raw).clone()
        logit = inner.feedback_logit.clone()
        for head, ratio in enumerate(roles):
            if ratio == 0.0:
                continue                                    # fast head: gradient only
            if ratio < 1.0:                                 # integrator: timescale edge
                g[head] = g[head] * math.exp(rate * float(1.0 - bal["G"][index, head]))
            else:                                           # latch: input edge, keep a basin
                u = float(bal["D"][index, head] / bal["T"][index, head].clamp_min(1e-6)) / input_target
                logit[head] = logit[head] + rate * max(-1.0, min(1.0, math.log(max(u, 1e-6))))
                if float(bal["kappa"][index, head]) < 1.2:
                    g[head] = g[head] * math.exp(rate)
        g = g.clamp(0.5, 64.0)
        inner.feedback_gain_raw.copy_(g + torch.log(-torch.expm1(-g)))  # softplus inverse
        inner.feedback_logit.copy_(logit.clamp(-7.0, 7.0))
    return torch.stack([bal["G"], bal["D"] / bal["T"].clamp_min(1e-6), bal["kappa"]])
