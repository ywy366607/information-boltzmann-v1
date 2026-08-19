"""Forward State Optimizer: treat Δ = S₂ − S as an innovation / pseudo-gradient.

S is an evolving belief, not a parameter. Surprise g(U) is the step size.
O(Δ, history) is temporal/geometric preconditioning:

  sgd        S + η g Δ
  rms        S + η g RMSNorm(Δ)          # amplitude only from g
  trust      S + g α Δ                   # α clips RMS(Δ)/RMS(S)
  momentum   S + η g m,  m = β m + (1-β) Δ     # across layers
  adam       S + η g m / (√v+ε)
  muon       S + η g NS(m)               # high-rank slice updates

Optional evidence decay (not weight decay):
  S ← S − λ (S − S_ev)   spring back to this layer's read of the raw field.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from fine_grain.models import newton_schulz

ALIASES = {
    "raw": "sgd",
    "a": "sgd",
    "sgd": "sgd",
    "none": "sgd",
    "rms_dir": "rms",
    "rms": "rms",
    "b": "rms",
    "trust": "trust",
    "trust_region": "trust",
    "c": "trust",
    "momentum": "momentum",
    "mom": "momentum",
    "adam": "adam",
    "muon": "muon",
}


def normalize_kind(kind: str) -> str:
    k = str(kind).lower()
    if k not in ALIASES:
        raise ValueError(f"unknown forward optimizer {kind!r}; choose {sorted(set(ALIASES.values()))}")
    return ALIASES[k]


class ForwardStateOpt(nn.Module):
    def __init__(
        self,
        d: int,
        kind: str = "sgd",
        eta_init: float = 1.0,
        beta: float = 0.9,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        trust_rho: float = 0.1,
        evidence_decay: float = 0.0,
    ):
        super().__init__()
        self.kind = normalize_kind(kind)
        self.beta = float(beta)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.trust_rho = float(trust_rho)
        self.evidence_decay = float(evidence_decay)
        self.eta = nn.Parameter(torch.tensor(float(eta_init)))
        self.rms_weight = nn.Parameter(torch.ones(d))
        self.rms_eps = 1e-6

    def _rms(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(dim=-1, keepdim=True)
        return self.rms_weight * x * torch.rsqrt(var + self.rms_eps)

    def _muon(self, M: torch.Tensor) -> torch.Tensor:
        # M [B,M,d] → polar on [B,d,M] (columns = slices), same as visual Stiefel
        U = newton_schulz(M.transpose(1, 2))
        return U.transpose(1, 2)

    def step(
        self,
        S: torch.Tensor,
        delta: torch.Tensor,
        gate: torch.Tensor,
        state: Optional[Dict[str, torch.Tensor]] = None,
        S_ev: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        state = state or {}
        m_prev = state.get("m")
        v_prev = state.get("v")
        kind = self.kind

        if m_prev is None or m_prev.shape != delta.shape:
            m_prev = torch.zeros_like(delta)
        if v_prev is None or v_prev.shape != delta.shape:
            v_prev = torch.zeros_like(delta)

        m = None
        v = None
        if kind == "sgd":
            step = delta
        elif kind == "rms":
            step = self._rms(delta)
        elif kind == "trust":
            rms_d = delta.pow(2).mean(dim=-1, keepdim=True).sqrt()
            rms_s = S.pow(2).mean(dim=-1, keepdim=True).sqrt()
            r = rms_d / rms_s.clamp_min(1e-6)
            alpha = (self.trust_rho / r.clamp_min(1e-6)).clamp(max=1.0)
            step = alpha * delta
            m = m_prev
            v = v_prev
        elif kind == "momentum":
            m = self.beta * m_prev + (1.0 - self.beta) * delta
            step = m
        elif kind == "adam":
            m = self.beta1 * m_prev + (1.0 - self.beta1) * delta
            v = self.beta2 * v_prev + (1.0 - self.beta2) * delta.square()
            step = m / (v.sqrt() + self.eps)
        elif kind == "muon":
            m = self.beta * m_prev + (1.0 - self.beta) * delta
            step = self._muon(m)
        else:
            raise RuntimeError(kind)

        S_new = S + self.eta * gate * step
        if self.evidence_decay > 0.0 and S_ev is not None:
            S_new = S_new - self.evidence_decay * (S - S_ev)

        if m is None:
            m = m_prev
        if v is None:
            v = v_prev
        meta = {
            "m": m,
            "v": v,
            "step": step.detach(),
            "rms_m": float(m.detach().pow(2).mean().sqrt()) if m is not None else 0.0,
        }
        return S_new, meta
