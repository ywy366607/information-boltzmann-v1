"""Differentiable, CPU-auditable matrix-momentum GDN candidate.

Not installed in the live body. S and P have shape [..., heads, keys, values].
All heads explicitly share key/value coordinate conventions. The symmetric
lift X(S) = [[0,S],[S.T,0]] / sqrt(2) preserves the Frobenius norm and allows
rectangular, non-symmetric GDN states without discarding their information.

H = (||P||^2 + omega^2 ||S||^2)/2
    + coupling/4 * sum_{i,j} ||[X(S_i), X(S_j)]||^2.

This is a classical, confined, BFSS-inspired restricted matrix Hamiltonian,
not full BFSS (no fermions or enforced zero Gauss charge). For coupling >= 0
and omega > 0, fixed-parameter continuous quiet flow conserves H and is bounded.
The symmetric split integrator is reversible/symplectic, not exactly energy
conserving. Large steps can be unstable; callers must audit numerical error.
"""

from __future__ import annotations

import math
import torch
from torch import Tensor


def symmetric_lift(s: Tensor) -> Tensor:
    """Reference/diagnostic only; dynamics use smaller blocks directly."""
    dk, dv = s.shape[-2:]
    prefix = s.shape[:-2]
    zz_k = s.new_zeros(*prefix, dk, dk)
    zz_v = s.new_zeros(*prefix, dv, dv)
    return torch.cat((torch.cat((zz_k, s), -1),
                      torch.cat((s.transpose(-1, -2), zz_v), -1)), -2) / math.sqrt(2)


def coupling_potential_and_gradient(
    s: Tensor, coupling: float | Tensor
) -> tuple[Tensor, Tensor]:
    """Exact block formula; no full Jacobian or autograd-inside-forward."""
    if s.ndim < 3 or s.shape[-3] < 1:
        raise ValueError("S must have at least one head: [..., heads, keys, values]")
    heads = s.unbind(-3)
    gradients = [torch.zeros_like(head) for head in heads]
    potential = s.new_zeros(s.shape[:-3])
    for i, si in enumerate(heads):
        for j in range(i + 1, len(heads)):
            sj = heads[j]
            a = si @ sj.transpose(-1, -2) - sj @ si.transpose(-1, -2)
            b = si.transpose(-1, -2) @ sj - sj.transpose(-1, -2) @ si
            potential = potential + coupling / 8 * (
                a.square().sum((-2, -1)) + b.square().sum((-2, -1)))
            gradients[i] = gradients[i] + coupling / 2 * (a @ sj - sj @ b)
            gradients[j] = gradients[j] + coupling / 2 * (si @ b - a @ si)
    return potential, torch.stack(gradients, -3)


def energy(s: Tensor, p: Tensor, *, frequency: float | Tensor,
           coupling: float | Tensor) -> Tensor:
    """One energy per leading batch element; not a reward or survival currency."""
    if s.shape != p.shape:
        raise ValueError("S and momentum must have identical shapes")
    potential, _ = coupling_potential_and_gradient(s, coupling)
    return (p.square() + frequency ** 2 * s.square()).sum((-3, -2, -1)) / 2 + potential


def rotation_charge(s: Tensor, p: Tensor) -> tuple[Tensor, Tensor]:
    """Two diagonal blocks of sum_h [X(S_h), X(P_h)].

    Conserved by quiet flow and its split integration, changed by boundary
    writes. This neural variant does NOT impose BFSS's zero Gauss constraint;
    nonzero charges are allowed. This is a candidate observable, not yet a
    learned semantic memory. No additional persistent state is allocated.
    """
    return (
        (s @ p.transpose(-1, -2) - p @ s.transpose(-1, -2)).sum(-3) / 2,
        (s.transpose(-1, -2) @ p - p.transpose(-1, -2) @ s).sum(-3) / 2,
    )


def quiet_step(s: Tensor, p: Tensor, *, dt: float | Tensor,
               frequency: float | Tensor, coupling: float | Tensor
               ) -> tuple[Tensor, Tensor]:
    """Half coupling kick, exact harmonic rotation, half coupling kick.

    No external-input gates, fake token, weight update, or autonomous scheduler.
    dt/frequency/coupling may be differentiable scalar tensors. They must stay
    fixed within the quiet trajectory for the stated energy interpretation.
    """
    if s.shape != p.shape:
        raise ValueError("S and momentum must have identical shapes")
    h = torch.as_tensor(dt, dtype=s.dtype, device=s.device)
    omega = torch.as_tensor(frequency, dtype=s.dtype, device=s.device)
    _, grad = coupling_potential_and_gradient(s, coupling)
    half_p = p - h / 2 * grad
    angle = omega * h
    cosine, sine = angle.cos(), angle.sin()
    drift = h * torch.sinc(angle / math.pi)
    next_s = cosine * s + drift * half_p
    next_p = cosine * half_p - omega * sine * s
    _, next_grad = coupling_potential_and_gradient(next_s, coupling)
    return next_s, next_p - h / 2 * next_grad


def event_step(s: Tensor, p: Tensor, k: Tensor, v: Tensor,
               alpha: Tensor, beta: Tensor, *, dt: float | Tensor,
               frequency: float | Tensor, coupling: float | Tensor
               ) -> tuple[Tensor, Tensor]:
    """Native decay-then-delta write plus the same innovation as momentum kick.

    k [...,heads,keys], v [...,heads,values], gates [...,heads]. Event time is
    normalized to one: P <- alpha P + delta_S. This is an explicit candidate
    boundary rule, not claimed to follow uniquely from BFSS. It is dissipative/
    driven and need not conserve H. dt=0 recovers the original S recurrence.
    """
    prior = alpha[..., None, None] * s
    residual = v - (prior.transpose(-1, -2) @ k.unsqueeze(-1)).squeeze(-1)
    innovation = beta[..., None, None] * k.unsqueeze(-1) * residual.unsqueeze(-2)
    written = prior + innovation
    kicked = alpha[..., None, None] * p + innovation
    return quiet_step(written, kicked, dt=dt, frequency=frequency, coupling=coupling)
