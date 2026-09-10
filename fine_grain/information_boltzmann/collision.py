"""Microreversible attention kernel and a mollified local particle jump process.

For each unordered pair the rate is K_h(x_i-x_j) B_f / N. This gives the
Boltzmann weak-form factor 1/2, rather than a density-independent per-token gate.
Finite h is a spatial approximation: only global momentum/energy are exact.
"""
import math

import torch
from torch import Tensor, nn

from .state import PhaseState


def reflect(v: Tensor, w: Tensor, normal: Tensor) -> tuple[Tensor, Tensor]:
    change = ((v - w) * normal).sum(-1, keepdim=True) * normal
    return v - change, w + change


def spatial_kernel(delta: Tensor, width: float) -> Tensor:
    """Normalized product triangular kernel on R^d; no velocity discretization."""
    return (1 - (delta / width).abs()).clamp_min(0).prod(-1) / width ** delta.shape[-1]


class CollisionKernel(nn.Module):
    def __init__(self, dim: int, hidden: int, max_rate: float = 1.0, width: float = 1.0):
        super().__init__()
        self.max_rate, self.width = max_rate, width
        self.query = nn.Linear(3 * dim, hidden)
        self.key = nn.Linear(2 * dim, hidden)
        self.value = nn.Linear(2 * dim, hidden)
        self.output = nn.Linear(hidden, 1)

    def rate(self, x: Tensor, v: Tensor, w: Tensor, normal: Tensor,
             context_x: Tensor, context_v: Tensor) -> Tensor:
        vp, wp = reflect(v, w, normal)
        # Average the entire exchange / reversal / normal-sign orbit. The same
        # background f is used for all eight queries, including reverse queries.
        orbit = torch.stack([torch.cat((a, b, sign * normal))
                             for a, b in ((v, w), (w, v), (vp, wp), (wp, vp))
                             for sign in (1, -1)])
        features = torch.cat((context_x - x, context_v), -1)
        q, k, value = self.query(orbit), self.key(features), self.value(features)
        weights = spatial_kernel(context_x - x, self.width)
        # Epsilon only stabilizes context attention; it does not alter pair rates.
        scores = q @ k.T / math.sqrt(q.shape[-1]) + weights.clamp_min(1e-12).log()
        attended = scores.softmax(-1) @ value
        score = self.output(torch.tanh(attended + q)).mean()
        return self.max_rate * torch.sigmoid(score.clamp(-12, 12))

    def forward(self, state: PhaseState, duration: float,
                generator: torch.Generator | None = None) -> tuple[PhaseState, Tensor, dict]:
        n, dim = state.v.shape
        zero = state.v.new_zeros(())
        if self.max_rate == 0 or duration == 0:
            return state, zero, {"candidates": 0, "accepted": 0, "cross_moment_change": 0.0}
        # A parameter-independent dominating Poisson rate. Never cap the count:
        # silently capping would change the collision generator.
        kernel_bound = self.width ** (-dim)
        total_rate = (n - 1) * self.max_rate * kernel_bound / 2
        count = int(torch.poisson(state.v.new_tensor(total_rate * duration), generator=generator).item())
        velocities, log_prob, accepted = state.v, zero, 0
        pairs = []
        context_x, context_v = state.x, state.v
        for _ in range(count):
            i = int(torch.randint(n, (), device=state.v.device, generator=generator).item())
            j = int(torch.randint(n - 1, (), device=state.v.device, generator=generator).item())
            j += int(j >= i)
            normal = torch.randn(dim, device=state.v.device, dtype=state.v.dtype, generator=generator)
            normal = normal / normal.norm().clamp_min(torch.finfo(normal.dtype).tiny)
            local = spatial_kernel(state.x[i] - state.x[j], self.width) / kernel_bound
            if local > 0:
                rate = self.rate((state.x[i] + state.x[j]) / 2, velocities[i], velocities[j],
                                 normal, context_x, context_v)
                probability = local * rate / self.max_rate
            else:
                probability = zero
            choose = bool((torch.rand((), device=state.v.device, generator=generator) < probability).item())
            # Pathwise gradients alone omit the Bernoulli event-probability term.
            # The streaming loss adds the causal likelihood-ratio estimator.
            log_prob = log_prob + (probability.clamp_min(1e-30).log() if choose
                                   else torch.log1p(-probability))
            if choose:
                vp, wp = reflect(velocities[i], velocities[j], normal)
                indices = torch.tensor([i, j], device=state.v.device)
                velocities = velocities.index_copy(0, indices, torch.stack((vp, wp)))
                pairs.append([i, j])
                accepted += 1
        cross = ((velocities - state.v) * state.x).sum(-1).mean().detach().item()
        return PhaseState(state.x, velocities, state.time), log_prob, {
            "candidates": count, "accepted": accepted, "cross_moment_change": cross,
            "pairs": pairs,
        }
