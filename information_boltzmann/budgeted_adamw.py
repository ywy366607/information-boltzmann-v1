"""Opt-in AdamW proposal with local budgets and recoverable coherence cooling.

Exact-real executed-step reference: GenerationBudgetedPlasticity.lean.
This candidate is NOT installed in the continuing language body. All incoming
gradients, including delayed credit, keep their existing causal provenance.
The controller estimates gradient coherence, not dopamine or causal truth.
"""
from __future__ import annotations

from typing import Callable

import torch


def tiled(x: torch.Tensor, size: int) -> torch.Tensor:
    """[rows,cols] -> [tile_rows,tile_cols,2,cell_rows,2,cell_cols]."""
    rows, cols = x.shape
    half = size // 2
    return x.reshape(rows // size, 2, half, cols // size, 2, half).permute(0, 3, 1, 2, 4, 5)


def untiled(x: torch.Tensor) -> torch.Tensor:
    a, b, _, r, _, c = x.shape
    return x.permute(0, 2, 3, 1, 4, 5).reshape(a * 2 * r, b * 2 * c)


def coherence_gate(signal: torch.Tensor, energy: torch.Tensor, noise_weight: float,
                   *, wiener: bool = False) -> torch.Tensor:
    """Finite-sample clipped signal fraction; cold startup is uninformed, neutral.

    With wiener=False (default), estimates per-step signal fraction
    (S - w*E) / ((1 - w) * E). With wiener=True, estimates momentum Wiener gain
    (S - w*E) / ((1 - w) * S), matching the EMA noise variance w*sigma^2 while
    keeping pure-noise rejection identical.
    """
    if noise_weight >= 1 - 1e-12:
        return torch.ones_like(energy)
    base = signal if wiener else energy
    denominator = (1 - noise_weight) * base
    return torch.where(base > 0,
                       (signal - noise_weight * energy) / denominator.clamp_min(
                           torch.finfo(energy.dtype).tiny),
                       torch.zeros_like(energy)).clamp(0, 1)


def noise_weight(beta: float, count: int) -> float:
    """Squared normalized EMA weights, with a zero observer initial state."""
    if beta == 0:
        return 1.0
    power = beta ** count
    return (1 - beta) / (1 + beta) * (1 + power) / (1 - power)


def balanced_cells(phase: torch.Tensor) -> torch.Tensor:
    return torch.stack((1 + phase, 1 - phase, 1 - phase, 1 + phase), -1).reshape(
        *phase.shape, 2, 2)


class BudgetedAdamW(torch.optim.AdamW):
    """Retain AdamW m/v; change its executed write, never its gradient evidence.

    Neutral allocation/cooling exactly recovers the ordinary AdamW formula.
    Extra state is four scalars per tensor tile, not copied weights or traces.
    Layers/heads/projections remain parameter identities; tile_size is a compute
    layout and cannot establish biological compartments or functional diversity.

    On adoption of old Adam state, store only its per-tile moment RMS. The
    triangle inequality bounds its surviving influence on the current moment.
    Cool against the UPPER recent-signal bound: uncertain inherited direction
    is not evidence of noise. Adam history is never
    reset, copied in full, or mislabelled as a newly observed variance history.
    """

    def __init__(self, params, lr=3e-4, betas=(.9, .999), eps=1e-8,
                 weight_decay=.01, *, tile_size=64, allocation=True, cooling=False,
                 fused=True, wiener_gate=False):
        if type(tile_size) is not int or tile_size < 2 or tile_size % 2:
            raise ValueError('tile_size must be a positive even layout size')
        if not 0 < betas[0] < 1:
            raise ValueError('coherence observer requires 0 < beta1 < 1')
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                         foreach=False)
        self.defaults.update(tile_size=tile_size, allocation=allocation, cooling=cooling,
                             budget_fused=fused, wiener_gate=bool(wiener_gate))
        for group in self.param_groups:
            group.update(tile_size=tile_size, allocation=allocation, cooling=cooling,
                         budget_fused=fused, wiener_gate=bool(wiener_gate))
        self._host_steps = {}

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self._host_steps = {}

    @torch.no_grad()
    def step(self, closure: Callable | None = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if any(group.get(k, False) for k in ('amsgrad', 'maximize', 'capturable', 'differentiable')):
                raise ValueError('candidate implements dense ordinary AdamW semantics only')
            beta1, beta2 = group['betas']
            allocation, cooling = group.get('allocation', False), group.get('cooling', False)
            wiener = bool(group.get('wiener_gate', False))
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse or p.dtype not in (torch.float32, torch.float64):
                    raise ValueError('candidate requires dense FP32/FP64 parameters')
                state = self.state[p]
                if not state:
                    state.update(step=torch.tensor(0.), exp_avg=torch.zeros_like(p),
                                 exp_avg_sq=torch.zeros_like(p))
                if p not in self._host_steps:
                    self._host_steps[p] = int(state['step'].item())
                count = self._host_steps[p] + 1
                size = group.get('tile_size', 64)
                matrix = p.ndim == 2 and p.shape[0] % size == 0 and p.shape[1] % size == 0
                shape = (p.shape[0] // size, p.shape[1] // size) if matrix else ()
                if (allocation or cooling) and 'budget_energy' not in state:
                    state['budget_energy'] = torch.zeros(shape, device=p.device, dtype=p.dtype)
                    state['budget_phase'] = torch.zeros(shape, device=p.device, dtype=p.dtype)
                    state['budget_gate'] = torch.ones(shape, device=p.device, dtype=p.dtype)
                    prior = state['exp_avg'].square()
                    state['budget_prior_rms'] = (
                        tiled(prior, size).mean((2, 3, 4, 5)) if matrix else prior.mean()).sqrt()
                    state['budget_count'] = 0
                observer_count = int(state.get('budget_count', 0)) + 1
                can_fuse = (group.get('budget_fused', True) and p.is_cuda and p.dtype == torch.float32
                            and p.is_contiguous() and g.is_contiguous() and
                            (matrix or p.numel() <= 4096) and (allocation or cooling))
                if can_fuse:
                    from information_boltzmann.budgeted_adamw_fused import fused_step
                    fused_step(p, g, state, count=count, beta1=beta1, beta2=beta2,
                               lr=group['lr'], decay=group['weight_decay'], eps=group['eps'],
                               tile_size=size, matrix=matrix, allocation=allocation, cooling=cooling,
                               observer_count=observer_count, noise=noise_weight(beta1, observer_count),
                               wiener=wiener)
                    self._host_steps[p] = count
                    state['budget_count'] = observer_count
                    state['step'].add_(1)
                    continue
                m, v = state['exp_avg'], state['exp_avg_sq']
                m.lerp_(g, 1 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                direction = (m / (1 - beta1 ** count)) / (
                    (v / (1 - beta2 ** count)).sqrt().add_(group['eps']))
                if allocation or cooling:
                    # These two observer quantities have the SAME beta1 weights.
                    dims = (2, 3, 4, 5)
                    energy = tiled(g.square(), size).mean(dims) if matrix else g.square().mean()
                    state['budget_energy'].lerp_(energy, 1 - beta1)
                    energy = state['budget_energy'] / (1 - beta1 ** observer_count)
                    signal = m.square()
                    signal = tiled(signal, size).mean(dims) if matrix else signal.mean()
                    recent_rms_upper = (signal.sqrt() + beta1 ** observer_count *
                                        state['budget_prior_rms']) / (1 - beta1 ** observer_count)
                    gate = coherence_gate(recent_rms_upper.square(), energy, noise_weight(
                        beta1, observer_count), wiener=wiener) if cooling else torch.ones_like(energy)
                    if matrix:
                        cells = tiled(direction, size)
                        power = cells.square().mean((3, 5))
                        total = power.sum((-2, -1))
                        contrast = power[..., 0, 0] + power[..., 1, 1] - power[..., 0, 1] - power[..., 1, 0]
                        cue = (contrast / total.clamp_min(torch.finfo(p.dtype).tiny)).clamp(-1, 1)
                        phase = state['budget_phase']
                        phase.lerp_(cue, 1 - beta1).clamp_(-1, 1)
                        a = balanced_cells(phase) if allocation else torch.ones_like(power)
                        proposal = untiled(cells * a[..., :, None, :, None])
                        gate_full = gate.repeat_interleave(size, 0).repeat_interleave(size, 1)
                    else:
                        proposal, gate_full = direction, gate
                    # The same gate acts on decay, otherwise a held weight drifts.
                    p.add_((proposal + group['weight_decay'] * p) * gate_full, alpha=-group['lr'])
                    # Store finite diagnostic state only; not an extra tensor field.
                    state['budget_gate'] = gate
                    state['budget_count'] = observer_count
                else:
                    p.mul_(1 - group['lr'] * group['weight_decay'])
                    p.add_(direction, alpha=-group['lr'])
                self._host_steps[p] = count
                state['step'].add_(1)
        return loss


def attach_budgeted_adamw(optimizer: torch.optim.AdamW, *, allocation=True,
                         cooling=True, tile_size=64, fused=True, wiener_gate=False) -> BudgetedAdamW:
    """Explicit maintenance-boundary attachment; never called by the live runner.

    Preserve optimizer identity, all groups/LRs, parameters, m/v, and clocks.
    Callers own the event boundary and checkpoint protocol; no loss/credit is
    consumed here. New observers start locally and bound inherited momentum.
    """
    if type(optimizer) is not torch.optim.AdamW:
        raise TypeError('attach only to a plain existing AdamW')
    if type(tile_size) is not int or tile_size < 2 or tile_size % 2:
        raise ValueError('tile_size must be positive and even')
    for group in optimizer.param_groups:
        if any(group.get(k, False) for k in ('amsgrad', 'maximize', 'capturable', 'differentiable')):
            raise ValueError('unsupported AdamW execution option')
        if not 0 < group['betas'][0] < 1:
            raise ValueError('coherence observer requires 0 < beta1 < 1')
    settings = dict(tile_size=tile_size, allocation=allocation, cooling=cooling,
                    budget_fused=fused, wiener_gate=bool(wiener_gate))
    optimizer.__class__ = BudgetedAdamW
    optimizer.defaults.update(settings)
    for group in optimizer.param_groups:
        group.update(settings)
    optimizer._host_steps = {}
    return optimizer
