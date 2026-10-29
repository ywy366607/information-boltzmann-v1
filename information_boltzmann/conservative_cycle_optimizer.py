"""Packed conservative-cycle references and the online ResourceCycle optimizer.

The same local phase/reserve state controls allocation and resource recovery.
Applicable arithmetic and interface contracts admit registered comparisons.
General lifelong learning, universal floating certificates and causal task
benefit remain open; see CGB_GENERIC_OPTIMIZER_SPEC.md.
"""
from __future__ import annotations

import math
from typing import Callable

import torch

FULL_LIFELONG_ADMITTED = False


def _cycle_chunk(code: torch.Tensor, gradient: torch.Tensor, lr: float, *, feedback: bool = True):
    """One bounded workspace: code [B] int16; gradient [B,4] float32/64.

    Four contiguous coordinates use implicit 2x2 order 00,01,10,11. Padded
    coordinates have zero gradient and are discarded. The reserve update uses
    the OLD phase, exactly as in integerCycleStep. No counter or index array.
    """
    phase = code.remainder(65) - 32
    reserve = torch.div(code, 65, rounding_mode="floor") + 8
    scale = gradient.abs().amax(dim=1, keepdim=True)
    normalized = gradient / torch.where(scale > 0, scale, torch.ones_like(scale))
    squares = normalized.square()
    diagonal = squares[:, 0] + squares[:, 3]
    cross = squares[:, 1] + squares[:, 2]
    total = diagonal + cross
    cue = 2 * (diagonal - cross) / torch.where(total > 0, total, torch.ones_like(total))
    cue_code = torch.floor(32 * cue.clamp(-2, 2) + 0.5).to(torch.int16)
    numerator = (2 * (4 + reserve) * phase if feedback else 8 * phase) + reserve * cue_code
    next_phase = torch.div(numerator + 8, 16, rounding_mode="floor").clamp(-32, 32)
    next_reserve = (reserve + torch.div(19 - reserve, 4, rounding_mode="floor")
                    - torch.div(reserve * phase.abs(), 128, rounding_mode="floor"))
    new_code = next_phase + 32 + 65 * (next_reserve - 8)
    diagonal_share = (32 + next_phase.to(gradient.dtype)) / 64
    cross_share = (32 - next_phase.to(gradient.dtype)) / 64
    shares = torch.stack((diagonal_share, cross_share, cross_share, diagonal_share), dim=1)
    update = gradient * shares * (lr * reserve.to(gradient.dtype) / 16)[:, None]
    return new_code, update


def _filtered_cycle_chunk(code: torch.Tensor, mean: torch.Tensor,
                          observed: torch.Tensor, gain: torch.Tensor,
                          scale: torch.Tensor, lr: float, *, feedback: bool = True):
    """Numerical reference for integer_preconditioned_step, not an optimizer.

    Coordinates are [blocks,4]; gain may be shared within a block. The caller
    supplies gain in [0,1] and positive scale, the square root of the final
    preconditioning denominator. This function does not estimate either.
    The mean stays in original gradient units when scale changes. Noise,
    drift and roundoff require filter_error_envelope's explicit assumptions.
    Both the allocation cue and final readout use the SAME scaled estimate.
    ResourceCycle uses this bridge without projecting parameters.
    """
    next_mean = (1 - gain) * mean + gain * observed
    next_code, scaled_update = _cycle_chunk(code, next_mean / scale, lr, feedback=feedback)
    return next_code, next_mean, scaled_update / scale


def _resource_observer_chunk(code: torch.Tensor, mean: torch.Tensor,
                             observed: torch.Tensor, rms: torch.Tensor,
                             mass: torch.Tensor, observer_rate: float, lr: float, *, feedback: bool = True):
    """Online numerical reference; not full optimizer/latch admission.

    code [G,C], mean/observed [G,C,4], rms/mass [G,1,1]. Each compartment
    shares two scalar statistics. H: its observation clock runs at
    observer_rate times the OLD mean available resource. Its normalized
    exposure generates gain; observed energy generates the denominator.
    Both moments use that same gain, so Jensen's energy bound applies.
    This is empirical gradient RMS, not calibrated noise or curvature.

    RMS is stored as a root and mixed in scaled units to avoid overflow or
    underflow when squaring finite extreme gradients. Zero energy needs no
    epsilon: a valid zero-energy state has zero mean and hence zero update.
    Inputs are not modified. Both feedback settings retain all live state;
    only the regenerative coupling in the phase transition is intervened on.
    """
    available = (torch.div(code, 65, rounding_mode="floor") + 8).to(mean.dtype)
    available = available.mean(dim=1, keepdim=True).unsqueeze(-1) / 16
    alpha = observer_rate * available
    next_mass = (1 - alpha) * mass + alpha
    gain = alpha / next_mass
    maximum = observed.abs().amax(dim=(1, 2), keepdim=True)
    normalized = observed / torch.where(maximum > 0, maximum, torch.ones_like(maximum))
    sample_rms = maximum * normalized.square().mean(dim=(1, 2), keepdim=True).sqrt()
    magnitude = torch.maximum(rms, sample_rms)
    safe = torch.where(magnitude > 0, magnitude, torch.ones_like(magnitude))
    next_rms = magnitude * ((1 - gain) * (rms / safe).square()
                            + gain * (sample_rms / safe).square()).sqrt()
    # scale^2 is the final preconditioning denominator (RMS), matching the
    # proved filtered readout rather than applying scaling after an old cue.
    scale = torch.where(next_rms > 0, next_rms, torch.ones_like(next_rms)).sqrt()
    next_code, next_mean, update = _filtered_cycle_chunk(
        code.reshape(-1), mean.reshape(-1, 4), observed.reshape(-1, 4),
        gain.expand_as(mean).reshape(-1, 4),
        scale.expand_as(mean).reshape(-1, 4), lr, feedback=feedback)
    return (next_code.reshape_as(code), next_mean.reshape_as(mean), next_rms,
            next_mass, update.reshape_as(mean))


class ConservativeCycle(torch.optim.Optimizer):
    """Numerical reference with two persistent bytes per four parameters.

    Accepts contiguous dense real parameters. The proof uses exact arithmetic,
    a valid smooth upper model and the declared cue error bound. Float32/64
    cue arithmetic and rounded parameter writes require separate validation.
    Zero/absent gradients still advance existing local state; no weight decay,
    momentum, shadow weights or hidden full-sized allocation matrix is added.
    ``chunk_cycles`` limits temporary storage and changes no mathematical law.
    """

    def __init__(self, params, lr: float, *, chunk_cycles: int = 65536, fused: bool = False):
        if not math.isfinite(lr) or lr < 0:
            raise ValueError("lr must be finite and nonnegative")
        if type(chunk_cycles) is not int or chunk_cycles < 1:
            raise ValueError("chunk_cycles must be a positive integer")
        if type(fused) is not bool:
            raise ValueError("fused must be boolean")
        super().__init__(params, dict(lr=lr, chunk_cycles=chunk_cycles, fused=fused))

    @staticmethod
    def _validate_parameter(parameter: torch.Tensor) -> None:
        if parameter.layout != torch.strided or not parameter.is_contiguous():
            raise ValueError("ConservativeCycle requires contiguous dense parameters")
        if parameter.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            raise ValueError("unsupported parameter dtype")
        gradient = parameter.grad
        if gradient is not None:
            if gradient.layout != torch.strided or not gradient.is_contiguous():
                raise ValueError("ConservativeCycle requires contiguous dense gradients")
            if gradient.numel():
                low, high = torch.aminmax(gradient)
                if not bool(torch.isfinite(low) & torch.isfinite(high)):
                    raise ValueError("nonfinite gradient")

    @torch.no_grad()
    def step(self, closure: Callable | None = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        # Validate before mutating any parameter or history. Reductions avoid
        # allocating a full-sized boolean finiteness array.
        for group in self.param_groups:
            if not math.isfinite(group["lr"]) or group["lr"] < 0:
                raise ValueError("lr must be finite and nonnegative")
            if type(group["chunk_cycles"]) is not int or group["chunk_cycles"] < 1:
                raise ValueError("chunk_cycles must be a positive integer")
            if type(group["fused"]) is not bool:
                raise ValueError("fused must be boolean")
            for parameter in group["params"]:
                self._validate_parameter(parameter)
        for group in self.param_groups:
            for parameter in group["params"]:
                count = parameter.numel()
                if not count:
                    continue
                state = self.state[parameter]
                if not state:
                    # K=0,R=16. This is the sole persistent tensor.
                    state["code"] = torch.full(((count + 3) // 4,), 552,
                                               dtype=torch.int16, device=parameter.device)
                code = state["code"]
                if (group["fused"] and parameter.is_cuda
                        and parameter.dtype in (torch.float32, torch.float64)):
                    from .conservative_cycle_cuda import fused_step
                    fused_step(parameter, parameter.grad, code, group["lr"])
                    continue
                flat = parameter.view(-1)
                grad = None if parameter.grad is None else parameter.grad.view(-1)
                dtype = torch.float64 if parameter.dtype == torch.float64 else torch.float32
                for start in range(0, code.numel(), group["chunk_cycles"]):
                    end = min(start + group["chunk_cycles"], code.numel())
                    first, last = 4 * start, min(4 * end, count)
                    work = torch.zeros((end - start, 4), dtype=dtype, device=parameter.device)
                    if grad is not None:
                        work.view(-1)[:last - first].copy_(grad[first:last])
                    new_code, update = _cycle_chunk(code[start:end], work, group["lr"])
                    flat[first:last].sub_(update.view(-1)[:last - first].to(parameter.dtype))
                    code[start:end].copy_(new_code)
        return loss

    def load_state_dict(self, state_dict):
        # Torch's default loader casts state tensors to parameter dtype. That
        # loses integer codes in BF16, so preserve the original packed payload
        # before the standard loader handles groups and metadata.
        groups = state_dict["param_groups"]
        if len(groups) != len(self.param_groups):
            raise ValueError("different number of parameter groups")
        saved = []
        for current, stored in zip(self.param_groups, groups):
            if len(current["params"]) != len(stored["params"]):
                raise ValueError("different parameter group size")
            for parameter, key in zip(current["params"], stored["params"]):
                payload = state_dict["state"].get(key, {})
                if not payload:
                    continue
                if set(payload) != {"code"}:
                    raise ValueError("unexpected optimizer state fields")
                code = payload["code"]
                if (not isinstance(code, torch.Tensor) or code.dtype != torch.int16
                        or tuple(code.shape) != ((parameter.numel() + 3) // 4,)):
                    raise ValueError("invalid packed optimizer state")
                if code.numel():
                    low, high = torch.aminmax(code)
                    if int(low) < 0 or int(high) >= 585:
                        raise ValueError("unused optimizer state code")
                saved.append((parameter, code.to(device=parameter.device).clone()))
        super().load_state_dict(state_dict)
        for parameter, code in saved:
            self.state[parameter]["code"] = code


def _matrix_allocation(phase, row, col, steps):
    allocation = phase.exp()
    row_budget = row / row.mean()
    col_budget = col / col.mean()
    for _ in range(steps):
        allocation = allocation * (row_budget / allocation.mean(dim=1, keepdim=True))
        allocation = allocation * (col_budget / allocation.mean(dim=0, keepdim=True))
    return allocation, row_budget, col_budget


def _previous_resource_update(state, parameter, lr, steps):
    """Reconstruct the previous direction from post-observer state and old reserves."""
    if 'previous_update' in state:  # checkpoints from the first feedback version
        return state['previous_update']
    if 'previous_row_reserve' not in state:
        return None
    row, col = state['previous_row_reserve'], state['previous_col_reserve']
    allocation, _, _ = _matrix_allocation(state['phase'], row, col, steps)
    scale = torch.where(state['rms'] > 0, state['rms'], torch.ones_like(state['rms'])).sqrt()
    update = lr * allocation * (row + col) / 2 * (state['mean'] / scale) / scale
    return update.view_as(parameter).to(parameter.dtype).to(state['mean'].dtype)


class ResourceCycle(torch.optim.Optimizer):
    """Matrix-coupled continuous-state optimizer; experimental, not certified.

    Actual tensor rows/columns share update-allocation budgets via finite
    Sinkhorn scaling. Signed means and RMS are per parameter. Phase is a
    persistent real-valued matrix; two reserve vectors replenish and deplete.
    Remaining tensors are viewed as one-row matrices. No parameter projection.
    Optional gradient-alignment feedback uses the previous applied direction.
    It is a directional proxy, not an observed loss decrease or a trust ratio.
    """
    def __init__(self, params, lr=1e-3, *, observer_rate=.125, feedback=True,
                 sinkhorn_steps=8, chunk_compartments=1024, auto_step=False,
                 rho_min=1e-4, rho_max=10.0, gamma_up=1.05, gamma_down=0.95, deadband_eps=1e-7,
                 adam_base=False, betas=(.9, .999), eps=1e-8, competition_strength=1.0,
                 latch_state=False, latch_strength=1.0, anchor_rate=.05,
                 resource_fatigue=False, resource_recovery_rate=.25,
                 outcome_modulation=False, outcome_deadband=1e-5):
        super().__init__(params, dict(lr=lr, observer_rate=observer_rate,
            feedback=feedback, sinkhorn_steps=sinkhorn_steps, chunk_compartments=chunk_compartments,
            auto_step=auto_step, rho=1.0, rho_min=rho_min, rho_max=rho_max,
            gamma_up=gamma_up, gamma_down=gamma_down, deadband_eps=deadband_eps,
            last_pred_decrease=0.0, adam_base=adam_base, betas=betas, eps=eps,
            competition_strength=competition_strength, latch_state=latch_state,
            latch_strength=latch_strength, anchor_rate=anchor_rate, resource_fatigue=resource_fatigue,
            resource_recovery_rate=resource_recovery_rate, outcome_modulation=outcome_modulation,
            outcome_deadband=outcome_deadband, last_outcome_evidence=0.0))
        for group in self.param_groups:
            self._validate_group(group)

    @staticmethod
    def _validate_group(group):
        if not math.isfinite(group['lr']) or group['lr'] < 0:
            raise ValueError('invalid lr')
        if not math.isfinite(group['observer_rate']) or not 0 < group['observer_rate'] <= 1:
            raise ValueError('invalid observer_rate')
        if type(group.get('feedback', True)) is not bool:
            raise ValueError('feedback must be boolean')
        if type(group.get('resource_fatigue', False)) is not bool:
            raise ValueError('resource_fatigue must be boolean')
        if type(group.get('outcome_modulation', False)) is not bool:
            raise ValueError('outcome_modulation must be boolean')
        if not math.isfinite(group.get('outcome_deadband', 1e-5)) or group.get('outcome_deadband', 1e-5) < 0:
            raise ValueError('invalid outcome_deadband')
        if (not math.isfinite(group.get('resource_recovery_rate', .25)) or
                not 0 < group.get('resource_recovery_rate', .25) <= 1):
            raise ValueError('invalid resource_recovery_rate')
        if type(group.get('sinkhorn_steps', 8)) is not int or group.get('sinkhorn_steps', 8) < 1:
            raise ValueError('sinkhorn_steps must be positive')
        if type(group.get('chunk_compartments', 1024)) is not int or group.get('chunk_compartments', 1024) < 1:
            raise ValueError('chunk_compartments must be positive')
        if type(group.get('auto_step', True)) is not bool:
            raise ValueError('auto_step must be boolean')
        if not math.isfinite(group.get('rho', 1.0)) or group.get('rho', 1.0) <= 0:
            raise ValueError('invalid rho')
        if not math.isfinite(group.get('rho_min', 1e-4)) or group.get('rho_min', 1e-4) <= 0:
            raise ValueError('invalid rho_min')
        if not math.isfinite(group.get('rho_max', 10.0)) or group.get('rho_max', 10.0) < group.get('rho_min', 1e-4):
            raise ValueError('invalid rho_max')
        if (not math.isfinite(group.get('gamma_up', 1.05)) or
                not math.isfinite(group.get('gamma_down', .95)) or
                not 1 <= group.get('gamma_up', 1.05) or not 0 < group.get('gamma_down', .95) <= 1):
            raise ValueError('invalid feedback multipliers')
        if not math.isfinite(group.get('deadband_eps', 1e-7)) or group.get('deadband_eps', 1e-7) < 0:
            raise ValueError('invalid deadband')
        if type(group.get('adam_base', False)) is not bool:
            raise ValueError('adam_base must be boolean')
        betas = group.get('betas', (.9, .999))
        if len(betas) != 2 or any(not math.isfinite(b) or not 0 <= b < 1 for b in betas):
            raise ValueError('invalid betas')
        if not math.isfinite(group.get('eps', 1e-8)) or group.get('eps', 1e-8) <= 0:
            raise ValueError('invalid eps')
        if not math.isfinite(group.get('competition_strength', 1.0)) or not 0 <= group.get('competition_strength', 1.0) <= 1:
            raise ValueError('invalid competition_strength')
        if group.get('adam_base', False) and group.get('auto_step', False):
            raise ValueError('directional auto_step is unavailable for adam_base')
        if type(group.get('latch_state', False)) is not bool:
            raise ValueError('latch_state must be boolean')
        if not math.isfinite(group.get('latch_strength', 1.0)) or group.get('latch_strength', 1.0) < 0:
            raise ValueError('invalid latch_strength')
        if not math.isfinite(group.get('anchor_rate', .05)) or not 0 < group.get('anchor_rate', .05) <= 1:
            raise ValueError('invalid anchor_rate')
        if group.get('latch_state', False) and group.get('auto_step', False):
            raise ValueError('directional auto_step is unavailable with latch_state')

    @staticmethod
    def _shape(p):
        return (p.shape[0], p.numel() // p.shape[0]) if p.ndim >= 2 else (1, p.numel())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        residuals = []
        predicted_decrease = 0.0
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            self._validate_group(group)
            for p in group['params']:
                ConservativeCycle._validate_parameter(p)

        # This is a directional derivative at the new point, not a measured
        # loss difference. The previous update includes allocation and reserves.
        for group in self.param_groups:
            if not group.get('auto_step', False):
                continue
            deadband_eps = group.get('deadband_eps', 1e-7)
            contributions = []
            for p in group['params']:
                if p.grad is None or p.numel() == 0:
                    continue
                previous = _previous_resource_update(self.state.get(p, {}), p,
                    group.get('previous_eff_lr', 0.0), group.get('previous_sinkhorn_steps', 8))
                if previous is not None:
                    contributions.append((p.grad.detach().to(previous.dtype) * previous).sum())
            if contributions:
                # One host synchronization per group; parameters may span devices.
                fresh_alignment = float(torch.stack([v.to(contributions[0].device) for v in contributions]).sum())
                if abs(fresh_alignment) > deadband_eps:
                    multiplier = group['gamma_up'] if fresh_alignment > 0 else group['gamma_down']
                    group['rho'] = min(group['rho_max'], max(group['rho_min'], group['rho'] * multiplier))

        for group in self.param_groups:
            # The trust gain is shared by the two feedback interfaces: the
            # legacy directional auto-step and the fresh-outcome third
            # factor.  Either one must affect the next continuous update;
            # otherwise outcome observation is diagnostic-only.
            uses_rho = group.get('auto_step', False) or group.get('outcome_modulation', False)
            eff_lr = group['lr'] * (group.get('rho', 1.0) if uses_rho else 1.0)
            for p in group['params']:
                if p.grad is None or p.numel() == 0:
                    if p in self.state:
                        self.state[p].pop('previous_update', None)
                        self.state[p].pop('previous_row_reserve', None)
                        self.state[p].pop('previous_col_reserve', None)
                    continue
                rows, cols = self._shape(p)
                dtype = torch.float64 if p.dtype == torch.float64 else torch.float32
                state = self.state[p]
                if not state:
                    state['mean'] = torch.zeros((rows, cols), device=p.device, dtype=dtype)
                    state['rms'] = torch.zeros_like(state['mean'])
                    state['phase'] = torch.zeros_like(state['mean'])
                    state['row_reserve'] = torch.ones((rows, 1), device=p.device, dtype=dtype)
                    state['col_reserve'] = torch.ones((1, cols), device=p.device, dtype=dtype)
                    state['resource_pool'] = torch.zeros((), device=p.device, dtype=dtype)
                    state['mass'] = torch.zeros((), device=p.device, dtype=dtype)
                    if group.get('adam_base', False):
                        state['adam_step'] = 0
                    if group.get('latch_state', False):
                        state['anchor'] = p.detach().view(rows, cols).to(dtype).clone()
                g = p.grad.view(rows, cols).to(dtype)
                mean, rms, phase = state['mean'], state['rms'], state['phase']
                row, col, mass = state['row_reserve'], state['col_reserve'], state['mass']
                resource_pool = state['resource_pool']
                alpha = group['observer_rate'] * (row.mean() + col.mean()) / 2
                nmass = (1-alpha)*mass + alpha
                gain = alpha/nmass
                adam = group.get('adam_base', False)
                if adam:
                    state['adam_step'] += 1
                    beta1, beta2 = group['betas']
                    nmean = beta1*mean + (1-beta1)*g
                    variance_gain = 1-beta2
                else:
                    nmean = (1-gain)*mean + gain*g
                    variance_gain = gain
                magnitude = torch.maximum(rms, g.abs())
                safe = torch.where(magnitude > 0, magnitude, torch.ones_like(magnitude))
                nrms = magnitude*((1-variance_gain)*(rms/safe).square()+variance_gain*(g/safe).square()).sqrt()
                scale = torch.where(nrms > 0, nrms, torch.ones_like(nrms)).sqrt()
                if adam:
                    count = state['adam_step']
                    direction = (nmean / (1-beta1**count)) / (nrms / math.sqrt(1-beta2**count) + group['eps'])
                    signal = direction
                else:
                    signal = nmean/scale
                peak = signal.abs().amax()
                normalized = signal / torch.where(peak > 0, peak, torch.ones_like(peak))
                power = normalized.square()
                avg = power.mean()
                cue = torch.where(avg > 0,
                    power / torch.where(avg > 0, avg, torch.ones_like(avg)) - 1,
                    torch.zeros_like(power)).clamp(-2, 2)
                drive = (row + col)/2 if group.get('feedback', True) else 0
                # A saturated phase must be releasable by sustained absent
                # evidence (cue=-1): with reserves in [.5,1], |phase|<=1
                # leaves a strictly contrary margin at the positive endpoint.
                nphase = ((.5+drive)*phase + cue).clamp(-1, 1)
                allocation, row_budget, col_budget = _matrix_allocation(
                    nphase, row, col, group.get('sinkhorn_steps', 8))
                residuals.append(torch.maximum((allocation.mean(dim=1, keepdim=True)-row_budget).abs().amax(),
                                               (allocation.mean(dim=0, keepdim=True)-col_budget).abs().amax()))
                use = allocation*power
                ru = use.mean(dim=1, keepdim=True)
                cu = use.mean(dim=0, keepdim=True)
                ru, cu = ru/(1+ru), cu/(1+cu)
                fatigue = 1 if group.get('resource_fatigue', False) else 0
                depletion_row = fatigue*.25*ru*row
                depletion_col = fatigue*.25*cu*col
                depleted_row = row - depletion_row
                depleted_col = col - depletion_col
                released = depletion_row.sum() + depletion_col.sum()
                available_pool = resource_pool + released
                row_need = (1 - depleted_row).clamp_min(0)
                col_need = (1 - depleted_col).clamp_min(0)
                total_need = row_need.sum() + col_need.sum()
                refill_fraction = group.get('resource_recovery_rate', .25) * torch.minimum(
                    torch.ones_like(available_pool),
                    available_pool / torch.where(total_need > 0, total_need, torch.ones_like(total_need)))
                nrow = depleted_row + refill_fraction*row_need
                ncol = depleted_col + refill_fraction*col_need
                nresource_pool = available_pool - refill_fraction*total_need
                if adam:
                    strength = group.get('competition_strength', 1.0)
                    factor = 1 + strength * (allocation * (row+col)/2 - 1)
                    update = eff_lr * factor * direction
                else:
                    update = eff_lr * allocation * (row+col)/2 * signal/scale
                if group.get('latch_state', False):
                    # A moving internal reference supplies a restorative path.
                    # Phase retains it; sustained gradient-driven displacement
                    # increases its rate of following the changing parameter.
                    anchor = state['anchor']
                    current = p.detach().view(rows, cols).to(dtype)
                    displacement = current - anchor
                    hold = nphase.clamp(0, 1)
                    reference_scale = anchor.square().mean().sqrt() + group['eps']
                    opposing = (-g * displacement).clamp_min(0) / (
                        g.abs() * (reference_scale + displacement.abs()) + group['eps'])
                    follow = (1 - hold + hold * opposing.clamp(0, 1)) * group['anchor_rate']
                    anchor.add_(follow * displacement)
                    restorative = eff_lr * allocation * (row + col) / 2 * hold * group['latch_strength'] * (current - anchor)
                    update = update + restorative
                predicted_decrease += float((g * update).sum().item())
                p.sub_(update.view_as(p).to(p.dtype))
                if group.get('auto_step', False):
                    # Retain only the reserve vectors, before overwriting them.
                    state['previous_row_reserve'] = row.clone()
                    state['previous_col_reserve'] = col.clone()
                    state.pop('previous_update', None)
                else:
                    state.pop('previous_update', None)
                    state.pop('previous_row_reserve', None)
                    state.pop('previous_col_reserve', None)
                for key, value in (('mean',nmean),('rms',nrms),('phase',nphase),
                                   ('row_reserve',nrow),('col_reserve',ncol),
                                   ('resource_pool',nresource_pool),('mass',nmass)):
                    state[key].copy_(value)
            # Retained for checkpoint compatibility; no Hessian is estimated here.
            group['last_pred_decrease'] = predicted_decrease
            group['previous_eff_lr'] = eff_lr
            group['previous_sinkhorn_steps'] = group.get('sinkhorn_steps', 8)
        self.last_budget_residual = torch.stack(residuals).amax() if residuals else None
        return loss

    @torch.no_grad()
    def observe_outcome(self, realized_decrease):
        """Update the group trust gain from a fresh post-update outcome.

        The caller must measure ``realized_decrease`` on data that was not used
        to form the preceding gradient.  This is a scalar group-level third
        factor; it adds no per-parameter state.
        """
        realized = float(realized_decrease)
        if not math.isfinite(realized):
            raise ValueError('nonfinite realized outcome')
        for group in self.param_groups:
            if not group.get('outcome_modulation', False):
                continue
            predicted = float(group.get('last_pred_decrease', 0.0))
            evidence = realized - predicted
            group['last_outcome_evidence'] = evidence
            if abs(evidence) <= group.get('outcome_deadband', 1e-5):
                continue
            multiplier = group.get('gamma_up', 1.05) if evidence > 0 else group.get('gamma_down', .95)
            group['rho'] = min(group.get('rho_max', 10.0),
                               max(group.get('rho_min', 1e-4), group.get('rho', 1.0) * multiplier))

    def load_state_dict(self, state_dict):
        groups = state_dict['param_groups']
        if len(groups) != len(self.param_groups):
            raise ValueError('different number of parameter groups')
        saved = []
        for current, stored in zip(self.param_groups, groups):
            stored.setdefault('auto_step', False)
            stored.setdefault('rho', 1.0)
            stored.setdefault('rho_min', 1e-4)
            stored.setdefault('rho_max', 10.0)
            stored.setdefault('gamma_up', 1.05)
            stored.setdefault('gamma_down', 0.95)
            stored.setdefault('deadband_eps', 1e-7)
            stored.setdefault('last_pred_decrease', 0.0)
            stored.setdefault('adam_base', False)
            stored.setdefault('betas', (.9, .999))
            stored.setdefault('eps', 1e-8)
            stored.setdefault('competition_strength', 1.0)
            stored.setdefault('latch_state', False)
            stored.setdefault('latch_strength', 1.0)
            stored.setdefault('anchor_rate', .05)
            stored.setdefault('resource_fatigue', False)
            stored.setdefault('resource_recovery_rate', .25)
            stored.setdefault('outcome_modulation', False)
            stored.setdefault('outcome_deadband', 1e-5)
            stored.setdefault('last_outcome_evidence', 0.0)
            self._validate_group(stored)
            if len(current['params']) != len(stored['params']):
                raise ValueError('different parameter group size')
            for p, key in zip(current['params'], stored['params']):
                payload = state_dict['state'].get(key, {})
                if not payload:
                    continue
                r,c = self._shape(p)
                dtype = torch.float64 if p.dtype == torch.float64 else torch.float32
                shapes = {'mean':(r,c), 'rms':(r,c), 'phase':(r,c),
                          'row_reserve':(r,1), 'col_reserve':(1,c),
                          'resource_pool':(), 'mass':()}
                if stored['latch_state']:
                    shapes['anchor'] = (r,c)
                if 'previous_update' in payload:
                    shapes['previous_update'] = tuple(p.shape)
                if 'previous_row_reserve' in payload and 'previous_col_reserve' in payload:
                    shapes['previous_row_reserve'] = (r, 1)
                    shapes['previous_col_reserve'] = (1, c)
                scalar_keys = {'adam_step'} if stored['adam_base'] else set()
                if set(payload) != set(shapes) | scalar_keys:
                    raise ValueError('incompatible optimizer state fields')
                if scalar_keys and (type(payload['adam_step']) is not int or payload['adam_step'] < 1):
                    raise ValueError('invalid adam step')
                for name, shape in shapes.items():
                    value = payload[name]
                    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or value.dtype != dtype:
                        raise ValueError('invalid optimizer state layout: '+name)
                    if not bool(torch.isfinite(value).all()):
                        raise ValueError('nonfinite optimizer state: '+name)
                if bool((payload['rms'] < 0).any()) or bool((payload['mass'] < 0) | (payload['mass'] > 1)):
                    raise ValueError('invalid optimizer statistics')
                if bool(((payload['rms'] == 0) & (payload['mean'] != 0)).any()):
                    raise ValueError('nonzero mean at zero RMS')
                saved.append((p, {k:(v.to(p.device).clone() if isinstance(v, torch.Tensor) else v) for k,v in payload.items()}))
        super().load_state_dict(state_dict)
        for p, payload in saved:
            self.state[p] = payload
