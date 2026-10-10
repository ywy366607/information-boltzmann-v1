"""Explicit coefficient-space structural filtering, separate from the solver.

One individual owns one diagonal Gaussian over the existing continuous Fourier
coefficients. The caller supplies the material basis, unit propagation directions
and quadrature volumes; this module never registers a second material/basis.
Window evaluation is pure. Only an explicit post-optimizer commit advances the
OU prior and evidence ledger. These identities do not establish task benefits.
"""
from __future__ import annotations

import math
from numbers import Integral

import torch
from torch import nn


class StructuralPosterior(nn.Module):
    """Gaussian coefficients for three installed capacities plus idle resource.

    All physical/statistical scales are explicit constructor arguments. Optional
    means use the zero-logit reference (equal four-way resource fractions), not a
    fitted spatial prior. ``prior_std`` is the stationary OU coefficient scale;
    ``initial_std`` initializes the trainable approximate posterior.

    Call ``begin_window(noise)`` once, reuse/reconstruct its reparameterization
    across BPTT chunks, score and backpropagate, and call ``record_window_evidence``
    once with actual totals. After the optimizer step, ``commit_window`` inherits
    the updated posterior through the actual elapsed OU interval. Forward and
    checkpoint recomputation never count evidence or advance the prior.
    """

    def __init__(self, coefficient_count: int, *, resource_density: float,
                 speed_reference: float, structure_time: float, prior_std: float,
                 initial_std: float, maintenance_supply: float, initial_dual: float,
                 initial_mean: torch.Tensor | None = None,
                 stationary_mean: torch.Tensor | None = None,
                 capacity_growth: dict | None = None):
        super().__init__()
        if not isinstance(coefficient_count, Integral) or isinstance(coefficient_count, bool) or coefficient_count < 1:
            raise ValueError('Positive coefficient count required')
        self.coefficient_count = int(coefficient_count)
        for name, value in (('resource_density', resource_density),
                            ('speed_reference', speed_reference),
                            ('structure_time', structure_time), ('prior_std', prior_std),
                            ('initial_std', initial_std)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'Finite positive {name} required')
        for name, value in (('maintenance_supply', maintenance_supply), ('initial_dual', initial_dual)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'Finite nonnegative {name} required')
        shape = (self.coefficient_count, 3)
        mean = torch.zeros(shape) if initial_mean is None else torch.as_tensor(initial_mean).detach().clone()
        if mean.shape != shape or mean.dtype not in (torch.float32, torch.float64) or not bool(torch.isfinite(mean).all()):
            raise ValueError('Initial mean must be finite float32/float64 [coefficient,3]')
        stationary = (torch.zeros_like(mean) if stationary_mean is None
                      else torch.as_tensor(stationary_mean, dtype=mean.dtype, device=mean.device).detach().clone())
        if stationary.shape != shape or not bool(torch.isfinite(stationary).all()):
            raise ValueError('Stationary mean must be finite [coefficient,3]')
        self.mean = nn.Parameter(mean)
        self.log_std = nn.Parameter(torch.full_like(mean, math.log(initial_std)))
        for name, value in (('resource_density', resource_density),
                            ('speed_reference', speed_reference), ('structure_time', structure_time),
                            ('maintenance_supply', maintenance_supply), ('dual', initial_dual)):
            tensor = mean.new_tensor(value)
            if not bool(torch.isfinite(tensor)) or (value > 0 and not bool(tensor > 0)):
                raise ValueError(f'{name} must be representable in coefficient precision')
            self.register_buffer(name, tensor)
        stationary_variance = torch.full_like(mean, prior_std * prior_std)
        initial_variance = (2 * self.log_std.detach()).exp()
        if not bool((torch.isfinite(stationary_variance) & (stationary_variance > 0)).all()):
            raise ValueError('Prior variance must be finite and positive in coefficient precision')
        if not bool((torch.isfinite(initial_variance) & (initial_variance > 0)).all()):
            raise ValueError('Initial variance must be finite and positive in coefficient precision')
        self.register_buffer('stationary_mean', stationary)
        self.register_buffer('stationary_variance', stationary_variance)
        self.register_buffer('prior_mean', stationary.clone())
        self.register_buffer('prior_variance', stationary_variance.clone())
        self.register_buffer('window_prior_mean', stationary.clone())
        self.register_buffer('window_prior_variance', stationary_variance.clone())
        # Store the exogenous noise, never a detached sample substituted for the
        # reparameterized graph. state_dict therefore replays pending windows.
        self.register_buffer('window_noise', torch.zeros_like(mean))
        self.register_buffer('window_active', torch.tensor(False, device=mean.device))
        self.register_buffer('window_evidence_recorded', torch.tensor(False, device=mean.device))
        for name in ('window_event_count', 'windows_committed', 'evidence_events'):
            self.register_buffer(name, torch.zeros((), dtype=torch.int64, device=mean.device))
        for name in ('window_duration', 'elapsed'):
            self.register_buffer(name, torch.zeros((), dtype=torch.float64, device=mean.device))
        self.register_buffer('window_maintenance', mean.new_zeros(()))
        self.capacity_growth = None
        if capacity_growth is not None:
            from .capacity_growth import SpectralCapacityGrowth
            self.capacity_growth = SpectralCapacityGrowth(**capacity_growth).to(mean.device)
            # Mean keeps its full task VJP, but has an independent update owner.
            self.log_std.requires_grad_(False)

    @staticmethod
    def _event_count(value: int) -> int:
        if not isinstance(value, Integral) or isinstance(value, bool) or value < 1:
            raise ValueError('Positive actual event count required')
        return int(value)

    def _posterior_variance(self):
        if self.log_std.dtype not in (torch.float32, torch.float64):
            raise ValueError('Structural posterior requires float32/float64 precision')
        variance = (2 * self.log_std).exp()
        torch._assert_async((torch.isfinite(variance) & (variance > 0)).all(),
                            'Posterior variance must be finite and positive')
        return variance

    def window_is_active(self) -> bool:
        """Host-side window flag. During CUDA-graph capture a device read is illegal, so the capturing
        code declares the (necessarily active) window through ``capture_window_active``."""
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return bool(getattr(self, 'capture_window_active', False))
        return bool(self.window_active)

    @torch.no_grad()
    def begin_window(self, noise: torch.Tensor) -> None:
        """Latch caller-owned noise and the fixed pre-window prior exactly once."""
        if bool(self.window_active):
            raise RuntimeError('A structural evidence window is already active')
        if noise.shape != self.mean.shape or noise.dtype != self.mean.dtype or noise.device != self.mean.device:
            raise ValueError('Window noise must match coefficient shape/dtype/device')
        if not bool(torch.isfinite(noise).all()):
            raise ValueError('Finite explicit window noise required')
        self.window_noise.copy_(noise)
        self.window_prior_mean.copy_(self.prior_mean)
        self.window_prior_variance.copy_(self.prior_variance)
        self.window_duration.zero_()
        self.window_event_count.zero_()
        self.window_maintenance.zero_()
        self.window_evidence_recorded.fill_(False)
        self.window_active.fill_(True)

    def sample_coefficients(self) -> torch.Tensor:
        """Rebuild the same-window sample graph; this performs no random draw."""
        torch._assert_async(self.window_active, 'Begin a structural window before sampling')
        torch._assert_async(torch.isfinite(self.mean).all(), 'Finite posterior mean required')
        self._posterior_variance()
        return self.mean + self.log_std.exp() * self.window_noise

    def allocation(self, basis: torch.Tensor, coefficients: torch.Tensor | None = None) -> torch.Tensor:
        """Return [..., active axis 0,1,2, idle] on the supplied continuous grid.

        Without an explicit sample, deployment/diagnostics use the posterior mean.
        Training callers explicitly pass ``sample_coefficients()`` and can reuse
        that tensor across all events of a differentiable chunk.
        """
        if basis.ndim < 2 or basis.shape[-1] != self.coefficient_count:
            raise ValueError('Basis must have shape [...,coefficient_count]')
        coefficient = self.mean if coefficients is None else coefficients
        if coefficient.shape != self.mean.shape:
            raise ValueError('Structural coefficients must have shape [coefficient,3]')
        if coefficient.dtype not in (torch.float32, torch.float64):
            raise ValueError('Structural allocation requires float32/float64 precision')
        torch._assert_async(torch.isfinite(basis).all(), 'Finite structural basis required')
        torch._assert_async(torch.isfinite(coefficient).all(), 'Finite structural coefficients required')
        # The installed budget is physical state. Ambient decoder autocast must
        # not lower its precision or relax direction/capacity constraints.
        with torch.autocast(device_type=coefficient.device.type, enabled=False):
            logits = basis.to(coefficient) @ coefficient
            logits = torch.cat((logits, torch.zeros_like(logits[..., :1])), -1)
            return self.resource_density.to(logits) * logits.softmax(-1)

    def slow_factor(self, basis: torch.Tensor, unit_directions: torch.Tensor,
                    coefficients: torch.Tensor | None = None) -> torch.Tensor:
        """Installed B=v_ref*diag(active allocation)*unit-direction rows.

        Direction rows are validated, not renormalized here. The medium derives
        them from its existing unit-lower-triangular shear map; there is no second
        free speed amplitude that can compensate for reduced charged capacity.
        """
        allocation = self.allocation(basis, coefficients)
        if unit_directions.shape != (*allocation.shape[:-1], 3, 3):
            raise ValueError('Direction rows must match the structural grid [...,3,3]')
        checked_dtype = torch.float64 if unit_directions.dtype == torch.float64 else torch.float32
        checked = unit_directions.to(dtype=checked_dtype, device=allocation.device)
        torch._assert_async(torch.isfinite(checked).all(), 'Finite direction rows required')
        tolerance = 32 * torch.finfo(checked_dtype).eps
        norm = torch.linalg.vector_norm(checked, dim=-1)
        torch._assert_async(torch.isclose(norm, torch.ones_like(norm), rtol=0, atol=tolerance).all(),
                            'Installed propagation directions must have unit row norms')
        directions = checked.to(allocation)
        return self.speed_reference.to(allocation) * allocation[..., :3, None] * directions

    def maintenance(self, allocation: torch.Tensor, cell_volume: float | torch.Tensor) -> torch.Tensor:
        """Volume-weighted installed capacity of one grid; idle is uncharged."""
        if allocation.ndim < 2 or allocation.shape[-1] != 4:
            raise ValueError('Allocation must have shape [...,4] for one individual')
        if isinstance(cell_volume, (int, float)):
            # A host scalar is validated on the host and multiplied in place: building a device tensor from it
            # would be a host-to-device copy, which a CUDA graph capture forbids.
            if not math.isfinite(cell_volume) or cell_volume <= 0:
                raise RuntimeError('Finite positive quadrature volumes required')      # same exception class as the device-side assert
            torch._assert_async((torch.isfinite(allocation) & (allocation >= 0)).all(),
                                'Finite nonnegative allocation required')
            return (allocation[..., :3].sum(-1) * float(cell_volume)).sum()
        volume = torch.as_tensor(cell_volume, dtype=allocation.dtype, device=allocation.device)
        if volume.numel() != 1 and volume.shape != allocation.shape[:-1]:
            raise ValueError('Cell volume must be scalar or match the spatial grid')
        torch._assert_async((torch.isfinite(volume) & (volume > 0)).all(),
                            'Finite positive quadrature volumes required')
        torch._assert_async((torch.isfinite(allocation) & (allocation >= 0)).all(),
                            'Finite nonnegative allocation required')
        return (allocation[..., :3].sum(-1) * volume).sum()

    def kl_divergence(self) -> torch.Tensor:
        """Full coefficient-space Gaussian KL against the latched window prior."""
        torch._assert_async(self.window_active, 'Begin a structural window before computing KL')
        self._posterior_variance()
        prior_variance = self.window_prior_variance
        torch._assert_async((torch.isfinite(prior_variance) & (prior_variance > 0)).all(),
                            'Window prior variance must be finite and positive')
        log_ratio = 2 * self.log_std - prior_variance.log()
        mean_cost = (self.mean - self.window_prior_mean).square() / prior_variance
        # expm1(r)-r is accurate near identical variances, unlike exp(r)-1-r.
        return .5 * (mean_cost + torch.expm1(log_ratio) - log_ratio).sum()

    def objective(self, mean_ce: torch.Tensor, maintenance: torch.Tensor,
                  event_count: int) -> torch.Tensor:
        """Pure window objective; chunk accumulation must share KL/L only once.

        For a window with L targets this is meanCE+(KL+lambda*(M-supply))/L.
        ``event_count`` is the whole-window L, including when this method is
        called for a chunk. The caller then weights the whole chunk objective
        by chunk_count/L. Calling this method never marks evidence.
        """
        count = self._event_count(event_count)
        if mean_ce.numel() != 1 or maintenance.numel() != 1:
            raise ValueError('Scalar mean CE and maintenance required')
        regularizer = self.kl_divergence() + self.dual * (maintenance - self.maintenance_supply)
        return mean_ce + regularizer / count

    @torch.no_grad()
    def record_window_evidence(self, event_count: int,
                               physical_duration: float | torch.Tensor,
                               maintenance: torch.Tensor) -> None:
        """Record actual whole-window totals once, independently of forward calls."""
        if not bool(self.window_active):
            raise RuntimeError('No active structural evidence window')
        if bool(self.window_evidence_recorded):
            raise RuntimeError('Structural window evidence was already recorded')
        count = self._event_count(event_count)
        duration = torch.as_tensor(physical_duration, dtype=torch.float64, device=self.mean.device)
        cost = torch.as_tensor(maintenance, dtype=self.mean.dtype, device=self.mean.device)
        if duration.numel() != 1 or not bool(torch.isfinite(duration) & (duration >= 0)):
            raise ValueError('Actual physical duration must be finite and nonnegative')
        if cost.numel() != 1 or not bool(torch.isfinite(cost) & (cost >= 0)):
            raise ValueError('Finite nonnegative actual maintenance required')
        self.window_event_count.fill_(count)
        self.window_duration.copy_(duration.reshape(()))
        self.window_maintenance.copy_(cost.reshape(()))
        self.window_evidence_recorded.fill_(True)

    @torch.no_grad()
    def validate_window_commit(self, *, posterior_mean: torch.Tensor | None = None,
                               dual_learning_rate: float | None = None):
        """Pure prospective OU/dual validation, also usable before a growth step."""
        if not bool(self.window_active) or not bool(self.window_evidence_recorded):
            raise RuntimeError('Record one active evidence window before committing')
        if dual_learning_rate is not None and (not math.isfinite(dual_learning_rate) or dual_learning_rate < 0):
            raise ValueError('Finite nonnegative dual learning rate required')
        variance = self._posterior_variance()
        interval = self.window_duration / self.structure_time.double()
        rho = (-interval).exp()
        relaxation = -torch.expm1(-2 * interval)
        mean = self.mean if posterior_mean is None else posterior_mean
        if mean.shape != self.mean.shape or mean.dtype != self.mean.dtype or mean.device != self.mean.device:
            raise ValueError('Prospective mean must match posterior coefficient storage')
        next_mean = (self.stationary_mean + rho * (mean - self.stationary_mean)).to(self.prior_mean)
        next_variance = (rho.square() * variance + relaxation * self.stationary_variance).to(self.prior_variance)
        next_dual = self.dual.clone()
        if dual_learning_rate is not None:
            next_dual = (next_dual + dual_learning_rate * (self.window_maintenance - self.maintenance_supply)).clamp_min(0)
        next_elapsed = self.elapsed + self.window_duration
        # Validate every prospective buffer before mutation. Optimizer failures
        # are fatal and resume from the last completed checkpoint; no in-place
        # rollback of an arbitrary optimizer is claimed here.
        valid = (torch.isfinite(next_mean).all()
                 & (torch.isfinite(next_variance) & (next_variance > 0)).all()
                 & torch.isfinite(next_dual) & (next_dual >= 0)
                 & torch.isfinite(next_elapsed) & (next_elapsed >= self.elapsed)
                 & torch.isfinite(interval) & (interval >= 0))
        if not bool(valid):
            raise FloatingPointError('Structural commit would produce invalid prior/dual/clock state')
        return next_mean, next_variance, next_dual, next_elapsed

    @torch.no_grad()
    def commit_window(self, *, dual_learning_rate: float | None = None) -> None:
        """Call only after the optimizer step; advance OU prior and ledger once."""
        next_mean, next_variance, next_dual, next_elapsed = self.validate_window_commit(
            dual_learning_rate=dual_learning_rate)
        self.prior_mean.copy_(next_mean)
        self.prior_variance.copy_(next_variance)
        self.dual.copy_(next_dual)
        self.elapsed.copy_(next_elapsed)
        self.windows_committed.add_(1)
        self.evidence_events.add_(self.window_event_count)
        self.window_active.fill_(False)
        self.window_evidence_recorded.fill_(False)
