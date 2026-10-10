"""Task-valued capacity adaptation in the existing continuous Fourier chart.

This is a pullback natural step, not independent voxel growth or the exact
unconstrained simplex mirror solution. Volume-weighted categorical geometry
couples the three active shares and idle. Gaussian prior curvature regularizes
unobserved Fourier directions without charging another loss. The physical
wave state, direction rows, posterior variance and random sample stay intact.
"""
from __future__ import annotations

import math

import torch
from torch import nn


def simplex_pullback_metric(basis: torch.Tensor, shares: torch.Tensor,
                            volumes: torch.Tensor) -> torch.Tensor:
    """Return [mode,active,mode,active] geometry flattened in mean's order."""
    if (basis.ndim != 2 or shares.shape != (basis.shape[0], 4)
            or volumes.shape != (basis.shape[0],)):
        raise ValueError('Expected [site,mode], [site,4] and [site] quadrature')
    if not all(bool(torch.isfinite(x).all()) for x in (basis, shares, volumes)):
        raise ValueError('Finite capacity geometry required')
    if not bool((volumes > 0).all()):
        raise ValueError('Positive physical quadrature required')
    if not bool((shares >= 0).all()) or not torch.allclose(
            shares.sum(-1), torch.ones_like(volumes)):
        raise ValueError('Normalized nonnegative four-way shares required')
    active = shares[:, :3]
    local = torch.diag_embed(active) - active[:, :, None] * active[:, None, :]
    weights = volumes / volumes.sum()
    lw = local * weights[:, None, None]
    m_count = basis.shape[1]
    metric = torch.zeros(m_count, 3, m_count, 3, dtype=basis.dtype, device=basis.device)
    for a in range(3):
        for b in range(3):
            metric[:, a, :, b] = basis.T @ (basis * lw[:, a, b, None])
    size = 3 * m_count
    return metric.reshape(size, size)


def allocation_log_prob(basis, coefficients):
    logits = basis @ coefficients
    return torch.cat((logits, torch.zeros_like(logits[:, :1])), -1).log_softmax(-1)


class SpectralCapacityGrowth(nn.Module):
    """One independent update owner for posterior.mean; no Adam moments.

    ``step_size`` is a dimensionless natural/proximal step. ``max_capacity_kl``
    is an explicit engineering trust allowance in nats per quadrature point,
    not a biological constant or task-performance claim. Positive prior
    curvature is exactly 1/(window_targets * window_prior_variance).
    Dense geometry/solve runs on CPU float64, keeping GPU workspace bounded.
    All regularization remains in the existing structural objective.
    """

    def __init__(self, *, step_size: float, max_capacity_kl: float,
                 solve_rtol: float = 1e-9, max_backtracks: int = 32):
        super().__init__()
        for name, value in (('step_size', step_size), ('max_capacity_kl', max_capacity_kl),
                            ('solve_rtol', solve_rtol)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'Finite positive {name} required')
            self.register_buffer(name, torch.tensor(value, dtype=torch.float64))
        if isinstance(max_backtracks, bool) or not isinstance(max_backtracks, int) or max_backtracks < 1:
            raise ValueError('Positive integer max_backtracks required')
        self.register_buffer('max_backtracks', torch.tensor(max_backtracks, dtype=torch.int64))
        for name in ('updates', 'zero_updates'):
            self.register_buffer(name, torch.zeros((), dtype=torch.int64))
        for name in ('last_gradient_norm', 'last_update_norm', 'last_capacity_kl',
                     'last_max_site_kl', 'last_gradient_dot_update', 'last_solve_residual',
                     'last_step_fraction'):
            self.register_buffer(name, torch.zeros((), dtype=torch.float64))

    @torch.no_grad()
    def propose(self, posterior, basis, cell_volume):
        """Pure proposal from complete accumulated, un-clipped window credit.

        Evaluation/recomputation never updates structure. The caller records
        the actual window, proposes once before any parameter step, applies
        after Adam succeeds, and then commits the one existing OU ledger.
        """
        if not bool(posterior.window_active) or not bool(posterior.window_evidence_recorded):
            raise RuntimeError('Growth requires one complete recorded evidence window')
        if posterior.mean.grad is None:
            raise RuntimeError('Missing task/structural capacity credit')
        if posterior.log_std.requires_grad:
            raise ValueError('Capacity growth freezes posterior log_std')
        if basis.shape[-1] != posterior.coefficient_count:
            raise ValueError('Capacity basis/continuous coefficient mismatch')
        phi = basis.detach().reshape(-1, posterior.coefficient_count).to('cpu', torch.float64)
        mean = posterior.mean.detach().to('cpu', torch.float64)
        gradient = posterior.mean.grad.detach().to('cpu', torch.float64)
        prior_variance = posterior.window_prior_variance.detach().to('cpu', torch.float64)
        if (not bool(torch.isfinite(gradient).all()) or
                not bool((torch.isfinite(prior_variance) & (prior_variance > 0)).all())):
            raise FloatingPointError('Finite capacity credit and positive prior variance required')
        volume = torch.as_tensor(cell_volume, dtype=torch.float64, device='cpu')
        volume = volume.expand(basis.shape[:-1]).reshape(-1)
        def represented_shares(coefficients):
            # Use exactly the installed allocation arithmetic, including its
            # parameter precision, basis matmul and softmax. Normalize only
            # after transfer to float64 to remove probability-sum roundoff.
            fractions = posterior.allocation(basis, coefficients).detach().reshape(-1, 4).to('cpu', torch.float64)
            return fractions / fractions.sum(-1, keepdim=True)

        shares = represented_shares(posterior.sample_coefficients())
        mean_shares = represented_shares(posterior.mean)
        if not bool((shares > 0).all()) or not bool((mean_shares > 0).all()):
            raise FloatingPointError('Underflowed capacity shares cannot support finite KL geometry')
        old_log_p, old_mean_log_p = shares.log(), mean_shares.log()
        metric = simplex_pullback_metric(phi, shares, volume)
        count = int(posterior.window_event_count)
        # Same prior Hessian as the meanCE + coefficient KL / count objective.
        precision = (1 / (count * prior_variance)).flatten()
        metric.diagonal().add_(precision)
        if not bool(torch.isfinite(metric).all()):
            raise FloatingPointError('Unrepresentable prior curvature')
        chol = torch.linalg.cholesky(metric)
        direction = torch.cholesky_solve(-gradient.flatten()[:, None], chol).flatten()
        residual = torch.linalg.vector_norm(metric @ direction + gradient.flatten())
        scale = torch.linalg.vector_norm(gradient)
        relative = float(residual / scale) if float(scale) else 0.
        if not math.isfinite(relative) or relative > float(self.solve_rtol):
            raise FloatingPointError(f'Capacity geometry solve failed: relative residual {relative}')
        weights = volume / volume.sum()
        maximum = float(self.max_capacity_kl)
        fraction = 1.
        for _ in range(int(self.max_backtracks)):
            # Quantize to the real parameter precision before checking the
            # candidate. Float64-only movement is not a physical update.
            delta = (float(self.step_size) * fraction * direction).reshape_as(mean)
            candidate = (mean + delta).to(dtype=posterior.mean.dtype).double()
            represented = candidate - mean
            dot = float((gradient * represented).sum())
            if float(represented.norm()) == 0:
                # Finite credit below parameter ULP is a valid no-op, not a
                # divergent learner. Count it explicitly so stalls are visible.
                return {'mean': candidate.to(posterior.mean), 'old_version': posterior.mean._version,
                        'gradient_norm': float(scale), 'update_norm': 0., 'capacity_kl': 0.,
                        'max_site_kl': 0., 'gradient_dot_update': 0.,
                        'solve_residual': relative, 'step_fraction': 0.}
            physical_mean = candidate.to(posterior.mean)
            physical_sample = physical_mean + posterior.log_std.exp() * posterior.window_noise
            new_shares = represented_shares(physical_sample)
            new_mean_shares = represented_shares(physical_mean)
            tiny = torch.finfo(torch.float64).tiny
            site_kl = (new_shares * (new_shares.clamp_min(tiny).log() - old_log_p)).sum(-1)
            mean_site_kl = (new_mean_shares * (new_mean_shares.clamp_min(tiny).log() - old_mean_log_p)).sum(-1)
            max_kl = float(torch.maximum(site_kl.max(), mean_site_kl.max()).clamp_min(0))
            if (math.isfinite(dot) and math.isfinite(max_kl) and max_kl <= maximum
                    and (dot < 0 or float(scale) == 0)):
                return {'mean': candidate.to(posterior.mean), 'old_version': posterior.mean._version,
                        'gradient_norm': float(scale), 'update_norm': float(represented.norm()),
                        'capacity_kl': float((weights * site_kl).sum().clamp_min(0)),
                        'max_site_kl': max_kl, 'gradient_dot_update': dot,
                        'solve_residual': relative, 'step_fraction': fraction}
            fraction *= .5
        raise FloatingPointError('No represented descending capacity step inside the KL allowance')

    @torch.no_grad()
    def apply(self, posterior, proposal):
        """Apply exactly one validated proposal; guards against double writers."""
        if posterior.mean._version != proposal['old_version']:
            raise RuntimeError('Capacity changed after proposal; independent update ownership violated')
        posterior.mean.copy_(proposal['mean'])
        self.updates.add_(1)
        if proposal['update_norm'] == 0:
            self.zero_updates.add_(1)
        for name in ('gradient_norm', 'update_norm', 'capacity_kl', 'max_site_kl',
                     'gradient_dot_update', 'solve_residual', 'step_fraction'):
            getattr(self, 'last_' + name).fill_(proposal[name])

    def summary(self):
        return {name: (int(value) if value.dtype == torch.int64 else float(value))
                for name, value in self.named_buffers()}
