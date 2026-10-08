"""Destination-conditioned local COBA sensitivities, constant in stream length.

Arriving delayed/STP pulses are conditioned on. The three derivatives per
edge differentiate conductance, post-reset voltage and ALIF adaptation exactly
under that condition; the global directional learning signal remains approximate.
"""
from dataclasses import dataclass
import torch


@torch.no_grad()
def compensated_add_(weight, residual, change, bounds=None):
    """Carry the rounding error of a float32 update into the next update."""
    total = residual + change
    old = weight.clone()
    new = old + total
    if bounds is not None:
        lower, upper = bounds
        unclipped = new
        new = new.clamp(lower, upper)
        valid = (unclipped >= lower) & (unclipped <= upper)
    else:
        valid = torch.ones_like(weight, dtype=torch.bool)
    residual.copy_(torch.where(valid, total - (new - old), 0))
    weight.copy_(new)


@dataclass
class LocalCobaEdgeCredit:
    conductance: torch.Tensor
    voltage: torch.Tensor
    adaptation: torch.Tensor
    rounding_residual: torch.Tensor

    @classmethod
    def zeros_like(cls, weight):
        return cls(*(torch.zeros_like(weight) for _ in range(4)))


def voltage_conductance_derivative(bio, old_voltage):
    """Partial dV_pre/dG_total for the actual clamped exponential integrator."""
    total = bio['g_total']
    alpha = bio['alpha_eff']
    denominator = total.clamp_min(1e-5)
    da = -alpha * ((total >= 1e-5) & (total <= 20)).to(total.dtype)
    db = (-da * denominator - (1-alpha) * (total >= 1e-5)) / denominator.square()
    force = (bio['v_pre'] - alpha * old_voltage) / bio['beta_int']
    return da * old_voltage + db * force


@torch.no_grad()
def update_local_edges_pytorch(state, weight, pending, pre, post, splits, pulses,
                               signal, bio, direct_factor, synaptic_decay,
                               spikes, psi, beta, rho, lr, apply):
    destination = post.long()
    source = pre.long()
    index = torch.arange(weight.numel(), device=weight.device)
    tier = sum((index >= cutoff).long() for cutoff in splits[1:4])
    arriving = torch.stack(pulses).reshape(4, -1)[tier, source]
    decay = synaptic_decay.reshape(-1)[destination]
    state.conductance.mul_(decay).add_((1-decay)*arriving)
    ev = bio['alpha_eff'].reshape(-1)[destination]*state.voltage + direct_factor.reshape(-1)[destination]*state.conductance
    old_b = state.adaptation
    ds = psi.reshape(-1)[destination]*(ev-beta.reshape(-1)[destination]*old_b)
    state.voltage.copy_((1-spikes.reshape(-1)[destination])*ev-bio['v_pre'].reshape(-1)[destination]*ds)
    r = rho.reshape(-1)[destination]
    state.adaptation.copy_(r*old_b+(1-r)*ds)
    pending.add_(-lr*signal.reshape(-1)[destination]*state.voltage)
    if apply:
        compensated_add_(weight, state.rounding_residual, pending, (0, 5))
        pending.zero_()
