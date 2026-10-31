"""Local calcium-dependent graded APL release on every supplied cable node.

Voltage, chemistry and vesicle availability remain separate physical states.
The positive kinetics are learnable effective quantities in model units, not
identified channel subtypes, molar calcium or molecular priming states. APL's
existing nonspiking cable supplies voltage; no somatic spike broadcast is added.
"""
import math

import torch
from torch import nn
from .fly_biophysical_operators import resource_recovery


APL_LOCAL_RELEASE_VERSION = 1
APL_LOCAL_RELEASE_PARAMETER_NAMES = (
    'apl_release_log_half_voltage', 'apl_release_logit_gate_decay',
    'apl_release_logit_calcium_decay', 'apl_release_log_rate',
    'apl_release_logit_resource_recovery',
)
APL_LOCAL_RELEASE_STATE_FIELDS = (
    'apl_release_gate', 'apl_release_calcium', 'apl_release_resources',
)


def initial_parameters(count, *, device=None, dtype=torch.float32):
    # Initialization scales in ticks, all subsequently learnable. They are not
    # asserted measurements of an APL calcium channel or a fixed world clock.
    def logit_decay(tau):
        rho = math.exp(-1. / tau)
        return math.log(rho / (1. - rho))
    values = (0., logit_decay(1.), logit_decay(8.), 0., logit_decay(16.))
    return {name: torch.full((count,), value, device=device, dtype=dtype)
            for name, value in zip(APL_LOCAL_RELEASE_PARAMETER_NAMES, values)}


def attach_apl_local_release(model, *, saved=None):
    if not hasattr(model, 'apl_cable_index'):
        raise ValueError('Local APL calcium release requires the native APL cable')
    if hasattr(model, APL_LOCAL_RELEASE_PARAMETER_NAMES[0]):
        raise ValueError('APL local release already attached')
    for name, value in initial_parameters(model.apl_cable_index.numel(),
            device=model.apl_log_axial.device, dtype=model.apl_log_axial.dtype).items():
        if saved is not None:
            if name not in saved or saved[name].shape != value.shape or not torch.isfinite(saved[name]).all():
                raise ValueError('Incomplete or invalid APL release parameters: ' + name)
            value = saved[name].to(value).clone()
        model.register_parameter(name, nn.Parameter(value))
    model.apl_local_release_enabled = True
    model.apl_calcium_entry_enabled = True
    model.apl_release_resource_enabled = True


def local_release_state(model, current):
    values = [getattr(current, name) for name in APL_LOCAL_RELEASE_STATE_FIELDS]
    shape = (1, model.aplc_owner.numel())
    if not values[0].numel():
        if any(value.numel() for value in values):
            raise ValueError('Incomplete local APL calcium/resource history')
        values = [current.h.new_zeros(shape), current.h.new_zeros(shape),
                  current.h.new_ones(shape)]
    if any(value.shape != shape for value in values):
        raise ValueError('Malformed local APL calcium/resource history')
    return values


def advance_apl_local_release(model, current, voltage_drive):
    """Consume local calcium-dependent resources and recover once per real tick.

    ``voltage_drive`` is the existing nonnegative, threshold-normalized local
    depolarization. No cell mean or externally chosen reward enters this law.
    Returned packet is consumed by the existing physical-delay ring.
    """
    gate, calcium, resources = local_release_state(model, current)
    owner = model.aplc_owner
    def spread(name):
        return getattr(model, name).index_select(0, owner).unsqueeze(0)
    half = spread('apl_release_log_half_voltage').exp()
    target = voltage_drive / (voltage_drive + half)
    rho_gate = spread('apl_release_logit_gate_decay').sigmoid()
    gate = rho_gate * gate + (1. - rho_gate) * target
    rho_calcium = spread('apl_release_logit_calcium_decay').sigmoid()
    entry = gate if model.apl_calcium_entry_enabled else gate * 0.
    calcium = rho_calcium * calcium + (1. - rho_calcium) * entry
    probability = -torch.expm1(-spread('apl_release_log_rate').exp() * calcium)
    available = resources if model.apl_release_resource_enabled else torch.ones_like(resources)
    released = available * probability
    recovery = spread('apl_release_logit_resource_recovery').sigmoid()
    resources = resource_recovery(available, released, recovery)
    return released, gate, calcium, resources
