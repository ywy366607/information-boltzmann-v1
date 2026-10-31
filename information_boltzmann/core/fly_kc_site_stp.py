"""Release-site resource conservation and use-dependent facilitation.

Coordinates are the actual shared presynaptic positions, not partner edges.
This extends the model's dynamic-synapse resource law to native KC sites.
Unknown kinetics are learned in model ticks; no biological time cap is imposed.
It is not a reconstruction of molecular priming or individual vesicle counts.
"""
import math

import torch
from torch import nn


KC_SITE_STP_VERSION = 1
KC_SITE_STP_PARAMETER_NAMES = (
    'kc_site_logit_u0', 'kc_site_log_tau_fac', 'kc_site_log_tau_rec',
)
KC_SITE_STP_STATE_FIELDS = ('kc_site_resources', 'kc_site_utilization')


def attach_kc_site_stp(model, *, saved=None):
    if not hasattr(model, 'kct_site_owner'):
        raise ValueError('KC site resources require actual terminal geometry')
    if model.use_stp:
        raise ValueError('Refuse simultaneous global and site-level KC STP')
    count = model.kct_index.numel()
    defaults = (math.log(.25 / .75), math.log(100.), math.log(200.))
    if saved is not None and not all(n in saved for n in KC_SITE_STP_PARAMETER_NAMES):
        raise ValueError('Incomplete KC site STP parameter archive')
    for name, value in zip(KC_SITE_STP_PARAMETER_NAMES, defaults):
        tensor = torch.full((count,), value, device=model.kct_index.device)
        if saved is not None:
            if saved[name].shape != tensor.shape:
                raise ValueError('KC site STP parameter identity/shape mismatch')
            if not torch.isfinite(saved[name]).all():
                raise ValueError('Nonfinite KC site STP parameters')
            tensor.copy_(saved[name].to(tensor))
        model.register_parameter(name, nn.Parameter(tensor))
    model.kc_site_stp_enabled = True


def site_parameters(model):
    owner = model.kct_site_owner
    u0 = torch.sigmoid(model.kc_site_logit_u0).index_select(0, owner).unsqueeze(0)
    fac = torch.exp(-torch.exp(-model.kc_site_log_tau_fac)).index_select(0, owner).unsqueeze(0)
    rec = torch.exp(-torch.exp(-model.kc_site_log_tau_rec)).index_select(0, owner).unsqueeze(0)
    return u0, fac, rec


def site_state(model, current):
    values = [getattr(current, name) for name in KC_SITE_STP_STATE_FIELDS]
    if not hasattr(model, 'kc_site_logit_u0'):
        if any(v.numel() for v in values):
            raise ValueError('Nonempty KC site resources require their saved parameters')
        return values
    shape = (1, model.kct_site_owner.numel())
    if bool(values[0].numel()) != bool(values[1].numel()):
        raise ValueError('Incomplete KC site resource history')
    if not values[0].numel():
        return [current.h.new_ones(shape), site_parameters(model)[0].detach().clone()]
    if any(v.shape != shape for v in values):
        raise ValueError('Incompatible KC site resource history')
    return values


def release_site_resources(model, current, raw_packet, gate, spikes):
    """One real tick: spike-triggered use, bounded release, then recovery.

    The electrical packet is expressed in the prior model's isolated-spike
    efficacy units: divide by U*(2-U), its rested first-spike use. Physical
    resource loss is the unnormalized released fraction. This preserves the
    existing reference edge efficacy without creating or overdrawing resources.
    Local receptor inhibition reduces both transmission and actual resource use.
    """
    resources, utilization = site_state(model, current)
    if not getattr(model, 'kc_site_stp_enabled', False):
        return raw_packet * gate, resources, utilization
    if spikes is None:
        raise ValueError('KC site resource updates require native spike events')
    u0, rho_fac, rho_rec = site_parameters(model)
    event = spikes.index_select(1, model.kct_index).index_select(1, model.kct_site_owner)
    active = utilization + u0 * (1. - utilization) * event
    released = active * resources * gate * event
    from .fly_biophysical_operators import resource_recovery
    next_resources = resource_recovery(resources, released, rho_rec)
    next_utilization = u0 + (active - u0) * rho_fac
    norm = (u0 * (2. - u0)).clamp_min(torch.finfo(u0.dtype).tiny)
    packet = raw_packet * gate * active * resources / norm
    return packet, next_resources, next_utilization
