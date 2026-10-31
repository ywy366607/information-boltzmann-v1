"""Persistent exact local-chemistry tangents, conditional on incoming packets.

Each cell owns its six materials. Transport stays within that cell, so one
row per material represents its own full spatial derivative without a dense
cell-by-cell Jacobian. Incoming KC/AP packets are held fixed for historical
credit; recurrent spike-mediated and earlier downstream history are not RTRL.
"""
import torch

from .fly_apl_cable import diffuse
from .fly_dan_release import DAN_RELEASE_PARAMETER_NAMES, DAN_RELEASE_STATE_FIELDS, dan_state, reciprocal_arrival

DAN_CREDIT_VERSION = 1
DAN_CREDIT_FIELDS = ('dan_credit_receptor', 'dan_credit_calcium',
                     'dan_credit_resources', 'dan_credit_packets')
DAN_CREDIT_PARAMETERS = DAN_RELEASE_PARAMETER_NAMES + ('log_kc_release',)
# Parameter rows: transport, affinity, off, calcium clock, release, recovery, gain.
RECEPTOR_ROWS = (1, 2)
CALCIUM_ROWS = (0, 1, 2, 3)


def credit_state(model, current):
    nodes, sites = model.danl_owner.numel(), model.danl_site_owner.numel()
    shapes = ((2, nodes), (4, nodes), (7, sites), (4, 7, sites))
    values = [getattr(current, name) for name in DAN_CREDIT_FIELDS]
    if not any(value.numel() for value in values):
        return [current.h.new_zeros(shape) for shape in shapes]
    if any(value.shape != shape for value, shape in zip(values, shapes)):
        raise ValueError('Incomplete or incompatible DAN carried-credit history')
    return values


@torch.no_grad()
def advance_credit(model, current, ring, kc_values, new, *, local_inputs=None):
    """Advance d(local states)/d(materials) once, without retaining a graph."""
    R, C, x, _ = [value.detach() for value in dan_state(model, current)]
    SR, SC, SX, SP = credit_state(model, current)
    owner, site_owner = model.danl_owner, model.danl_site_owner
    signal = reciprocal_arrival(model, [v.detach() for v in kc_values])
    if local_inputs and 'dan_receptor_input' in local_inputs:
        from .fly_native_runtime import local_input
        signal = signal + local_input(local_inputs, 'dan_receptor_input', signal).detach()
    affinity = model.dan_local_log_affinity.exp()[owner][None]
    off = model.dan_local_log_off.exp()[owner][None]
    equilibrium = signal / (signal + affinity)
    exponent = off * (1 + signal / affinity)
    decay = (-exponent).exp()
    SR1 = decay * SR + torch.cat((
        -(1-decay)*equilibrium*(1-equilibrium) + (R-equilibrium)*decay*off*signal/affinity,
        -(R-equilibrium)*decay*exponent), 0)
    rho = model.dan_local_logit_calcium_decay.sigmoid()[owner][None]
    R1, C1 = new[0].detach(), new[1].detach()
    z = rho*C + (1-rho)*R1
    tangent = rho * SC
    tangent[1:3] += (1-rho) * SR1
    tangent[3:4] += rho*(1-rho)*(C-R1)
    # A C1 = V z; dC1/dlogD = A^-1 V(C1-z). No new spatial approximation.
    tangent[0:1] += C1-z
    SC1 = diffuse(tangent, model.dan_local_log_transport.detach(), model.danl_geometry)
    gain = model.log_kc_release.exp()[model.superclass_id[model.danl_index]][site_owner][None]
    local = gain*C1[:, model.danl_site_node]
    drive = ring[0].detach()[:, model.danl_index][:, site_owner] + local
    rate = model.dan_local_log_release_rate.exp()[site_owner][None]
    survival = (-rate*drive).exp()
    probability = -torch.expm1(-rate*drive)
    dp = torch.zeros_like(SX)
    dp[:4] = rate*survival*gain*SC1[:, model.danl_site_node]
    dp[4:5] = rate*drive*survival
    dp[6:7] = rate*local*survival
    norm = (-torch.expm1(-rate)).clamp_min(torch.finfo(rate.dtype).tiny)
    SY = (probability*SX + x*dp)/norm
    SY[4:5] -= x*probability*rate*(-rate).exp()/norm.square()
    recovery = model.dan_local_logit_resource_recovery.sigmoid()[site_owner][None]
    SX1 = recovery*(survival*SX-x*dp)
    SX1[5:6] += (x*survival-1)*recovery*(1-recovery)
    return [SR1, SC1, SX1, torch.cat((SY[None], SP[:3]), 0)]


def carried_gradients(model, gradients, start):
    """Contract opening-state VJPs with saved tangents; never mutate .grad."""
    if not getattr(model, 'dan_carried_credit_enabled', False):
        return {}
    if not any(start.get(n) is not None and start[n].numel() for n in DAN_CREDIT_FIELDS):
        return {}
    if any(start.get(n) is None or not start[n].numel() for n in DAN_CREDIT_FIELDS):
        raise ValueError('Incomplete DAN opening credit')
    totals = model.dan_local_log_transport.new_zeros(7, model.danl_index.numel())
    for name, credit, rows, owner in zip(DAN_RELEASE_STATE_FIELDS, DAN_CREDIT_FIELDS,
            (RECEPTOR_ROWS, CALCIUM_ROWS, tuple(range(7)), tuple(range(7))),
            (model.danl_owner, model.danl_owner, model.danl_site_owner, model.danl_site_owner)):
        G = gradients.get(name)
        if G is None:
            continue
        S = start[credit]
        for row, parameter_row in enumerate(rows):
            value = ((G.detach()*S[:, row]).sum(0) if name == 'dan_local_packets'
                     else G.detach()[0]*S[row])
            totals[parameter_row].index_add_(0, owner, value)
    from .fly_dan_receiver_credit import contract_receiver_credit
    totals = totals + contract_receiver_credit(model, gradients, start)
    result = {name: totals[i].reshape(getattr(model, name).shape)
              for i, name in enumerate(DAN_RELEASE_PARAMETER_NAMES)}
    classes = model.superclass_id[model.danl_index]
    result['log_kc_release'] = torch.zeros_like(model.log_kc_release).index_add(0, classes, totals[6])
    return result
