"""Source-resolved historical credit through the existing global DA filters.

DA concentration is linear in incoming release at fixed filter coefficients.
An exact contribution per existing source-target edge therefore carries every
source's seven local material derivatives without a dense source-neuron array.
This does not transport these derivatives through the separate KC spatial field.
"""
import numpy as np
import torch

DAN_RECEIVER_CREDIT_VERSION = 1
DAN_RECEIVER_CREDIT_FIELD = 'dan_credit_dopamine'


def attach_receiver_credit(model):
    if not getattr(model, 'dan_carried_credit_enabled', False):
        raise ValueError('Receiver credit needs persistent local DAN source credit')
    inverse = torch.full((model.n_neurons,), -1, device=model.danl_index.device, dtype=torch.long)
    inverse[model.danl_index] = torch.arange(model.danl_index.numel(), device=inverse.device)
    owner = inverse[model.dan_edge_pre.long()]
    if bool((owner < 0).any()):
        raise ValueError('Receiver credit edge belongs to an unknown source')
    model.register_buffer('danrc_edge_owner', owner, persistent=False)
    for lag in range(1, 5):
        edge = getattr(model, f'danl_out_{lag}_edge').cpu().numpy()
        order = np.argsort(edge, kind='stable')
        target, count = np.unique(edge[order], return_counts=True)
        for name, value in (('order', order), ('target', target), ('count', count)):
            model.register_buffer(f'danrc_{lag}_{name}', torch.as_tensor(value, device=inverse.device), persistent=False)
    model.dan_receiver_credit_enabled = True


def receiver_state(model, current):
    value = getattr(current, DAN_RECEIVER_CREDIT_FIELD)
    shape = (2, 7, model.dan_edge_pre.numel())
    if not value.numel():
        return current.h.new_zeros(shape)
    if value.shape != shape:
        raise ValueError('Malformed source-resolved dopamine credit history')
    return value


@torch.no_grad()
def edge_arrival_credit(model, packets):
    """Use both actual release positions and original per-edge delays/weights."""
    result = packets.new_zeros(model.dan_edge_pre.numel(), 7)
    for lag in range(1, 5):
        route = f'danl_out_{lag}'
        site = getattr(model, route+'_site')
        order = getattr(model, f'danrc_{lag}_order')
        if not order.numel():
            continue
        edge = getattr(model, route+'_edge')
        weight = model.dan_edge_weight[edge]/model.dan_scale * getattr(model, route+'_fraction')
        values = (packets[lag-1, :, site] * weight).T.contiguous()
        reduced = torch.segment_reduce(values[order], 'sum', lengths=getattr(model, f'danrc_{lag}_count'), axis=0)
        result.index_add_(0, getattr(model, f'danrc_{lag}_target'), reduced)
    return result.T


@torch.no_grad()
def advance_receiver_credit(model, current, dan_values):
    history = receiver_state(model, current)
    if model.dan_local_source_enabled and dan_values is not None and len(dan_values) == 8:
        arrival = edge_arrival_credit(model, dan_values[7])
    else:
        arrival = torch.zeros_like(history[0])
    fast, slow = model.dopamine_logit_fast.sigmoid(), model.dopamine_logit_slow.sigmoid()
    concentration = fast*history[0] + (1-fast)*arrival
    baseline = slow*history[1] + (1-slow)*concentration
    return torch.stack((concentration, baseline))


def contract_receiver_credit(model, gradients, start):
    """Sum recipient-state VJPs by original source identity, not by target only."""
    result = model.dan_local_log_transport.new_zeros(7, model.danl_index.numel())
    if not getattr(model, 'dan_receiver_credit_enabled', False):
        return result
    S, G = start.get(DAN_RECEIVER_CREDIT_FIELD), gradients.get('dopamine')
    if S is None or not S.numel() or G is None:
        return result
    if S.shape != (2, 7, model.dan_edge_pre.numel()):
        raise ValueError('Malformed source-resolved opening dopamine credit')
    recipient = model.dan_edge_post.long()
    for row in range(7):
        value = (G.detach()[:, recipient] * S[:, row]).sum(0)
        result[row].index_add_(0, model.danrc_edge_owner, value)
    return result
