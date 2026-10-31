"""Sensory axon terminals: actual axo-axonal return contacts gate local release.

A sensory neuron's dendrites are peripheral, so every connectome synapse onto
it sits on its central axon terminals (FLY_CONNECTOME_AUDIT_20261012): it
modulates release, it does not make the cell fire. These 486k pairs were
dropped from the point graph (coba_r2 no_sensory_input) and pinned at zero.

Here they act where they are. Each actual contact delivers its source's
delayed packet (an external cell's pulse, or another sensory cell's packet at
its own actual release site) to its actual node on the supplied morphology,
one synapse of conductance in the coba_r2 unit. Excitatory and inhibitory
terminal conductances decay on the cell's own synaptic clocks and spread over
the full forest (every node kept; learnable per-cell transport).

Release at each actual site is shunted by the local open conductance,
    fraction = g_L / (g_L + g_E * s_E + g_I * s_I),
the same conductance algebra as the soma, with the cell's own leak and gains
(per-cell learnable terminal gains start at 1). With infinite transport the
terminal sees exactly the conductance the point model would have given the
soma. Mechanism hypothesis: presynaptic inhibition of fly sensory terminals is
GABAergic (Olsen & Wilson 2008; Root et al. 2008); that depolarizing
(cholinergic) terminal conductance also reduces release by shunting the
invading spike is assumed, not measured. Facilitation and receptor subtypes
are not modelled. Amine/DA contacts are excluded by the builder.

The cell's outputs leave from its actual sites: each live model edge is split
evenly over its actual contacts (contacts == synapses), weighted by the
edge's current learned weight, so with no terminal input the delivery equals
the point-cell delivery up to summation order.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .fly_apl_cable import diffuse

SENSORY_TERMINAL_VERSION = 1
SENSORY_TERMINAL_PARAMETER_NAMES = ('sensory_terminal_log_transport', 'sensory_terminal_log_gain')
SENSORY_TERMINAL_STATE_FIELDS = ('sensory_terminal_signal', 'sensory_terminal_packets')
SENSORY_TERMINAL_ID_NAMES = ('sensory_terminal_digest',)


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            digest.update(block)
    return digest.digest()


def initial_parameters(cells, *, device=None):
    return {name: torch.zeros(2, cells, device=device) for name in SENSORY_TERMINAL_PARAMETER_NAMES}


def _segments(target):
    order = np.argsort(target, kind='stable')
    unique, count = np.unique(np.asarray(target)[order], return_counts=True)
    return order, unique, count


def attach_sensory_terminals(model, path, *, saved=None):
    path = Path(path)
    digest = torch.tensor(list(file_digest(path)), dtype=torch.uint8)
    if saved is not None and 'sensory_terminal_log_gain' in saved:
        if 'sensory_terminal_digest' not in saved or not torch.equal(saved['sensory_terminal_digest'].cpu(), digest):
            raise ValueError('Sensory terminal anatomy differs from the archived individual')
    with np.load(path, allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    device = model.edge_pre_e.device
    for channel in ('e', 'i'):
        edge = torch.as_tensor(data[f'verify_{channel}_edge'].astype(np.int64), device=device)
        for end in ('pre', 'post'):
            actual = getattr(model, f'edge_{end}_{channel}').index_select(0, edge).cpu().numpy()
            if not np.array_equal(actual, data[f'verify_{channel}_{end}']):
                raise ValueError('Sensory terminal contacts do not match the actual edge order')
    cells = data['cell_index'].astype(np.int64)
    pre = {c: getattr(model, f'edge_pre_{c}').cpu().numpy() for c in 'ei'}
    slot = np.full(model.n_neurons, -1, np.int64)
    slot[cells] = np.arange(len(cells))
    for c in 'ei':
        live = np.flatnonzero(slot[pre[c]] >= 0)
        if not np.array_equal(np.sort(live), np.sort(data[f'verify_{c}_edge'].astype(np.int64))):
            raise ValueError('A native sensory output edge has no actual release site')
    if np.any(data['out_share'] <= 0) or np.any(data['in_weight'] <= 0):
        raise ValueError('Invalid contact share or synaptic unit')
    for key in ('in_delay', 'out_delay'):
        if np.any((data[key] < 1) | (data[key] > 4)):
            raise ValueError('Invalid physical delay')

    def buf(name, value, dtype=torch.long):
        model.register_buffer(name, torch.as_tensor(np.asarray(value), dtype=dtype, device=device), persistent=False)

    model.register_buffer('sensory_terminal_digest', digest.to(device), persistent=True)
    n_nodes = len(data['volume'])
    buf('st_cell', cells)
    buf('st_owner', data['owner'])
    buf('st_density', data['density'], torch.float32)
    buf('st_site_node', data['site_node'])
    buf('st_site_owner', data['site_owner'])
    buf('st_site_cell', cells[data['site_owner']])
    for lag in range(1, 5):
        sel = data['in_delay'] == lag
        native = sel & (data['in_site'] >= 0)
        external = sel & (data['in_site'] < 0)
        for label, chosen, source in (('ext', external, data['in_source']), ('nat', native, data['in_site'])):
            target = data['in_kind'][chosen].astype(np.int64) * n_nodes + data['in_node'][chosen]
            order, unique, count = _segments(target)
            buf(f'st_in_{label}_{lag}_source', source[chosen])
            buf(f'st_in_{label}_{lag}_weight', data['in_weight'][chosen], torch.float32)
            buf(f'st_in_{label}_{lag}_order', order)
            buf(f'st_in_{label}_{lag}_target', unique)
            buf(f'st_in_{label}_{lag}_count', count)
        for c, channel in enumerate('ei'):
            chosen = (data['out_delay'] == lag) & (data['out_channel'] == c)
            edge = data['out_edge'][chosen].astype(np.int64)
            post = getattr(model, f'edge_post_{channel}').cpu().numpy()[edge]
            order, unique, count = _segments(post)
            buf(f'st_out_{channel}_{lag}_site', data['out_site'][chosen])
            buf(f'st_out_{channel}_{lag}_edge', edge)
            buf(f'st_out_{channel}_{lag}_share', data['out_share'][chosen], torch.float32)
            buf(f'st_out_{channel}_{lag}_order', order)
            buf(f'st_out_{channel}_{lag}_target', unique)
            buf(f'st_out_{channel}_{lag}_count', count)
    from .fly_kc_terminals import CellwiseGeometry
    model.st_geometry = CellwiseGeometry(data['volume'], data['u'], data['v'], data['g'], data['owner'])
    for name, value in initial_parameters(len(cells), device=device).items():
        if saved is not None and 'sensory_terminal_log_gain' in saved:
            if name not in saved or saved[name].shape != value.shape or not torch.isfinite(saved[name]).all():
                raise ValueError('Incomplete sensory terminal physiological archive: ' + name)
            value = saved[name].to(value).clone()
        model.register_parameter(name, nn.Parameter(value))
    manifest = path.with_suffix('.json')
    model.sensory_terminal_manifest = json.loads(manifest.read_text(encoding='utf-8')) if manifest.exists() else {}
    model.sensory_terminal_enabled = True
    model.sensory_terminal_gating_enabled = True
    model.sensory_terminal_input_enabled = True
    model.sensory_terminal_transport_enabled = True
    return dict(model.sensory_terminal_manifest)


def terminal_values(model, current):
    """Persistent terminal conductances and site packets; legacy states start from the ring."""
    signal, packets = (getattr(current, name, current.h.new_empty(0)) for name in SENSORY_TERMINAL_STATE_FIELDS)
    shape = (2, model.st_owner.numel())
    if not signal.numel():
        if packets.numel():
            raise ValueError('Incomplete sensory terminal history')
        signal = current.h.new_zeros(shape)
        # Packets already in flight were released before sites existed:
        # each site carries its cell's pulse, exactly as the point graph had it.
        packets = torch.cat([p.index_select(1, model.st_site_cell) for p in current.ring], 0)
    if signal.shape != shape or packets.shape != (4, model.st_site_cell.numel()):
        raise ValueError('Malformed sensory terminal history')
    return signal, packets


def _reduce(model, prefix, value, size):
    order = getattr(model, prefix + '_order')
    if not order.numel():
        return value.new_zeros(size)
    sums = torch.segment_reduce(value.index_select(0, order), 'sum', lengths=getattr(model, prefix + '_count'))
    return value.new_zeros(size).index_copy(0, getattr(model, prefix + '_target'), sums)


def sensory_delivery(model, packets):
    """Delayed site packets -> conductance increments on the actual targets ([1, N] each)."""
    out = []
    for channel in ('e', 'i'):
        total = packets.new_zeros(model.n_neurons)
        weight = getattr(model, f'edge_weight_{channel}')
        for lag in range(1, 5):
            prefix = f'st_out_{channel}_{lag}'
            site = getattr(model, prefix + '_site')
            if not site.numel():
                continue
            value = (packets[lag - 1].index_select(0, site) * getattr(model, prefix + '_share')
                     * weight.index_select(0, getattr(model, prefix + '_edge')))
            total = total + _reduce(model, prefix, value, model.n_neurons)
        out.append(total.unsqueeze(0))
    return tuple(out)


def release_fraction(model, signal, base_rates, conductance_gains):
    """Local shunting of release at every actual site, from the cell's own conductance law."""
    cell = model.st_site_cell
    owner = model.st_site_owner
    reference = signal.new_zeros(1, model.n_neurons)
    leak = base_rates[0].expand_as(reference)[0].index_select(0, cell)
    g_leak = -torch.log(leak.clamp(1e-5, 1 - 1e-7))
    gains = torch.exp(model.sensory_terminal_log_gain).index_select(1, owner)
    g_e = conductance_gains[0].expand_as(reference)[0].index_select(0, cell) * gains[0]
    g_i = conductance_gains[1].expand_as(reference)[0].index_select(0, cell) * gains[1]
    local = signal.index_select(1, model.st_site_node)
    return g_leak / (g_leak + g_e * local[0] + g_i * local[1])


def advance_sensory_terminals(model, current, raw_packet, *, base_rates=None, conductance_gains=None):
    """OLD pulses/packets update terminal conductance; NEW sensory pulses leave from sites."""
    signal, packets = terminal_values(model, current)
    rates = base_rates if base_rates is not None else model.get_decay_rates()
    gains = conductance_gains if conductance_gains is not None else model.get_conductance_gains()
    n_nodes = model.st_owner.numel()
    arrivals = signal.new_zeros(2 * n_nodes)
    if model.sensory_terminal_input_enabled:
        for lag in range(1, 5):
            ext = f'st_in_ext_{lag}'
            source = getattr(model, ext + '_source')
            if source.numel():
                value = current.ring[lag - 1][0].index_select(0, source) * getattr(model, ext + '_weight')
                arrivals = arrivals + _reduce(model, ext, value, 2 * n_nodes)
            nat = f'st_in_nat_{lag}'
            site = getattr(model, nat + '_source')
            if site.numel():
                value = packets[lag - 1].index_select(0, site) * getattr(model, nat + '_weight')
                arrivals = arrivals + _reduce(model, nat, value, 2 * n_nodes)
    arrivals = arrivals.view(2, n_nodes) * model.st_density
    reference = signal.new_zeros(1, model.n_neurons)
    rho = torch.stack([rates[1].expand_as(reference)[0], rates[2].expand_as(reference)[0]], 0)
    rho = rho.index_select(1, model.st_cell).index_select(1, model.st_owner)
    signal = rho * signal + (1 - rho) * arrivals
    if model.sensory_terminal_transport_enabled:
        signal = torch.cat([diffuse(signal[k:k + 1], model.sensory_terminal_log_transport[k], model.st_geometry)
                            for k in range(2)], 0)
    raw = raw_packet[0].index_select(0, model.st_site_cell)
    if model.sensory_terminal_gating_enabled:
        raw = raw * release_fraction(model, signal, rates, gains)
    packets = torch.cat((raw.unsqueeze(0), packets[:3]), 0)
    return [signal, packets]
