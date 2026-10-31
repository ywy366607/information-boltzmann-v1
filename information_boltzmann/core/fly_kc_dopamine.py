"""DAN contacts drive local KC receptor signals on the full supplied morphology.

The transported quantity is an effective postsynaptic receptor-driving signal,
not extracellular dopamine diffusion or an identified molecular receptor.
Source release remains the existing spike/terminal-feedback mechanism.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import torch
from torch import nn

from .fly_apl_cable import diffuse
from .fly_kc_terminals import _sum

KC_DOPAMINE_VERSION = 1
KC_DOPAMINE_PARAMETER_NAMES = ('kc_dopamine_log_transport',)
KC_DOPAMINE_ID_NAMES = ('kc_dopamine_digest',)
KC_DOPAMINE_STATE_FIELDS = ('kc_dopamine_signal', 'kc_dopamine_credit')


def attach_kc_dopamine(model, path, *, saved=None):
    if not getattr(model, 'kc_contact_memory_enabled', False):
        raise ValueError('Spatial KC teacher requires actual plastic contacts')
    if getattr(model, 'dopamine_sensitivity', None) is None:
        raise ValueError('Spatial KC teacher requires the existing dopamine clocks and source route')
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    device = model.kct_index.device
    def buf(name, value, dtype=torch.long, persistent=False):
        model.register_buffer(name, torch.as_tensor(value, device=device, dtype=dtype), persistent=persistent)
    digest = np.frombuffer(hashlib.sha256(path.read_bytes()).digest(), dtype=np.uint8).copy()
    if saved is not None and ('kc_dopamine_digest' not in saved or not torch.equal(
            saved['kc_dopamine_digest'].cpu(), torch.from_numpy(digest))):
        raise ValueError('DAN-to-KC geometry identity mismatch')
    if not np.array_equal(data['kc_index'], model.kct_index.cpu().numpy()):
        raise ValueError('DAN target KC order mismatch')
    if len(data['node_ids']) != model.kct_owner.numel():
        raise ValueError('DAN target morphology size mismatch')
    if not np.array_equal(model.dan_edge_pre.cpu().numpy()[data['edge']], data['pre']) or not np.array_equal(
            model.dan_edge_post.cpu().numpy()[data['edge']], data['post']):
        raise ValueError('DAN graph edge identity/order mismatch')
    counts = np.diff([int(v) for v in model.dan_delay_splits])
    delays = np.repeat(np.arange(1,5),counts)
    if not np.array_equal(delays[data['edge']],data['delay']):
        raise ValueError('DAN physical transmission delay mismatch')
    if not np.isfinite(data['fraction']).all() or np.any(data['fraction']<=0):
        raise ValueError('Invalid DAN physical contact fractions')
    manifest = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    if manifest['kc_geometry_sha256'] != bytes(model.kc_terminal_digest.cpu().tolist()).hex():
        raise ValueError('DAN contacts belong to a different KC morphology')
    if not np.array_equal(model.kct_index.cpu().numpy()[model.kct_owner.cpu().numpy()[data['node']]], data['post']):
        raise ValueError('DAN contact crosses a KC cell boundary')
    unique = np.unique(data['edge'])
    expected = np.flatnonzero(np.isin(model.dan_edge_post.cpu().numpy(), data['kc_index']))
    if not np.array_equal(unique, expected):
        raise ValueError('Missing active DAN-to-KC graph edges')
    total = np.bincount(data['edge'], weights=data['fraction'], minlength=model.dan_edge_pre.numel())
    if not np.allclose(total[expected], 1., rtol=0, atol=1e-7):
        raise ValueError('DAN physical contact fractions do not conserve graph edges')
    buf('kc_dopamine_digest', digest, torch.uint8, True)
    for lag in range(1,5):
        pick = data['delay'] == lag
        prefix = f'kcdop_{lag}'
        for key in ('node','pre','edge','fraction'):
            buf(prefix+'_'+key, data[key][pick], torch.float32 if key == 'fraction' else torch.long)
        target = data['node'][pick]
        order = np.argsort(target, kind='stable')
        nodes, count = np.unique(target[order], return_counts=True)
        for key, value in [('order',order),('target',nodes),('count',count)]:
            buf(prefix+'_'+key,value)
    model.kc_dopamine_log_transport = nn.Parameter(model.kc_terminal_log_diffusion[1].detach().clone())
    if saved is not None:
        model.load_state_dict({n:saved[n] for n in KC_DOPAMINE_PARAMETER_NAMES}, strict=False)
    model.kc_dopamine_enabled = model.kc_dopamine_delivery_enabled = model.kc_dopamine_teacher_enabled = True
    model.kc_dopamine_manifest = manifest


def dopamine_state(model, current):
    signal, credit = [getattr(current,n) for n in KC_DOPAMINE_STATE_FIELDS]
    count = model.kct_owner.numel()
    if signal.numel() == 0:
        if credit.numel():
            raise ValueError('Incomplete spatial dopamine history')
        target = model.kct_index.index_select(0, model.kct_owner)
        signal = current.dopamine.index_select(1,target)
        credit = (current.credit_dopamine.index_select(1,target) if current.credit_dopamine.numel()
                  else signal.new_zeros(3,count))
    if signal.shape != (2,count) or credit.shape != (3,count):
        raise ValueError('Malformed spatial dopamine signal/history')
    return signal, credit


def spatial_arrival(model, ring, terminal_release=None, *, dan_values=None):
    if dan_values is not None and model.dan_local_source_enabled:
        from .fly_dan_release import dan_arrival
        result = dan_arrival(model,dan_values,'kc')
        if not model.kc_dopamine_delivery_enabled:
            result = result*0.
        return result*model.kct_density
    out = ring[0].new_zeros(model.kct_owner.numel())
    for lag in range(1,5):
        prefix = f'kcdop_{lag}'
        pre, edge = getattr(model,prefix+'_pre'), getattr(model,prefix+'_edge')
        amount = ring[lag-1][0].index_select(0,pre)
        if terminal_release is not None:
            amount = amount + terminal_release[0].index_select(0,pre)
        weight = model.dan_edge_weight.index_select(0,edge) / model.dan_scale
        out = out + _sum(model,prefix,amount*weight*getattr(model,prefix+'_fraction'),out.numel())
    if not model.kc_dopamine_delivery_enabled:
        out = out * 0.
    # Normalize by native compartment volume. Weighted cell averages reproduce
    # the original graph arrivals; spatial peaks are not arbitrary extra gain.
    return out.unsqueeze(0)*model.kct_density


def advance_kc_dopamine(model, current, ring, terminal_release=None, *, dan_values=None):
    before, history = dopamine_state(model,current)
    arrival = spatial_arrival(model,ring,terminal_release,dan_values=dan_values)
    fast, slow = torch.sigmoid(model.dopamine_logit_fast), torch.sigmoid(model.dopamine_logit_slow)
    material = model.kc_dopamine_log_transport
    geometry = model.kct_geometries[0]
    # Same full-tree solve, two RHS together: concentration and its conditional
    # clock sensitivity. Avoid a second 4M-node traversal/host transfer.
    with torch.no_grad():
        direct = fast.detach()*(1-fast.detach())*(before[:1].detach()-arrival.detach())
        clock_rhs = fast.detach()*history[:1]+direct
    transported = diffuse(torch.cat((fast*before[:1]+(1-fast)*arrival,clock_rhs),0),material,geometry)
    concentration = transported[:1]
    signal = torch.cat((concentration,slow*before[1:]+(1-slow)*concentration),0)
    with torch.no_grad():
        sf = transported[1:2].detach()
        slf = slow.detach()*history[1:2]+(1-slow.detach())*sf
        sls = slow.detach()*history[2:3]+slow.detach()*(1-slow.detach())*(before[1:].detach()-concentration.detach())
        history = torch.cat((sf,slf,sls),0)
    return signal, history


def contact_teacher(model, values):
    nodes = model.kct_site_node.index_select(0,model.kcm_site)
    return tuple(value.index_select(1,nodes) for value in values)
