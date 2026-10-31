"""Anatomical DPM/APL axonal inhibition and DPM serotonin delivery.

Somatodendritic GABA already travels through the native inhibitory edge list.
Only its measured axonal share enters this additional terminal compartment.
Reconstructed contacts retain source identity, target identity and delay. No
population average changes a threshold or a forgetting constant. The model's
one axonal output per KC remains a spatial-resolution limit: it does not claim
to recover APL's within-neurite cable dynamics or individual KC bouton fields.
When fly_kc_terminals is attached, this pooled operator is inactive and its
historical physical arrays remain archived; local site chemistry drives release.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


MB_TERMINAL_PARAMETER_NAMES = (
    'mb_terminal_log_gaba_gain', 'mb_terminal_log_5ht_gain',
    'mb_dpm_log_tau', 'mb_dpm_log_affinity', 'mb_dpm_log_off',
)
MB_TERMINAL_VERSION = 1
MB_TERMINAL_ID_NAMES = ('mb_terminal_source_body_ids', 'mb_terminal_kc_body_ids')
MB_TERMINAL_STATE_FIELDS = ('mb_terminal_gaba', 'mb_dpm_concentration', 'mb_dpm_receptor')


def attach_mb_terminal(model, graph_npz, annotations, shares, *, saved=None):
    """Add measured axonal contacts, leaving the old fast edges/moments alone."""
    if hasattr(model, 'mb_terminal_kc_index'):
        return dict(model.mb_terminal_manifest)
    if model.synapse_model != 'coba':
        raise ValueError('MB terminal conductances require native COBA physics')
    graph = np.load(Path(graph_npz), allow_pickle=False)
    ids = graph['neuron_body_ids']
    ann = pd.read_feather(annotations, columns=['bodyId', 'type', 'class']).set_index('bodyId').reindex(ids)
    types = ann['type'].fillna('').to_numpy()
    kc = np.flatnonzero(ann['class'].fillna('').to_numpy() == 'Kenyon_Cell')
    dpm, apl = np.flatnonzero(types == 'DPM'), np.flatnonzero(types == 'APL')
    sources = np.concatenate((dpm, apl))
    if not (kc.size and dpm.size and apl.size):
        raise ValueError('Annotated KC, DPM and APL populations are required')
    index = np.full(ids.size, -1, dtype=np.int64)
    index[kc] = np.arange(kc.size)
    pre, post = graph['edge_pre_i'].astype(np.int64), graph['edge_post_i'].astype(np.int64)
    selected = np.isin(pre, sources) & (index[post] >= 0)
    pre, post = pre[selected], post[selected]
    weight = np.abs(graph['edge_weight_i'][selected]).astype(np.float32)
    delay = graph['edge_delay_i'][selected].astype(np.int64)
    evidence = pd.read_feather(shares).set_index(['body_pre', 'body_post'])
    pairs = pd.MultiIndex.from_arrays((ids[pre], ids[post]), names=['body_pre', 'body_post'])
    row = evidence.reindex(pairs)
    # Unknown localization is not silently assigned to an axon. For known
    # contacts, dendritic + axonal shares exactly sum to the original weight.
    valid = row['total'].notna().to_numpy() & (row['total'].fillna(0).to_numpy() > 0)
    fraction = np.zeros(pre.size, dtype=np.float32)
    fraction[valid] = (1. - row['dendritic'].to_numpy()[valid] / row['total'].to_numpy()[valid])
    if np.any((fraction < 0) | (fraction > 1)):
        raise ValueError('Invalid axonal/dendritic contact split')
    active = (fraction > 0) & (weight > 0)
    source_slot = np.full(ids.size, -1, dtype=np.int64)
    source_slot[sources] = np.arange(sources.size)
    slots = source_slot[pre[active]]
    target = index[post[active]]
    weight, delay, pre = weight[active] * fraction[active], delay[active], pre[active]
    if np.any((delay < 1) | (delay > 4)):
        raise ValueError('Axonal contact delay exceeds the actual native pulse ring')
    device = model.edge_pre_i.device

    def buffer(name, value, dtype=torch.long, persistent=False):
        model.register_buffer(name, torch.as_tensor(value, dtype=dtype, device=device), persistent=persistent)

    buffer('mb_terminal_kc_index', kc)
    buffer('mb_terminal_source_index', sources)
    buffer('mb_terminal_source_body_ids', ids[sources], persistent=True)
    buffer('mb_terminal_kc_body_ids', ids[kc], persistent=True)
    for name in MB_TERMINAL_ID_NAMES:
        if saved is not None and name in saved and not torch.equal(getattr(model, name).cpu(), saved[name].cpu()):
            raise ValueError('MB terminal anatomical identity mismatch: '+name)
        if saved is not None and any(k in saved for k in MB_TERMINAL_PARAMETER_NAMES) and name not in saved:
            raise ValueError('MB terminal archive is missing anatomical identities')
    n_dpm = int(dpm.size)
    is_dpm = slots < n_dpm
    if not np.any(is_dpm):
        raise ValueError('No measured DPM axonal contacts; cannot invent a serotonin route')
    denominator = np.bincount(target[is_dpm], weights=weight[is_dpm], minlength=kc.size)
    serotonin_weight = np.zeros(weight.size, dtype=np.float32)
    serotonin_weight[is_dpm] = weight[is_dpm] / np.maximum(denominator[target[is_dpm]], 1e-12)
    for lag in range(1, 5):
        sel = delay == lag
        buffer(f'mbt_pre_{lag}', pre[sel])
        buffer(f'mbt_post_{lag}', slots[sel] * kc.size + target[sel])
        buffer(f'mbt_gaba_weight_{lag}', weight[sel], torch.float32)
        # All slots share a layout; APL entries have zero serotonergic weight.
        buffer(f'mbt_5ht_weight_{lag}', serotonin_weight[sel], torch.float32)
    model.mb_terminal_log_gaba_gain = nn.Parameter(torch.zeros(kc.size, device=device))
    model.mb_terminal_log_5ht_gain = nn.Parameter(torch.zeros(kc.size, device=device))
    # Reuse the trained 5-HT kinetics as initialization, then allow DPM's own
    # clearance times to learn. No upper lifetime or biological-ms clamp.
    def previous(name, default):
        value = getattr(model, name, None)
        return (value.detach().clone() if value is not None else torch.tensor(default, device=device))
    model.mb_dpm_log_tau = nn.Parameter(previous('amine_log_clearance_5ht', float(np.log(120.))).expand(n_dpm).clone())
    model.mb_dpm_log_affinity = nn.Parameter(previous('amine_log_affinity_5ht', float(np.log(.5))))
    model.mb_dpm_log_off = nn.Parameter(previous('amine_log_off_5ht', float(np.log(18.))))
    if saved is not None:
        entries = {k:saved[k] for k in MB_TERMINAL_PARAMETER_NAMES if k in saved}
        if entries and len(entries) != len(MB_TERMINAL_PARAMETER_NAMES):
            raise ValueError('Partial MB terminal parameter archive')
        model.load_state_dict(entries, strict=False)
    model.mb_terminal_n_dpm = n_dpm
    model.mb_terminal_enabled = model.mb_terminal_gaba_enabled = model.mb_dpm_5ht_enabled = True
    model.mb_pooled_quarantined = True
    model.mb_terminal_manifest = {
        'dpm_cells':n_dpm, 'apl_cells':int(apl.size), 'kc_targets':int(kc.size),
        'axonal_gaba_contacts':int(pre.size), 'dpm_5ht_contacts':int(is_dpm.sum()),
        'source_body_ids':ids[sources].tolist(),
        'unknown_localization_contacts_not_added':int((~valid).sum()),
        'scope':'individual source/target axonal output; sub-bouton spatial fields not reconstructed'}
    return dict(model.mb_terminal_manifest)


def step_mb_terminal(model, current, *, base_rates=None, conductance_gains=None):
    """Advance actual contact chemistry and return the per-KC release fraction."""
    fields = [getattr(current, name, current.h.new_empty(0)) for name in MB_TERMINAL_STATE_FIELDS]
    if not getattr(model, 'mb_terminal_enabled', False):
        return None, *fields
    gaba, concentration, receptor = fields
    nk, ns = model.mb_terminal_kc_index.numel(), model.mb_terminal_source_index.numel()
    nd = model.mb_terminal_n_dpm
    def initialize_empty(value, shape, label):
        if value.shape == shape:
            return value
        if value.numel():
            raise ValueError(f'Nonempty MB terminal {label} state has incompatible shape')
        return current.h.new_zeros(shape)

    gaba = initialize_empty(gaba, (ns,nk), 'GABA')
    concentration = initialize_empty(concentration, (nd,nk), '5HT concentration')
    receptor = initialize_empty(receptor, (1,nk), '5HT receptor')
    incoming_gaba = current.h.new_zeros((ns,nk))
    incoming_5ht = current.h.new_zeros((ns,nk))
    for lag in range(1,5):
        pre = getattr(model, f'mbt_pre_{lag}')
        release = current.ring[lag-1].index_select(1,pre)[0]
        if hasattr(model, 'apl_cable_index'):
            # APL now releases at local cable sites; never count its old pooled
            # readout packet as a second physical source.
            release = release * (~torch.isin(pre, model.apl_cable_index)).to(release.dtype)
        posts = getattr(model, f'mbt_post_{lag}')
        incoming_gaba.view(-1).index_add_(0,posts,release*getattr(model,f'mbt_gaba_weight_{lag}'))
        incoming_5ht.view(-1).index_add_(0,posts,release*getattr(model,f'mbt_5ht_weight_{lag}'))
    if hasattr(model, 'apl_cable_index'):
        from .fly_apl_cable import apl_mb_arrival
        incoming_gaba = incoming_gaba + apl_mb_arrival(model, current).reshape_as(incoming_gaba)
    rates = base_rates if base_rates is not None else model.get_decay_rates()
    kc = model.mb_terminal_kc_index
    rho_i = rates[2].expand_as(current.h).index_select(1,kc)
    if getattr(model,'mb_terminal_gaba_enabled',True):
        gaba = rho_i*gaba+(1.-rho_i)*incoming_gaba
    if getattr(model,'mb_dpm_5ht_enabled',True):
        alpha = -torch.expm1(-torch.exp(-model.mb_dpm_log_tau)).unsqueeze(1)
        concentration = concentration+alpha*(incoming_5ht[:nd]-concentration)
        total = concentration.sum(dim=0,keepdim=True)
        affinity = torch.exp(model.mb_dpm_log_affinity)
        from .fly_biophysical_operators import ligand_binding
        receptor = ligand_binding(receptor,total,affinity,torch.exp(-model.mb_dpm_log_off),clamp=True)
    gains = conductance_gains if conductance_gains is not None else model.get_conductance_gains()
    g_i = gains[1].expand_as(current.h).index_select(1,kc)
    leak = -torch.log(rates[0].expand_as(current.h).index_select(1,kc).clamp(1e-5,1.-1e-7))
    fraction = current.h.new_ones((1,nk))
    if getattr(model,'mb_terminal_gaba_enabled',True):
        fraction = leak/(leak+g_i*torch.exp(model.mb_terminal_log_gaba_gain)*gaba.sum(0,keepdim=True))
    if getattr(model,'mb_dpm_5ht_enabled',True):
        fraction = fraction*torch.exp(-torch.exp(model.mb_terminal_log_5ht_gain)*receptor)
    return fraction,gaba,concentration,receptor


def apply_mb_terminal_release(model, ring, vesicles, fraction):
    """Gate each newly emitted packet ONCE; keep past packets and vesicles exact."""
    if fraction is None:
        return ring,vesicles
    kc = model.mb_terminal_kc_index
    packet = ring[0].index_select(1,kc)
    released = ring[0].index_copy(1,kc,packet*fraction)
    if model.use_stp:
        # Inhibited vesicles did not fuse. Retain them instead of silently
        # consuming the pre-gate packet from the presynaptic resource pool.
        vesicles = vesicles.index_add(1,kc,packet*(1.-fraction))
    return (released,*ring[1:]),vesicles


def quarantine_mb_pooled_credit(learner):
    """Archive obsolete proxy derivatives, preserving their real-world history."""
    if not getattr(learner.model,'mb_pooled_quarantined',False):
        return False
    from .fly_mb_consolidation import MB_CONSOLIDATION_PARAMETER_NAMES
    consequence = getattr(learner,'consequence_state',None)
    if not isinstance(consequence,dict) or not consequence.get('native_credit'):
        return False
    credit=consequence['native_credit']
    traces={k:credit.get(k) for k in ('E','response_score')}
    traces.update({'deferred:'+k:v for k,v in credit.get('deferred_response_scores',{}).items()})
    archived={}
    for key,trace in traces.items():
        if trace is None:
            continue
        removed={k:v.detach().cpu().clone() for k,v in trace.items() if k in MB_CONSOLIDATION_PARAMETER_NAMES}
        if removed:
            kept={k:v for k,v in trace.items() if k not in removed}
            if key.startswith('deferred:'):
                credit['deferred_response_scores'][key[9:]]=kept
            else:
                credit[key]=kept
            archived[key]=removed
    if archived:
        consequence.setdefault('quarantined_mb_pooled_credit',[]).append({
            'clock_ticks':credit.get('clock_ticks'),'traces':archived,
            'reason':'Pooled DPM/APL threshold/forgetting proxy replaced; experience preserved'})
    return bool(archived)
