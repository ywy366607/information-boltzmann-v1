"""Axo-axonal KC mAChR-B-like inhibition of release and cAMP teaching.

Actual delayed terminal packets drive receptors at actual recipient positions.
The intracellular inhibitory effector uses the unchanged full KC forest. This
is an effective Gi/o pathway, not identified receptor density or a VGCC cable.
Fast somatic KC weights and their calyx restriction are never modified here.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .fly_apl_cable import diffuse
from .fly_biophysical_operators import ligand_binding


KC_LATERAL_VERSION = 1
KC_LATERAL_PARAMETER_NAMES = ('kc_lateral_log_affinity', 'kc_lateral_log_off',
    'kc_lateral_logit_decay', 'kc_lateral_log_transport',
    'kc_lateral_log_calcium_gain', 'kc_lateral_log_camp_gain')
KC_LATERAL_STATE_FIELDS = ('kc_lateral_receptor', 'kc_lateral_signal')
KC_LATERAL_ID_NAMES = ('kc_lateral_digest',)


def initial_parameters(count, *, device=None):
    rho = math.exp(-1. / 8.)
    defaults = (0., 0., math.log(rho / (1 - rho)), 0., 0., 0.)
    return {name: torch.full((count,), value, device=device)
            for name, value in zip(KC_LATERAL_PARAMETER_NAMES, defaults)}


def attach_kc_lateral(model, path, *, saved=None):
    if not getattr(model, 'kc_contact_memory_enabled', False):
        raise ValueError('KC lateral modulation needs native terminal/contact coordinates')
    path = Path(path)
    manifest = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    if manifest['kc_geometry_sha256'] != bytes(model.kc_terminal_digest.cpu().tolist()).hex():
        raise ValueError('KC lateral contacts belong to a different morphology')
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    if not np.array_equal(data['kc_index'], model.kct_index.cpu().numpy()):
        raise ValueError('KC lateral cell order mismatch')
    edge, site, node = data['edge'], data['site'], data['node']
    source = model.kct_index.cpu().numpy()[model.kct_site_owner.cpu().numpy()[site]]
    target = model.kct_index.cpu().numpy()[model.kct_owner.cpu().numpy()[node]]
    if not np.array_equal(model.edge_pre_e.cpu().numpy()[edge], source) or not np.array_equal(
            model.edge_post_e.cpu().numpy()[edge], target):
        raise ValueError('KC lateral native endpoint identity mismatch')
    if np.any(data['count'] <= 0) or np.any((data['delay'] < 1) | (data['delay'] > 4)):
        raise ValueError('Invalid local contact mass or physical delay')
    digest = torch.tensor(list(hashlib.sha256(path.read_bytes()).digest()), dtype=torch.uint8)
    if saved is not None and ('kc_lateral_digest' not in saved or not torch.equal(saved['kc_lateral_digest'].cpu(), digest)):
        raise ValueError('KC lateral contact digest mismatch')
    device = model.kct_index.device
    model.register_buffer('kc_lateral_digest', digest.to(device), persistent=True)
    for lag in range(1, 5):
        selected = data['delay'] == lag
        targets = node[selected]
        order = np.argsort(targets, kind='stable')
        unique, count = np.unique(targets[order], return_counts=True)
        for key, value, dtype in (
            ('site', site[selected], torch.long), ('order', order, torch.long),
            ('target', unique, torch.long), ('count', count, torch.long),
            ('mass', data['count'][selected] / model.kct_geometries[0].area[targets], torch.float32)):
            model.register_buffer(f'kcl_{lag}_{key}', torch.as_tensor(value, device=device, dtype=dtype), persistent=False)
    for name, value in initial_parameters(model.kct_index.numel(), device=device).items():
        if saved is not None:
            if name not in saved or saved[name].shape != value.shape or not torch.isfinite(saved[name]).all():
                raise ValueError('Incomplete KC lateral physiological archive: ' + name)
            value = saved[name].to(value).clone()
        model.register_parameter(name, nn.Parameter(value))
    model.kc_lateral_enabled = True
    model.kc_lateral_delivery_enabled = True
    model.kc_lateral_calcium_enabled = True
    model.kc_lateral_camp_enabled = True
    model.kc_lateral_manifest = manifest


def lateral_state(model, current):
    values = [getattr(current, name) for name in KC_LATERAL_STATE_FIELDS]
    shape = (1, model.kct_owner.numel())
    if not values[0].numel():
        if values[1].numel():
            raise ValueError('Incomplete KC lateral receptor/effector history')
        values = [current.h.new_zeros(shape), current.h.new_zeros(shape)]
    if any(value.shape != shape for value in values):
        raise ValueError('Malformed KC lateral receptor/effector history')
    return values


def advance_kc_lateral(model, current, packets):
    receptor, signal = lateral_state(model, current)
    arrival = receptor.new_zeros(receptor.shape[1])
    for lag in range(1, 5):
        site = getattr(model, f'kcl_{lag}_site')
        if not site.numel():
            continue
        value = packets[lag - 1, site] * getattr(model, f'kcl_{lag}_mass')
        value = torch.segment_reduce(value[getattr(model, f'kcl_{lag}_order')], 'sum',
                                     lengths=getattr(model, f'kcl_{lag}_count'))
        arrival.index_add_(0, getattr(model, f'kcl_{lag}_target'), value)
    if not model.kc_lateral_delivery_enabled:
        arrival = arrival * 0.
    owner = model.kct_owner
    affinity = model.kc_lateral_log_affinity.exp()[owner][None]
    off = model.kc_lateral_log_off.exp()[owner][None]
    receptor = ligand_binding(receptor, arrival[None], affinity, off)
    rho = model.kc_lateral_logit_decay.sigmoid()[owner][None]
    signal = diffuse(rho * signal + (1 - rho) * receptor,
                     model.kc_lateral_log_transport, model.kct_geometries[0])
    return receptor, signal


def effector_gate(model, signal, sites, kind):
    if not getattr(model, f'kc_lateral_{kind}_enabled'):
        return signal.new_ones(1, sites.numel())
    owner = model.kct_site_owner[sites]
    strength = getattr(model, f'kc_lateral_log_{kind}_gain').exp()[owner][None]
    local = signal.index_select(1, model.kct_site_node[sites])
    return torch.exp(-strength * local)
