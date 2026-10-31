"""Native slow amine delivery for the fly connectome.

This is deliberately a small physical closure around the existing network:
annotated octopaminergic/serotonergic cells release into immutable anatomical
contacts, concentration relaxes, receptor occupancy follows mass action, and
occupancy changes the existing membrane threshold by a learnable cell-class
response.  It does not reopen the fast edges which were pinned by the
connectome audit and it does not write a reward or policy logit.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch import nn


AMINE_PARAMETER_NAMES = (
    "amine_log_clearance_oa", "amine_log_clearance_5ht",
    "amine_log_affinity_oa", "amine_log_affinity_5ht",
    "amine_log_off_oa", "amine_log_off_5ht",
    "amine_response_oa", "amine_response_5ht",
)
AMINE_KINETICS_VERSION = 2


def attach_amine_modulation(model, graph_npz: str | Path, modulatory_feather: str | Path,
                            *, tau_clearance=(80.0, 120.0), affinity=(0.5, 0.5),
                            tau_off=(12.0, 18.0), response_init=0.0,
                            transmitter_names=("octopamine", "serotonin")):
    """Attach a fixed anatomical delivery graph and learnable native response.

    The graph is built from the original edge arrays, before any fast
    modulatory edges are pinned to zero.  Contact weights are non-negative and
    normalized per target/channel, so concentration is a normalized local
    exposure rather than a claim about absolute molecule counts.

    The historical ``log_clearance`` and ``log_off`` parameter names store log
    *time constants*, in real graph ticks. Rates are exp(-parameter). Version 1
    accidentally used the reciprocal kinetics; enabled v1 archives require an
    explicit repair, never a silent reinterpretation of their learned state.
    """
    for name, values in (("tau_clearance", tau_clearance), ("tau_off", tau_off),
                         ("affinity", affinity)):
        if len(values) != 2 or any(not math.isfinite(float(v)) or float(v) <= 0 for v in values):
            raise ValueError(f'{name} must contain two finite positive values')
    if not math.isfinite(float(response_init)):
        raise ValueError('response_init must be finite')
    if getattr(model, "amine_response_oa", None) is not None:
        return int(model.amine_target_count)
    import pandas as pd

    graph = np.load(graph_npz, allow_pickle=False)
    table = pd.read_feather(modulatory_feather)
    ids = graph["neuron_body_ids"]
    body_by_channel = []
    for nt in transmitter_names:
        body_by_channel.append({int(b) for b in table[table.consensusNt == nt].bodyId})

    if "edge_pre_e" in graph:
        edges = [(graph["edge_pre_e"], graph["edge_post_e"], np.abs(graph["edge_weight_e"])),
                 (graph["edge_pre_i"], graph["edge_post_i"], np.abs(graph["edge_weight_i"]))]
    else:
        edges = [(graph["edge_pre"], graph["edge_post"], np.abs(graph["edge_weight"]))]
    pre_parts, post_parts, weight_parts, channel_parts = [], [], [], []
    for channel, bodies in enumerate(body_by_channel):
        for pre, post, weight in edges:
            mask = np.isin(ids[pre.astype(np.int64)], list(bodies))
            if np.any(mask):
                pre_parts.append(pre[mask].astype(np.int64))
                post_parts.append(post[mask].astype(np.int64))
                weight_parts.append(weight[mask].astype(np.float32))
                channel_parts.append(np.full(int(mask.sum()), channel, dtype=np.int64))
    device = model.edge_pre_e.device if hasattr(model, "edge_pre_e") else model.edge_pre.device
    if pre_parts:
        pre = np.concatenate(pre_parts); post = np.concatenate(post_parts)
        weight = np.concatenate(weight_parts); channel = np.concatenate(channel_parts)
        # Normalize the anatomical exposure into [0, 1] per target and amine.
        denom = np.zeros((2, model.n_neurons), dtype=np.float32)
        np.add.at(denom, (channel, post), weight)
        weight = weight / np.maximum(denom[channel, post], 1e-8)
    else:
        pre = post = channel = np.empty(0, dtype=np.int64)
        weight = np.empty(0, dtype=np.float32)
    model.register_buffer("amine_pre", torch.as_tensor(pre, dtype=torch.long, device=device), persistent=False)
    model.register_buffer("amine_post", torch.as_tensor(post, dtype=torch.long, device=device), persistent=False)
    model.register_buffer("amine_channel", torch.as_tensor(channel, dtype=torch.long, device=device), persistent=False)
    # One scatter over the two separate transmitter-by-cell coordinates. This
    # preserves each contact and avoids per-tick channel masks / CPU-GPU syncs.
    model.register_buffer("amine_target_flat", torch.as_tensor(
        channel * model.n_neurons + post, dtype=torch.long, device=device), persistent=False)
    model.register_buffer("amine_contact_weight", torch.as_tensor(weight, dtype=torch.float32, device=device), persistent=False)
    model.register_buffer("amine_source_mask", torch.as_tensor(
        np.array([any(int(body) in bodies for bodies in body_by_channel) for body in ids], dtype=np.bool_),
        dtype=torch.bool, device=device), persistent=False)
    n_class = int(model.n_superclasses)
    for name, value in zip(("amine_log_clearance_oa", "amine_log_clearance_5ht"), tau_clearance):
        setattr(model, name, nn.Parameter(torch.tensor(math.log(float(value)), device=device)))
    for name, value in zip(("amine_log_affinity_oa", "amine_log_affinity_5ht"), affinity):
        setattr(model, name, nn.Parameter(torch.tensor(math.log(float(value)), device=device)))
    for name, value in zip(("amine_log_off_oa", "amine_log_off_5ht"), tau_off):
        setattr(model, name, nn.Parameter(torch.tensor(math.log(float(value)), device=device)))
    model.amine_response_oa = nn.Parameter(torch.full((n_class,), float(response_init), device=device))
    model.amine_response_5ht = nn.Parameter(torch.full((n_class,), float(response_init), device=device))
    model.amine_target_count = model.n_neurons
    model.amine_transmitters = tuple(transmitter_names)
    model.amine_kinetics_version = AMINE_KINETICS_VERSION
    model.amine_enabled = True
    return int(pre.size)


def step_amine_modulation(model, current, thresholds):
    """Advance concentration/occupancy and return thresholds for this tick."""
    concentration = getattr(current, "amine_concentration", torch.empty(0))
    receptor = getattr(current, "amine_receptor", torch.empty(0))
    n = current.h.shape[1]
    if not getattr(model, "amine_enabled", False):
        return thresholds, concentration, receptor
    if concentration.shape != (2, n):
        concentration = current.h.new_zeros((2, n))
    if receptor.shape != (2, n):
        receptor = current.h.new_zeros((2, n))
    release = current.h.new_zeros((2, n))
    if model.amine_pre.numel():
        source = current.ring[0].index_select(1, model.amine_pre)
        values = source * model.amine_contact_weight.unsqueeze(0)
        release.view(-1).index_add_(0, model.amine_target_flat, values[0])
    next_c = []
    next_r = []
    for i, suffix in enumerate(("oa", "5ht")):
        clearance_rate = torch.exp(-getattr(model, f"amine_log_clearance_{suffix}"))
        alpha = -torch.expm1(-clearance_rate)
        c = concentration[i:i + 1] + alpha * (release[i:i + 1] - concentration[i:i + 1])
        K = torch.exp(getattr(model, f"amine_log_affinity_{suffix}"))
        from .fly_biophysical_operators import ligand_binding
        b = ligand_binding(receptor[i:i+1],c,K,torch.exp(-getattr(model,f"amine_log_off_{suffix}")))
        next_c.append(c)
        next_r.append(b.clamp(0.0, 1.0))
    concentration = torch.cat(next_c, dim=0)
    receptor = torch.cat(next_r, dim=0)
    class_id = model.superclass_id.to(torch.long)
    response = (model.amine_response_oa[class_id].unsqueeze(0) * receptor[0]
                + model.amine_response_5ht[class_id].unsqueeze(0) * receptor[1])
    thresholds = thresholds * torch.exp(-response)
    return thresholds, concentration, receptor
