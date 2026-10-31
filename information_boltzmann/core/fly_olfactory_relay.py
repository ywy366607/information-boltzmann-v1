"""Connectome-grounded antennal-lobe local inhibition and PN adaptation.

The graph already contains AL local neurons and projection neurons. This module
adds only the missing slow local state: spikes of annotated ALIN/ALON/ALLN/LN
cells are integrated and delivered through the original inhibitory AL->PN
contacts. The contact topology and weights remain fixed; the local decay and
net postsynaptic response are learnable and start at a frozen-equivalent gain.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


AL_PARAMETER_NAMES = ("al_local_logit_decay", "al_pn_feedback")
LOCAL_PATTERN = r"ALIN|ALON|ALLN|lLN|mAL|vLN|LNO|AL-AST"


def attach_olfactory_relay(model, graph_npz: str | Path, annotations: str | Path,
                           *, tau_ticks: float = 16.0, feedback_init: float = 0.0):
    if getattr(model, "olfactory_relay_enabled", False):
        return int(model.al_local_index.numel()), int(model.al_pn_index.numel()), int(model.al_contact_weight.numel())
    graph = np.load(graph_npz, allow_pickle=False)
    ann = pd.read_feather(annotations).set_index("bodyId").reindex(graph["neuron_body_ids"])
    typ = ann["type"].fillna("").astype(str)
    superclass = ann["superclass"].fillna("").astype(str)
    local = (typ.str.contains(LOCAL_PATTERN, regex=True) & superclass.eq("cb_intrinsic")).to_numpy()
    pn = (typ.str.contains("PN", regex=False) & superclass.eq("cb_intrinsic")).to_numpy()
    # Local inhibition is represented by the graph's inhibitory edge list only.
    pre = graph["edge_pre_i"].astype(np.int64)
    post = graph["edge_post_i"].astype(np.int64)
    weight = np.abs(graph["edge_weight_i"]).astype(np.float32)
    keep = local[pre] & pn[post]
    pre, post, weight = pre[keep], post[keep], weight[keep]
    if not local.any() or not pn.any() or not keep.any():
        raise ValueError("The connectome has no annotated AL local inhibitory contacts into PNs")
    # Normalize each PN's local inhibition to a dimensionless exposure.
    denom = np.zeros(model.n_neurons, dtype=np.float32)
    np.add.at(denom, post, weight)
    weight = weight / np.maximum(denom[post], 1e-8)
    device = model.edge_pre_e.device if hasattr(model, "edge_pre_e") else model.edge_pre.device
    model.register_buffer("al_local_index", torch.as_tensor(np.flatnonzero(local), dtype=torch.long, device=device), persistent=False)
    model.register_buffer("al_pn_index", torch.as_tensor(np.flatnonzero(pn), dtype=torch.long, device=device), persistent=False)
    local_index = np.flatnonzero(local)
    pn_index = np.flatnonzero(pn)
    model.register_buffer("al_contact_local_slot", torch.as_tensor(np.searchsorted(local_index, pre), dtype=torch.long, device=device), persistent=False)
    model.register_buffer("al_contact_pn_slot", torch.as_tensor(np.searchsorted(pn_index, post), dtype=torch.long, device=device), persistent=False)
    model.register_buffer("al_contact_weight", torch.as_tensor(weight, dtype=torch.float32, device=device), persistent=False)
    decay = math.exp(-1.0 / max(float(tau_ticks), 1.0))
    model.al_local_logit_decay = nn.Parameter(torch.tensor(math.log(decay / (1.0 - decay)), device=device))
    # Positive gain raises PN threshold, matching the inhibitory contact sign;
    # zero is the frozen-equivalence attachment point.
    model.al_pn_feedback = nn.Parameter(torch.full((int(pn.sum()),), float(feedback_init), device=device))
    model.olfactory_relay_enabled = True
    return int(local.sum()), int(pn.sum()), int(pre.size)


def step_olfactory_relay(model, current, thresholds):
    trace = getattr(current, "al_local_trace", torch.empty(0))
    drive = getattr(current, "al_pn_drive", torch.empty(0))
    if not getattr(model, "olfactory_relay_enabled", False):
        return thresholds, trace, drive
    n_local = model.al_local_index.numel()
    n_pn = model.al_pn_index.numel()
    if trace.shape != (1, n_local):
        trace = current.h.new_zeros((1, n_local))
    if drive.shape != (1, n_pn):
        drive = current.h.new_zeros((1, n_pn))
    decay = torch.sigmoid(model.al_local_logit_decay)
    local_spikes = current.ring[0].index_select(1, model.al_local_index)
    trace = decay * trace + (1.0 - decay) * local_spikes
    exposure = trace.index_select(1, model.al_contact_local_slot) * model.al_contact_weight.unsqueeze(0)
    drive = current.h.new_zeros((1, n_pn))
    drive.index_add_(1, model.al_contact_pn_slot, exposure)
    thresholds = thresholds.clone()
    thresholds[:, model.al_pn_index] = thresholds[:, model.al_pn_index] * torch.exp(
        model.al_pn_feedback.unsqueeze(0) * drive)
    return thresholds, trace, drive
