"""Native DPM/APL slow memory regulation around the existing MB compartment state."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


MB_CONSOLIDATION_PARAMETER_NAMES = (
    "mb_dpm_logit_decay", "mb_apl_logit_decay",
    "mb_dpm_consolidation", "mb_apl_feedback",
)


def _logit(value: float) -> float:
    value = min(max(value, 1e-6), 1.0 - 1e-6)
    return math.log(value / (1.0 - value))


def attach_mb_consolidation(model, graph_npz: str | Path, annotations: str | Path,
                            *, tau_dpm: float = 256.0, tau_apl: float = 16.0,
                            dpm_consolidation_init: float = 0.0,
                            apl_feedback_init: float = 0.0):
    if getattr(model, "mb_consolidation_enabled", False):
        return int(model.dpm_index.numel()), int(model.apl_index.numel())
    graph = np.load(graph_npz, allow_pickle=False)
    ann = pd.read_feather(annotations).set_index("bodyId").reindex(graph["neuron_body_ids"])
    types = ann["type"].fillna("").astype(str).to_numpy()
    dpm = np.flatnonzero(types == "DPM")
    apl = np.flatnonzero(types == "APL")
    device = model.edge_pre_e.device if hasattr(model, "edge_pre_e") else model.edge_pre.device
    model.register_buffer("dpm_index", torch.as_tensor(dpm, dtype=torch.long, device=device), persistent=False)
    model.register_buffer("apl_index", torch.as_tensor(apl, dtype=torch.long, device=device), persistent=False)
    if not dpm.size:
        raise ValueError("The connectome has no DPM neurons")
    model.mb_dpm_logit_decay = nn.Parameter(torch.tensor(_logit(math.exp(-1.0 / max(tau_dpm, 1.0))), device=device))
    model.mb_apl_logit_decay = nn.Parameter(torch.tensor(_logit(math.exp(-1.0 / max(tau_apl, 1.0))), device=device))
    model.mb_dpm_consolidation = nn.Parameter(torch.tensor(float(dpm_consolidation_init), device=device))
    model.mb_apl_feedback = nn.Parameter(torch.tensor(float(apl_feedback_init), device=device))
    model.mb_consolidation_enabled = True
    return int(dpm.size), int(apl.size)


def step_mb_consolidation(model, current, thresholds):
    dpm_trace = getattr(current, "dpm_trace", torch.empty(0))
    apl_trace = getattr(current, "apl_trace", torch.empty(0))
    if (not getattr(model, "mb_consolidation_enabled", False)
            or getattr(model, 'mb_pooled_quarantined', False)):
        return thresholds, dpm_trace, apl_trace
    if dpm_trace.numel() != 1:
        dpm_trace = current.h.new_zeros(1, 1)
    if apl_trace.numel() != 1:
        apl_trace = current.h.new_zeros(1, 1)
    dpm_spikes = current.ring[0][:, model.dpm_index].mean(dim=1, keepdim=True)
    dpm_decay = torch.sigmoid(model.mb_dpm_logit_decay)
    dpm_trace = dpm_decay * dpm_trace + (1.0 - dpm_decay) * dpm_spikes
    if model.apl_index.numel():
        # APL is graded in the corrected graph; use its positive membrane
        # activity as the native slow inhibitory signal rather than a fake spike.
        apl_level = current.h[:, model.apl_index].clamp_min(0.0).mean(dim=1, keepdim=True)
        apl_level = torch.sigmoid(apl_level)
        apl_decay = torch.sigmoid(model.mb_apl_logit_decay)
        apl_trace = apl_decay * apl_trace + (1.0 - apl_decay) * apl_level
        if getattr(model, "comp_kc", None) is not None and model.comp_kc.numel():
            thresholds = thresholds.clone()
            kc = torch.unique(model.comp_kc.long())
            thresholds[:, kc] = thresholds[:, kc] * torch.exp(-model.mb_apl_feedback * apl_trace)
    return thresholds, dpm_trace, apl_trace


def dpm_forgetting_factor(model, dpm_trace):
    """Native DPM-dependent retention multiplier for compartment forgetting."""
    if (not getattr(model, "mb_consolidation_enabled", False) or dpm_trace.numel() == 0
            or getattr(model, 'mb_pooled_quarantined', False)):
        return 1.0
    return 1.0 - torch.tanh(model.mb_dpm_consolidation) * dpm_trace
