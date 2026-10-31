"""Connectome-grounded persistent central-complex state.

The CX cells already participate in the full graph. This module gives their
activity the one missing native role: a slow, local state trace that changes
their own excitability and therefore propagates through the existing CX and
descending connections. No heading label, action reward, or decoder shortcut
is introduced.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


CX_PARAMETER_NAMES = ("cx_logit_decay", "cx_feedback")


def attach_cx_state(model, graph_npz: str | Path, annotations: str | Path,
                    *, tau_ticks: float = 64.0, feedback_init: float = 0.0):
    if getattr(model, "cx_index", None) is not None:
        return int(model.cx_index.numel())
    graph = np.load(graph_npz, allow_pickle=False)
    ann = pd.read_feather(annotations).set_index("bodyId").reindex(graph["neuron_body_ids"])
    mask = ann["class"].fillna("").astype(str).to_numpy() == "CX"
    device = model.edge_pre_e.device if hasattr(model, "edge_pre_e") else model.edge_pre.device
    index = torch.as_tensor(np.flatnonzero(mask), dtype=torch.long, device=device)
    model.register_buffer("cx_index", index, persistent=False)
    if index.numel() == 0:
        raise ValueError("The connectome has no annotated CX neurons")
    tau = max(float(tau_ticks), 1.0)
    decay = math.exp(-1.0 / tau)
    model.cx_logit_decay = nn.Parameter(torch.full((index.numel(),), math.log(decay / (1.0 - decay)), device=device))
    # Zero coupling is the frozen-equivalence control. Native CX activity still
    # accumulates in the persistent trace and wakes the gradient of this path.
    model.cx_feedback = nn.Parameter(torch.full((index.numel(),), float(feedback_init), device=device))
    model.cx_enabled = True
    return int(index.numel())


def step_cx_state(model, current, thresholds):
    trace = getattr(current, "cx_trace", torch.empty(0))
    if not getattr(model, "cx_enabled", False):
        return thresholds, trace
    idx = model.cx_index
    if trace.shape != (1, idx.numel()):
        trace = current.h.new_zeros((1, idx.numel()))
    spikes = current.ring[0].index_select(1, idx)
    decay = torch.sigmoid(model.cx_logit_decay).unsqueeze(0)
    trace = decay * trace + (1.0 - decay) * spikes
    # The trace modifies the CX membrane threshold. Its downstream influence
    # is therefore carried by the fixed anatomical CX -> brain/descending/VNC
    # edges, rather than a private action head.
    effective = torch.exp(-model.cx_feedback * trace[0]).unsqueeze(0)
    thresholds = thresholds.clone()
    thresholds[:, idx] = thresholds[:, idx] * effective
    return thresholds, trace
