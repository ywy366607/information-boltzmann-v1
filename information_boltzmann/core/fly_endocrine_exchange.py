"""Evidence-scoped LK secretion/receptor axis, not a synthetic hunger reward.

LK-producing cells and IPC/ITP targets come from MaleCNS annotations; Lkr
expression in IPC and ITP populations is supported by Zandawala et al. 2018.
Kinetics, normalized units and effect sizes are model assumptions, not measured
cell-specific hormone concentrations. No ENS receptor identity is invented.
"""
import math
import numpy as np
import pandas as pd
import torch
from torch import nn

ENDOCRINE_PARAMETER_NAMES = ('endocrine_log_tau', 'endocrine_log_affinity', 'endocrine_response')


def attach_endocrine_exchange(model, graph_npz, annotations, *, tau_ticks=(32., 256., 64.)):
    if getattr(model, 'endocrine_enabled', False):
        return int(model.endocrine_lk_index.numel()), int(model.endocrine_target_index.numel())
    if len(tau_ticks) != 3 or any(not math.isfinite(t) or t <= 0 for t in tau_ticks):
        raise ValueError('Endocrine clocks must be three finite positive values')
    graph = np.load(graph_npz, allow_pickle=False)
    ann = pd.read_feather(annotations).set_index('bodyId').reindex(graph['neuron_body_ids'])
    typ = ann['type'].fillna('').astype(str).to_numpy()
    lk = np.flatnonzero(typ == 'LK')
    targets = np.flatnonzero(np.isin(typ, ['IPC', 'ITP']))
    if not lk.size or not targets.size:
        raise ValueError('LK and IPC/ITP target annotations are required')
    device = model.edge_pre_e.device
    model.register_buffer('endocrine_lk_index', torch.as_tensor(lk, dtype=torch.long, device=device), persistent=False)
    model.register_buffer('endocrine_target_index', torch.as_tensor(targets, dtype=torch.long, device=device), persistent=False)
    model.endocrine_log_tau = nn.Parameter(torch.tensor([math.log(t) for t in tau_ticks], device=device))
    model.endocrine_log_affinity = nn.Parameter(torch.tensor(math.log(.5), device=device))
    model.endocrine_response = nn.Parameter(torch.zeros(targets.size, device=device))
    model.endocrine_enabled = True
    return int(lk.size), int(targets.size)


def step_endocrine_exchange(model, current, thresholds):
    calcium = current.endocrine_calcium
    concentration = current.endocrine_lk
    occupancy = current.endocrine_receptor
    if not getattr(model, 'endocrine_enabled', False):
        return thresholds, calcium, concentration, occupancy
    if calcium.shape != (1, model.endocrine_lk_index.numel()):
        calcium = current.h.new_zeros((1, model.endocrine_lk_index.numel()))
    if concentration.shape != (1, 1):
        concentration = current.h.new_zeros((1, 1))
        occupancy = torch.zeros_like(concentration)
    # Exact exponential relaxation per physical tick, not per word/update.
    decay = torch.exp(-torch.exp(-model.endocrine_log_tau))
    spikes = current.ring[0].index_select(1, model.endocrine_lk_index).clamp_min(0.)
    calcium = decay[0] * calcium + (1.-decay[0]) * spikes
    release = calcium.square().mean(dim=1, keepdim=True)
    concentration = decay[1] * concentration + (1.-decay[1]) * release
    affinity = model.endocrine_log_affinity.exp()
    equilibrium = concentration / (concentration + affinity)
    occupancy = decay[2] * occupancy + (1.-decay[2]) * equilibrium
    result = thresholds.expand_as(current.h).clone()
    idx = model.endocrine_target_index
    result[:, idx] = result[:, idx] * torch.exp(torch.tanh(model.endocrine_response) * occupancy)
    return result, calcium, concentration, occupancy
