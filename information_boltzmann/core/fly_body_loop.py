"""Historical body proxies, quarantined from normal native-brain execution.

Neither a descending population mean nor a scalar speech trace is a native
body. These equations remain only to reproduce historical frozen diagnostics.
Normal execution retains the original CNS synapses and injects no proxy return.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


BODY_LOOP_PARAMETER_NAMES = ("body_logit_decay", "body_feedback")
BODY_ACTUATOR_PARAMETER_NAMES = ('body_muscle_logit_decay', 'body_proprioceptive_gain')


def quarantine_body_proxy_credit(learner):
    """Archive proxy derivatives while preserving all real pending credit.

    The old physical state, parameters, moments and world are left intact. Only
    derivatives with respect to now-inactive synthetic return parameters are
    removed from the active native score trace and retained in its archive.
    """
    if getattr(learner.model, 'body_proxy_compatibility', False):
        return False
    consequence = getattr(learner, 'consequence_state', None)
    if not isinstance(consequence, dict) or not consequence.get('native_credit'):
        return False
    credit = consequence['native_credit']
    names = set(BODY_LOOP_PARAMETER_NAMES + BODY_ACTUATOR_PARAMETER_NAMES)
    replacements, archived = {}, {}
    traces = {key: credit.get(key) for key in ('E', 'response_score')}
    traces.update({f'deferred:{key}': value for key, value in
                   credit.get('deferred_response_scores', {}).items()})
    for key, trace in traces.items():
        if trace is None:
            continue
        if not isinstance(trace, dict):
            raise ValueError('Native score trace must be a dictionary')
        kept, removed = {}, {}
        for name, value in trace.items():
            if name in names:
                removed[name] = value.detach().cpu().clone()
            else:
                kept[name] = value
        if removed:
            replacements[key], archived[key] = kept, removed
    if not archived:
        return False
    for key, value in replacements.items():
        if key.startswith('deferred:'):
            credit['deferred_response_scores'][key[len('deferred:'):]] = value
        else:
            credit[key] = value
    consequence.setdefault('quarantined_body_proxy_credit', []).append({
        'clock_ticks': credit.get('clock_ticks'), 'traces': archived,
        'reason': 'Synthetic body return rejected as native closure; histories retained'})
    return True


def attach_body_actuator(model, graph_npz, annotations, *, tau_ticks=8.):
    """Language-body approximation: executed speech -> actuator -> proprioception.

    This is an information-organism interface, not a fly muscle biomechanics
    model. Only executed self-hearing events drive it, never policy propensity.
    """
    if getattr(model, 'body_actuator_enabled', False):
        return int(model.body_proprioceptive_index.numel())
    if not getattr(model, 'body_loop_enabled', False):
        raise ValueError('Body actuator needs the persistent body interface')
    if not math.isfinite(tau_ticks) or tau_ticks <= 0:
        raise ValueError('Actuator clock must be finite and positive')
    graph = np.load(graph_npz, allow_pickle=False)
    ann = pd.read_feather(annotations).set_index('bodyId').reindex(graph['neuron_body_ids'])
    idx = np.flatnonzero(ann['class'].fillna('').to_numpy() == 'mechanosensory_proprioceptive')
    if not idx.size:
        raise ValueError('No annotated proprioceptive cells in graph')
    device = model.edge_pre_e.device
    model.register_buffer('body_proprioceptive_index', torch.as_tensor(idx, dtype=torch.long, device=device), persistent=False)
    model.body_muscle_logit_decay = nn.Parameter(torch.tensor(_logit(math.exp(-1./tau_ticks)), device=device))
    model.body_proprioceptive_gain = nn.Parameter(torch.zeros(2, idx.size, device=device))
    model.body_actuator_enabled = True
    return int(idx.size)


def step_body_actuator(model, current, source, executed_action=False):
    muscle, feedback = current.body_muscle, current.body_reafference
    if not getattr(model, 'body_proxy_compatibility', False):
        return source, muscle, feedback
    if not (getattr(model, 'body_actuator_enabled', False) and getattr(model, 'body_loop_enabled', False)):
        return source, muscle, feedback
    if not isinstance(executed_action, bool):
        raise ValueError('Executed action must be the recorded boolean command, not a policy logit')
    if muscle.shape != (1, 1):
        muscle = current.h.new_zeros((1, 1))
    decay = torch.sigmoid(model.body_muscle_logit_decay)
    previous = muscle
    muscle = decay * muscle + (1.-decay) * float(executed_action)
    # Position and signed movement are consequences, not labels of the word.
    feedback = torch.cat([muscle, muscle - previous], dim=1)
    drive = feedback @ model.body_proprioceptive_gain
    source = torch.zeros_like(current.h) if source is None else source.clone()
    source[:, model.body_proprioceptive_index] = source[:, model.body_proprioceptive_index] + drive
    return source, muscle, feedback


def _logit(value: float) -> float:
    value = min(max(value, 1e-6), 1.0 - 1e-6)
    return math.log(value / (1.0 - value))


def attach_body_loop(model, graph_npz: str | Path, annotations: str | Path,
                     *, tau_ticks: float = 32.0, feedback_init: float = 0.0):
    if getattr(model, "body_loop_enabled", False):
        return int(model.body_descending_index.numel()), int(model.body_ascending_index.numel())
    graph = np.load(graph_npz, allow_pickle=False)
    ann = pd.read_feather(annotations).set_index("bodyId").reindex(graph["neuron_body_ids"])
    superclass = ann["superclass"].fillna("").astype(str).to_numpy()
    descending = np.flatnonzero(superclass == "descending_neuron")
    ascending = np.flatnonzero(superclass == "ascending_neuron")
    device = model.edge_pre_e.device if hasattr(model, "edge_pre_e") else model.edge_pre.device
    model.register_buffer("body_descending_index", torch.as_tensor(descending, dtype=torch.long, device=device), persistent=False)
    model.register_buffer("body_ascending_index", torch.as_tensor(ascending, dtype=torch.long, device=device), persistent=False)
    if not descending.size or not ascending.size:
        raise ValueError("The connectome needs descending and ascending neurons for the body loop")
    decay = math.exp(-1.0 / max(float(tau_ticks), 1.0))
    model.body_logit_decay = nn.Parameter(torch.tensor(_logit(decay), device=device))
    # Per ascending-cell response is learnable and zero-equivalent at birth.
    model.body_feedback = nn.Parameter(torch.full((ascending.size,), float(feedback_init), device=device))
    model.body_loop_enabled = True
    return int(descending.size), int(ascending.size)


def step_body_loop(model, current, source):
    action_trace = getattr(current, "body_action_trace", torch.empty(0))
    feedback_trace = getattr(current, "body_feedback_trace", torch.empty(0))
    if not getattr(model, 'body_proxy_compatibility', False):
        return source, action_trace, feedback_trace
    if not getattr(model, "body_loop_enabled", False):
        return source, action_trace, feedback_trace
    if action_trace.numel() != 1:
        action_trace = current.h.new_zeros(1, 1)
    if feedback_trace.numel() != 1:
        feedback_trace = current.h.new_zeros(1, 1)
    decay = torch.sigmoid(model.body_logit_decay)
    descending = current.ring[0].index_select(1, model.body_descending_index).mean(dim=1, keepdim=True)
    ascending = current.ring[0].index_select(1, model.body_ascending_index).mean(dim=1, keepdim=True)
    action_trace = decay * action_trace + (1.0 - decay) * descending
    feedback_trace = decay * feedback_trace + (1.0 - decay) * ascending
    if getattr(model, 'body_actuator_enabled', False):
        # Preserve historical state, but replace the direct synthetic return
        # with execution-derived feedback through native sensory cells.
        return source, action_trace, feedback_trace
    if source is None:
        source = torch.zeros_like(current.h)
    # Historical synthetic feedback, NOT a native body consequence. It bypasses
    # motor neurons, peripheral effectors and real sensory transduction.
    # Its historical sign/strength were learned, but that does not establish a
    # biological peripheral route.
    source = source.clone()
    source[:, model.body_ascending_index] = source[:, model.body_ascending_index] + (
        action_trace * model.body_feedback.unsqueeze(0))
    return source, action_trace, feedback_trace
