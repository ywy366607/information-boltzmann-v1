"""A language peripheral attached to individually annotated native motor cells.

This defines a digital peripheral protocol, not fly muscle biomechanics. No
body state or fabricated proprioceptive current is generated. Native release
events carry commands, and the real world supplies their sensory consequences.
"""
from __future__ import annotations

import copy
import numpy as np
import pandas as pd
import torch
from torch import nn


MOTOR_PORT_PARAMETER_NAMES = ('motor_port_projection.weight', 'motor_port_norm.weight',
    'motor_port_norm.bias', 'motor_port_decoder.weight', 'motor_port_decoder.bias')


def checkpoint_model_weights(model):
    """Retain the historical parameter payload plus the motor-channel identities.

    Anatomical indices are rebuilt from the source graph on restore; their body
    IDs must be saved independently to detect a reordered or replaced graph.
    """
    weights = dict(model.named_parameters())
    from .fly_native_runtime import identity_names
    for name in identity_names():
        if hasattr(model, name):
            weights[name] = getattr(model, name).detach().clone()
    return weights


def attach_native_motor_port(model, graph_npz, annotations, *, saved=None):
    graph = np.load(graph_npz, allow_pickle=False)
    ann = pd.read_feather(annotations).set_index('bodyId').reindex(graph['neuron_body_ids'])
    index = np.flatnonzero(ann['superclass'].isin(['cb_motor', 'vnc_motor']).to_numpy())
    if not index.size:
        raise ValueError('Native execution requires annotated motor neurons')
    read = model.read_indices.detach().cpu().numpy()
    positions = {int(cell): i for i, cell in enumerate(read)}
    if any(int(cell) not in positions for cell in index):
        raise ValueError('Native motor cells must be represented on the existing read surface')
    device = model.output_read.weight.device
    pos = torch.tensor([positions[int(cell)] for cell in index], device=device)
    model.register_buffer('native_motor_index', torch.as_tensor(index, device=device), persistent=False)
    body_ids = torch.as_tensor(graph['neuron_body_ids'][index], device=device)
    if saved and any(k in saved for k in MOTOR_PORT_PARAMETER_NAMES) and 'native_motor_body_ids' not in saved:
        raise ValueError('Saved native motor port lacks anatomical identities; explicit migration required')
    if saved and 'native_motor_body_ids' in saved and not torch.equal(saved['native_motor_body_ids'].to(device), body_ids):
        raise ValueError('Native motor identities changed; an explicit anatomical migration is required')
    model.register_buffer('native_motor_body_ids', body_ids, persistent=True)
    model.register_buffer('native_motor_positions', pos, persistent=False)
    mask = torch.zeros(model.n_read, device=device)
    mask[pos] = 1.
    model.register_buffer('native_motor_mask', mask, persistent=False)
    # Preserve the source RNG: initialization copies retained language knowledge
    # without tying the prediction head to subsequent action learning.
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        model.motor_port_projection = nn.Linear(len(index), model.output_read.out_features,
                                                bias=False, device=device)
    with torch.no_grad():
        model.motor_port_projection.weight.copy_(model.output_read.weight[:, pos])
    model.motor_port_norm = copy.deepcopy(model.read_norm)
    model.motor_port_decoder = copy.deepcopy(model.decoder)
    if saved:
        own = {k: v for k, v in saved.items() if k in MOTOR_PORT_PARAMETER_NAMES}
        expected = {k for k, _ in model.named_parameters() if k in MOTOR_PORT_PARAMETER_NAMES}
        if own and set(own) != expected:
            raise ValueError('Partial native motor port checkpoint is not resumable')
        if own:
            loaded = model.load_state_dict(own, strict=False)
            if loaded.unexpected_keys:
                raise ValueError('Native motor port checkpoint mismatch')
    model.native_motor_port_enabled = True
    model.native_motor_terminal_enabled = True
    return int(index.size)


def motor_availability(model, physical):
    """A terminal packet exists iff a native MN actually released this tick.

    It is a stateless digital-device event, not a motor population mean, a
    reward, a body variable or a desired speaking rate.
    """
    if not getattr(model, 'native_motor_terminal_enabled', True):
        return torch.zeros(physical.h.shape[0], dtype=torch.bool, device=physical.h.device)
    return physical.ring[0].index_select(1, model.native_motor_index).gt(0).any(1)


def read_action(model, physical):
    if not getattr(model, 'native_motor_port_enabled', False):
        from .fly_bptt_learning import read_fly_motor
        return read_fly_motor(model, physical)
    # The peripheral receives actual axonal release, not an invasive read of
    # membrane potentials or the prediction head's accumulated expectation.
    motor = physical.ring[0].index_select(1, model.native_motor_index)
    if not getattr(model, 'native_motor_terminal_enabled', True):
        motor = motor * 0.
    return model.motor_port_projection(motor)


def action_logits(model, latent):
    if getattr(model, 'native_motor_port_enabled', False):
        return model.motor_port_decoder(model.motor_port_norm(latent))
    return model.decoder(model.read_norm(latent))
