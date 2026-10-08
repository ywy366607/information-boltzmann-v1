"""Explicit regularization of prediction weights, preserving physical coordinates."""
from __future__ import annotations

import math

import torch
from torch import nn


def medium_parameter_groups(model, weight_decay: float = 0.01):
    """Decay affine/embedding weights, excluding physical and precision maps.

    A zero log time/rate is a physical reference, not a smaller physical quantity.
    Spatial coordinates, material coefficients, norm gains, biases and attention
    scales likewise have no generic weight-norm shrinkage interpretation.
    """
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError('Weight decay must be finite and nonnegative')
    decay_ids = set()
    for name, module in model.named_modules():
        physical = name.startswith('medium.') and not name.startswith('medium.collision_rate.')
        precision = name.startswith(('write_agent.process_variance',
                                     'write_agent.observation_precision'))
        if isinstance(module, (nn.Linear, nn.Embedding)) and not physical and not precision:
            decay_ids.add(id(module.weight))
    groups = [dict(params=[], param_names=[], group_name='prediction_weights', weight_decay=weight_decay),
              dict(params=[], param_names=[], group_name='physical_scales_norms_biases', weight_decay=0.0)]
    candidates = (model.learning_named_parameters() if hasattr(model, 'learning_named_parameters')
                  else model.named_parameters())
    for name, parameter in candidates:
        if parameter.requires_grad:
            group = groups[0 if id(parameter) in decay_ids else 1]
            group['params'].append(parameter)
            group['param_names'].append(name)
    return groups


def make_medium_optimizer(model, *, lr: float, weight_decay: float = 0.01, saved_state=None):
    """New runs use explicit groups; exact resume keeps the saved regularization.

    Legacy one-group AdamW moments remain associated with their original parameter
    order. Adopting the new head/group policy is an explicit initialize-from branch,
    with fresh optimizer/eligibility, rather than a silent change during resume.
    """
    legacy = (saved_state is not None and len(saved_state['param_groups']) == 1
              and 'param_names' not in saved_state['param_groups'][0])
    if saved_state is not None and not legacy:
        # Restore the exact saved names/order, including old disconnected entries.
        # Filtering them is a new-run optimization, never a resume mutation.
        named = dict(model.named_parameters())
        parameters = []
        for group in saved_state['param_groups']:
            names = group.get('param_names')
            if names is None or any(name not in named for name in names):
                raise ValueError('Saved optimizer parameters are absent from this architecture')
            parameters.append({**{k: v for k, v in group.items() if k != 'params'},
                               'params': [named[name] for name in names]})
    else:
        parameters = model.parameters() if legacy else medium_parameter_groups(model, weight_decay)
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay, foreach=False)
    if saved_state is not None:
        if not legacy:
            expected = [g['param_names'] for g in optimizer.param_groups]
            actual = [g.get('param_names') for g in saved_state['param_groups']]
            if actual != expected:
                raise ValueError('Optimizer continuation parameter names/groups changed')
        optimizer.load_state_dict(saved_state)
    return optimizer


def optimizer_policy(optimizer):
    return [{'name': g.get('group_name', 'legacy_all_parameters'),
             'weight_decay': g['weight_decay'],
             'parameters': sum(p.numel() for p in g['params'])}
            for g in optimizer.param_groups]


def load_medium_branch_weights(model, saved_weights):
    """Permit only the new all-ones norm gain when initializing an old branch."""
    missing = set(model.state_dict()) - set(saved_weights)
    unexpected = set(saved_weights) - set(model.state_dict())
    if unexpected or missing - {'read_norm.weight'}:
        raise ValueError(f'Incompatible branch weights: missing={missing}, unexpected={unexpected}')
    model.load_state_dict(saved_weights, strict=not missing)


def initialize_dynamic_read_branch(model, saved_weights):
    """Explicit weights-only migration; old optimizer/continuation is not resumed.

    New additive paths start at zero, preserving the old prediction exactly.
    A fresh optimizer must include their parameters. Persisted physical state
    can be copied explicitly into this new branch without resetting experience.
    """
    if model.read_mode != 'dynamic':
        raise ValueError('Destination must explicitly enable dynamic read')
    new_keys = {'readout.' + name + '.weight'
                for name in ('motion_policy', 'motion_keys', 'motion_merge')}
    missing = set(model.state_dict()) - set(saved_weights)
    unexpected = set(saved_weights) - set(model.state_dict())
    if missing != new_keys or unexpected:
        raise ValueError(f'Expected an exact instantaneous source: missing={missing}, unexpected={unexpected}')
    model.load_state_dict(saved_weights, strict=False)
    with torch.no_grad():
        for name in ('motion_policy', 'motion_keys', 'motion_merge'):
            getattr(model.readout, name).weight.zero_()
