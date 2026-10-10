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
    posterior = getattr(getattr(model, 'medium', None), 'structural_posterior', None)
    independent = ({id(posterior.mean), id(posterior.log_std)}
                   if posterior is not None and posterior.capacity_growth is not None else set())
    for name, parameter in candidates:
        if parameter.requires_grad and id(parameter) not in independent:
            group = groups[0 if id(parameter) in decay_ids else 1]
            group['params'].append(parameter)
            group['param_names'].append(name)
    return groups


def make_medium_optimizer(model, *, lr: float, weight_decay: float = 0.01, saved_state=None,
                          fused: bool = False):
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
        posterior = getattr(getattr(model, 'medium', None), 'structural_posterior', None)
        excluded = ({'medium.structural_posterior.mean', 'medium.structural_posterior.log_std'}
                    if posterior is not None and posterior.capacity_growth is not None else set())
        parameters = []
        for group in saved_state['param_groups']:
            names = group.get('param_names')
            if names is None or any(name not in named for name in names):
                raise ValueError('Saved optimizer parameters are absent from this architecture')
            if set(names) & excluded:
                raise ValueError('Saved Adam still owns structural capacity; explicit growth migration required')
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
    if fused:
        # Preserve groups, moments, counters and hyperparameters; only replace
        # elementwise Adam arithmetic with the native fused CUDA kernel.
        for group in optimizer.param_groups:
            if any(p.device.type != 'cuda' for p in group['params']):
                raise ValueError('Fused medium AdamW requires CUDA parameters')
            group.update(fused=True, foreach=False)
            for parameter in group['params']:
                state = optimizer.state.get(parameter)
                if state and isinstance(state.get('step'), torch.Tensor):
                    state['step'] = state['step'].to(parameter.device)
        optimizer.defaults.update(fused=True, foreach=False)
    if any(group.get('fused') for group in optimizer.param_groups):
        def track_fused_mutations(optimizer, args, kwargs):
            # This PyTorch fused kernel updates storage without bumping the
            # Tensor version. Replay guards and norm caches need that signal.
            torch.autograd.graph.increment_version([
                p for group in optimizer.param_groups for p in group['params']
                if p.grad is not None])
        optimizer.register_step_post_hook(track_fused_mutations)
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


def initialize_capacity_growth_branch(model, saved_weights, saved_optimizer, learner_state):
    """Explicit completed-window migration, preserving every other Adam moment.

    Source weights, physical state, RNG, OU prior and all exposure ledgers remain
    the caller's continuous individual. Only two old structural Adam entries are
    removed. Growth statistics begin at zero. A pending window cannot switch
    learning rules; ordinary growth checkpoints can still resume mid-window.
    """
    posterior = getattr(getattr(model, 'medium', None), 'structural_posterior', None)
    if posterior is None or posterior.capacity_growth is None:
        raise ValueError('Destination must explicitly enable capacity growth')
    prefix = 'medium.structural_posterior.'
    if learner_state['pending'] != 0 or bool(saved_weights[prefix + 'window_active']):
        raise ValueError('Capacity growth migration requires a completed structural window')
    if any(name.startswith(prefix + 'capacity_growth.') for name in saved_weights):
        raise ValueError('Growth individual requires normal exact resume, not rule migration')
    new_keys = {prefix + 'capacity_growth.' + name
                for name in posterior.capacity_growth.state_dict()}
    missing, unexpected = set(model.state_dict()) - set(saved_weights), set(saved_weights) - set(model.state_dict())
    if missing != new_keys or unexpected:
        raise ValueError(f'Capacity migration has unrelated architecture changes: {missing}, {unexpected}')
    excluded = {prefix + 'mean', prefix + 'log_std'}
    groups, state = [], dict(saved_optimizer['state'])
    found = set()
    for group in saved_optimizer['param_groups']:
        names = group.get('param_names')
        if names is None or len(names) != len(group['params']):
            raise ValueError('Capacity migration requires named optimizer groups')
        pairs = [(name, index) for name, index in zip(names, group['params']) if name not in excluded]
        for name, index in zip(names, group['params']):
            if name in excluded:
                state.pop(index, None)
                found.add(name)
        groups.append({**group, 'params': [index for _, index in pairs],
                       'param_names': [name for name, _ in pairs]})
    if found != excluded:
        raise ValueError('Source optimizer must own exactly the expected mean and log_std entries')
    model.load_state_dict(saved_weights, strict=False)
    return {'state': state, 'param_groups': groups}


def initialize_hopf_branch(model, saved_weights, saved_optimizer, learner_state):
    """Exact-identity physical-junction adoption at a completed credit boundary.

    All old named Adam entries and physical state stay intact. New junction
    gates start at zero, without perturbing weights or consuming random noise.
    Caller restores the checkpoint RNG/cursors/belief after constructing model.
    """
    if not getattr(model, 'hopf_recomposition', False) or model.hopf_pathway is None:
        raise ValueError('Destination must explicitly enable physical junctions')
    if learner_state['pending'] != 0:
        raise ValueError('Physical junction adoption requires a completed credit window')
    window_key = 'medium.structural_posterior.window_active'
    if window_key in saved_weights and bool(saved_weights[window_key]):
        raise ValueError('Physical junction adoption requires a completed structural window')
    prefix = 'medium.hopf_pathway.'
    expected_keys = {prefix + key for key in model.hopf_pathway.state_dict()}
    missing = set(model.state_dict()) - set(saved_weights)
    unexpected = set(saved_weights) - set(model.state_dict())
    if missing != expected_keys or unexpected:
        raise ValueError(f'Unexpected junction adoption architecture: missing={missing}, unexpected={unexpected}')
    source_groups = saved_optimizer['param_groups']
    if any('param_names' not in g or len(g['param_names']) != len(g['params']) for g in source_groups):
        raise ValueError('Physical junction adoption requires named source optimizer groups')
    physical_groups = [g for g in source_groups if g.get('group_name') == 'physical_scales_norms_biases']
    if len(physical_groups) != 1 or physical_groups[0].get('weight_decay') != 0:
        raise ValueError('Source must have one zero-decay physical optimizer group')
    names = [name for g in source_groups for name in g['param_names']]
    indices = [i for g in source_groups for i in g['params']]
    named = dict(model.named_parameters())
    if len(set(names)) != len(names) or len(set(indices)) != len(indices):
        raise ValueError('Source optimizer has duplicate parameter ownership')
    if any(name not in named or name.startswith(prefix) for name in names):
        raise ValueError('Source optimizer names do not match the baseline architecture')
    if set(saved_optimizer['state']) - set(indices):
        raise ValueError('Source optimizer has moments without parameter owners')
    destination_state = model.state_dict()
    if any(isinstance(destination_state[name], torch.Tensor) and
           (not isinstance(value, torch.Tensor) or destination_state[name].shape != value.shape)
           for name, value in saved_weights.items()):
        raise ValueError('Source baseline tensors have incompatible shapes')
    for group in source_groups:
        for name, index in zip(group['param_names'], group['params']):
            for key in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
                value = saved_optimizer['state'].get(index, {}).get(key)
                if value is not None and value.shape != named[name].shape:
                    raise ValueError('Source Adam moment shape does not match its named parameter')
    new_names = sorted(name for name, _ in model.named_parameters() if name.startswith(prefix))
    groups, state = [], dict(saved_optimizer['state'])
    next_id = 1 + max((i for g in saved_optimizer['param_groups'] for i in g['params']), default=-1)
    assigned = False
    for group in source_groups:
        new_group = dict(group)
        new_group['params'] = list(group['params'])
        new_group['param_names'] = list(group['param_names'])
        if group.get('group_name') == 'physical_scales_norms_biases':
            for name in new_names:
                new_group['params'].append(next_id)
                new_group['param_names'].append(name)
                next_id += 1
            assigned = True
        groups.append(new_group)
    # Validation above is read-only. Apply the exact-weight adoption only after
    # the complete parameter/moment/structural-window contract is accepted.
    model.load_state_dict(saved_weights, strict=False)
    with torch.no_grad():
        model.hopf_pathway.gate.weight.zero_()
        model.hopf_pathway.gate.bias.zero_()
    return {'state': state, 'param_groups': groups}
