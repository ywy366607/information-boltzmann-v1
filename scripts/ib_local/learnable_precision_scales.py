"""Opt-in learnable positive scales for relation ridge and memory read temperature.

Both parameters are parameterized in log-space with bounded positive clamps and
zero weight decay so only task gradients move them from their exact inherited
defaults (0.1 and 0.2). When disabled, helpers return the inherited float values
bit-for-bit identically.
"""
from __future__ import annotations

import math

import torch
from torch import nn

RIDGE_MIN = 1e-3
RIDGE_MAX = 10.0
TEMP_MIN = 0.02
TEMP_MAX = 2.0


def effective_ridge(model, fallback=0.1):
    param = getattr(model, 'log_relation_ridge', None)
    if param is None:
        return fallback
    if not torch.is_grad_enabled():
        from scripts.ib_local.preview_runtime import _scope, tensor_key
        active = _scope.get()
        if active is not None and active[0] is model:
            cache = active[1]
            key = ('effective_ridge', tensor_key(param))
            entry = cache.get(key)
            if entry is not None:
                return entry[0]
            out = param.exp().clamp(RIDGE_MIN, RIDGE_MAX)
            out._cached_float = float(out.detach())
            cache[key] = (out, (param,))
            return out
    return param.exp().clamp(RIDGE_MIN, RIDGE_MAX)


def effective_temperature(memory):
    param = getattr(memory, 'log_temperature', None)
    if param is None:
        return memory.temperature
    if not torch.is_grad_enabled():
        from scripts.ib_local.preview_runtime import _scope, tensor_key
        active = _scope.get()
        if active is not None and getattr(active[0], 'memory', None) is memory:
            cache = active[1]
            key = ('effective_temperature', tensor_key(param))
            entry = cache.get(key)
            if entry is not None:
                return entry[0]
            out = param.exp().clamp(TEMP_MIN, TEMP_MAX)
            cache[key] = (out, (param,))
            return out
    return param.exp().clamp(TEMP_MIN, TEMP_MAX)


def configure(engine, spec=None, *, initialize=True):
    """Attach learnable log_relation_ridge and/or memory.log_temperature.

    Follows the same parameter-group and pending-participation extension
    contract as AdaptiveHamiltonian and ContactMomentum. When `ridge_lr` or
    `temperature_lr` is present in `spec`, each parameter gets its own
    zero-weight-decay param_group in the same order `[ridge, temperature]`.
    """
    spec = dict(schema=1, ridge=True, temperature=True) if spec is None else dict(spec)
    if spec.get('schema', 1) != 1:
        raise ValueError('unknown learnable_precision_scales schema')
    learn_ridge = bool(spec.get('ridge', True))
    learn_temp = bool(spec.get('temperature', True))
    ridge_lr = spec.get('ridge_lr')
    temp_lr = spec.get('temperature_lr')
    model = engine.model
    ref = model.encoder.weight
    added = []
    if learn_ridge and not hasattr(model, 'log_relation_ridge'):
        init_ridge = float(getattr(engine, 'relation_ridge', getattr(model, 'causal_relation_ridge', 0.1)))
        model.log_relation_ridge = nn.Parameter(
            torch.tensor(math.log(init_ridge), device=ref.device, dtype=ref.dtype))
        added.append(('ridge', model.log_relation_ridge))
    if learn_temp and not hasattr(model.memory, 'log_temperature'):
        init_temp = float(model.memory.temperature)
        model.memory.log_temperature = nn.Parameter(
            torch.tensor(math.log(init_temp), device=ref.device, dtype=ref.dtype))
        added.append(('temperature', model.memory.log_temperature))
    if not added:
        return
    base_lr = engine.optimizer.param_groups[0]['lr']
    params_only = [p for _, p in added]
    old_count = len(engine.params)
    if ridge_lr is None and temp_lr is None:
        engine.optimizer.add_param_group(dict(
            params=params_only, lr=base_lr, weight_decay=0.0))
    else:
        for role, p in added:
            lr = float(base_lr if ridge_lr is None else ridge_lr) if role == 'ridge' else float(
                base_lr if temp_lr is None else temp_lr)
            engine.optimizer.add_param_group(dict(params=[p], lr=lr, weight_decay=0.0))
    engine.params = [*engine.params, *params_only]
    engine.groups = (*engine.groups, params_only)
    if initialize:
        def extend(value):
            if isinstance(value, (list, tuple)) and len(value) == old_count and all(
                    v is None or torch.is_tensor(v) for v in value):
                return type(value)([*value, *([None] * len(params_only))])
            if isinstance(value, dict):
                return {k: extend(v) for k, v in value.items()}
            return value
        for name in ('prev', 'internal_pending', 'fire_grad'):
            if hasattr(engine, name):
                setattr(engine, name, extend(getattr(engine, name)))
        if hasattr(engine, 'head_mask'):
            engine.head_mask = [*engine.head_mask, *([True] * len(params_only))]
    engine.learnable_precision_scales_schema = 1


def _find_param_group(optimizer, param):
    for g_idx, group in enumerate(optimizer.param_groups):
        for p_idx, p in enumerate(group['params']):
            if p is param:
                return g_idx, p_idx
    raise RuntimeError('parameter not found in optimizer.param_groups')


def set_scale_learning_rates(engine, config, *, ridge_lr=None, temperature_lr=None):
    """Give log_relation_ridge and/or memory.log_temperature explicit scalar LRs.

    Splits a shared scale param_group in-place into `[log_relation_ridge]` and
    `[memory.log_temperature]` while preserving EVERY optimizer state tensor
    (exp_avg, exp_avg_sq, step, budget_*) and all other param_groups verbatim.
    """
    if ridge_lr is not None and ridge_lr <= 0:
        raise ValueError('ridge_lr must be positive')
    if temperature_lr is not None and temperature_lr <= 0:
        raise ValueError('temperature_lr must be positive')
    model = engine.model
    opt = engine.optimizer
    p_ridge = getattr(model, 'log_relation_ridge', None)
    p_temp = getattr(model.memory, 'log_temperature', None)
    if ridge_lr is not None and p_ridge is None:
        raise ValueError('ridge_lr requires learnable log_relation_ridge')
    if temperature_lr is not None and p_temp is None:
        raise ValueError('temperature_lr requires learnable memory.log_temperature')
    if p_ridge is not None and p_temp is not None:
        gr_idx, _ = _find_param_group(opt, p_ridge)
        gt_idx, _ = _find_param_group(opt, p_temp)
        if gr_idx == gt_idx:
            old_grp = opt.param_groups[gr_idx]
            if len(old_grp['params']) != 2:
                raise RuntimeError('expected dedicated scale param_group of size 2')
            base_meta = {k: v for k, v in old_grp.items() if k != 'params'}
            grp_r = dict(base_meta, params=[p_ridge],
                         lr=float(old_grp['lr'] if ridge_lr is None else ridge_lr),
                         weight_decay=0.0)
            grp_t = dict(base_meta, params=[p_temp],
                         lr=float(old_grp['lr'] if temperature_lr is None else temperature_lr),
                         weight_decay=0.0)
            opt.param_groups[gr_idx:gr_idx + 1] = [grp_r, grp_t]
        else:
            if ridge_lr is not None:
                opt.param_groups[gr_idx]['lr'] = float(ridge_lr)
                opt.param_groups[gr_idx]['weight_decay'] = 0.0
            if temperature_lr is not None:
                opt.param_groups[gt_idx]['lr'] = float(temperature_lr)
                opt.param_groups[gt_idx]['weight_decay'] = 0.0
    else:
        if ridge_lr is not None:
            gr_idx, _ = _find_param_group(opt, p_ridge)
            opt.param_groups[gr_idx]['lr'] = float(ridge_lr)
            opt.param_groups[gr_idx]['weight_decay'] = 0.0
        if temperature_lr is not None:
            gt_idx, _ = _find_param_group(opt, p_temp)
            opt.param_groups[gt_idx]['lr'] = float(temperature_lr)
            opt.param_groups[gt_idx]['weight_decay'] = 0.0
    spec = dict(config.get('learnable_precision_scales',
                           dict(schema=1, ridge=p_ridge is not None, temperature=p_temp is not None)))
    if ridge_lr is not None:
        spec['ridge_lr'] = float(ridge_lr)
    if temperature_lr is not None:
        spec['temperature_lr'] = float(temperature_lr)
    config['learnable_precision_scales'] = spec


@torch.no_grad()
def diagnostics(engine):
    model = engine.model
    opt = engine.optimizer
    out = {}
    if hasattr(model, 'log_relation_ridge'):
        p = model.log_relation_ridge
        gate = opt.state.get(p, {}).get('budget_gate')
        gr_idx, _ = _find_param_group(opt, p)
        out['relation_ridge'] = float(effective_ridge(model))
        out['relation_ridge_gate'] = None if gate is None else float(gate.detach())
        out['relation_ridge_lr'] = float(opt.param_groups[gr_idx]['lr'])
    if hasattr(model.memory, 'log_temperature'):
        p = model.memory.log_temperature
        gate = opt.state.get(p, {}).get('budget_gate')
        gt_idx, _ = _find_param_group(opt, p)
        out['memory_temperature'] = float(effective_temperature(model.memory))
        out['memory_temperature_gate'] = None if gate is None else float(gate.detach())
        out['memory_temperature_lr'] = float(opt.param_groups[gt_idx]['lr'])
    return out
