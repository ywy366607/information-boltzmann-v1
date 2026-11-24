"""Explicit function-preserving growth of the continuous navigation individual.

Duplicate channels/heads, split incoming weights, and append identity residual
layers. Retain the 128-dimensional learned fact chart through trainable bridges:
no cold archive rewrite and no new copy of old facts is needed. All physical
layers and both bridges train. No old layer or memory is frozen.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from scripts.ib_local.gated_deltanet import GatedDeltaNetLM
from scripts.ib_local.scoped_decision_graph import ScopedDecisionGraph
from scripts.ib_local.train_gdn_fast_table import TableGDN


def parameter_lift(name, value, target_shape, *, covector=False, second=False):
    """Channel duplication map, or its minimum-norm cotangent transport.

    Old Adam moments and saved score gradients are covectors, not parameters.
    Newly introduced directions have no historical participation. This preserves
    the old restricted directional derivative, not the unavailable full new-body
    historical derivative. The inherited delayed estimator remains approximate.
    """
    if tuple(value.shape) == tuple(target_shape):
        return value.clone()
    old = value.shape
    if name == 'momentum_to_boundary.weight':
        result = value.new_zeros(target_shape)
        block = value.reshape(value.shape[0], 4, 128)
        result[:, :4*256] = block.repeat(1, 1, 2).reshape(value.shape[0], -1) / 2
        if covector:
            # Derivative of each half-weight on the duplicate submanifold is
            # the original derivative, including second moments.
            result = value.new_zeros(target_shape)
            result[:, :1024] = block.repeat(1, 1, 2).reshape(value.shape[0], -1)
        return result
    if len(old) != len(target_shape) or any(n % o for n, o in zip(target_shape, old)):
        raise ValueError(f'unsupported growth map: {name}: {old} -> {target_shape}')
    repeats = tuple(n//o for n, o in zip(target_shape, old))
    factor = 1.
    if name.endswith('qkv_conv.weight'):
        # Q,K,V occupy separate contiguous blocks, not interleaved whole QKV.
        blocks = value.chunk(3, 0)
        result = torch.cat([b.repeat(repeats[0], 1, 1) for b in blocks], 0)
    else:
        result = value.repeat(repeats)
    if value.ndim == 2 and name.endswith('weight'):
        if name == 'body.base.source.embedding.weight' or name == 'body.base.decoder.weight':
            factor = 1.
        elif name == 'body.mix.weight':
            factor = 1.  # final_norm is halved to preserve the tied decoder
        else:
            factor = 1/repeats[1]
    if name == 'body.base.final_norm.weight':
        factor = .5
    if name == 'body.key_norm.weight':
        factor = 1/math.sqrt(2)  # preserve dot/sqrt(width) table gate
    if covector:
        factor = factor / (math.prod(repeats)*factor*factor)
    if second:
        factor *= factor
    return result*factor


def lift_packed(old_base, new_base, recurrent):
    result = new_base.initial_state(recurrent.shape[1], device=recurrent.device,
                                    dtype=recurrent.dtype)
    for i, layer in enumerate(old_base.layers):
        following = new_base.layers[i]
        matrices = recurrent[i, :, :layer.matrix_size].reshape(-1, layer.heads, layer.dk, layer.dv)
        result[i, :, :following.matrix_size] = matrices.repeat(1, 2, 1, 1).flatten(1)
        history = recurrent[i, :, layer.matrix_size:].reshape(-1, 3, old_base.d, old_base.conv_kernel-1)
        result[i, :, following.matrix_size:] = history.repeat(1, 1, 2, 1).flatten(1)
    return result


def lift_read(value):
    if value.shape[-1] != 512:
        raise ValueError('only the inherited four-layer P receipt can be lifted')
    result = value.new_zeros(*value.shape[:-1], 32*256)
    result[..., :1024] = value.reshape(*value.shape[:-1], 4, 128).repeat(
        *([1]*(value.ndim-1)), 1, 2).reshape(*value.shape[:-1], 1024)
    return result


def configure_body_growth(engine, spec, *, initialize=True):
    if spec.get('schema') != 1:
        raise ValueError('unknown body growth mapping')
    model, old_body = engine.model, engine.model.body
    old_base = old_body.base
    if old_base.d != 128 or len(old_base.layers) != 4 or old_base.heads != 4:
        raise ValueError('growth requires the known 5356 four-layer source')
    if hasattr(model, 'head_coupling'):
        raise ValueError('learned coupling needs its own explicit migration')
    old_named = dict(model.named_parameters())
    old_params = list(engine.params)
    old_names = {id(p): n for n, p in old_named.items()}
    old_groups = [[old_names[id(p)] for p in g['params']] for g in engine.optimizer.param_groups]
    device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
    with torch.random.fork_rng(devices=[device] if device.type == 'cuda' else []):
        torch.manual_seed(spec.get('seed', 111435))
        new_base = GatedDeltaNetLM(**spec['backbone']).to(device=device, dtype=dtype)
        new_body = TableGDN(new_base, old_body.use_table, old_body.inject_layer).to(device=device,dtype=dtype)
        read = nn.Linear(256, 128, bias=False, device=device, dtype=dtype)
        write = nn.Linear(128, 256, bias=False, device=device, dtype=dtype)
        momentum = nn.Linear(32*256, model.encoder.out_features, bias=False, device=device,dtype=dtype)
    if initialize:
        with torch.no_grad():
            for name, parameter in new_body.named_parameters():
                key = 'body.'+name
                old = old_named.get(key)
                if old is not None:
                    parameter.copy_(parameter_lift(key, old, parameter.shape))
            for index in range(4, 32):
                layer = new_base.layers[index]
                # Residual identity at birth; gradient immediately opens both
                # output projections, then learns the complete new layer.
                layer.o_proj.weight.zero_()
                layer.mlp['down'].weight.zero_()
            eye = torch.eye(128, device=device,dtype=dtype)
            read.weight.copy_(torch.cat((eye, eye), 1)/2)
            write.weight.copy_(torch.cat((eye, eye), 0))
            momentum.weight.copy_(parameter_lift('momentum_to_boundary.weight',
                                   model.momentum_to_boundary.weight, momentum.weight.shape))
    model.body, model.feature_read, model.feature_write = new_body, read, write
    model.momentum_to_boundary = momentum
    new_named = dict(model.named_parameters())
    inherited = set(old_named)
    # Reuse group order for existing parameters. Newly born layers join the
    # original body group, bridges the extension group; deterministic on restore.
    extras = [n for n in new_named if n not in inherited]
    new_body_names = [n for n in extras if n.startswith('body.')]
    extension_names = [n for n in extras if not n.startswith('body.')]
    optimizer = engine.optimizer
    old_state = optimizer.state
    from collections import defaultdict
    optimizer.state = defaultdict(dict)
    for i, (group, names) in enumerate(zip(optimizer.param_groups, old_groups)):
        names = names + (new_body_names if i == 0 else [])
        group['params'] = [new_named[n] for n in names]
    # Match the loader's deterministic extra group, without resetting settings.
    optimizer.add_param_group(dict(params=[new_named[n] for n in extension_names],
                                   lr=optimizer.param_groups[0]['lr']))
    if initialize:
        for name, old in old_named.items():
            if old not in old_state:
                continue
            current = new_named[name]
            for field, value in old_state[old].items():
                optimizer.state[current][field] = (parameter_lift(name, value, current.shape,
                    covector=True, second=field in ('exp_avg_sq','max_exp_avg_sq'))
                    if torch.is_tensor(value) and value.ndim else value.clone()
                    if torch.is_tensor(value) else value)
    engine.params = list(new_named.values())
    engine.groups = ([p for n,p in new_named.items() if n == 'body.base.source.embedding.weight'],
                     [p for n,p in new_named.items() if n.startswith('body.') and n != 'body.base.source.embedding.weight'],
                     [p for n,p in new_named.items() if not n.startswith('body.')])
    if initialize:
        engine.recurrent = lift_packed(old_base, new_base, engine.recurrent)
        engine.bfss_momenta = [p.repeat(1, 2, 1, 1) for p in engine.bfss_momenta] + [
            torch.zeros(1, 8, 32, 32, device=device,dtype=dtype) for _ in range(28)]
        # Duplicating heads doubles pair-force sums; halve coupling to preserve
        # the old quiet flow on its duplicate submanifold, not its raw constant.
        engine.bfss_coupling /= 2
        indices = {id(p): i for i,p in enumerate(engine.params)}
        def remap(value, key=''):
            if key in ('momentum_read','_momentum_decision_read') and torch.is_tensor(value):
                return lift_read(value)
            if isinstance(value, (list,tuple)) and len(value) == len(old_params) and all(
                    x is None or torch.is_tensor(x) for x in value):
                result = [None]*len(engine.params)
                for p, x in zip(old_params, value):
                    if x is not None:
                        name = old_names[id(p)]
                        result[indices[id(new_named[name])]] = parameter_lift(
                            name, x, new_named[name].shape, covector=True)
                return tuple(result) if isinstance(value,tuple) else result
            if isinstance(value, dict):
                return {k:remap(v,k) for k,v in value.items()}
            if isinstance(value, list):
                return [remap(v) for v in value]
            if isinstance(value, tuple):
                return tuple(remap(v) for v in value)
            return value
        for name,value in list(vars(engine).items()):
            if name not in ('model','optimizer','params','groups','decision','recurrent','bfss_momenta','world','table'):
                setattr(engine,name,remap(value,name))
    model._momentum_current = engine.bfss_momenta
    engine.decision = ScopedDecisionGraph(model, capture=device.type=='cuda')
    engine.policy_replay_backend = 'batched_guarded'
    engine.body_growth_schema = 1
    validate_pending_participation(engine)
    return dict(parameters=sum(p.numel() for p in model.parameters()),
                backbone=new_base.config_dict(), fact_width=model.memory.width,
                coupling=engine.bfss_coupling, all_parameters_trainable=all(
                    p.requires_grad for p in model.parameters()),
                identity_layers=28, old_moment_states=len(old_state),
                archive_schema_unchanged=True)


def validate_pending_participation(engine):
    """Reject a stale gradient chart before any external event can happen."""
    pending=getattr(engine,'prev',None)
    actor=pending.get('actor') if isinstance(pending,dict) else None
    if actor is None:return
    if len(actor)!=len(engine.params):
        raise ValueError('pending actor length differs from current parameter chart')
    for i,(parameter,gradient) in enumerate(zip(engine.params,actor)):
        if gradient is not None and gradient.shape!=parameter.shape:
            raise ValueError(f'pending actor shape differs at parameter {i}: {gradient.shape} / {parameter.shape}')
