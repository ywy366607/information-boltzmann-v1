"""Nonreentrant checkpointing with tensor-only inputs for nested physical state."""
from __future__ import annotations

from dataclasses import fields, is_dataclass

import torch
from torch.utils.checkpoint import checkpoint


def _flatten(value, tensors):
    if isinstance(value, torch.Tensor):
        index = len(tensors)
        tensors.append(value)
        return ('tensor', index)
    if is_dataclass(value) and not isinstance(value, type):
        return ('dataclass', type(value), tuple(
            (field.name, _flatten(getattr(value, field.name), tensors))
            for field in fields(value)))
    if isinstance(value, tuple):
        return ('tuple', tuple(_flatten(item, tensors) for item in value))
    if isinstance(value, list):
        return ('list', tuple(_flatten(item, tensors) for item in value))
    if isinstance(value, dict):
        return ('dict', tuple((_flatten(key, tensors), _flatten(item, tensors))
                              for key, item in value.items()))
    if value is None or isinstance(value, (bool, int, float, complex, str, bytes,
                                           torch.dtype, torch.device)):
        return ('constant', value)
    raise TypeError(f'Unsupported checkpoint-state component: {type(value).__name__}')


def _restore(schema, tensors):
    kind = schema[0]
    if kind == 'tensor':
        return tensors[schema[1]]
    if kind == 'dataclass':
        return schema[1](**{name: _restore(child, tensors) for name, child in schema[2]})
    if kind == 'tuple':
        return tuple(_restore(child, tensors) for child in schema[1])
    if kind == 'list':
        return [_restore(child, tensors) for child in schema[1]]
    if kind == 'dict':
        return {_restore(key, tensors): _restore(child, tensors) for key, child in schema[1]}
    return schema[1]


def checkpoint_state(function, state, *args, preserve_rng_state=False):
    """Checkpoint a dataclass state without retaining its hidden tensor inputs.

    PyTorch saves only top-level tensor arguments through saved-tensor hooks;
    passing an entire dataclass captures it strongly in ``ctx.get_args``. For
    nested checkpoints that bypasses the outer recomputation boundary and keeps
    every inner physical state resident. This wrapper flattens all tensor leaves
    before checkpointing. Its closure contains only the callable and a tensor-
    free reconstruction schema, never the incoming state or tensor tuple.

    Dataclasses, tuples, lists, dictionaries and scalar constants are supported.
    Tensor values, gradients, aliases, dtypes and clocks are preserved; no leaf
    is detached or cloned. The callable must not independently capture state.
    """
    tensors = []
    schema = _flatten((state, args), tensors)

    def execute(*leaves):
        current, arguments = _restore(schema, leaves)
        return function(current, *arguments)

    return checkpoint(execute, *tensors, use_reentrant=False,
                      preserve_rng_state=preserve_rng_state)


class _StateVJPCheckpoint(torch.autograd.Function):
    """One independent recomputation engine per event, with no retained graph."""

    @staticmethod
    def forward(ctx, function, input_schema, metadata, input_count, parameters, *inputs):
        ctx.set_materialize_grads(False)
        ctx.function = function
        ctx.input_schema = input_schema
        ctx.input_count = input_count
        ctx.input_requires_grad = tuple(value.requires_grad for value in inputs[:input_count])
        # The callable reads these exact module leaves. Saved-tensor hooks may
        # unpack a different Tensor object, so keep the original leaf identities
        # as well as saving their values for ordinary autograd version checks.
        ctx.parameters = parameters
        ctx.parameter_versions = tuple(value._version for value in parameters)
        ctx.save_for_backward(*inputs)
        current, arguments = _restore(input_schema, inputs[:input_count])
        with torch.no_grad():
            result = function(current, *arguments)
        outputs = []
        ctx.output_schema = _flatten(result, outputs)
        ctx.output_specs = tuple((value.shape, value.dtype, value.device, value.layout)
                                 for value in outputs)
        metadata['schema'] = ctx.output_schema
        nondifferentiable = tuple(value for value in outputs
                                  if not (value.is_floating_point() or value.is_complex()))
        if nondifferentiable:
            ctx.mark_non_differentiable(*nondifferentiable)
        return tuple(outputs)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, *output_gradients):
        saved = ctx.saved_tensors  # Also checks versions of all event inputs.
        if any(value._version != version for value, version in
               zip(ctx.parameters, ctx.parameter_versions)):
            raise RuntimeError('A checkpoint_state_vjp parameter changed before backward')
        leaves = [value.detach().requires_grad_(requires_grad)
                  for value, requires_grad in
                  zip(saved[:ctx.input_count], ctx.input_requires_grad)]
        targets, positions = [], []
        for index, value in enumerate((*leaves, *ctx.parameters)):
            if value.requires_grad:
                targets.append(value)
                positions.append(index)
        gradients = [None] * (ctx.input_count + len(ctx.parameters))
        with torch.enable_grad():
            current, arguments = _restore(ctx.input_schema, leaves)
            replayed = ctx.function(current, *arguments)
            outputs = []
            output_schema = _flatten(replayed, outputs)
            output_specs = tuple((value.shape, value.dtype, value.device, value.layout)
                                 for value in outputs)
            if output_schema != ctx.output_schema or output_specs != ctx.output_specs:
                raise RuntimeError('checkpoint_state_vjp output structure changed during replay')
            selected = [(value, gradient) for value, gradient in zip(outputs, output_gradients)
                        if gradient is not None and value.requires_grad]
            if selected and targets:
                local_gradients = torch.autograd.grad(
                    tuple(value for value, _ in selected), targets,
                    grad_outputs=tuple(gradient for _, gradient in selected),
                    allow_unused=True, retain_graph=False, create_graph=False)
                for index, gradient in zip(positions, local_gradients):
                    gradients[index] = gradient
        return (None, None, None, None, None, *gradients)


def checkpoint_state_vjp(function, state, *args, parameters=()):
    """Recompute one deterministic event for an exact first-order state/parameter VJP.

    Unlike :func:`checkpoint_state`, the backward replay runs in an independent
    local autograd engine and releases that event's graph before the preceding
    event is visited. Every state and argument cotangent returns to the outer
    engine, so chaining calls retains the complete declared BPTT interval.

    ``parameters`` must contain the original module leaf tensors read by the
    callable; repeated identities are counted once. All differentiable nonleaf
    values (prepared coefficients, normalized tables, durations) must instead
    be explicit state/arguments. Capturing their shared graph in the callable
    would bypass this boundary. Duplicate ordinary inputs remain separate
    branches and their cotangents are combined by the outer engine.

    The callable must be deterministic and free of in-place input/parameter or
    buffer changes, hidden randomness, and object-identity-dependent branches.
    Parameter gradient hooks are unsupported: local and outer reverse engines
    would otherwise apply the same hook twice.
    Noise must be an explicit input. This helper supports first-order reverse
    differentiation only. Floating/complex outputs expose a checkpoint node;
    an explicitly detached output still returns no gradient on replay. Integer
    and boolean outputs stay nondifferentiable. Nested pure tensor containers
    use the same schema contract as :func:`checkpoint_state`.
    """
    if not torch.is_grad_enabled():
        return function(state, *args)
    unique_parameters, seen = [], set()
    for parameter in parameters:
        if not isinstance(parameter, torch.Tensor) or not parameter.is_leaf:
            raise TypeError('checkpoint_state_vjp parameters must be original module leaf tensors')
        if (getattr(parameter, '_backward_hooks', None)
                or getattr(parameter, '_post_accumulate_grad_hooks', None)):
            raise ValueError('checkpoint_state_vjp does not support parameter gradient hooks')
        if id(parameter) not in seen:
            seen.add(id(parameter))
            unique_parameters.append(parameter)
    inputs = []
    input_schema = _flatten((state, args), inputs)
    metadata = {}
    outputs = _StateVJPCheckpoint.apply(
        function, input_schema, metadata, len(inputs), tuple(unique_parameters),
        *inputs, *unique_parameters)
    return _restore(metadata['schema'], outputs)
