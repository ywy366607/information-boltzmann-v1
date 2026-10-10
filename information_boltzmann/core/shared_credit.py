"""One physical replay with separate task and writer cotangent lanes.

The auxiliary lane propagates through every physical state, but updates only
its declared writer leaves. It never supplies structural likelihood evidence.
This is exact first-order differentiation, not a shorter credit window.
"""
import torch

from .state_checkpoint import _flatten, _restore


class _SharedCreditCheckpoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, function, metadata, main_schema, aux_schema, main_count,
                state_count, parameters, auxiliary_ids, shared_indices, *inputs):
        ctx.set_materialize_grads(False)
        ctx.function = function
        ctx.main_schema, ctx.aux_schema = main_schema, aux_schema
        ctx.main_count, ctx.state_count = main_count, state_count
        ctx.parameters, ctx.auxiliary_ids = parameters, auxiliary_ids
        ctx.shared_indices = shared_indices
        ctx.requirements = tuple(x.requires_grad for x in inputs[:main_count + state_count])
        ctx.versions = tuple(p._version for p in parameters)
        ctx.saved_devices = tuple(value.device for value in inputs)
        history = {}
        saved = []
        for i, value in enumerate(inputs):
            is_state = i < state_count or main_count <= i < main_count + state_count
            if metadata.get('history_offload', False) and is_state and value.is_cuda and value.numel() >= 4096:
                # The two cotangent lanes share the same primal storage. Copy
                # identical state views once, retaining exact FP32 history on
                # the host instead of ~300 MiB of inactive GPU field history.
                key = (value.data_ptr(), value.shape, value.stride(), value.dtype)
                if key not in history:
                    history[key] = value.detach().to('cpu', copy=True)
                saved.append(history[key])
            else:
                saved.append(value)
        ctx.save_for_backward(*saved)
        state, arguments = _restore(main_schema, inputs[:main_count])
        with torch.no_grad():
            result = function(state, *arguments)
        primary = []
        ctx.output_schema = _flatten(result, primary)
        auxiliary = []
        ctx.aux_output_schema = _flatten((result[1], result[2]['_write_free_energy']), auxiliary)
        # Separate output identities preserve two independent adjoint lanes;
        # views share physical storage and do not duplicate the field history.
        auxiliary = [x.view_as(x) for x in auxiliary]
        ctx.primary_count = len(primary)
        metadata.update(primary=ctx.output_schema, auxiliary=ctx.aux_output_schema)
        outputs = primary + auxiliary
        ctx.output_specs = tuple((x.shape, x.dtype, x.device) for x in outputs)
        ctx.mark_non_differentiable(*(x for x in outputs
                                      if not (x.is_floating_point() or x.is_complex())))
        return tuple(outputs)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, *cotangents):
        restored = {}
        saved = []
        for value, device in zip(ctx.saved_tensors, ctx.saved_devices):
            if value.device != device:
                key = (value.data_ptr(), value.shape, value.stride(), device)
                if key not in restored:
                    restored[key] = value.to(device=device, copy=True)
                saved.append(restored[key])
            else:
                saved.append(value)
        if any(p._version != version for p, version in zip(ctx.parameters, ctx.versions)):
            raise RuntimeError('Shared-credit parameter changed before backward')
        leaves = []
        for i, value in enumerate(saved[:ctx.main_count]):
            required = ctx.requirements[i]
            if i < ctx.state_count:
                required |= ctx.requirements[ctx.main_count + i]
            leaves.append(value.detach().requires_grad_(required))
        gradients = [None] * (ctx.main_count + ctx.state_count + len(ctx.parameters))
        with torch.enable_grad():
            state, arguments = _restore(ctx.main_schema, leaves)
            # The first retained AOT VJP consumes private workspace; the final
            # VJP then uses the original saved buffers and releases the graph.
            # Reusing those buffers on both passes is unsafe on this compiler.
            parameter_storage = {p.untyped_storage().data_ptr() for p in ctx.parameters}
            # Explicit coefficients/table arguments are caller-owned immutable
            # inputs, not donated backward workspace. In particular, copying
            # the 147-MiB frozen word table per token would erase the speed gain.
            parameter_storage.update(x.untyped_storage().data_ptr()
                                     for x in leaves[ctx.state_count:])
            preserve_saved = [True]
            def unpack(value):
                if (not preserve_saved[0]
                        or value.untyped_storage().data_ptr() in parameter_storage
                        or any(s == 0 for s in value.stride())):
                    return value
                # Inductor's padded saved buffers have an explicit stride
                # contract; clone() may compact them and break that contract.
                owned = torch.empty_strided(value.shape, value.stride(),
                                            dtype=value.dtype, device=value.device)
                return owned.copy_(value)
            # Retain storage, not the source Tensor's grad_fn: returning the
            # original Tensor from a pack hook creates an autograd reference
            # cycle and can retain an entire previous window on CUDA.
            with torch.autograd.graph.saved_tensors_hooks(lambda x: x.detach(), unpack):
                result = ctx.function(state, *arguments)
            primary, auxiliary = [], []
            schema = _flatten(result, primary)
            aux_schema = _flatten((result[1], result[2]['_write_free_energy']), auxiliary)
            specs = tuple((x.shape, x.dtype, x.device) for x in primary + auxiliary)
            if (schema != ctx.output_schema or aux_schema != ctx.aux_output_schema
                    or specs != ctx.output_specs):
                raise RuntimeError('Shared-credit replay schema changed')
            selected_main = [(x, g) for x, g in zip(primary, cotangents[:ctx.primary_count])
                             if g is not None and x.requires_grad]
            selected_aux = [(x, g) for x, g in zip(auxiliary, cotangents[ctx.primary_count:])
                            if g is not None and x.requires_grad]
            aux_targets, aux_destinations = [], []
            for i, leaf in enumerate(leaves):
                if i < ctx.state_count and ctx.requirements[ctx.main_count + i]:
                    aux_targets.append(leaf)
                    aux_destinations.append(ctx.main_count + i)
                elif i in ctx.shared_indices and leaf.requires_grad:
                    aux_targets.append(leaf)
                    aux_destinations.append(i)
            for i, p in enumerate(ctx.parameters):
                if p.requires_grad and id(p) in ctx.auxiliary_ids:
                    aux_targets.append(p)
                    aux_destinations.append(ctx.main_count + ctx.state_count + i)
            if selected_aux and aux_targets:
                local = torch.autograd.grad(tuple(x for x, _ in selected_aux), aux_targets,
                    grad_outputs=tuple(g for _, g in selected_aux),
                    retain_graph=bool(selected_main), allow_unused=True)
                for i, gradient in zip(aux_destinations, local):
                    gradients[i] = gradient
            preserve_saved[0] = False
            main_targets, main_destinations = [], []
            for i, leaf in enumerate(leaves):
                if ctx.requirements[i]:
                    main_targets.append(leaf)
                    main_destinations.append(i)
            for i, p in enumerate(ctx.parameters):
                if p.requires_grad:
                    main_targets.append(p)
                    main_destinations.append(ctx.main_count + ctx.state_count + i)
            if selected_main and main_targets:
                local = torch.autograd.grad(tuple(x for x, _ in selected_main), main_targets,
                    grad_outputs=tuple(g for _, g in selected_main), allow_unused=True)
                from collections import Counter
                ownership = Counter(x.untyped_storage().data_ptr() for x in gradients if x is not None)
                protected = {x.untyped_storage().data_ptr() for x in (*leaves, *ctx.parameters)}
                protected.update(g.untyped_storage().data_ptr() for g in cotangents if g is not None)
                for i, gradient in zip(main_destinations, local):
                    if gradient is not None:
                        old = gradients[i]
                        if old is None:
                            gradients[i] = gradient
                        elif (i >= ctx.main_count + ctx.state_count and old.ndim >= 2
                              and old._base is None
                              and ownership[old.untyped_storage().data_ptr()] == 1
                              and old.untyped_storage().data_ptr() not in protected):
                            # These first-order parameter gradients are private
                            # accumulators, not primal/cotangent state. Reuse
                            # them rather than keeping a third large matrix.
                            old.add_(gradient)
                        else:
                            gradients[i] = old + gradient
        return (None,) * 9 + tuple(gradients)


def checkpoint_shared_credit(function, state, auxiliary_state, *args,
                             parameters, auxiliary_parameters, shared_auxiliary_tensors=(),
                             history_offload=False):
    """Execute once; replay once; keep task and local-writer VJPs separate.

    Both state lanes must have identical physical values. The auxiliary copy
    carries only its separate cotangent history. Callers supply the same initial
    state, and thereafter only the paired outputs of this function.
    """
    if not torch.is_grad_enabled():
        raise ValueError('Shared credit is a first-order training execution path')
    # Two cotangent lanes reuse one AOT graph. Buffer donation would consume
    # saved activations on the first VJP and invalidate the second one.
    import torch._functorch.config as aot_config
    aot_config.donated_buffer = False
    parameters = tuple(dict.fromkeys(parameters))
    tensors = []
    main_schema = _flatten((state, args), tensors)
    main_count = len(tensors)
    state_tensors = []
    _flatten(state, state_tensors)
    aux_schema = _flatten(auxiliary_state, tensors)
    if len(tensors) - main_count != len(state_tensors):
        raise ValueError('Shared-credit state schemas differ')
    shared_ids = {id(x) for x in shared_auxiliary_tensors}
    shared_indices = tuple(i for i, x in enumerate(tensors[:main_count]) if id(x) in shared_ids)
    auxiliary_ids = frozenset(id(p) for p in auxiliary_parameters)
    if not auxiliary_ids.issubset({id(p) for p in parameters}):
        raise ValueError('Auxiliary parameters must belong to the task replay leaves')
    metadata = {'history_offload': history_offload}
    outputs = _SharedCreditCheckpoint.apply(function, metadata, main_schema, aux_schema,
        main_count, len(state_tensors), parameters, auxiliary_ids, shared_indices,
        *tensors, *parameters)
    primary = []
    _flatten(_restore(metadata['primary'], outputs), primary)
    return (_restore(metadata['primary'], outputs),
            _restore(metadata['auxiliary'], outputs[len(primary):]))
