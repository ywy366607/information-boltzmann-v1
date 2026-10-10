"""Exact first-order single-row linear gradients, accumulated by one GEMM.

Store small input/cotangent factors during a full recurrent backward. For a
linear map, sum_t dy_t.T @ x_t == cat(dy).T @ cat(x). No credit is shortened.
Caller flushes before clipping/Adam; parameters and state_dict stay unchanged.
"""
from types import MethodType
import copy
import weakref

import torch
from torch import nn
from torch.nn import functional as F


_COLLECTORS = weakref.WeakValueDictionary()


@torch.library.custom_op('ib_credit::linear_input_and_factors', mutates_args=('receipt',))
def _linear_input_and_factors(inputs: torch.Tensor, gradient: torch.Tensor,
                             weight: torch.Tensor, receipt: torch.Tensor,
                             collector_id: int, weight_index: int) -> torch.Tensor:
    collector = _COLLECTORS[collector_id]
    collector.record(collector.weights[weight_index], inputs, gradient)
    # An explicit mutation keeps this credit side effect in AOT's backward
    # even when only the weight (not the frozen input) requires a gradient.
    receipt.add_(0)
    return gradient @ weight


@_linear_input_and_factors.register_fake
def _linear_input_and_factors_fake(inputs, gradient, weight, receipt, collector_id, weight_index):
    return torch.empty_like(inputs)


class _DeferredLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weight, bias, receipt, collector_id, weight_index):
        ctx.receipt, ctx.collector_id, ctx.weight_index = receipt, collector_id, weight_index
        ctx.save_for_backward(inputs, weight)
        ctx.has_bias = bias is not None
        return F.linear(inputs, weight, bias)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, gradient):
        inputs, weight = ctx.saved_tensors
        if ctx.needs_input_grad[1]:
            input_gradient = _linear_input_and_factors(inputs, gradient, weight,
                ctx.receipt, ctx.collector_id, ctx.weight_index)
        else:
            input_gradient = gradient @ weight if ctx.needs_input_grad[0] else None
        if not ctx.needs_input_grad[0]:
            input_gradient = None
        bias_gradient = (gradient.reshape(-1, gradient.shape[-1]).sum(0)
                         if ctx.has_bias and ctx.needs_input_grad[2] else None)
        return input_gradient, None, bias_gradient, None, None, None


class DeferredWriterCredit:
    def __init__(self, writer, reader=None):
        self.factors = {}
        self.flushes = 0
        self.factor_rows = 0
        self.modules = []
        self.enabled = True
        self.weights = {}
        self.collector_id = id(self)
        _COLLECTORS[self.collector_id] = self
        modules = [('write_agent.' + n, m) for n, m in writer.named_modules()]
        if reader is not None:
            modules += [('readout.' + n, m) for n, m in reader.named_modules()]
        for name, module in modules:
            # Local-content rows already form a useful spatial batch. The
            # symbolic single-individual policy MLPs have one row per event.
            if not isinstance(module, nn.Linear) or name == 'write_agent.local_content':
                continue
            module._deferred_credit_owner = self
            module._deferred_weight_index = len(self.weights)
            self.weights[module._deferred_weight_index] = module.weight
            module.register_buffer('_deferred_receipt', module.weight.new_zeros(()), persistent=False)
            def forward(layer, inputs):
                if (layer._deferred_credit_owner.enabled and torch.is_grad_enabled() and layer.weight.requires_grad
                        and inputs.numel() == inputs.shape[-1]):
                    return _DeferredLinear.apply(inputs, layer.weight, layer.bias,
                        layer._deferred_receipt, layer._deferred_credit_owner.collector_id,
                        layer._deferred_weight_index)
                return F.linear(inputs, layer.weight, layer.bias)
            module.forward = MethodType(forward, module)
            self.modules.append(name)

    def __deepcopy__(self, memo):
        clone = self.__class__.__new__(self.__class__)
        memo[id(self)] = clone
        for name, value in self.__dict__.items():
            setattr(clone, name, copy.deepcopy(value, memo))
        clone.collector_id = id(clone)
        _COLLECTORS[clone.collector_id] = clone
        return clone

    def record(self, weight, inputs, cotangent):
        entry = self.factors.setdefault(id(weight), (weight, [], []))
        # Own small factors: a subsequent reverse kernel may reuse its
        # cotangent workspace, but cannot overwrite this window's credit.
        entry[1].append(inputs.detach().reshape(-1, inputs.shape[-1]).clone())
        entry[2].append(cotangent.detach().reshape(-1, cotangent.shape[-1]).clone())

    @torch.no_grad()
    def flush(self):
        for weight, inputs, cotangents in self.factors.values():
            x, dy = torch.cat(inputs), torch.cat(cotangents)
            # addmm writes directly into the existing pending credit buffer.
            # A genuinely unused weight has no factors and retains None.
            if weight.grad is None:
                weight.grad = dy.transpose(0, 1) @ x
            else:
                weight.grad.addmm_(dy.transpose(0, 1), x)
            self.factor_rows += x.shape[0]
        self.factors.clear()
        self.flushes += 1

    def assert_flushed(self):
        if self.factors:
            raise RuntimeError('Deferred writer credit must be flushed before state serialization')
