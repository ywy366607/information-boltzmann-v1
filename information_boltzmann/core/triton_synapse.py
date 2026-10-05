"""Fused Triton kernel for 25-million-edge synaptic transmission.

Fuses the gather, elementwise multiplication, and scatter/index_add into a
single kernel, eliminating the 102 MB intermediate VRAM allocations per token.
Uses float32 CUDA atomic reductions (summation order is nondeterministic).
The backward returns both pulse and connection-weight gradients without
retaining an edge-sized activation for each time step.
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def _sum_contiguous_keys(left_value, left_start, right_value, right_start):
        # Carry a reset flag rather than comparing segment endpoints: a
        # repeated key in unsorted input must not bridge intervening runs.
        return tl.where(right_start, right_value, left_value + right_value), left_start | right_start

    @triton.jit
    def _synaptic_joint_bwd_kernel(
        grad_current_ptr, spikes_ptr, pre_ptr, post_ptr, weight_ptr,
        grad_spikes_ptr, grad_weight_ptr, N_EDGES: tl.constexpr,
        SPIKE_GRAD: tl.constexpr, WEIGHT_GRAD: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N_EDGES
        pre = tl.load(pre_ptr + offsets, mask=mask, other=0)
        post = tl.load(post_ptr + offsets, mask=mask, other=0)
        gc = tl.load(grad_current_ptr + post, mask=mask, other=0.0)
        if SPIKE_GRAD:
            weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
            # Edges are grouped by presynaptic cell within each delay tier.
            # Reduce each contiguous run before issuing an atomic write.
            # Also correct for unsorted input: separated runs add independently.
            previous_pre = tl.load(pre_ptr + offsets - 1,
                                   mask=(offsets > 0) & mask, other=-1)
            run_start = (previous_pre != pre) | (offsets % BLOCK_SIZE == 0)
            sums, _ = tl.associative_scan((gc * weight, run_start), 0, _sum_contiguous_keys)
            next_pre = tl.load(pre_ptr + offsets + 1,
                               mask=(offsets + 1) < N_EDGES, other=-1)
            run_end = (next_pre != pre) | ((offsets + 1) % BLOCK_SIZE == 0)
            tl.atomic_add(grad_spikes_ptr + pre, sums, mask=mask & run_end)
        if WEIGHT_GRAD:
            spikes = tl.load(spikes_ptr + pre, mask=mask, other=0.0)
            tl.store(grad_weight_ptr + offsets, gc * spikes, mask=mask)

    @triton.jit
    def _synaptic_fwd_kernel(
        spikes_ptr, edge_pre_ptr, edge_post_ptr, edge_weight_ptr, current_ptr,
        N_EDGES: tl.constexpr, BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N_EDGES

        pre = tl.load(edge_pre_ptr + offsets, mask=mask, other=0)
        s = tl.load(spikes_ptr + pre, mask=mask, other=0.0)
        # Silent presynaptic pulses contribute exactly zero. Avoid their
        # destination reads and atomic writes, but keep the full backward:
        # a silent pulse can still receive a nonzero surrogate derivative.
        active = mask & (s != 0.0)
        post = tl.load(edge_post_ptr + offsets, mask=active, other=0)
        w = tl.load(edge_weight_ptr + offsets, mask=active, other=0.0)
        contrib = s * w
        tl.atomic_add(current_ptr + post, contrib, mask=active)

    @triton.jit
    def _synaptic_bwd_kernel(
        grad_current_ptr, edge_pre_ptr, edge_post_ptr, edge_weight_ptr, grad_spikes_ptr,
        N_EDGES: tl.constexpr, BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N_EDGES

        pre = tl.load(edge_pre_ptr + offsets, mask=mask, other=0)
        post = tl.load(edge_post_ptr + offsets, mask=mask, other=0)
        w = tl.load(edge_weight_ptr + offsets, mask=mask, other=0.0)
        gc = tl.load(grad_current_ptr + post, mask=mask, other=0.0)

        contrib = gc * w
        tl.atomic_add(grad_spikes_ptr + pre, contrib, mask=mask)


    class TritonSynapticTransmission(torch.autograd.Function):
        """Fused Triton synaptic transmission with forward & backward atomic operations."""

        @staticmethod
        def forward(ctx, spikes, edge_pre, edge_post, edge_weight):
            ctx.save_for_backward(edge_pre, edge_post, edge_weight, spikes)
            ctx.spikes_shape = spikes.shape
            n_neurons = spikes.shape[-1]
            current = torch.zeros(n_neurons, dtype=torch.float32, device=spikes.device)
            n_edges = edge_pre.shape[0]
            BLOCK_SIZE = 1024
            grid = (triton.cdiv(n_edges, BLOCK_SIZE),)
            _synaptic_fwd_kernel[grid](
                spikes.flatten(), edge_pre, edge_post, edge_weight, current,
                n_edges, BLOCK_SIZE
            )
            return current.unsqueeze(0)

        @staticmethod
        def backward(ctx, grad_current):
            edge_pre, edge_post, edge_weight, spikes = ctx.saved_tensors
            n_neurons = grad_current.shape[-1]
            grad_current = grad_current.contiguous()
            gs_needed, gw_needed = ctx.needs_input_grad[0], ctx.needs_input_grad[3]
            grad_spikes = torch.zeros_like(spikes) if gs_needed else None
            grad_weight = torch.empty_like(edge_weight) if gw_needed else None
            n_edges = edge_pre.shape[0]
            BLOCK_SIZE = 1024
            grid = (triton.cdiv(n_edges, BLOCK_SIZE),)
            if n_edges:
                _synaptic_joint_bwd_kernel[grid](
                    grad_current.flatten(), spikes.flatten(), edge_pre, edge_post,
                    edge_weight, grad_spikes if gs_needed else grad_current,
                    grad_weight if gw_needed else grad_current, n_edges,
                    gs_needed, gw_needed, BLOCK_SIZE)
            return grad_spikes, None, None, grad_weight

    class TritonDelayedSynapticTransmission(torch.autograd.Function):
        """Fused Triton multi-delay synaptic transmission using segmented edge groups."""

        @staticmethod
        def forward(ctx, s1, s2, s3, s4, edge_pre, edge_post, edge_weight, delay_splits):
            ctx.save_for_backward(edge_pre, edge_post, edge_weight, s1, s2, s3, s4)
            ctx.delay_splits = delay_splits
            n_neurons = s1.shape[-1]
            current = torch.zeros(n_neurons, dtype=torch.float32, device=s1.device)
            spikes_list = [s1, s2, s3, s4]
            BLOCK_SIZE = 1024
            for k in range(len(delay_splits) - 1):
                start = int(delay_splits[k])
                end = int(delay_splits[k+1])
                n_sub = end - start
                if n_sub == 0:
                    continue
                grid = (triton.cdiv(n_sub, BLOCK_SIZE),)
                _synaptic_fwd_kernel[grid](
                    spikes_list[k].flatten(),
                    edge_pre[start:end],
                    edge_post[start:end],
                    edge_weight[start:end],
                    current,
                    n_sub, BLOCK_SIZE
                )
            return current.unsqueeze(0)

        @staticmethod
        def backward(ctx, grad_current):
            edge_pre, edge_post, edge_weight, *spikes = ctx.saved_tensors
            delay_splits = ctx.delay_splits
            n_neurons = grad_current.shape[-1]
            BLOCK_SIZE = 1024
            grad_spikes = []
            grad_current = grad_current.contiguous()
            gw_needed = ctx.needs_input_grad[6]
            grad_weight = torch.empty_like(edge_weight) if gw_needed else None
            for k in range(len(delay_splits) - 1):
                start = int(delay_splits[k])
                end = int(delay_splits[k+1])
                n_sub = end - start
                gs_needed = ctx.needs_input_grad[k]
                gs = torch.zeros_like(spikes[k]) if gs_needed else None
                if n_sub > 0:
                    grid = (triton.cdiv(n_sub, BLOCK_SIZE),)
                    _synaptic_joint_bwd_kernel[grid](
                        grad_current.flatten(), spikes[k].flatten(),
                        edge_pre[start:end], edge_post[start:end], edge_weight[start:end],
                        gs if gs_needed else grad_current,
                        grad_weight[start:end] if gw_needed else grad_current,
                        n_sub, gs_needed, gw_needed, BLOCK_SIZE)
                grad_spikes.append(gs)
            return *grad_spikes, None, None, grad_weight, None


class PyTorchSynapticTransmission(torch.autograd.Function):
    """Fallback PyTorch implementation for CPU or platforms without Triton."""

    @staticmethod
    def forward(ctx, spikes, edge_pre, edge_post, edge_weight):
        contribution = spikes.t()[edge_pre] * edge_weight[:, None]
        current = torch.zeros_like(spikes).t()
        current.index_add_(0, edge_post, contribution)
        ctx.save_for_backward(edge_pre, edge_post, edge_weight, spikes)
        ctx.spikes_shape = spikes.shape
        return current.t()

    @staticmethod
    def backward(ctx, grad_current):
        edge_pre, edge_post, edge_weight, spikes = ctx.saved_tensors
        grad_spikes = torch.zeros(ctx.spikes_shape, dtype=grad_current.dtype,
                                  device=grad_current.device)
        grad_spikes = grad_spikes.t().index_add(
            0, edge_pre,
            grad_current.t()[edge_post] * edge_weight[:, None]).t()
        grad_weight = (grad_current[:, edge_post] * spikes[:, edge_pre]).sum(0) if ctx.needs_input_grad[3] else None
        return grad_spikes, None, None, grad_weight


class PyTorchDelayedSynapticTransmission(torch.autograd.Function):
    """Fallback PyTorch multi-delay transmission for CPU or platforms without Triton."""

    @staticmethod
    def forward(ctx, s1, s2, s3, s4, edge_pre, edge_post, edge_weight, delay_splits):
        spikes_list = [s1, s2, s3, s4]
        current = torch.zeros_like(s1).t()
        for k in range(len(delay_splits) - 1):
            start = int(delay_splits[k])
            end = int(delay_splits[k+1])
            if end > start:
                sub_pre = edge_pre[start:end]
                sub_post = edge_post[start:end]
                sub_w = edge_weight[start:end]
                contrib = spikes_list[k].t()[sub_pre] * sub_w[:, None]
                current.index_add_(0, sub_post, contrib)
        ctx.save_for_backward(edge_pre, edge_post, edge_weight, s1, s2, s3, s4)
        ctx.delay_splits = delay_splits
        ctx.s_shape = s1.shape
        return current.t()

    @staticmethod
    def backward(ctx, grad_current):
        edge_pre, edge_post, edge_weight, *spikes = ctx.saved_tensors
        delay_splits = ctx.delay_splits
        grad_spikes = []
        grad_weight = torch.empty_like(edge_weight) if ctx.needs_input_grad[6] else None
        for k in range(len(delay_splits) - 1):
            start = int(delay_splits[k])
            end = int(delay_splits[k+1])
            gs = torch.zeros(ctx.s_shape, dtype=grad_current.dtype, device=grad_current.device).t()
            if end > start:
                sub_pre = edge_pre[start:end]
                sub_post = edge_post[start:end]
                sub_w = edge_weight[start:end]
                contrib = grad_current.t()[sub_post] * sub_w[:, None]
                gs = gs.index_add(0, sub_pre, contrib)
                if grad_weight is not None:
                    grad_weight[start:end] = (grad_current[:, sub_post] * spikes[k][:, sub_pre]).sum(0)
            grad_spikes.append(gs.t())
        return *grad_spikes, None, None, grad_weight, None


def execute_synaptic_transmission(spikes: torch.Tensor, edge_pre: torch.Tensor,
                                  edge_post: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
    if HAS_TRITON and spikes.is_cuda and edge_pre.is_cuda:
        if spikes.shape[0] == 1:
            return TritonSynapticTransmission.apply(spikes, edge_pre, edge_post, edge_weight)
        else:
            outs = [TritonSynapticTransmission.apply(spikes[b:b+1], edge_pre, edge_post, edge_weight)
                    for b in range(spikes.shape[0])]
            return torch.cat(outs, dim=0)
    return PyTorchSynapticTransmission.apply(spikes, edge_pre, edge_post, edge_weight)


def execute_delayed_synaptic_transmission(
    spikes_ring: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | list[torch.Tensor],
    edge_pre: torch.Tensor,
    edge_post: torch.Tensor,
    edge_weight: torch.Tensor,
    delay_splits: list[int] | tuple[int, ...] | torch.Tensor,
) -> torch.Tensor:
    """Execute biological multi-delay synaptic transmission along segmented edges."""
    if isinstance(delay_splits, torch.Tensor):
        delay_splits = tuple(int(x) for x in delay_splits.cpu().tolist())
    elif not isinstance(delay_splits, (tuple, list)):
        delay_splits = tuple(int(x) for x in delay_splits)
    s1, s2, s3, s4 = spikes_ring[0], spikes_ring[1], spikes_ring[2], spikes_ring[3]
    if HAS_TRITON and s1.is_cuda and edge_pre.is_cuda:
        if s1.shape[0] == 1:
            return TritonDelayedSynapticTransmission.apply(s1, s2, s3, s4, edge_pre, edge_post, edge_weight, delay_splits)
        else:
            outs = [TritonDelayedSynapticTransmission.apply(
                s1[b:b+1], s2[b:b+1], s3[b:b+1], s4[b:b+1], edge_pre, edge_post, edge_weight, delay_splits)
                for b in range(s1.shape[0])]
            return torch.cat(outs, dim=0)
    return PyTorchDelayedSynapticTransmission.apply(s1, s2, s3, s4, edge_pre, edge_post, edge_weight, delay_splits)
