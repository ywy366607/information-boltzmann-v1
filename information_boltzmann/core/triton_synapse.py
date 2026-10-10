"""Fused Triton kernel for 25-million-edge synaptic transmission.

Fuses the gather, elementwise multiplication, and scatter/index_add into a
single kernel, eliminating the 102 MB intermediate VRAM allocations per token.
Uses float32 CUDA atomic reductions (summation order is nondeterministic).
The backward returns both pulse and connection-weight gradients without
retaining an edge-sized activation for each time step.
"""
from __future__ import annotations

import os

import torch
import numpy as np

# Fused deterministic incoming forward (one program per postsynaptic row, fixed order, no atomics, silent presynaptic
# edges skip their weight load). Set FLY_FUSED_INCOMING=0 to reproduce the previous gather/segment_reduce arithmetic.
FUSED_INCOMING = os.environ.get('FLY_FUSED_INCOMING', '1') != '0'
INCOMING_ROWS, INCOMING_BLOCK = 4, 64      # rows per program, edges per vector step (measured on the GTX 1650)
# Event-driven forward: only presynaptic cells that spiked push along their outgoing edges. Deterministic because the
# contributions are summed as 64-bit fixed-point integers (exact, order-independent). FLY_EVENT_DRIVEN=0 disables it.
EVENT_DRIVEN = os.environ.get('FLY_EVENT_DRIVEN', '1') != '0'
EVENT_ROWS, EVENT_BLOCK = 16, 128   # measured on the GTX 1650
FIXED_SCALE = float(2 ** 40)
_OUTGOING_CACHE = {}
# Set by the learner around its own loss.backward(): the event-driven edge-weight gradient is accumulated straight into
# edge_weight.grad instead of returning a full edge-sized tensor per tick. Off everywhere else (autograd.grad keeps the
# ordinary contract).
DIRECT_EDGE_GRAD = False


# Checkpoint recompute reuse: during a token's first forward the transmission outputs are RECORDED; when the checkpoint
# recomputes that token for backward they are REPLAYED instead of recomputed (same values, the kernel is skipped; the
# pulses are still saved for the exact backward). Driven by advance_fly_token_adaptive. FLY_REUSE_TRANSMISSION=0 disables.
REUSE_TRANSMISSION = os.environ.get('FLY_REUSE_TRANSMISSION', '1') != '0'
TRANSMISSION_RECORD = None      # list being filled during a first forward, or None
TRANSMISSION_REPLAY = None      # iterator over recorded outputs during a recompute, or None

# Dense pulse gradient (grad_spikes[tier][pre] = sum over that cell's tier edges of w * grad_current[post]) as run sums:
# edges are grouped by (tier, presynaptic cell), so each edge stores only a uint8 run index inside its 256-edge block
# instead of its 4-byte presynaptic id; every run piece is stored once and the pieces of a cell are then added in a fixed
# order (no atomics, bitwise reproducible). FLY_PULSE_RUNS=0 returns to the per-edge atomic kernel.
PULSE_RUNS = os.environ.get('FLY_PULSE_RUNS', '1') != '0'
PULSE_BLOCK, PULSE_SLOTS = 256, 256
# Event-driven forward over a compacted list of (tier, cell) slots that spiked, built on the device (no host sync), instead
# of scanning every cell of every tier. Uses the same (tier, cell)-grouped edge layout; FLY_EVENT_COMPACT=0 disables.
EVENT_COMPACT = os.environ.get('FLY_EVENT_COMPACT', '1') != '0'
COMPACT_BLOCK, EVENT_PROGRAMS = 1024, 1792      # measured on the GTX 1650
_EVENT_SCRATCH = {}      # per (device, n): int64 accumulator, slot counter and slot list, reused between calls      # edges per block (keeps run indices below 256), output slots per combine program
_PULSE_CACHE = {}


class direct_edge_grad:
    """Context manager for a learner's own loss.backward(): edge-weight gradients go straight into .grad."""
    def __enter__(self):
        global DIRECT_EDGE_GRAD
        self.previous, DIRECT_EDGE_GRAD = DIRECT_EDGE_GRAD, True

    def __exit__(self, *exc):
        global DIRECT_EDGE_GRAD
        DIRECT_EDGE_GRAD = self.previous


def outgoing_layout(edge_pre, delay_splits, n_neurons):
    """Per-tier outgoing CSR (edge indices ordered by presynaptic cell) on the edges' device; topology only, cached."""
    splits = _canonical_delay_splits(delay_splits, edge_pre.numel())
    key = (edge_pre.data_ptr(), edge_pre.numel(), tuple(int(x) for x in splits), int(n_neurons), str(edge_pre.device))
    if key not in _OUTGOING_CACHE:
        tiers = []
        for start, end in zip(splits, splits[1:]):
            pre = edge_pre[start:end].long()
            order = (torch.argsort(pre, stable=True) + start).to(torch.int32)
            counts = torch.bincount(pre, minlength=n_neurons)
            ptr = torch.zeros(n_neurons + 1, dtype=torch.int32, device=edge_pre.device)
            ptr[1:] = torch.cumsum(counts, 0).to(torch.int32)
            tiers.append((order, ptr))
        _OUTGOING_CACHE[key] = tuple(tiers)
    return _OUTGOING_CACHE[key]

def pulse_grad_layout(edge_pre, delay_splits, n_neurons):
    """Static run layout for the pulse gradient, cached per edge array: (uint8 run index of each edge within its block,
    first piece of each block, piece range of each output slot tier * N + pre, slot of each piece, edge range of each
    slot).  None if the edges are not grouped by
    (tier, presynaptic cell) in ascending order; the caller then falls back to the atomic kernel."""
    splits = _canonical_delay_splits(delay_splits, edge_pre.numel())
    key_id = (edge_pre.data_ptr(), edge_pre.numel(), tuple(int(x) for x in splits), int(n_neurons), str(edge_pre.device))
    if key_id not in _PULSE_CACHE:
        n_edges, block = edge_pre.numel(), PULSE_BLOCK
        key = edge_pre.long().clone()
        for k, (a, b) in enumerate(zip(splits, splits[1:])):
            key[int(a):int(b)] += k * n_neurons
        if n_edges == 0 or bool((key[1:] < key[:-1]).any()):
            _PULSE_CACHE[key_id] = None
        else:
            position = torch.arange(n_edges, device=key.device)
            new = torch.ones(n_edges, dtype=torch.bool, device=key.device)
            new[1:] = (key[1:] != key[:-1]) | (position[1:] % block == 0)    # a piece never crosses a block edge
            piece = torch.cumsum(new.long(), 0) - 1
            first = piece[::block]
            local = (piece - first[position // block]).to(torch.uint8)
            counts = torch.bincount(key[new], minlength=4 * n_neurons)
            slot_ptr = torch.zeros(4 * n_neurons + 1, dtype=torch.int32, device=key.device)
            slot_ptr[1:] = torch.cumsum(counts, 0).to(torch.int32)
            edge_ptr = torch.zeros(4 * n_neurons + 1, dtype=torch.int32, device=key.device)
            edge_ptr[1:] = torch.cumsum(torch.bincount(key, minlength=4 * n_neurons), 0).to(torch.int32)
            _PULSE_CACHE[key_id] = (local, first.to(torch.int32), slot_ptr, key[new].to(torch.int32), edge_ptr)
    return _PULSE_CACHE[key_id]


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


    @triton.jit
    def _incoming_fwd_kernel(pulse_ptr, source_ptr, order_ptr, weight_ptr, offsets_ptr, current_ptr,
                             BLOCK: tl.constexpr):
        # One postsynaptic row per program. Its incoming edges are contiguous in this tier's layout; they are reduced in
        # a fixed order (blocked elementwise accumulation, then one fixed tree sum), so the result is deterministic.
        row = tl.program_id(0)
        start = tl.load(offsets_ptr + row)
        end = tl.load(offsets_ptr + row + 1)
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for base in range(start, end, BLOCK):
            idx = base + tl.arange(0, BLOCK)
            mask = idx < end
            src = tl.load(source_ptr + idx, mask=mask, other=0)
            pulse = tl.load(pulse_ptr + src, mask=mask, other=0.0)
            active = mask & (pulse != 0.0)
            edge = tl.load(order_ptr + idx, mask=active, other=0)
            weight = tl.load(weight_ptr + edge, mask=active, other=0.0)
            acc += pulse * weight
        total = tl.sum(acc, axis=0)
        tl.store(current_ptr + row, tl.load(current_ptr + row) + total)

    @triton.jit
    def _incoming_tier(acc, row, pulse_ptr, source_ptr, order_ptr, weight_ptr, offsets_ptr, BLOCK: tl.constexpr):
        start = tl.load(offsets_ptr + row)
        end = tl.load(offsets_ptr + row + 1)
        for base in range(start, end, BLOCK):
            idx = base + tl.arange(0, BLOCK)
            mask = idx < end
            src = tl.load(source_ptr + idx, mask=mask, other=0)
            pulse = tl.load(pulse_ptr + src, mask=mask, other=0.0)
            active = mask & (pulse != 0.0)
            edge = tl.load(order_ptr + idx, mask=active, other=0)
            weight = tl.load(weight_ptr + edge, mask=active, other=0.0)
            acc += pulse * weight
        return acc

    @triton.jit
    def _incoming4_fwd_kernel(p1, s1, o1, f1, p2, s2, o2, f2, p3, s3, o3, f3, p4, s4, o4, f4,
                              weight_ptr, current_ptr, n_rows, ROWS: tl.constexpr, BLOCK: tl.constexpr):
        # All four delay tiers of ROWS postsynaptic rows in one program; each row is reduced in a fixed order.
        for r in range(ROWS):
            row = tl.program_id(0) * ROWS + r
            if row < n_rows:
                acc = tl.zeros((BLOCK,), dtype=tl.float32)
                acc = _incoming_tier(acc, row, p1, s1, o1, weight_ptr, f1, BLOCK)
                acc = _incoming_tier(acc, row, p2, s2, o2, weight_ptr, f2, BLOCK)
                acc = _incoming_tier(acc, row, p3, s3, o3, weight_ptr, f3, BLOCK)
                acc = _incoming_tier(acc, row, p4, s4, o4, weight_ptr, f4, BLOCK)
                tl.store(current_ptr + row, tl.sum(acc, axis=0))

    @triton.jit
    def _outgoing_tier(row, pulse_ptr, order_ptr, ptr_ptr, post_ptr, weight_ptr, acc_ptr,
                       SCALE: tl.constexpr, BLOCK: tl.constexpr):
        pulse = tl.load(pulse_ptr + row)
        if pulse != 0.0:
            start = tl.load(ptr_ptr + row)
            end = tl.load(ptr_ptr + row + 1)
            for base in range(start, end, BLOCK):
                idx = base + tl.arange(0, BLOCK)
                mask = idx < end
                edge = tl.load(order_ptr + idx, mask=mask, other=0)
                post = tl.load(post_ptr + edge, mask=mask, other=0)
                weight = tl.load(weight_ptr + edge, mask=mask, other=0.0)
                value = (pulse * weight * SCALE).to(tl.int64)
                tl.atomic_add(acc_ptr + post, value, mask=mask)

    @triton.jit
    def _outgoing4_fwd_kernel(p1, o1, r1, p2, o2, r2, p3, o3, r3, p4, o4, r4, post_ptr, weight_ptr, acc_ptr, n_rows,
                              SCALE: tl.constexpr, ROWS: tl.constexpr, BLOCK: tl.constexpr):
        for r in range(ROWS):
            row = tl.program_id(0) * ROWS + r
            if row < n_rows:
                _outgoing_tier(row, p1, o1, r1, post_ptr, weight_ptr, acc_ptr, SCALE, BLOCK)
                _outgoing_tier(row, p2, o2, r2, post_ptr, weight_ptr, acc_ptr, SCALE, BLOCK)
                _outgoing_tier(row, p3, o3, r3, post_ptr, weight_ptr, acc_ptr, SCALE, BLOCK)
                _outgoing_tier(row, p4, o4, r4, post_ptr, weight_ptr, acc_ptr, SCALE, BLOCK)

    @triton.jit
    def _outgoing_wgrad_tier(row, pulse_ptr, order_ptr, ptr_ptr, post_ptr, grad_current_ptr, grad_weight_ptr,
                             BLOCK: tl.constexpr):
        pulse = tl.load(pulse_ptr + row)
        if pulse != 0.0:
            start = tl.load(ptr_ptr + row)
            end = tl.load(ptr_ptr + row + 1)
            for base in range(start, end, BLOCK):
                idx = base + tl.arange(0, BLOCK)
                mask = idx < end
                edge = tl.load(order_ptr + idx, mask=mask, other=0)
                post = tl.load(post_ptr + edge, mask=mask, other=0)
                gc = tl.load(grad_current_ptr + post, mask=mask, other=0.0)
                old = tl.load(grad_weight_ptr + edge, mask=mask, other=0.0)
                tl.store(grad_weight_ptr + edge, old + gc * pulse, mask=mask)   # each edge once per call: no atomics

    @triton.jit
    def _outgoing4_wgrad_kernel(p1, o1, r1, p2, o2, r2, p3, o3, r3, p4, o4, r4, post_ptr, grad_current_ptr,
                                grad_weight_ptr, n_rows, ROWS: tl.constexpr, BLOCK: tl.constexpr):
        for r in range(ROWS):
            row = tl.program_id(0) * ROWS + r
            if row < n_rows:
                _outgoing_wgrad_tier(row, p1, o1, r1, post_ptr, grad_current_ptr, grad_weight_ptr, BLOCK)
                _outgoing_wgrad_tier(row, p2, o2, r2, post_ptr, grad_current_ptr, grad_weight_ptr, BLOCK)
                _outgoing_wgrad_tier(row, p3, o3, r3, post_ptr, grad_current_ptr, grad_weight_ptr, BLOCK)
                _outgoing_wgrad_tier(row, p4, o4, r4, post_ptr, grad_current_ptr, grad_weight_ptr, BLOCK)

    @triton.jit
    def _pulse_grad_pieces_kernel(local_ptr, first_ptr, piece_slot_ptr, post_ptr, weight_ptr, grad_current_ptr,
                                  pieces_ptr, p1, p2, p3, p4, grad_weight_ptr, n_edges, n_rows,
                                  WEIGHT_GRAD: tl.constexpr, BLOCK: tl.constexpr):
        # One block of edges; per edge only its uint8 run index, target and weight are read (plus the gather).  The
        # same gathered grad_current also gives the edge-weight gradient of edges whose presynaptic cell spiked
        # (read-add-store, each edge once per call: no atomics).
        lane = tl.arange(0, BLOCK)
        idx = tl.program_id(0) * BLOCK + lane
        mask = idx < n_edges
        local = tl.load(local_ptr + idx, mask=mask, other=0).to(tl.int32)
        post = tl.load(post_ptr + idx, mask=mask, other=0)
        gc = tl.load(grad_current_ptr + post, mask=mask, other=0.0)
        value = tl.load(weight_ptr + idx, mask=mask, other=0.0) * gc
        previous = tl.load(local_ptr + idx - 1, mask=mask & (lane > 0), other=0).to(tl.int32)
        start = (lane == 0) | (previous != local)
        sums, _ = tl.associative_scan((value, start), 0, _sum_contiguous_keys)
        following = tl.load(local_ptr + idx + 1, mask=(lane < BLOCK - 1) & (idx + 1 < n_edges), other=0).to(tl.int32)
        end = mask & ((lane == BLOCK - 1) | (idx + 1 >= n_edges) | (following != local))
        piece = tl.load(first_ptr + tl.program_id(0)) + local
        tl.store(pieces_ptr + piece, sums, mask=end)
        if WEIGHT_GRAD:
            slot = tl.load(piece_slot_ptr + piece, mask=mask, other=0)
            tier = slot // n_rows
            pre = slot - tier * n_rows
            pulse = (tl.load(p1 + pre, mask=mask & (tier == 0), other=0.0) + tl.load(p2 + pre, mask=mask & (tier == 1), other=0.0)
                     + tl.load(p3 + pre, mask=mask & (tier == 2), other=0.0) + tl.load(p4 + pre, mask=mask & (tier == 3), other=0.0))
            active = mask & (pulse != 0.0)
            old = tl.load(grad_weight_ptr + idx, mask=active, other=0.0)
            tl.store(grad_weight_ptr + idx, old + gc * pulse, mask=active)

    @triton.jit
    def _compact_active_kernel(p1, p2, p3, p4, slots_ptr, count_ptr, n_rows, BLOCK: tl.constexpr):
        # Append every (tier, cell) whose pulse is nonzero to the slot list; one atomic reservation per block.  The list
        # order varies between runs, the forward result does not (exact integer sums).
        tier = tl.program_id(1)
        row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = row < n_rows
        pulse = (tl.load(p1 + row, mask=mask & (tier == 0), other=0.0) + tl.load(p2 + row, mask=mask & (tier == 1), other=0.0)
                 + tl.load(p3 + row, mask=mask & (tier == 2), other=0.0) + tl.load(p4 + row, mask=mask & (tier == 3), other=0.0))
        active = (mask & (pulse != 0.0)).to(tl.int32)
        total = tl.sum(active, axis=0)
        if total > 0:
            base = tl.atomic_add(count_ptr, total)
            position = base + tl.cumsum(active, axis=0) - active
            tl.store(slots_ptr + position, tier * n_rows + row, mask=active != 0)

    @triton.jit
    def _event_slots_fwd_kernel(slots_ptr, count_ptr, p1, p2, p3, p4, edge_ptr, post_ptr, weight_ptr, acc_ptr, n_rows,
                                SCALE: tl.constexpr, BLOCK: tl.constexpr):
        # Persistent programs walk the compacted list; each spiking (tier, cell) pushes along its contiguous edges.
        count = tl.load(count_ptr)
        for i in range(tl.program_id(0), count, tl.num_programs(0)):
            slot = tl.load(slots_ptr + i)
            tier = slot // n_rows
            row = slot - tier * n_rows
            if tier == 0:
                pulse = tl.load(p1 + row)
            elif tier == 1:
                pulse = tl.load(p2 + row)
            elif tier == 2:
                pulse = tl.load(p3 + row)
            else:
                pulse = tl.load(p4 + row)
            start = tl.load(edge_ptr + slot)
            end = tl.load(edge_ptr + slot + 1)
            for base in range(start, end, BLOCK):
                idx = base + tl.arange(0, BLOCK)
                mask = idx < end
                post = tl.load(post_ptr + idx, mask=mask, other=0)
                weight = tl.load(weight_ptr + idx, mask=mask, other=0.0)
                tl.atomic_add(acc_ptr + post, (pulse * weight * SCALE).to(tl.int64), mask=mask)

    @triton.jit
    def _finish_fixed_kernel(acc_ptr, out_ptr, n_rows, INV_SCALE: tl.constexpr, BLOCK: tl.constexpr):
        # Fixed-point sum -> float32 current (exact scaling by a power of two).  Zeroing the accumulator in this same
        # kernel would race: the zero store may be laid out on other threads than the load.
        row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = row < n_rows
        acc = tl.load(acc_ptr + row, mask=mask, other=0)
        tl.store(out_ptr + row, (acc.to(tl.float64) * INV_SCALE).to(tl.float32), mask=mask)

    @triton.jit
    def _pulse_grad_combine_kernel(pieces_ptr, slot_ptr, out_ptr, n_slots, SLOTS: tl.constexpr):
        # Each output slot adds its pieces in edge order: fixed arithmetic, no atomics.
        slot = tl.program_id(0) * SLOTS + tl.arange(0, SLOTS)
        mask = slot < n_slots
        start = tl.load(slot_ptr + slot, mask=mask, other=0)
        count = tl.load(slot_ptr + slot + 1, mask=mask, other=0) - start
        acc = tl.zeros((SLOTS,), dtype=tl.float32)
        for j in range(0, tl.max(count, axis=0)):
            acc += tl.load(pieces_ptr + start + j, mask=j < count, other=0.0)
        tl.store(out_ptr + slot, acc, mask=mask)

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
            delay_splits = _canonical_delay_splits(delay_splits, edge_weight.numel())
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


def _canonical_delay_splits(delay_splits, n_edges):
    """Four slots, including empty tiers; metadata is constant during capture."""
    if isinstance(delay_splits, torch.Tensor):
        delay_splits = delay_splits.cpu().tolist()
    splits = tuple(int(x) for x in delay_splits)
    if (not splits or len(splits) > 5 or splits[0] != 0
            or splits[-1] != n_edges
            or any(a > b for a, b in zip(splits, splits[1:]))):
        raise ValueError('Delay splits must partition all edges into at most four tiers')
    return splits + (n_edges,) * (5 - len(splits))


class PyTorchDelayedSynapticTransmission(torch.autograd.Function):
    """Fallback PyTorch multi-delay transmission for CPU or platforms without Triton."""

    @staticmethod
    def forward(ctx, s1, s2, s3, s4, edge_pre, edge_post, edge_weight, delay_splits):
        delay_splits = _canonical_delay_splits(delay_splits, edge_weight.numel())
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


def build_incoming_layout(edge_pre, edge_post, delay_splits, n_neurons):
    """Stable incoming-row layout; original trainable edge order is unchanged.

    Cache topology only. Returned int32 indices are moved with the model;
    weights are gathered afresh every call. Construct outside CUDA capture.
    """
    pre = edge_pre.detach().cpu().numpy()
    post = edge_post.detach().cpu().numpy()
    splits = _canonical_delay_splits(delay_splits, len(pre))
    layouts = []
    for start, end in zip(splits, splits[1:]):
        order = np.argsort(post[start:end], kind='stable') + start
        counts = np.bincount(post[start:end].astype(np.int64), minlength=n_neurons)
        offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int32)
        layouts.append(tuple(torch.from_numpy(value) for value in (
            order.astype(np.int32), pre[order].astype(np.int32), offsets)))
    return tuple(layouts)


class IncomingDelayedTransmission(torch.autograd.Function):
    """Fixed incoming summation order forward; complete original edge VJP.

    Use two-dimensional[E,B] data: each output row is reduced in its fixed
    offset order, avoiding CUDA atomic forward summation. No edge-sized
    activation is retained per tick. Backward may still use CUDA atomics.
    """
    @staticmethod
    def forward(ctx, s1, s2, s3, s4, edge_pre, edge_post, edge_weight,
                delay_splits, layouts):
        ctx.delay_splits = _canonical_delay_splits(delay_splits, edge_weight.numel())
        ctx.save_for_backward(edge_pre, edge_post, edge_weight, s1, s2, s3, s4)
        ctx.edge_weight_leaf = edge_weight
        ctx.s_shape = s1.shape
        if TRANSMISSION_REPLAY is not None:
            return next(TRANSMISSION_REPLAY).clone()
        out = IncomingDelayedTransmission._compute(s1, s2, s3, s4, edge_pre, edge_post, edge_weight, ctx.delay_splits,
                                                   layouts)
        if TRANSMISSION_RECORD is not None:
            TRANSMISSION_RECORD.append(out.detach())
        return out

    @staticmethod
    def _compute(s1, s2, s3, s4, edge_pre, edge_post, edge_weight, delay_splits, layouts):
        if (HAS_TRITON and FUSED_INCOMING and s1.is_cuda and s1.shape[0] == 1
                and s1.dtype == torch.float32 and edge_weight.dtype == torch.float32):
            n_rows = s1.shape[-1]
            weight = edge_weight.contiguous()
            runs = pulse_grad_layout(edge_pre, delay_splits, n_rows) if EVENT_DRIVEN and EVENT_COMPACT else None
            if runs is not None:
                pulses = [pulse.contiguous().view(-1) for pulse in (s1, s2, s3, s4)]
                key = (str(s1.device), n_rows)
                if key not in _EVENT_SCRATCH:
                    _EVENT_SCRATCH[key] = (torch.empty(n_rows, dtype=torch.int64, device=s1.device),
                                           torch.empty(1, dtype=torch.int32, device=s1.device),
                                           torch.empty(4 * n_rows, dtype=torch.int32, device=s1.device))
                acc, count, slots = _EVENT_SCRATCH[key]
                acc.zero_()
                count.zero_()
                _compact_active_kernel[(triton.cdiv(n_rows, COMPACT_BLOCK), 4)](
                    *pulses, slots, count, n_rows, BLOCK=COMPACT_BLOCK)
                _event_slots_fwd_kernel[(EVENT_PROGRAMS,)](
                    slots, count, *pulses, runs[4], edge_post, weight, acc, n_rows, SCALE=FIXED_SCALE, BLOCK=EVENT_BLOCK)
                out = torch.empty(1, n_rows, dtype=torch.float32, device=s1.device)
                _finish_fixed_kernel[(triton.cdiv(n_rows, COMPACT_BLOCK),)](
                    acc, out, n_rows, INV_SCALE=1.0 / FIXED_SCALE, BLOCK=COMPACT_BLOCK)
                return out
            if EVENT_DRIVEN:
                tiers = outgoing_layout(edge_pre, delay_splits, n_rows)
                args = []
                for pulse, (order, ptr) in zip((s1, s2, s3, s4), tiers):
                    if not order.numel():
                        order = ptr.new_zeros(1)
                    args += [pulse.contiguous().view(-1), order, ptr]
                acc = torch.zeros(n_rows, dtype=torch.int64, device=s1.device)
                _outgoing4_fwd_kernel[(triton.cdiv(n_rows, EVENT_ROWS),)](
                    *args, edge_post, weight, acc, n_rows, SCALE=FIXED_SCALE, ROWS=EVENT_ROWS, BLOCK=EVENT_BLOCK)
                return (acc.to(torch.float64) / FIXED_SCALE).to(torch.float32).view(1, -1)
            current = torch.zeros(n_rows, dtype=torch.float32, device=s1.device)
            args = []
            for pulse, (order, source, offsets) in zip((s1, s2, s3, s4), layouts):
                if not order.numel():                    # empty tier: offsets are all zero, nothing is read
                    order = source = offsets.new_zeros(1)
                args += [pulse.contiguous().view(-1), source, order, offsets]
            rows_per_program = INCOMING_ROWS
            _incoming4_fwd_kernel[(triton.cdiv(n_rows, rows_per_program),)](
                *args, weight, current, n_rows, ROWS=rows_per_program, BLOCK=INCOMING_BLOCK)
            return current.view(1, -1)
        current = torch.zeros_like(s1).t()
        for pulse, (order, source, offsets) in zip((s1, s2, s3, s4), layouts):
            if not order.numel():
                continue
            contribution = torch.index_select(pulse, 1, source).t().contiguous()
            contribution.mul_(torch.index_select(edge_weight, 0, order)[:, None])
            current.add_(torch.segment_reduce(contribution, 'sum', offsets=offsets,
                                             axis=0, unsafe=True))
        return current.t()

    @staticmethod
    def backward(ctx, grad_current):
        if (HAS_TRITON and EVENT_DRIVEN and grad_current.is_cuda and ctx.s_shape[0] == 1
                and grad_current.dtype == torch.float32 and ctx.edge_weight_leaf.dtype == torch.float32):
            edge_pre, edge_post, edge_weight, *spikes = ctx.saved_tensors      # unpacked exactly once
            return IncomingDelayedTransmission._event_backward(ctx, grad_current, edge_pre, edge_post, edge_weight, spikes)
        if HAS_TRITON and grad_current.is_cuda and ctx.s_shape[0] == 1:
            result = TritonDelayedSynapticTransmission.backward(ctx, grad_current)
        else:
            result = PyTorchDelayedSynapticTransmission.backward(ctx, grad_current)
        return (*result, None)

    @staticmethod
    def _event_backward(ctx, grad_current, edge_pre, edge_post, edge_weight, spikes):
        n_rows = grad_current.shape[-1]
        gc = grad_current.contiguous().view(-1)
        splits = ctx.delay_splits
        grad_spikes = [None] * 4                             # dense pulse gradient: silent cells keep their surrogate VJP
        grad_weight = target = None
        if ctx.needs_input_grad[6]:
            leaf = ctx.edge_weight_leaf
            direct = (DIRECT_EDGE_GRAD and leaf.is_leaf and not torch.cuda.is_current_stream_capturing())
            if direct:
                if leaf.grad is None:
                    leaf.grad = torch.zeros_like(leaf)
                target = leaf.grad
            else:
                grad_weight = torch.zeros_like(edge_weight)
                target = grad_weight
        runs = pulse_grad_layout(edge_pre, splits, n_rows) if PULSE_RUNS else None
        if runs is not None:
            local, first, slot_ptr, piece_slot, _ = runs
            n_edges = edge_weight.numel()
            pieces = torch.empty(piece_slot.numel(), dtype=torch.float32, device=gc.device)
            pulses = [pulse.contiguous().view(-1) for pulse in spikes]
            _pulse_grad_pieces_kernel[(triton.cdiv(n_edges, PULSE_BLOCK),)](
                local, first, piece_slot, edge_post, edge_weight, gc, pieces, *pulses,
                target if target is not None else gc, n_edges, n_rows, WEIGHT_GRAD=target is not None, BLOCK=PULSE_BLOCK)
            if any(ctx.needs_input_grad[:4]):
                out = torch.empty(4, n_rows, dtype=torch.float32, device=gc.device)
                _pulse_grad_combine_kernel[(triton.cdiv(4 * n_rows, PULSE_SLOTS),)](
                    pieces, slot_ptr, out, 4 * n_rows, SLOTS=PULSE_SLOTS)
                grad_spikes = [out[k].view(ctx.s_shape) if ctx.needs_input_grad[k] else None for k in range(4)]
            return (*grad_spikes, None, None, grad_weight, None, None)
        for k in range(4):
            start, end = int(splits[k]), int(splits[k + 1])
            gs = torch.zeros_like(spikes[k]) if ctx.needs_input_grad[k] else None
            if gs is not None and end > start:
                _synaptic_joint_bwd_kernel[(triton.cdiv(end - start, 1024),)](
                    gc, spikes[k].reshape(-1), edge_pre[start:end], edge_post[start:end], edge_weight[start:end],
                    gs, gc, end - start, True, False, 1024)
            grad_spikes[k] = gs
        if target is not None:
            tiers = outgoing_layout(edge_pre, splits, n_rows)
            args = []
            for pulse, (order, ptr) in zip(spikes, tiers):
                if not order.numel():
                    order = ptr.new_zeros(1)
                args += [pulse.contiguous().view(-1), order, ptr]
            _outgoing4_wgrad_kernel[(triton.cdiv(n_rows, EVENT_ROWS),)](
                *args, edge_post, gc, target, n_rows, ROWS=EVENT_ROWS, BLOCK=EVENT_BLOCK)
        return (*grad_spikes, None, None, grad_weight, None, None)


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
    delay_splits = _canonical_delay_splits(delay_splits, edge_weight.numel())
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
