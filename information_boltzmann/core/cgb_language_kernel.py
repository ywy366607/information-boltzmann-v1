"""Checked transport and finite local edits from ``LanguageLifeKernel.lean``."""

from __future__ import annotations

from dataclasses import dataclass
import torch


class KernelContractViolation(ValueError):
    """Raised when an integer reference-kernel contract is violated."""


@dataclass(frozen=True)
class KernelState:
    tick: int
    charge: torch.Tensor
    reserve: int


@dataclass(frozen=True)
class TernaryGraph:
    """Explicit small-graph oracle; production graphs need not store endpoints."""

    source: torch.Tensor
    target: torch.Tensor
    nodes: int

    def __post_init__(self) -> None:
        if (self.source.dtype != torch.int64 or self.target.dtype != torch.int64
                or self.source.shape != self.target.shape or self.source.ndim != 1):
            raise KernelContractViolation("explicit oracle needs aligned one-dimensional int64 indices")
        if self.nodes < 1:
            raise KernelContractViolation("graph needs at least one node")
        if bool(((self.source < 0) | (self.source >= self.nodes) |
                 (self.target < 0) | (self.target >= self.nodes)).any()):
            raise KernelContractViolation("graph index outside node range")


@dataclass(frozen=True)
class ImplicitBipartiteGraph:
    """Implicit source j -> target cols+i; only two dimensions encode topology."""

    rows: int
    cols: int

    def __post_init__(self) -> None:
        if any(type(n) is not int or n < 1 for n in (self.rows, self.cols)):
            raise KernelContractViolation("block dimensions must be positive integers")

    @property
    def nodes(self) -> int:
        return self.rows + self.cols


@torch.no_grad()
def charge_update(graph: TernaryGraph | ImplicitBipartiteGraph,
                  charge: torch.Tensor, flow, *, tile_shape=None) -> torch.Tensor:
    """Exact transport; producer(old_charge, row_slice, col_slice) must be bounded.
    Producer allocations are its external contract, not enforced by this loop."""
    if charge.dtype not in (torch.int32, torch.int64) or charge.shape != (graph.nodes,):
        raise KernelContractViolation("charge shape does not match graph")
    implicit = isinstance(graph, ImplicitBipartiteGraph)
    if not implicit and (callable(flow) or tile_shape is not None):
        raise KernelContractViolation("tile producers require implicit topology")
    if implicit:
        if isinstance(flow, torch.Tensor) and flow.shape != (graph.rows, graph.cols):
            raise KernelContractViolation("flow shape does not match graph")
        tile_shape = (min(64, graph.rows), min(64, graph.cols)) if tile_shape is None else tile_shape
        if not isinstance(tile_shape, tuple) or len(tile_shape) != 2 or any(
                type(n) is not int or n < 1 for n in tile_shape):
            raise KernelContractViolation("tile dimensions must be positive integers")
        tiles = ((slice(i, min(i + tile_shape[0], graph.rows)),
                  slice(j, min(j + tile_shape[1], graph.cols)))
                 for i in range(0, graph.rows, tile_shape[0]) for j in range(0, graph.cols, tile_shape[1]))
    else:
        tiles = ((slice(None), slice(None)),)
    snapshot, out = charge.clone(), charge.clone()
    limits = torch.iinfo(charge.dtype)
    for rs, cs in tiles:
        packet = flow(snapshot, rs, cs) if callable(flow) else flow[rs, cs] if implicit else flow
        expected = (rs.stop - rs.start, cs.stop - cs.start) if implicit else graph.source.shape
        if not isinstance(packet, torch.Tensor) or packet.shape != expected or packet.dtype not in (
                torch.int8, torch.int16, torch.int32, torch.int64) or packet.device != charge.device:
            raise KernelContractViolation("tile needs aligned shape, device and signed integer dtype")
        if not torch.equal(snapshot, charge):
            raise KernelContractViolation("producer modified the frozen source state")
        lo, hi = torch.aminmax(packet) if packet.numel() else (0, 0)
        degree = max(packet.shape) if implicit else 2 * packet.numel()
        extent = degree * max(abs(int(lo)), abs(int(hi)))
        lo, hi = torch.aminmax(out)
        if int(lo) - extent < limits.min or int(hi) + extent > limits.max:
            raise KernelContractViolation(f"transport exceeds certified {charge.dtype} range")
        if implicit:
            out[cs].sub_(packet.sum(dim=0, dtype=charge.dtype))
            out[graph.cols + rs.start:graph.cols + rs.stop].add_(packet.sum(dim=1, dtype=charge.dtype))
        else:
            packet = packet.to(dtype=charge.dtype)
            out.index_add_(0, graph.target, packet)
            out.index_add_(0, graph.source, -packet)
    return out


def reserve_update(reserve: int, incoming: int, outgoing: int) -> int:
    if any(type(v) is not int or v < 0 for v in (reserve, incoming, outgoing)):
        raise KernelContractViolation("reserve ledger values must be nonnegative integers")
    if outgoing > reserve + incoming:
        raise KernelContractViolation("outgoing exceeds available reserve and incoming supply")
    return reserve + incoming - outgoing


def step(graph: TernaryGraph | ImplicitBipartiteGraph, state: KernelState, flow,
         incoming: int, outgoing: int, *, tile_shape=None) -> KernelState:
    return KernelState(
        tick=state.tick + 1,
        charge=charge_update(graph, state.charge, flow, tile_shape=tile_shape),
        reserve=reserve_update(state.reserve, incoming, outgoing),
    )


def local_masses(code, candidate, context, configurations, observed, rule):
    """Exact sums from one pure closed rule; callback allocations are external.
    Context contains all dependencies. Labels identify the same old/new event."""
    if (any(type(k) is not int or not -8 <= k <= 7 for k in (code, candidate))
            or type(configurations) is not int or not 1 <= configurations <= 256
            or type(observed) is not int or not -(2**31) <= observed < 2**31
            or not isinstance(context, tuple) or len(context) > 64
            or any(type(k) is not int or not -(2**31) <= k < 2**31 for k in context)):
        raise KernelContractViolation("invalid bounded local rule domain")
    totals = [0, 0, 0, 0]
    for config in range(configurations):
        old, label = rule(code, config, context)
        new, new_label = rule(candidate, config, context)
        if (any(type(m) is not int or m <= 0 for m in (old, new))
                or type(label) is not int or type(new_label) is not int or label != new_label
                or not -(2**31) <= label < 2**31):
            raise KernelContractViolation("rule needs positive integer masses and unchanged labels")
        for i, value in enumerate((old, old if label == observed else 0,
                                   new, new if label == observed else 0)):
            totals[i] += value
        if max(totals) > 2**31 - 1:
            raise KernelContractViolation("local mass exceeds int32 range")
    return tuple(totals)


def local_sample(code, context, configurations, rule, ticket):
    """Inverse CDF of the same pure finite rule; caller supplies the integer ticket."""
    total = local_masses(code, code, context, configurations, 0, rule)[0]
    if type(ticket) is not int or not 0 <= ticket < total:
        raise KernelContractViolation("ticket outside local mass")
    for config in range(configurations):
        mass, label = rule(code, config, context)
        ticket -= mass
        if ticket < 0:
            return config, label
    raise KernelContractViolation("sampling rule changed")


@torch.no_grad()
def local_edit(packed, index, delta, context, configurations, observed, rule):
    """Single-owner offset-binary nibble edit; shared scales stay fixed externally."""
    if (not isinstance(packed, torch.Tensor) or packed.dtype != torch.uint8 or packed.ndim != 1
            or not packed.is_contiguous() or type(index) is not int or not 0 <= index < 2 * packed.numel()
            or type(delta) is not int or delta not in (-1, 1)):
        raise KernelContractViolation("local edit needs a contiguous byte vector and legal nibble/delta")
    byte_index, shift = index // 2, 4 * (index % 2)
    previous = int(packed[byte_index])
    code = ((previous >> shift) & 15) - 8
    masses = local_masses(code, code + delta, context, configurations, observed, rule)
    total, event, new_total, new_event = masses
    if not event:
        raise KernelContractViolation("observed label absent from local rule")
    accepted = new_event * total > event * new_total
    if accepted:
        if int(packed[byte_index]) != previous:
            raise KernelContractViolation("packed byte changed before commit")
        packed[byte_index] = (previous & ~(15 << shift)) | ((code + delta + 8) << shift)
    return accepted, masses
