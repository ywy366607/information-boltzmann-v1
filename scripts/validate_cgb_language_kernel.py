"""Deterministic Torch reference checks for the Lean language-life kernel."""

from __future__ import annotations

from pathlib import Path
from fractions import Fraction
import sys
import importlib.util

import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "cgb_language_kernel", ROOT / "information_boltzmann" / "core" / "cgb_language_kernel.py"
)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load cgb_language_kernel")
kernel = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = kernel
spec.loader.exec_module(kernel)


def reject(operation, message: str = "") -> None:
    try:
        operation()
    except kernel.KernelContractViolation as exc:
        assert message in str(exc)
    else:
        raise AssertionError("invalid kernel operation was admitted")


def local_rule(code, config, context):
    hidden, label = divmod(config, 2)
    mass = (code + 9 if hidden else 1) * (abs(context[0])
        if (hidden == label) == (context[0] > 0) else 1)
    return mass, label


def validate_local_edit() -> None:
    for byte in range(256):
        for index in (0, 1):
            code = ((byte >> (4 * index)) & 15) - 8
            for delta in (-1, 1):
                for coupling, observed in ((k, b) for k in (3, -3, 1) for b in (0, 1)):
                    packed = torch.tensor([byte, 0x5A], dtype=torch.uint8)
                    args = (packed, index, delta, (coupling,), 4, observed, local_rule)
                    if not -8 <= code + delta <= 7:
                        reject(lambda: kernel.local_edit(*args))
                        assert packed.tolist() == [byte, 0x5A]
                        continue
                    a, b, c, d = (coupling, 1, 1, coupling) if coupling > 0 else (1, -coupling, -coupling, 1)
                    u, v = code + 9, code + delta + 9
                    p, q = (Fraction(b + k * d, a + b + k * (c + d)) for k in (u, v))
                    expected = q > p if observed else q < p
                    accepted, (z, e, nz, ne) = kernel.local_edit(*args)
                    assert accepted == expected and ne * z - e * nz == delta * (a * d - b * c) * (1 if observed else -1)
                    assert (Fraction(e, z), Fraction(ne, nz)) == ((p, q) if observed else (1 - p, 1 - q))
                    expected_byte = (byte & ~(15 << (4 * index))) | ((code + delta + 8) << (4 * index))
                    assert packed.tolist() == [expected_byte if accepted else byte, 0x5A]
    alias = [kernel.local_masses(-8, -7, (k,), 4, 1, local_rule) for k in (3, -3)]
    assert alias == [(8, 4, 12, 7), (8, 4, 12, 5)]  # Same marginals, opposite edit direction.
    for code in range(-8, 8):
        a = code + 9
        counts = [0] * 4
        for ticket in range(4 * (a + 1)):
            config, label = kernel.local_sample(code, (3,), 4, local_rule, ticket)
            counts[config] += 1
            assert label == config % 2
        assert counts == [3, 1, a, 3 * a]
    for rule, context, observed in ((lambda *_: (0, 0), (), 0),
            (lambda c, *_: (1, c), (), 0), (local_rule, (3,), 9),
            (local_rule, (True,), 0), (local_rule, (2**31,), 0),
            (lambda *_: (2**31, 0), (), 0), (lambda *_: (2**30, 0), (), 0)):
        packed = torch.tensor([0x88], dtype=torch.uint8)
        reject(lambda: kernel.local_edit(packed, 0, 1, context, 4, observed, rule))
        assert int(packed[0]) == 0x88
    for ticket in (-1, 40):
        reject(lambda: kernel.local_sample(0, (3,), 4, local_rule, ticket))


def validate_implicit_graph() -> None:
    graph = kernel.ImplicitBipartiteGraph(5, 7)
    assert vars(graph) == {"rows": 5, "cols": 7}
    oracle = kernel.TernaryGraph(torch.arange(7).repeat(5),
                                 torch.arange(5).repeat_interleave(7) + 7, 12)
    charge = torch.arange(12, dtype=torch.int32).remainder(5) - 2

    def producer(old, rs, cs):
        assert not torch.is_grad_enabled() and torch.equal(old, charge)
        return (old[7 + rs.start:7 + rs.stop, None] - old[None, cs]).to(torch.int8)

    flow = (charge[7:, None] - charge[None, :7]).to(torch.int8)
    expected = kernel.charge_update(oracle, charge, flow.flatten())
    for shape in ((2, 3), (3, 4), (4, 5)):
        result = kernel.charge_update(graph, charge, producer, tile_shape=shape)
        assert torch.equal(result, expected) and result.dtype == torch.int32
        assert int(result.sum()) == int(charge.sum()) and result.grad_fn is None
        assert torch.equal(result, kernel.charge_update(graph, charge, producer, tile_shape=shape))
    state = kernel.step(graph, kernel.KernelState(0, charge, 1), producer, 1, 1, tile_shape=(2, 3))
    assert state.tick == 1 and state.reserve == 1 and torch.equal(state.charge, expected)
    assert torch.equal(expected, kernel.charge_update(graph, charge, flow))
    for bad in (torch.ones((2, 3)), torch.zeros((4, 7), dtype=torch.int8)):
        reject(lambda: kernel.charge_update(graph, charge, lambda *_: bad, tile_shape=(2, 3)))
    reject(lambda: kernel.charge_update(graph, charge.float(), flow))
    reject(lambda: kernel.charge_update(graph, charge, producer, tile_shape=(0, 3)))

    def mutator(old, rs, cs):
        old.add_(1)
        return torch.ones((rs.stop - rs.start, cs.stop - cs.start), dtype=torch.int8)

    reject(lambda: kernel.charge_update(graph, charge, mutator), "frozen")
    assert torch.equal(charge, torch.arange(12, dtype=torch.int32).remainder(5) - 2)
    block = kernel.ImplicitBipartiteGraph(1, 2)
    near = torch.tensor([0, 0, torch.iinfo(torch.int32).max - 1], dtype=torch.int32)
    reject(lambda: kernel.charge_update(block, near, torch.ones((1, 2), dtype=torch.int8),
                                        tile_shape=(1, 1)), "int32 range")
    assert int(near[-1]) == torch.iinfo(torch.int32).max - 1


def main() -> None:
    validate_local_edit()
    validate_implicit_graph()
    graph = kernel.TernaryGraph(torch.tensor([0, 1, 2, 2]), torch.tensor([1, 2, 0, 2]), 3)
    state = kernel.KernelState(4, torch.tensor([7, -2, 5]), 3)
    flow = torch.tensor([2, -3, 4, 9], dtype=torch.int64)
    next_state = kernel.step(graph, state, flow, incoming=2, outgoing=4)
    assert next_state.tick == 5
    assert int(next_state.charge.sum()) == int(state.charge.sum())
    assert next_state.reserve + 4 == state.reserve + 2
    limits = torch.iinfo(torch.int64)
    for bad_charge, bad_flow in (([limits.max, 0, 0], [-1, 0, 0, 0]),
                                 ([0, 0, 0], [limits.min, 0, 0, 0])):
        reject(lambda: kernel.charge_update(graph, torch.tensor(bad_charge),
                                            torch.tensor(bad_flow)), "int64 range")
    reject(lambda: kernel.step(graph, state, flow, incoming=0, outgoing=4), "exceeds")
    print("cgb_language_kernel_ok")


if __name__ == "__main__":
    main()
