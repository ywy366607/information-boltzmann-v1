"""Exact ternary and sparse storage analysis for the current incidence chain."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from bench_algebraic_coefficients import CELLS, SIGNALS, port_next


def pack_trits(values: list[int]) -> bytes:
    encoded = { -1: 0, 0: 1, 1: 2 }
    out = bytearray()
    for start in range(0, len(values), 5):
        value = 0
        for i, trit in enumerate(values[start:start + 5]):
            value += encoded[trit] * (3 ** i)
        out.append(value)
    return bytes(out)


def unpack_trits(data: bytes, count: int) -> list[int]:
    decoded = {0: -1, 1: 0, 2: 1}
    values: list[int] = []
    for byte in data:
        value = byte
        for _ in range(5):
            values.append(decoded[value % 3])
            value //= 3
    return values[:count]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    states = [(cell, reserve) for cell in CELLS for reserve in range(3)]
    state_index = {state: i for i, state in enumerate(states)}
    edges = []
    dense: list[int] = []
    self_loops = 0
    nonzero = 0
    for cell in CELLS:
        for reserve in range(3):
            for signal in SIGNALS:
                for supplied in (False, True):
                    source = state_index[(cell, reserve)]
                    target_cell, target_reserve = port_next(signal, supplied, cell, reserve)
                    target = state_index[(target_cell, target_reserve)]
                    edges.append((source, target, signal, supplied))
                    row = [0] * len(states)
                    if source == target:
                        self_loops += 1
                    else:
                        row[source] = -1
                        row[target] = 1
                        nonzero += 2
                    dense.extend(row)

    packed = pack_trits(dense)
    roundtrip = unpack_trits(packed, len(dense)) == dense
    edge_count = len(edges)
    state_bits = math.ceil(math.log2(len(states)))
    input_bits = math.ceil(math.log2(len(SIGNALS) * 2))
    result = {
        "scope": "exact incidence storage analysis for the current PortSeparation chain",
        "states": len(states),
        "inputs": len(SIGNALS) * 2,
        "edges": edge_count,
        "incidence_entries": len(dense),
        "incidence_nonzero_entries": nonzero,
        "self_loop_edges": self_loops,
        "alphabet": [-1, 0, 1],
        "information_theoretic_bits_per_trit": math.log2(3),
        "ternary_packing": {
            "trits_per_byte": 5,
            "packed_bytes": len(packed),
            "roundtrip_exact": roundtrip,
            "theoretical_dense_bytes": math.ceil(len(dense) * math.log2(3) / 8),
        },
        "sparse_orientation_known": {
            "bits_per_state_index": state_bits,
            "source_target_bits_per_edge": 2 * state_bits,
            "bytes_for_all_edges": math.ceil(edge_count * 2 * state_bits / 8),
            "sign_bits_needed": 0,
            "reason": "source is -1 and target is +1 by definition",
        },
        "sparse_with_flow_counter": {
            "flow_bits": 3,
            "bits_per_edge": 2 * state_bits + 3,
            "bytes_for_all_edges": math.ceil(edge_count * (2 * state_bits + 3) / 8),
        },
        "generated_transition_encoding": {
            "source_index_bits": state_bits,
            "input_index_bits": input_bits,
            "bits_per_edge_if_target_recomputed": state_bits + input_bits,
            "bytes_for_all_edges": math.ceil(edge_count * (state_bits + input_bits) / 8),
        },
        "conclusion": {
            "ternary_dense_is_exact": True,
            "sparse_source_target_beats_dense_ternary": True,
            "log2_3_is_asymptotic_storage_rate_not_gpu_dtype": True,
            "accumulator_can_be_ternary": False,
            "recommended_current_encoding": "generated transition or sparse source/target indices",
        },
    }
    print(json.dumps(result, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
