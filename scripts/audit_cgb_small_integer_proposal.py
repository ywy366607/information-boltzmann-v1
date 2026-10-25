"""Arithmetic/interface audit of bounded storage proposals, not a capability test.

Uses Python exact integers as the reference. Byte counts exclude indexing,
allocator overhead, event queues, optimizer state, and GPU temporary workspace.
No production dynamics, checkpoint, or clock is changed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def pack_nibbles(values: list[int]) -> bytes:
    if any(not 0 <= value < 16 for value in values):
        raise ValueError("unsigned nibble outside 0..15")
    packed = bytearray((len(values) + 1) // 2)
    for index, value in enumerate(values):
        packed[index // 2] |= value << (4 * (index % 2))
    return bytes(packed)


def unpack_nibbles(packed: bytes, count: int) -> list[int]:
    return [(packed[i // 2] >> (4 * (i % 2))) & 15 for i in range(count)]


def audit() -> dict:
    carry_cases = 0
    hard_reset_mismatches = 0
    for threshold in (15, 16):
        for residual in range(threshold):
            for incoming in range(257):
                total = residual + incoming
                pulses, remainder = divmod(total, threshold)
                assert 0 <= remainder < threshold
                assert total == threshold * pulses + remainder
                # Proposed hard reset: emit only one pulse and discard overshoot.
                hard_pulse = int(total >= threshold)
                hard_remainder = 0 if hard_pulse else total
                hard_reset_mismatches += (
                    total != threshold * hard_pulse + hard_remainder
                )
                carry_cases += 1

    packed_cases = 0
    for left in range(16):
        for right in range(16):
            values = [left, right]
            packed = pack_nibbles(values)
            assert len(packed) == 1
            assert unpack_nibbles(packed, len(values)) == values
            packed_cases += 1
    odd = [0, 15, 7]
    assert unpack_nibbles(pack_nibbles(odd), len(odd)) == odd

    clock_cases = 0
    for start_phase in range(256):
        for age in range(256):
            now_phase = (start_phase + age) % 256
            assert (now_phase - start_phase) % 256 == age
            clock_cases += 1
    assert 1 % 256 == 257 % 256
    assert (257 - 1) % 256 == 0

    # An infinite-duration path needs more bits if its exact net total grows.
    history_bits = {str(k): k.bit_length() for k in (15, 255, 65535)}
    assert history_bits == {"15": 4, "255": 8, "65535": 16}

    scalar_counts = (1_000_000, 1_000_000_000)
    return {
        "status": "arithmetic_and_encoding_verified_only",
        "production_changed": False,
        "gpu_memory_or_speed_measured": False,
        "checks": {
            "exact_carry_cases": carry_cases,
            "hard_reset_nonconservation_cases": hard_reset_mismatches,
            "nibble_pair_roundtrips": packed_cases,
            "odd_length_roundtrip": True,
            "bounded_modulo_age_cases": clock_cases,
        },
        "reset_counterexample": {
            "threshold": 15, "residual": 14, "incoming": 6,
            "exact": {"pulse_count": 1, "remainder": 5},
            "hard_reset": {"pulse_count": 1, "remainder": 0},
            "unaccounted_amount": 5,
        },
        "multiple_pulse_counterexample": {
            "threshold": 16, "total": 32,
            "required_pulse_count": 2, "one_bit_per_tick_sufficient": False,
        },
        "clock_counterexample": {
            "timestamps": [1, 257], "phases": [1, 1],
            "true_elapsed": 256, "modulo_elapsed": 0,
        },
        "clock_exactness_condition": "0 <= known elapsed < 256; phase-only dynamics must also factor through modulo",
        "accumulated_value_bit_lengths": history_bits,
        "storage_payload_bytes": {
            str(count): {
                "int64": 8 * count, "int32": 4 * count,
                "uint8": count, "packed_uint4": (count + 1) // 2,
            } for count in scalar_counts
        },
        "acceptance": {
            "narrow_storage": "prove reachable range and exact encode/decode plus transition equivalence",
            "carry_events": "preserve count, residual, sign, timing and boundary ledger",
            "modulo_phase": "periodic use or bounded-age contract with stale-event rejection",
            "full_model_lossless": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "checks": result["checks"]}))


if __name__ == "__main__":
    main()
