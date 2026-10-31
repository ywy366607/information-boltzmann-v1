"""Quantization and bit-width analysis for the current finite CGB witness."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def bits_for_unsigned_max(max_value: int) -> int:
    return max(1, math.ceil(math.log2(max_value + 1)))


def quantize_error(
    values: list[float], bits: int, lo: float, hi: float, levels: int | None = None
) -> dict[str, float]:
    levels = levels if levels is not None else 2**bits - 1
    reconstructed = [lo + round((v - lo) / (hi - lo) * levels) / levels * (hi - lo) for v in values]
    errors = [abs(a - b) for a, b in zip(values, reconstructed)]
    return {
        "max_abs_error": max(errors),
        "mean_abs_error": sum(errors) / len(errors),
        "step": (hi - lo) / levels,
    }


def bytes_for(bits: int) -> int:
    return (bits + 7) // 8


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    states = 24
    inputs = 16
    edges = states * inputs
    h = 3
    dyadic_h3 = [0.0, 1 / 8, 1 / 4, 1 / 2, 1.0]
    flow_max = 6
    reserve_max = 2
    result = {
        "scope": "exact bit-width and quantization analysis for the current finite witness",
        "states": states,
        "inputs": inputs,
        "transition_edges": edges,
        "exact_ranges": {
            "binary_transition_coefficients": {"values": [0, 1], "minimum_bits": 1},
            "material_flow_counters": {"range": [0, flow_max], "minimum_unsigned_bits": bits_for_unsigned_max(flow_max)},
            "reserve": {"range": [0, reserve_max], "minimum_unsigned_bits": bits_for_unsigned_max(reserve_max)},
            "cell_index": {"range": [0, 7], "minimum_unsigned_bits": 3},
            "state_index": {"range": [0, states - 1], "minimum_unsigned_bits": 5},
        },
        "quantization": {
            "current_binary_transition": {
                "int1_max_abs_error": 0.0,
                "int4_max_abs_error": 0.0,
                "int8_max_abs_error": 0.0,
            },
            "three_tick_dyadic_probabilities": {
                "values": dyadic_h3,
                "q3_fixed_scale_8_in_4bit": quantize_error(dyadic_h3, 4, 0.0, 1.0, levels=8),
                "q4_fixed_scale_15": quantize_error(dyadic_h3, 4, 0.0, 1.0),
                "q8_fixed_scale_255": quantize_error(dyadic_h3, 8, 0.0, 1.0),
            },
            "arbitrary_unit_interval_worst_case": {
                "q4": {"max_abs_error_bound": 1 / 30, "step": 1 / 15},
                "q8": {"max_abs_error_bound": 1 / 510, "step": 1 / 255},
            },
        },
        "storage": {
            "dense_transition_binary_bytes": math.ceil(inputs * states * states * 1 / 8),
            "sparse_target_only_5bit_bytes": math.ceil(edges * 5 / 8),
            "sparse_target_plus_3bit_flow_bytes": math.ceil(edges * (5 + 3) / 8),
            "dense_transition_q4_bytes": math.ceil(inputs * states * states * 4 / 8),
            "dense_transition_q8_bytes": math.ceil(inputs * states * states * 8 / 8),
            "one_state_vector_q4_bytes": math.ceil(states * 4 / 8),
            "one_state_vector_q8_bytes": states,
            "one_state_vector_float16_bytes": states * 2,
            "one_state_vector_float32_bytes": states * 4,
        },
        "recommendation": {
            "exact_current_transition_storage": "1-bit dense or 5-bit sparse target index",
            "exact_current_flow_storage": "3-bit unsigned counters; int4 is sufficient with margin",
            "exact_three_tick_dyadic_probability_storage": "Q3 fixed-point (scale 8 in a 4-bit unsigned container)",
            "accumulator": "int32 for current 24-state kernel; widen if batch or path sums grow",
            "universal_real_valued_claim": False,
        },
    }
    print(json.dumps(result, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
