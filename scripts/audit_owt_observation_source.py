"""Read-only audit that the real OWT token source fits the CGB stream boundary."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


def load_stream_module(root: Path):
    contract_path = root / "information_boltzmann" / "core" / "cgb_low_resource_life.py"
    stream_path = root / "information_boltzmann" / "core" / "cgb_observation_stream.py"
    contract_spec = importlib.util.spec_from_file_location("cgb_low_resource_life", contract_path)
    if contract_spec is None or contract_spec.loader is None:
        raise RuntimeError(f"cannot load {contract_path}")
    contract = importlib.util.module_from_spec(contract_spec)
    sys.modules[contract_spec.name] = contract
    contract_spec.loader.exec_module(contract)
    stream_spec = importlib.util.spec_from_file_location("cgb_observation_stream", stream_path)
    if stream_spec is None or stream_spec.loader is None:
        raise RuntimeError(f"cannot load {stream_path}")
    stream = importlib.util.module_from_spec(stream_spec)
    sys.modules[stream_spec.name] = stream
    stream_spec.loader.exec_module(stream)
    return stream


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "data" / "ib_owt_gpt2" / "train.npy"
    if not path.exists():
        raise FileNotFoundError(path)
    tokens = np.load(path, mmap_mode="r")
    if tokens.ndim != 1 or len(tokens) < 3:
        raise AssertionError(f"expected a one-dimensional stream with at least 3 tokens: {tokens.shape}")
    stream = load_stream_module(root)
    sample = list(stream.token_observations(tokens[:1024], source="owt_train"))
    assert len(sample) == min(1024, len(tokens))
    assert sample[0].tick == 0
    assert sample[-1].tick == len(sample) - 1
    assert all(event.source == "owt_train" for event in sample)
    assert all(event.observation().arrived for event in sample)
    print(
        "cgb_owt_source_ok",
        f"total_tokens={len(tokens)}",
        f"sampled={len(sample)}",
        f"first_token={sample[0].payload}",
        f"last_token={sample[-1].payload}",
    )


if __name__ == "__main__":
    main()
