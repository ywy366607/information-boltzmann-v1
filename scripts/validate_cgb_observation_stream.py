"""Validate the dependency-free JSONL-to-observation boundary."""

from __future__ import annotations

import importlib.util
import hashlib
import io
import json
import sys
from pathlib import Path


def load_modules():
    root = Path(__file__).resolve().parents[1]
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
    return contract, stream


def expect_rejection(stream_module, text: str, fragment: str) -> None:
    try:
        list(stream_module.JsonlObservationStream(io.StringIO(text)))
    except stream_module.StreamFormatError as exc:
        assert fragment in str(exc), str(exc)
    else:
        raise AssertionError(f"stream was accepted: {text!r}")


def validate_native_bytes(stream_module) -> None:
    import numpy as np

    documents = ["é汉🙂\x00\x01\x02".encode(), b"B\n", "中Z".encode()]
    records = [{"sha256": hashlib.sha256(doc).hexdigest()} for doc in documents]
    raw = np.asarray([value for doc in documents for value in [*(b + 4 for b in doc), 2, 1]])
    view = stream_module.NativeByteView(raw, records, "a" * 64)
    expected = b"".join(documents)
    assert len(view) == len(expected) and view[:].dtype == np.uint8
    assert view[:].tobytes() == expected and bytes(view[i] for i in range(len(view))) == expected
    events = list(view.observations(start_tick=7))
    assert [e.tick for e in events] == list(range(7, 7 + len(expected)))
    assert [e.payload for e in events] == list(expected)
    assert [i for i, e in enumerate(events) if e.metadata["document_start"]] == [0, len(documents[0]), len(documents[0] + documents[1])]
    assert view.cursor(len(documents[0]))["raw_offset"] == len(documents[0]) + 2
    assert view.cursor(len(view))["raw_offset"] == len(raw)
    for offset in range(len(view)):
        cursor = json.loads(json.dumps(view.cursor(offset)))
        resumed = list(view.observations(cursor=cursor, start_tick=offset, limit=len(view) - offset))
        assert bytes(e.payload for e in resumed) == expected[offset:]
        assert [e.metadata for e in resumed] == [e.metadata for e in events[offset:]]
    assert view[1] == documents[0][1]  # Resume inside a multibyte UTF-8 character.

    def rejected(operation):
        try:
            operation()
        except stream_module.StreamFormatError:
            pass
        else:
            raise AssertionError("invalid byte source/cursor was accepted")

    rejected(lambda: stream_module.NativeByteView(raw[:-1], records, "a" * 64))
    corrupt = raw.copy()
    corrupt[0] += 1
    rejected(lambda: stream_module.NativeByteView(corrupt, records, "a" * 64))
    rejected(lambda: list(view.observations(cursor={**view.cursor(1), "raw_offset": 0})))
    rejected(lambda: list(view.observations(cursor={**view.cursor(1), "source_sha256": "b" * 64})))
    rejected(lambda: list(view.observations(start_tick=-1)))
    long_doc = b"A" * 999_999 + "é".encode()
    large = np.concatenate((np.frombuffer(long_doc, dtype=np.uint8).astype(np.int64) + 4, [2, 1]))
    bounded = stream_module.NativeByteView(large, [{"sha256": hashlib.sha256(long_doc).hexdigest()}], "c" * 64)
    assert bounded[-2:].tobytes() == "é".encode()
    assert bounded[:1_000_000].tobytes() == long_doc[:1_000_000]
    rejected(lambda: bounded[:])
    # EOS ends a read block; the matching BOS begins the next block.
    edge_doc = b"A" * 999_999
    edge_raw = np.concatenate((np.full(len(edge_doc), 69), [2, 1, 70, 2, 1]))
    edge_records = [{"sha256": hashlib.sha256(doc).hexdigest()} for doc in (edge_doc, b"B")]
    edge = stream_module.NativeByteView(edge_raw, edge_records, "d" * 64)
    assert edge[999_998:].tobytes() == b"AB"
    assert edge.cursor(999_999)["raw_offset"] == 1_000_001


def main() -> None:
    contract, stream_module = load_modules()
    validate_native_bytes(stream_module)
    source = io.StringIO(
        '{"tick": 0, "token": "a", "source": "owt"}\n'
        '{"tick": 1, "payload": {"token": "b"}, "source": "owt", "loss": true}\n'
    )
    events = list(stream_module.JsonlObservationStream(source))
    assert [event.tick for event in events] == [0, 1]
    assert [event.payload for event in events] == ["a", {"token": "b"}]
    assert events[1].metadata == {"loss": True}
    observations = list(
        stream_module.JsonlObservationStream(
            io.StringIO('{"tick": 0, "token": "a"}\n')
        ).observations()
    )
    assert isinstance(observations[0], contract.Observation)
    assert observations[0].arrived and observations[0].tick == 0

    token_events = list(
        stream_module.token_observations([11, 50256], source="owt", start_tick=7)
    )
    assert [event.tick for event in token_events] == [7, 8]
    assert [event.payload for event in token_events] == [11, 50256]
    assert token_events[0].metadata == {"token_offset": 0}
    expect_rejection(stream_module, "", "no events")
    try:
        list(stream_module.token_observations([1, -1]))
    except stream_module.StreamFormatError as exc:
        assert "nonnegative" in str(exc)
    else:
        raise AssertionError("negative token was accepted")

    expect_rejection(
        stream_module,
        '{"tick": 0, "token": "a"}\n{"tick": 2, "token": "c"}\n',
        "expected 1",
    )
    expect_rejection(
        stream_module,
        '{"tick": 0, "token": "a", "arrived": false}\n',
        "did not arrive",
    )
    expect_rejection(stream_module, "{\"tick\": 0}\n", "exactly one")
    print("cgb_observation_stream_ok")


if __name__ == "__main__":
    main()
