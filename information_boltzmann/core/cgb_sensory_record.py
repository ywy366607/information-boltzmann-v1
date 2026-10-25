"""Exact finite record of the existing Lean sensory-maintenance body.

This is a reference/encoding algorithm, not a learned brain or CUDA kernel.
Record values are unsigned nibbles; Python objects themselves are not 4-bit
allocations. Only pack_records() produces packed payload bytes. World state,
absolute time and physical external-I/O costs are not included in this record.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .cgb_causal_record import DelayedSourceCarrier

CELL_NAMES = ("vacuum", "search", "aSpent", "bSpent", "aReady", "bReady")
STIMULI = ("missing", "calm", "answerA", "answerB", "feed", "scarce", "miss", "probe")
BASE_CONTENT = (2, 1, 1, 4, 4)
CREDIT = (1, 0, 0, 2, 2)


def _integer(value: int, upper: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < upper:
        raise ValueError(f"expected integer in 0..{upper - 1}")


def read_record(record: int) -> tuple[str, int]:
    _integer(record, 16)
    if record == 0:
        return "vacuum", 0
    group, reserve = divmod(record - 1, 3)
    return CELL_NAMES[group + 1], reserve


def encode_record(cell: str, reserve: int) -> int:
    _integer(reserve, 3)
    if cell == "vacuum":
        if reserve:
            raise ValueError("nonzero-reserve vacuum is outside the record domain")
        return 0
    if cell not in CELL_NAMES[1:]:
        raise ValueError("cell is outside the proved record domain")
    return 1 + 3 * (CELL_NAMES.index(cell) - 1) + reserve


def write_record(record: int, stimulus: str) -> int:
    """Advance using only the previous record and an actual boundary stimulus."""
    cell, reserve = read_record(record)
    if stimulus not in STIMULI:
        raise ValueError("unknown boundary stimulus")
    if record == 0 or stimulus == "scarce":
        return 0
    if stimulus == "calm":
        return record
    if stimulus == "missing":
        return record - 1 if reserve else 0
    if stimulus == "miss":
        return encode_record("search", reserve) if cell in ("aReady", "bReady") else 0
    if cell == "search" and stimulus in ("answerA", "answerB"):
        cell = "aSpent" if stimulus == "answerA" else "bSpent"
    elif stimulus == "feed" and cell in ("aSpent", "bSpent"):
        cell = "aReady" if cell == "aSpent" else "bReady"
    # These registered sensed events supply resources in the original model.
    return encode_record(cell, 2)


def action(record: int) -> int:
    cell, _ = read_record(record)
    return 0 if cell.startswith("a") else 1 if cell.startswith("b") else 2


def content(record: int) -> int:
    read_record(record)
    if not record:
        return 0
    group, reserve = divmod(record - 1, 3)
    return BASE_CONTENT[group] + reserve


def ports(record: int, stimulus: str) -> tuple[int, int, int, int, int]:
    """Original body ledger: incoming, outgoing, writes, topups, lost credit.

    These are the existing discrete physical-model units. Serializing a
    record to disk has separate, uncalibrated costs and is not charged here.
    """
    cell, reserve = read_record(record)
    write_record(record, stimulus)  # Validate the stimulus as well.
    if stimulus == "calm":
        return 0, 0, 0, 0, 0
    if not record:
        return 0, 0, 0, 0, 0
    group = (record - 1) // 3
    credit = CREDIT[group]
    if stimulus == "missing":
        return 0, 1 if reserve else content(record), 0, 0, credit if not reserve else 0
    if stimulus == "scarce":
        return 0, content(record), 0, 0, credit
    if stimulus == "miss":
        ready = cell in ("aReady", "bReady")
        return 0, 2 if ready else content(record), int(ready), 0, 0 if ready else credit
    spent_feed = stimulus == "feed" and cell in ("aSpent", "bSpent")
    incoming = (4 if spent_feed else 1) + 2 - reserve
    writes = int(cell == "search" and stimulus in ("answerA", "answerB"))
    return incoming, 2 if writes else 1, writes, 2 if spent_feed else 0, 0


def pack_records(records: list[int]) -> bytes:
    data = bytearray((len(records) + 1) // 2)
    for index, record in enumerate(records):
        _integer(record, 16)
        data[index // 2] |= record << (4 * (index % 2))
    return bytes(data)


def unpack_records(data: bytes, count: int) -> list[int]:
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("record count must be a nonnegative integer")
    if len(data) != (count + 1) // 2:
        raise ValueError("payload length does not match record count")
    if count % 2 and data[-1] >> 4:
        raise ValueError("noncanonical padding nibble")
    return [(data[index // 2] >> (4 * (index % 2))) & 15 for index in range(count)]


def write_archive(path: Path, records: list[int], model_sha256: str) -> int:
    """Serialize actual records, including format/model binding and integrity.

    The caller supplies the expected model version. Hashes detect accidental
    corruption, not malicious replacement. Actual file bytes include metadata.
    """
    payload = pack_records(records)
    header = json.dumps({
        "format": "CGB.SensoryRecord.archive.v1", "count": len(records),
        "model_sha256": model_sha256,
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
    }, sort_keys=True, separators=(",", ":")).encode("utf-8")
    data = b"CGBREC1\n" + len(header).to_bytes(4, "little") + header + payload
    path.write_bytes(data)
    return len(data)


def read_archive(path: Path, expected_model_sha256: str) -> list[int]:
    data = path.read_bytes()
    if len(data) < 12 or data[:8] != b"CGBREC1\n":
        raise ValueError("invalid archive header")
    size = int.from_bytes(data[8:12], "little")
    if size > len(data) - 12:
        raise ValueError("truncated archive header")
    header = json.loads(data[12:12 + size])
    if header.get("format") != "CGB.SensoryRecord.archive.v1":
        raise ValueError("unsupported archive format")
    if header.get("model_sha256") != expected_model_sha256:
        raise ValueError("record belongs to a different model")
    payload = data[12 + size:]
    if header.get("payload_sha256") != hashlib.sha256(payload).hexdigest():
        raise ValueError("record payload checksum mismatch")
    return unpack_records(payload, header["count"])


class SensoryRecordObserver:
    """Natural one-hop boundary pairing with one paid carrier slot.

    Call once per original world tick, even during sensory loss. The only
    world input is the actual boundary stimulus; no hidden regime is supplied.
    The physical body update always runs. This observer does not control it.
    """

    def __init__(self, record: int, carrier: DelayedSourceCarrier):
        read_record(record)
        if carrier.status != "idle":
            raise ValueError("a new observer requires an idle carrier")
        self.record = record
        self.carrier = carrier

    def advance(self, stimulus: str, *, deliver: bool = False) -> tuple[int, bool, tuple | None]:
        next_record = write_record(self.record, stimulus)  # Validate before mutation.
        if not isinstance(deliver, bool):
            raise ValueError("delivery request must be boolean")
        executed_action = action(self.record)
        live = self.record != 0
        self.carrier.tick()  # Account for the preceding full world interval.
        if self.carrier.status == "awaiting":
            self.carrier.complete(None if stimulus == "missing" else stimulus)
        result = self.carrier.read() if deliver else None
        if live and self.carrier.status not in ("awaiting", "waiting"):
            # Candidate prior uses only this tick's observation. It deliberately
            # makes no unverified claim to track the evolving world posterior.
            labels = ("feed", "feed", "miss", "miss", "probe", "probe",
                      "answerA", "answerB", "scarce")
            prior = 511 if stimulus == "missing" else sum(
                1 << i for i, label in enumerate(labels) if label == stimulus)
            self.carrier.begin(prior, executed_action)
        self.record = next_record
        return executed_action, live, result
