"""Recorded JSONL/token boundaries and bounded native UTF-8 byte views.

The adapter owns only the boundary between a recorded stream and the runtime
contract.  It does not tokenize, train, or infer capability.  Every accepted
line denotes one physical tick; gaps, duplicates, and explicit non-arrival are
rejected instead of being silently converted into a pause.
"""

from __future__ import annotations

from dataclasses import dataclass
from bisect import bisect_right
import hashlib
import json
from numbers import Integral
from pathlib import Path
from typing import Any, Iterable, Iterator, TextIO

try:  # Normal package import.
    from .cgb_low_resource_life import Observation
except ImportError:  # Direct loading used by standalone validators.
    from cgb_low_resource_life import Observation


class StreamFormatError(ValueError):
    """Raised when a recorded stream violates the continuous-input contract."""


@dataclass(frozen=True)
class StreamEvent:
    """One validated source event and its CGB observation view."""

    tick: int
    payload: Any
    source: str | None
    metadata: dict[str, Any]

    def observation(self) -> Observation:
        return Observation(tick=self.tick, payload=self.payload, arrived=True)


class NativeByteView:
    """Exact offset-four corpus view; EOS/BOS stay metadata, never fake bytes.

    Retains one end offset per verified document, not a second payload array.
    A byte tick is the declared native world's event, not a GPT-2 token tick.
    """

    def __init__(self, values, documents, source_sha256: str):
        import numpy as np
        if values.ndim != 1 or values.dtype.kind not in "iu" or not len(values):
            raise StreamFormatError("byte source needs a nonempty integer vector")
        self.values, self.documents, self.source_sha256 = values, documents, source_sha256
        self.ends, digest, total, expect_bos = [], hashlib.sha256(), 0, False
        for start in range(0, len(values), 1_000_000):
            block, low = values[start:start + 1_000_000], 0
            for cut in np.flatnonzero((block < 4) | (block >= 260)):
                piece = block[low:cut]
                if expect_bos and len(piece):
                    raise StreamFormatError("bytes occurred before document BOS")
                digest.update((piece - 4).astype(np.uint8).tobytes())
                total += len(piece)
                if expect_bos:
                    if int(block[cut]) != 1:
                        raise StreamFormatError("expected document BOS=1")
                else:
                    if (int(block[cut]) != 2 or len(self.ends) >= len(documents)
                            or digest.hexdigest() != documents[len(self.ends)]["sha256"]):
                        raise StreamFormatError("document bytes/hash or EOS=2 disagree")
                    self.ends.append(total)
                    digest = hashlib.sha256()
                expect_bos = not expect_bos
                low = int(cut) + 1
            piece = block[low:]
            if expect_bos and len(piece):
                raise StreamFormatError("bytes occurred before document BOS")
            digest.update((piece - 4).astype(np.uint8).tobytes())
            total += len(piece)
        if (expect_bos or len(self.ends) != len(documents) or not self.ends
                or total != self.ends[-1] or any(b <= a for a, b in zip([0] + self.ends, self.ends))):
            raise StreamFormatError("incomplete or empty document in byte source")

    def __len__(self):
        return self.ends[-1]

    def cursor(self, offset: int) -> dict:
        if type(offset) is not int or not 0 <= offset <= len(self):
            raise StreamFormatError("byte cursor outside source")
        return {"format": "offset4-byte-v1", "source_sha256": self.source_sha256,
                "byte_offset": offset, "raw_offset": offset + 2 * bisect_right(self.ends, offset)}

    def __getitem__(self, key):
        import numpy as np
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            if step != 1 or stop - start > 1_000_000:
                raise StreamFormatError("byte slices require unit stride and at most 1000000 bytes")
            raw = self.values[self.cursor(start)["raw_offset"]:self.cursor(stop)["raw_offset"]]
            return (raw[raw >= 4] - 4).astype(np.uint8)
        if type(key) is not int or not 0 <= key < len(self):
            raise StreamFormatError("byte index outside source")
        return int(self.values[self.cursor(key)["raw_offset"]]) - 4

    def observations(self, *, start=0, limit=None, start_tick=0, cursor=None):
        if cursor is not None:
            start = cursor.get("byte_offset")
            if cursor != self.cursor(start):
                raise StreamFormatError("restored byte cursor disagrees with source")
        self.cursor(start)
        limit = len(self) - start if limit is None else limit
        if type(limit) is not int or limit < 1 or start + limit > len(self):
            raise StreamFormatError("byte observation range outside source")
        if type(start_tick) is not int or start_tick < 0:
            raise StreamFormatError("byte start tick must be nonnegative")
        for delta in range(limit):
            index = start + delta
            doc = bisect_right(self.ends, index)
            yield StreamEvent(start_tick + delta, self[index], "owt_utf8_bytes",
                {"byte_offset": index, "document_start": index == (self.ends[doc - 1] if doc else 0),
                 "document_sha256": self.documents[doc]["sha256"], "next_cursor": self.cursor(index + 1)})


def load_owt_sources(train_path: Path, validation_path: Path, alphabet: str):
    """Verify file provenance and document splitting before exposing real events."""
    import numpy as np

    def digest(path):
        result = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                result.update(block)
        return result.hexdigest()

    if train_path.resolve().parent != validation_path.resolve().parent:
        raise StreamFormatError("source splits must share their manifest")
    folder = train_path.parent
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    byte = alphabet == "byte"
    if (alphabet not in ("byte", "gpt2") or manifest.get("source") != "openwebtext"
            or not manifest.get("revision") or manifest.get("tokenizer") != ("byte" if byte else "gpt2")
            or manifest.get("vocab_size") != (260 if byte else 50257)
            or manifest.get("split_rule") != "NFC_document_sha256_mod100_98_1_1"):
        raise StreamFormatError("unrecognized OWT source, alphabet or split schema")
    documents = manifest["documents"]
    if len({row["sha256"] for row in documents}) != len(documents):
        raise StreamFormatError("duplicate document in source manifest")
    for row in documents:
        bucket = int(row["sha256"], 16) % 100
        if row["split"] != ("train" if bucket < 98 else "validation" if bucket == 98 else "test"):
            raise StreamFormatError("document split disagrees with its hash")
    hashes = {"tokenizer.json": digest(folder / "tokenizer.json")}
    tokenizer = json.loads((folder / "tokenizer.json").read_text(encoding="utf-8"))
    valid = (tokenizer.get("type") == "utf8_bytes" and tokenizer.get("offset") == 4 if byte else
             tokenizer.get("encoding") == "gpt2" and tokenizer.get("n_vocab") == 50257)
    if not valid or hashes["tokenizer.json"] != manifest["files"].get("tokenizer.json"):
        raise StreamFormatError("tokenizer metadata or checksum disagrees with source")
    arrays = []
    for path, split in ((train_path, "train"), (validation_path, "validation")):
        hashes[path.name] = digest(path)
        if path.name != split + ".npy" or hashes[path.name] != manifest["files"].get(path.name):
            raise StreamFormatError("source filename or checksum mismatch")
        values = np.load(path, mmap_mode="r")
        if (values.ndim != 1 or values.dtype.kind not in "iu" or not len(values)
                or len(values) != manifest["token_counts"][split]
                or int(values.min()) < 0 or int(values.max()) >= manifest["vocab_size"]):
            raise StreamFormatError("invalid declared source array")
        arrays.append(NativeByteView(values, [r for r in documents if r["split"] == split],
                                     hashes[path.name]) if byte else values)
    provenance = {"manifest": digest(folder / "manifest.json"), "files": hashes}
    if byte:
        provenance.update(verified_document_counts={s: len(a.documents) for s, a in zip(("train", "validation"), arrays)},
                          native_byte_counts={s: len(a) for s, a in zip(("train", "validation"), arrays)})
    return (*arrays, manifest, provenance)


def token_observations(
    tokens: Iterable[int],
    *,
    source: str = "token_stream",
    start_tick: int = 0,
) -> Iterator[StreamEvent]:
    """Adapt an existing integer token stream without changing its clock.

    This is the bridge for the read-only OWT ``train.npy`` source and for
    future token iterators.  ``start_tick`` is explicit so a resumed stream
    cannot silently restart its physical clock at zero.
    """

    if isinstance(start_tick, bool) or not isinstance(start_tick, int) or start_tick < 0:
        raise StreamFormatError("start_tick must be a nonnegative integer")
    if not isinstance(source, str) or not source:
        raise StreamFormatError("source must be a non-empty string")
    seen_event = False
    for offset, token in enumerate(tokens):
        if isinstance(token, bool) or not isinstance(token, Integral):
            raise StreamFormatError(f"token offset {offset}: token must be an integer")
        token_value = int(token)
        if token_value < 0:
            raise StreamFormatError(f"token offset {offset}: token must be nonnegative")
        seen_event = True
        yield StreamEvent(
            tick=start_tick + offset,
            payload=token_value,
            source=source,
            metadata={"token_offset": offset},
        )
    if not seen_event:
        raise StreamFormatError("token stream contains no events")


class JsonlObservationStream(Iterable[StreamEvent]):
    """Validate and expose a JSONL stream as contiguous ``StreamEvent`` values.

    Each non-empty line must be an object with an integer ``tick`` and exactly
    one of ``payload`` or ``token``.  ``arrived`` may be omitted, but when it is
    present it must be ``true``.  The first event must have tick zero and every
    following event must advance by one.
    """

    def __init__(self, stream: TextIO):
        self._stream = stream

    @classmethod
    def from_path(cls, path: str | Path) -> "JsonlObservationStream":
        return cls(Path(path).open("r", encoding="utf-8"))

    def __iter__(self) -> Iterator[StreamEvent]:
        expected_tick = 0
        seen_event = False
        for line_number, raw_line in enumerate(self._stream, start=1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise StreamFormatError(
                    f"line {line_number}: invalid JSON"
                ) from exc
            if not isinstance(record, dict):
                raise StreamFormatError(f"line {line_number}: event must be an object")
            tick = record.get("tick")
            if isinstance(tick, bool) or not isinstance(tick, int):
                raise StreamFormatError(f"line {line_number}: tick must be an integer")
            if tick != expected_tick:
                relation = "missing or out-of-order" if tick > expected_tick else "duplicate or out-of-order"
                raise StreamFormatError(
                    f"line {line_number}: {relation} tick {tick}; expected {expected_tick}"
                )
            if record.get("arrived", True) is not True:
                raise StreamFormatError(f"line {line_number}: event did not arrive")
            has_payload = "payload" in record
            has_token = "token" in record
            if has_payload == has_token:
                raise StreamFormatError(
                    f"line {line_number}: provide exactly one of payload or token"
                )
            source = record.get("source")
            if source is not None and not isinstance(source, str):
                raise StreamFormatError(f"line {line_number}: source must be a string")
            payload = record["payload"] if has_payload else record["token"]
            metadata = {
                key: value
                for key, value in record.items()
                if key not in {"tick", "payload", "token", "source", "arrived"}
            }
            yield StreamEvent(tick, payload, source, metadata)
            seen_event = True
            expected_tick += 1
        if not seen_event:
            raise StreamFormatError("stream contains no events")

    def observations(self) -> Iterator[Observation]:
        """Yield the contract observations while retaining stream validation."""

        for event in self:
            yield event.observation()
