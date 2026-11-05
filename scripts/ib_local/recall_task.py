"""Key-value recall embedded in real text, to ask whether a latch is learned and used (2026-11-03).

User-requested diagnostic task, not an admission test.  OpenWebText bytes stay the bulk of
the stream; records are inserted between them:

    fact:   "\n[#Kq7 = mango]\n"      ... G bytes of ordinary text ...
    query:  "\n[#Kq7 = mango]\n"

The query repeats the fact's format, so the value after " = " can only be predicted by
recalling the association written G bytes earlier.  Keys are 3 random characters from 62,
values are drawn from 512 common words of the text itself.

Training stream: gaps log-uniform in [16, 448], so fact and query usually fall inside one
512-byte backpropagation window.  Evaluation streams: one per gap G in {16, ..., 16384},
fresh keys, unseen text (train split from byte 3M on), frozen weights, live state; gaps beyond 512 were never inside one credit
window during training (the user's "train at 128, infer longer").
"""

from __future__ import annotations

import collections
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
KEY_CHARS = np.frombuffer(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789", np.uint8)
TRAIN_GAPS = (16, 448)
EVAL_GAPS = (16, 64, 256, 1024, 4096, 16384)


def owt_bytes(split: str, n: int) -> np.ndarray:
    data = ROOT / "data/ib_owt_10000"
    offset = json.loads((data / "tokenizer.json").read_text(encoding="utf-8"))["offset"]
    raw = np.load(data / f"{split}.npy", mmap_mode="r")
    out, at = [], 0
    while sum(len(x) for x in out) < n:
        block = np.asarray(raw[at:at + 4 * n])
        out.append((block[block >= offset] - offset).astype(np.uint8))
        at += 4 * n
        if at >= len(raw) and sum(len(x) for x in out) < n:
            raise ValueError("not enough source bytes")
    return np.concatenate(out)[:n]


def value_words(text: np.ndarray, count: int = 512) -> list[bytes]:
    words = re.findall(rb"(?<![A-Za-z])([a-z]{4,8})(?![A-Za-z])", text.tobytes())
    return [w for w, _ in collections.Counter(words).most_common(count)]


def record(key: bytes, value: bytes) -> tuple[bytes, int, int]:
    """Bytes of one record and the [start, stop) span of its value inside it."""
    head = b"\n[#" + key + b" = "
    return head + value + b"]\n", len(head), len(head) + len(value)


class Builder:
    def __init__(self, base: np.ndarray, words: list[bytes], seed: int):
        self.base, self.words = base, words
        self.rng = np.random.default_rng(seed)
        self.cursor = 0
        self.used: set[bytes] = set()

    def text(self, n: int) -> bytes:
        if self.cursor + n > len(self.base):
            self.cursor = 0
        out = self.base[self.cursor:self.cursor + n].tobytes()
        self.cursor += n
        return out

    def key(self) -> bytes:
        while True:
            key = KEY_CHARS[self.rng.integers(0, len(KEY_CHARS), 3)].tobytes()
            if key not in self.used:
                self.used.add(key)
                return key

    def pairs(self, gaps, spacer=64, limit=None):
        """Yield (bytes, value spans) for fact, gap text, query, spacer; spans index the query."""
        stream, spans, produced = bytearray(), [], 0
        for gap in gaps:
            value = self.words[self.rng.integers(0, len(self.words))]
            key = self.key()
            fact, _, _ = record(key, value)
            query, lo, hi = record(key, value)
            stream += fact + self.text(int(gap))
            spans.append((len(stream) + lo, len(stream) + hi, int(gap)))
            stream += query + self.text(spacer)
            if limit is not None and len(stream) >= limit:
                break
        return bytes(stream), spans


def train_stream(n: int, seed: int = 0, spacer: int = 128):
    base = owt_bytes("train", n)
    builder = Builder(base, value_words(base[:4_000_000]), seed)
    lo, hi = np.log(TRAIN_GAPS[0]), np.log(TRAIN_GAPS[1])
    gaps = np.exp(builder.rng.uniform(lo, hi, size=n // 64)).astype(int)
    stream, spans = builder.pairs(gaps, spacer=spacer, limit=n + 1)
    return np.frombuffer(stream[:n + 1], np.uint8).copy(), [s for s in spans if s[1] <= n]


def eval_stream(gap: int, probes: int, seed: int = 1, warmup: int = 2048,
                source_start: int = 3_000_000):
    # Text from source_start on must lie beyond what training read (2.56M for 5000 x 512).
    source = owt_bytes("train", source_start + 1_400_000)
    base, words = source[source_start:], value_words(source[:4_000_000])
    builder = Builder(base, words, seed)
    head = builder.text(warmup)
    stream, spans = builder.pairs([gap] * probes)
    spans = [(a + len(head), b + len(head), g) for a, b, g in spans]
    return np.frombuffer(head + stream, np.uint8).copy(), spans
