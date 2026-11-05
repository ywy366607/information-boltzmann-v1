"""Mode task: one bit of situation state that must be held, with dense gradient (2026-11-03).

User-requested diagnostic for the latch (what a biological latch holds is self/situation
state: which mode am I in).  OpenWebText bytes with inserted

    marker:  "\n<mode:A>\n"  or  "\n<mode:B>\n"   sets the mode until the next marker
    query:   "{?" then "}"                         asks for the current mode

At the query position after "?" the training target is the mode letter, but the next
input byte is "}": the answer is never shown, so after a marker nothing in the text reveals
the mode again.  Queries are frequent (gaps log-uniform in [32, 256] bytes), so every few
dozen bytes the loss asks whether the bit is still held; inside a 512-byte credit window,
holding it is rewarded, and holding it across every window is holding it for good.

Training segments (marker to next marker) are log-uniform in [256, 4096] bytes; evaluation
segments are 32768 bytes long, so distances beyond 4096 were never seen in training.
Streams are (inputs, targets) with targets[i] the byte to predict after inputs[i].
"""

from __future__ import annotations

import numpy as np

from scripts.ib_local.recall_task import owt_bytes

QUERY_GAPS = (32, 256)
TRAIN_SEGMENTS = (256, 4096)
EVAL_SEGMENT = 32768
# Bootstrap stage: the first run gave flip_effect 0.000 even 2 bytes after the marker (the
# read-write circuit never formed: a saddle where q-k alignment, value write and readout are
# each useless without the others).  Short dense segments come first to form the circuit.
EASY_QUERY_GAPS = (4, 32)
EASY_SEGMENTS = (16, 128)


def marker(mode: int) -> bytes:
    return b"\n<mode:" + (b"A" if mode == 0 else b"B") + b">\n"


def _build(base: np.ndarray, n: int, rng: np.random.Generator, segment_sampler,
           query_gaps=QUERY_GAPS):
    inputs, targets, queries = bytearray(), [], []  # queries: (position, mode, distance)
    cursor, mode = 0, int(rng.integers(2))

    def text(k):
        nonlocal cursor
        if cursor + k > len(base):
            cursor = 0
        out = base[cursor:cursor + k].tobytes()
        cursor += k
        return out

    while len(inputs) < n + 1:
        mode = int(rng.integers(2))  # independent each segment: half the markers switch
        start = len(inputs)
        inputs += marker(mode)
        length = segment_sampler()
        while len(inputs) - start < length:
            inputs += text(int(np.exp(rng.uniform(*np.log(query_gaps)))))
            inputs += b"{?"
            queries.append((len(inputs) - 1, mode, len(inputs) - start))
            inputs += b"}"
    inputs = np.frombuffer(bytes(inputs[:n + 1]), np.uint8).copy()
    targets = inputs[1:].copy()
    for position, mode, _ in queries:
        if position < n:
            targets[position] = ord("A") if mode == 0 else ord("B")
    queries = [q for q in queries if q[0] < n]
    return inputs[:n], targets[:n], queries


def train_stream(n: int, seed: int = 0, easy: int = 0, longest: int = TRAIN_SEGMENTS[1]):
    """The first `easy` bytes use the bootstrap schedule, the rest the long one, with
    segments log-uniform in [256, longest]."""
    rng = np.random.default_rng(seed)
    base = owt_bytes("train", n)
    parts = []
    if easy:
        lo, hi = np.log(EASY_SEGMENTS)
        parts.append(_build(base[:easy], easy, rng, lambda: int(np.exp(rng.uniform(lo, hi))),
                            EASY_QUERY_GAPS))
    if n > easy:
        lo, hi = np.log((TRAIN_SEGMENTS[0], longest))
        parts.append(_build(base[easy:], n - easy, rng, lambda: int(np.exp(rng.uniform(lo, hi)))))
    offset, queries = 0, []
    for inputs, _, qs in parts:
        queries += [(p + offset, m, d) for p, m, d in qs]
        offset += len(inputs)
    return (np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]), queries)


def eval_stream(n: int = 6 * EVAL_SEGMENT, seed: int = 1, source_start: int = 12_000_000,
                segment: int = EVAL_SEGMENT):
    rng = np.random.default_rng(seed)
    base = owt_bytes("train", source_start + n + 200_000)[source_start:]
    return _build(base, n, rng, lambda: segment)
