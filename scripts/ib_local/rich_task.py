"""Rich-world task: two held states at two timescales plus decoys, in one OWT stream (2026-11-04).

User's goal: no single mechanism should take everything ("一家独大"); in a world that asks for
several timescales, fast heads, slowly decaying heads and latching heads should each be used,
each in its place, and latching plus recovery should be needed (an agent must hold what it is
doing, and get back to it after a disturbance).

Stream (bytes; targets[i] is the byte to predict after inputs[i]):
    long state  M in {A, B}: marker "\n<mode:A>\n", segments log-uniform [1024, 32768] bytes
                (evaluation: 131072, 4x the longest trained segment);
    medium state P in {x, y}: marker "\n<who:x>\n", segments log-uniform [64, 1024];
    decoys: "\n<mood:B>\n" with a random letter, about one per 1500 bytes; byte-wise close to a
                mode marker but must not change M (a disturbance the long state must survive);
    queries every 32-256 bytes, half "{?" (answer: M) and half "{@" (answer: P); the answer is
                the target at the query byte and is never shown: the next input byte is "}".
M and P are independent, and nothing in recent text reveals either once its marker is past.

Curriculum (developmental order). The first run (rich_latch_s11) learned P perfectly and M not
at all, even at 0-1 kB: a "copy the letter of the latest <...:X> marker" circuit solves P and
leaves M, whose marker is older and of another kind, at the read-write saddle. So the dense
bootstrap first teaches M alone (as in mode_task, where it forms), then both, then the long
world: the first half of the `easy` bytes has no P markers or queries.
"""

from __future__ import annotations

import numpy as np

from scripts.ib_local.recall_task import owt_bytes

QUERY_GAPS = (32, 256)
LONG_SEGMENTS = (1024, 32768)
MEDIUM_SEGMENTS = (64, 1024)
EVAL_LONG = 131072
DECOY_RATE = 1 / 1500
EASY_QUERY_GAPS = (4, 32)
EASY_LONG = (32, 256)
EASY_MEDIUM = (16, 128)


def mode_marker(m: int) -> bytes:
    return b"\n<mode:" + b"AB"[m:m + 1] + b">\n"


def who_marker(p: int) -> bytes:
    return b"\n<who:" + b"xy"[p:p + 1] + b">\n"


def decoy(letter: int) -> bytes:
    return b"\n<mood:" + b"AB"[letter:letter + 1] + b">\n"


def _loguniform(rng, bounds):
    return int(np.exp(rng.uniform(*np.log(bounds))))


def _build(base, n, rng, long_len, medium_len, query_gaps, decoys=True, with_p=True):
    """queries: (position, kind 0=M 1=P, answer, distance from that state's marker,
    bytes since the last decoy or None)."""
    out, queries = bytearray(), []
    cursor = 0

    def text(k):
        nonlocal cursor
        if cursor + k > len(base):
            cursor = 0
        chunk = base[cursor:cursor + k].tobytes()
        cursor += k
        return chunk

    m, p = int(rng.integers(2)), int(rng.integers(2))
    m_start = p_start = 0
    m_end, p_end = long_len(), medium_len()
    out += mode_marker(m) + (who_marker(p) if with_p else b"")
    last_decoy = None
    while len(out) < n + 1:
        gap = _loguniform(rng, query_gaps)
        out += text(gap)
        if decoys and rng.random() < gap * DECOY_RATE:
            out += decoy(int(rng.integers(2)))
            last_decoy = len(out)
        if len(out) - m_start >= m_end:
            m, m_start, m_end = int(rng.integers(2)), len(out), long_len()
            out += mode_marker(m)
        if with_p and len(out) - p_start >= p_end:
            p, p_start, p_end = int(rng.integers(2)), len(out), medium_len()
            out += who_marker(p)
        kind = int(rng.integers(2)) if with_p else 0
        out += b"{?" if kind == 0 else b"{@"
        position = len(out) - 1
        answer = (b"AB"[m] if kind == 0 else b"xy"[p])
        start = m_start if kind == 0 else p_start
        queries.append((position, kind, answer, position - start,
                        None if last_decoy is None else position - last_decoy))
        out += b"}"
    inputs = np.frombuffer(bytes(out[:n + 1]), np.uint8).copy()
    targets = inputs[1:].copy()
    queries = [q for q in queries if q[0] < n]
    for position, _, answer, _, _ in queries:
        targets[position] = answer
    return inputs[:n], targets[:n], queries


def train_stream(n: int, seed: int = 0, easy: int = 0):
    rng = np.random.default_rng(seed)
    base = owt_bytes("train", n)
    parts = []
    if easy:
        half = easy // 2
        parts.append(_build(base[:half], half, rng, lambda: _loguniform(rng, EASY_LONG),
                            lambda: _loguniform(rng, EASY_MEDIUM), EASY_QUERY_GAPS, decoys=False,
                            with_p=False))
        parts.append(_build(base[half:easy], easy - half, rng, lambda: _loguniform(rng, EASY_LONG),
                            lambda: _loguniform(rng, EASY_MEDIUM), EASY_QUERY_GAPS, decoys=False))
    if n > easy:
        parts.append(_build(base[easy:], n - easy, rng, lambda: _loguniform(rng, LONG_SEGMENTS),
                            lambda: _loguniform(rng, MEDIUM_SEGMENTS), QUERY_GAPS))
    offset, queries = 0, []
    for inputs, _, qs in parts:
        queries += [(q[0] + offset,) + q[1:] for q in qs]
        offset += len(inputs)
    return (np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]), queries)


def eval_stream(n: int, seed: int = 1, source_start: int = 12_000_000, long_len: int = EVAL_LONG):
    rng = np.random.default_rng(seed)
    base = owt_bytes("train", source_start + n + 200_000)[source_start:]
    return _build(base, n, rng, lambda: long_len, lambda: _loguniform(rng, MEDIUM_SEGMENTS),
                  QUERY_GAPS)


def flip_markers(stream: np.ndarray, which: str) -> np.ndarray:
    """Swap the letter of every mode marker (which='M') or who marker (which='P')."""
    out = stream.copy()
    raw = stream.tobytes()
    prefix, pair = (b"\n<mode:", b"AB") if which == "M" else (b"\n<who:", b"xy")
    at = raw.find(prefix)
    while at >= 0:
        i = at + len(prefix)
        if i < len(out):
            out[i] = pair[1] if out[i] == pair[0] else pair[0]
        at = raw.find(prefix, at + 1)
    return out
