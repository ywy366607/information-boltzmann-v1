"""Byte n-gram fast table: hashed one-hot-key delta memory that lives in RAM or on disk (2026-11-03).

The table is the Qwen3.8 / Engram hashed n-gram lookup turned into a test-time-trained
memory.  With one-hot (hashed) keys the delta rule is exact per-slot assimilation
(`mixed_key_delta_differs_from_slot_assimilation`), so each slot is an online estimate of
the next-byte distribution after its context:

* address: for each order n and hash head h, slot = hash_h(last n bytes); a 16-bit
  fingerprint marks which context owns the slot (online stand-in for MultiHashFormer's
  rehashing, which needs a closed vocabulary).
* write: when the next byte y arrives, row <- row + gain (onehot(y) - row), with a per-slot
  Kalman gain max(1/(n+2), floor) from the slot's own visit count (occurrence clock,
  process-noise floor).  Slots that are not visited do not decay.
* collisions: a different owner decrements the count; at zero it takes the slot
  (space-saving eviction: frequent contexts keep their slots, rare ones are released).
* read: heads and orders are combined by precision, count / (count + 1) / running Brier
  error, so a crowded or unreliable slot loses weight.

Everything the table does depends only on the byte stream, not on any network, so it can
run on the CPU ahead of the network and its arrays can be ordinary RAM or `np.memmap`
files.  `run()` returns the table's own prequential NLL per byte and its read distribution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numba
import numpy as np

MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)


@numba.njit(cache=True, inline="always")
def _mix(x):
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


@numba.njit(cache=True)
def _run(stream, rows, count, tag, err, orders, heads, slots, gain_floor, err_floor,
         prior_weight, history, nll, out_probs, keep_probs):
    """Prequential pass: predict stream[t+1] from contexts ending at t, then write it."""
    n_orders = orders.shape[0]
    vocab = rows.shape[1]
    p = np.empty(vocab, dtype=np.float64)
    owner = np.empty((n_orders, heads), dtype=np.int64)
    sig = np.empty((n_orders, heads), dtype=np.uint16)
    for t in range(stream.shape[0] - 1):
        history = ((history << np.uint64(8)) | np.uint64(stream[t])) & MASK64
        for j in range(vocab):
            p[j] = prior_weight / vocab
        total = prior_weight
        for o in range(n_orders):
            n = orders[o]
            key = history & ((np.uint64(1) << np.uint64(8 * n)) - np.uint64(1)) if n < 8 else history
            for h in range(heads):
                code = _mix(key ^ _mix(np.uint64(o * 131 + h * 7919 + 1)))
                slot = (o * heads + h) * slots + np.int64(code & np.uint64(slots - 1))
                fingerprint = np.uint16(((code >> np.uint64(40)) & np.uint64(0xFFFF)) | np.uint64(1))
                owner[o, h] = slot
                sig[o, h] = fingerprint
                if tag[slot] == fingerprint and count[slot] > 0:
                    c = count[slot]
                    w = c / (c + 1.0) / (err[slot] + err_floor)
                    total += w
                    for j in range(vocab):
                        p[j] += w * rows[slot, j]
        y = stream[t + 1]
        for j in range(vocab):
            p[j] /= total
        nll[t] = -math.log(p[y])
        if keep_probs:
            for j in range(vocab):
                out_probs[t, j] = p[j]
        # Write the arrived byte into every addressed slot.
        for o in range(n_orders):
            for h in range(heads):
                slot = owner[o, h]
                fingerprint = sig[o, h]
                if tag[slot] != fingerprint:
                    if count[slot] > 0:
                        count[slot] -= 1
                        continue
                    tag[slot] = fingerprint
                    for j in range(vocab):
                        rows[slot, j] = 1.0 / vocab
                    err[slot] = 1.0
                c = count[slot]
                brier = 0.0
                for j in range(vocab):
                    d = rows[slot, j] - (1.0 if j == y else 0.0)
                    brier += d * d
                gain = max(1.0 / (c + 2.0), gain_floor)
                err[slot] += max(1.0 / (c + 1.0), gain_floor) * (brier - err[slot])
                for j in range(vocab):
                    rows[slot, j] *= 1.0 - gain
                rows[slot, y] += gain
                count[slot] = c + 1
    return history


@dataclass
class ByteNgramFastTable:
    orders: tuple = (0, 1, 2, 3, 4, 5, 6)
    heads: int = 2
    slots_log2: int = 14
    gain_floor: float = 1.0 / 256
    err_floor: float = 0.02
    prior_weight: float = 0.05
    storage: Path | None = None  # directory for np.memmap arrays; None keeps them in RAM
    history: np.uint64 = field(default=np.uint64(0))

    def __post_init__(self):
        n = len(self.orders) * self.heads * (1 << self.slots_log2)
        shapes = dict(rows=((n, 256), np.float32), count=((n,), np.int64),
                      tag=((n,), np.uint16), err=((n,), np.float32))
        for name, (shape, dtype) in shapes.items():
            if self.storage is None:
                array = np.zeros(shape, dtype=dtype)
            else:
                self.storage.mkdir(parents=True, exist_ok=True)
                array = np.lib.format.open_memmap(self.storage / f"{name}.npy", mode="w+",
                                                  dtype=dtype, shape=shape)
            setattr(self, name, array)

    @property
    def nbytes(self) -> int:
        return sum(getattr(self, k).nbytes for k in ("rows", "count", "tag", "err"))

    def run(self, stream: np.ndarray, keep_probs: bool = False):
        """Consume stream (uint8); return per-byte NLL of stream[1:] and optional read distributions."""
        stream = np.ascontiguousarray(stream, dtype=np.uint8)
        nll = np.empty(len(stream) - 1, dtype=np.float64)
        probs = np.empty((len(stream) - 1 if keep_probs else 0, 256), dtype=np.float32)
        self.history = _run(stream, self.rows, self.count, self.tag, self.err,
                            np.asarray(self.orders, dtype=np.int64), self.heads,
                            1 << self.slots_log2, self.gain_floor, self.err_floor,
                            self.prior_weight, np.uint64(self.history), nll, probs, keep_probs)
        return nll, (probs if keep_probs else None)
