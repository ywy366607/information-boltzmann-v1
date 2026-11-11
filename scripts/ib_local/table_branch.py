"""Exact reversible table branches for synchronous imagination/evaluation.

Only touched rows are journaled. The live learner must not execute concurrently
inside a branch; finally restores every shared row even if the branch raises.
History/event scalars belong to the branch, so they never change on the parent.
"""
from contextlib import contextmanager

import numba
import numpy as np

from scripts.ib_local.ngram_fast_table import MASK64, _mix
from scripts.ib_local.retrieved_table_credit import RetrievedCreditTable

FIELDS = ('rows', 'count', 'tag', 'err', 'epoch', 'last_write')


@numba.njit(cache=True)
def touched_slots(stream, history, orders, heads, slots):
    result = np.empty((len(stream)-1)*len(orders)*heads, np.int64)
    at = 0
    for byte in stream[:-1]:
        history = ((history << np.uint64(8)) | np.uint64(byte)) & MASK64
        for o in range(len(orders)):
            n = orders[o]
            key = history & ((np.uint64(1) << np.uint64(8*n))-np.uint64(1)) if n < 8 else history
            for h in range(heads):
                code = _mix(key ^ _mix(np.uint64(o*131+h*7919+1)))
                result[at] = (o*heads+h)*slots + np.int64(code & np.uint64(slots-1))
                at += 1
    return result


class JournaledTable(RetrievedCreditTable):
    def __init__(self, parent):
        self.__dict__.update(parent.__dict__)
        self._saved_mask = np.zeros(len(self.count), dtype=np.bool_)
        self._saved_blocks = []
        self._closed = False

    def run_credit(self, stream, max_window=512):
        if self._closed:
            raise RuntimeError('table branch is closed')
        stream = np.ascontiguousarray(stream, dtype=np.uint8)
        if not 0 < len(stream)-1 <= max_window:
            raise ValueError('branch window must be bounded')
        slots = np.unique(touched_slots(stream, np.uint64(self.history),
            np.asarray(self.orders, np.int64), self.heads, 1 << self.slots_log2))
        fresh = slots[~self._saved_mask[slots]]
        if fresh.size:
            self._saved_blocks.append((fresh, tuple(getattr(self, name)[fresh].copy() for name in FIELDS)))
            self._saved_mask[fresh] = True
        return super().run_credit(stream, max_window)

    def settle(self, *args, **kwargs):
        # Occupancy reinvestment can touch an entire pool outside the rows
        # journaled by assimilation. These branches support run/read only.
        raise RuntimeError('table branch does not journal occupancy reinvestment')

    @property
    def journal_bytes(self):
        return self._saved_mask.nbytes + sum(index.nbytes + sum(a.nbytes for a in arrays)
                                            for index, arrays in self._saved_blocks)

    def rollback(self):
        if not self._closed:
            for slots, arrays in self._saved_blocks:
                for name, array in zip(FIELDS, arrays):
                    getattr(self, name)[slots] = array
            self._saved_blocks.clear()
            self._closed = True


@contextmanager
def table_branch(parent):
    branch = JournaledTable(parent)
    try:
        yield branch
    finally:
        branch.rollback()
