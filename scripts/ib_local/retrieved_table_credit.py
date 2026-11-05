"""Optional retrieved-count credit on the existing byte table.

Frozen-read eligibility is d log(p_table) / d count at each actual retrieval.
Later GDN loss returns a factor through its table input; no past GDN graph is kept.
A bounded, compartment-local reinvestment changes the SAME count already used
for read precision, assimilation and eviction. This is a conditional local
learning rule, not exact historical-write credit or a complete organism proof.
"""
from dataclasses import dataclass
import math
import numpy as np
import numba
from scripts.ib_local.ngram_fast_table import ByteNgramFastTable, MASK64, _mix

MAX_OCCUPANCY = float(2**24)

@numba.njit(cache=True)
def _run_credit(stream, rows, count, tag, err, orders, heads, slots, gain_floor, err_floor,
         prior_weight, history, nll, out_probs, keep_probs, epoch, last_write, events,
                packet_slot, packet_epoch, packet_age, eligibility):
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
        # Derivative of log mixture with respect to an existing count, BEFORE
        # the target writes this slot. Snapshot is bounded by one training window.
        for o in range(n_orders):
            for h in range(heads):
                slot = owner[o, h]
                jh = o * heads + h
                if tag[slot] == sig[o, h] and count[slot] > 0:
                    c = count[slot]
                    packet_slot[t, jh] = slot
                    packet_epoch[t, jh] = epoch[slot]
                    packet_age[t, jh] = events + t + 1 - last_write[slot]
                    dw = 1.0 / ((c + 1.0)**2 * (err[slot] + err_floor))
                    for j in range(vocab):
                        mixed = p[j] / total
                        eligibility[t, jh, j] = dw / total * (rows[slot, j] / mixed - 1.0)
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
                        count[slot] = max(0.0, count[slot] - 1.0)
                        continue
                    epoch[slot] += np.uint64(1)
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
                count[slot] = min(c + 1, MAX_OCCUPANCY)
                last_write[slot] = events + t + 1
    return history


@dataclass
class ReadCredit:
    slots: np.ndarray
    epochs: np.ndarray
    ages: np.ndarray
    eligibility: np.ndarray

    @property
    def nbytes(self):
        return sum(x.nbytes for x in (self.slots, self.epochs, self.ages, self.eligibility))


class RetrievedCreditTable(ByteNgramFastTable):
    """Fixed-size table plus owner incarnation and last-write time (16 B/slot).

    Count is now a real-valued occupancy, not an integer visit statistic.
    With credit disabled it matches original counts until the explicit 2**24 cap.
    Epoch prevents a late receipt from paying a replacement with the same tag.
    Last-write time is diagnostic only, never used to select what is rewarded.
    """
    def __post_init__(self):
        if self.storage is not None:
            raise ValueError('credit prototype uses bounded RAM; memmap restore not implemented')
        if not (0 < self.gain_floor < 1 and self.err_floor > 0 and self.prior_weight > 0):
            raise ValueError('positive read denominator and gain in (0,1) required')
        super().__post_init__()
        self.count = self.count.astype(np.float64)
        self.epoch = np.zeros(self.count.shape, dtype=np.uint64)
        self.last_write = np.zeros(self.count.shape, dtype=np.uint64)
        self.events = 0

    @property
    def nbytes(self):
        return super().nbytes + self.epoch.nbytes + self.last_write.nbytes

    def run(self, stream, keep_probs=False):
        """Ordinary assimilation must also advance owner/time metadata."""
        nll, probs, _ = self.run_credit(stream)
        return nll, probs if keep_probs else None

    def run_credit(self, stream, max_window=512):
        stream = np.ascontiguousarray(stream, dtype=np.uint8)
        n = len(stream)-1
        if not 0 < n <= max_window:
            raise ValueError('receipt must fit one declared bounded window')
        if self.events+n >= np.iinfo(np.uint64).max:
            raise OverflowError('event counter exhausted; cannot silently alias receipts')
        h = len(self.orders)*self.heads
        packet = ReadCredit(np.full((n,h), -1, np.int64),
                            np.zeros((n,h), np.uint64), np.zeros((n,h), np.uint64),
                            np.zeros((n,h,256), np.float32))
        nll, probs = np.empty(n), np.empty((n,256), np.float32)
        self.history = _run_credit(stream, self.rows, self.count, self.tag, self.err,
            np.asarray(self.orders,np.int64), self.heads, 1<<self.slots_log2,
            self.gain_floor, self.err_floor, self.prior_weight, np.uint64(self.history),
            nll, probs, True, self.epoch, self.last_write, self.events,
            packet.slots, packet.epochs, packet.ages, packet.eligibility)
        self.events += n
        return nll, probs, packet

    def settle(self, packet, dloss_dlogp, mode='aligned', rng=None, rate=None):
        """Settle AFTER neural backward, before the next table window.

        dloss_dlogp is for SUM loss, not mean loss. Clipping the exponent is
        numerical stabilization, not a proved optimal learning rate.
        All modes retain identical state/receipt storage. Shuffle permutes
        aggregated signals among valid addresses WITHIN each order/hash pool.
        """
        if mode not in ('aligned','off','shuffled'):
            raise ValueError(mode)
        rate = self.gain_floor if rate is None else rate
        if not 0 <= rate < 1:
            raise ValueError('exchange rate must be in [0,1)')
        g = np.asarray(dloss_dlogp,dtype=np.float64)
        if g.shape != (packet.slots.shape[0],256) or not np.isfinite(g).all():
            raise ValueError('invalid third factor')
        local = np.einsum('thv,tv->th',packet.eligibility,g)
        slots = packet.slots.ravel()
        valid = slots >= 0
        safe = np.maximum(slots,0)
        valid &= packet.epochs.ravel() == self.epoch[safe]
        valid &= self.count[safe] > 0
        ids, inverse = np.unique(slots[valid], return_inverse=True)
        derivative = np.bincount(inverse, weights=local.ravel()[valid], minlength=len(ids))
        gross = 0.0
        max_error = 0.0
        bounded_pools = 0
        for group in np.unique(ids // (1<<self.slots_log2)):
            mask = ids // (1<<self.slots_log2) == group
            index = ids[mask]
            c = self.count[index].copy()
            score = -derivative[mask]
            if mode == 'shuffled':
                if rng is None: raise ValueError('shuffle requires explicit RNG')
                score = rng.permutation(score)
            if mode == 'off' or rate == 0: continue
            w = np.exp(np.clip(score,-20,20))
            if np.ptp(w) == 0: continue
            proposed = c*w
            delta = c.sum()*proposed/proposed.sum() - c
            # Uniformly shorten this pool's exchange at the numerical boundary.
            # Clipping coordinates separately would destroy conservation.
            rising = delta > 0
            step = rate
            if np.any(rising):
                step = min(rate, float(np.min((MAX_OCCUPANCY-c[rising])/delta[rising])))
            bounded_pools += int(step < rate)
            updated = c + step*delta
            # Same sum in real arithmetic; count changes future write and eviction.
            self.count[index] = updated
            gross += float(np.abs(updated-c).sum()/2)
            max_error = max(max_error,abs(float(updated.sum()-c.sum())))
        return dict(eligible_reads=int(valid.sum()), eligible_slots=len(ids),
                    reads_older_than_window=int((valid & (packet.ages.ravel()>packet.slots.shape[0])).sum()),
                    derivative_abs_sum=float(np.abs(derivative).sum()),
                    gross_count_exchange=gross, pool_total_error=max_error,
                    boundary_limited_pools=bounded_pools)

    def state_dict(self):
        return dict(arrays={k:getattr(self,k).copy() for k in
                    ('rows','count','tag','err','epoch','last_write')},
                    history=int(self.history), events=self.events,
                    config=dict(orders=self.orders,heads=self.heads,slots_log2=self.slots_log2,
                                gain_floor=self.gain_floor,err_floor=self.err_floor,
                                prior_weight=self.prior_weight))

    @classmethod
    def from_state_dict(cls, state):
        obj=cls(**state['config'])
        for k,v in state['arrays'].items():
            if getattr(obj,k).shape != v.shape: raise ValueError(k)
            getattr(obj,k)[:] = v
        obj.history=np.uint64(state['history']); obj.events=state['events']
        return obj
