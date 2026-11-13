"""Blocked NumPy companion to the dense eligibility reference.

U/V are updated IN PLACE, only after selection against the old basis. Source
and readout providers must be held stable throughout a call. No D-sized FP64
copy, D-by-P oracle, or state Jacobian is constructed. Scalar field ledgers
and resident U/V are still proportional to their declared fixed layout.
All numerical error/margin claims remain conditional or empirical.
"""
from dataclasses import dataclass
import math

import numpy as np

from scripts.ib_local.bounded_causal_tag_reference import ErrorTerm, NumericalBudget


@dataclass
class ArraySource:
    values: np.ndarray
    token: object = None               # caller-held time/version, not a history tape

    def __call__(self, start, stop):
        return self.values[start:stop]


@dataclass
class ScanStats:
    source_calls: int = 0
    source_bytes: int = 0
    basis_bytes: int = 0
    max_block: int = 0

    def read(self, source, lo, hi):
        value = np.asarray(source(lo, hi))
        if value.shape != (hi - lo,):
            raise ValueError('source provider returned the wrong block')
        self.source_calls += 1
        self.source_bytes += value.nbytes
        self.max_block = max(self.max_block, hi - lo)
        return value.astype(np.float64)


def blocks(layout, block_size):
    if block_size <= 0:
        raise ValueError('block size must be positive')
    for f, name in enumerate(layout.fields):
        sl = layout.slices[name]
        for lo in range(sl.start, sl.stop, block_size):
            yield f, lo, min(lo + block_size, sl.stop)


def workspace_budget(rank, owners, fields, block_size):
    """Explicit conservative array budget, not peak RSS/BLAS/provider memory."""
    return dict(block_vector_bytes=12 * block_size * 8,
                owner_row_bytes=6 * owners * 8,
                rank_scalar_bytes=8 * rank * 8,
                field_scalar_bytes=(2 * rank + 12) * fields * 8,
                excludes='resident U/V, provider-owned state, Python metadata and library internals')


def _residual(code, source, a, lo, hi, stop, stats):
    b = stats.read(source, lo, hi)
    r = b.copy()
    for j in range(stop):
        u = code.U[j, lo:hi].astype(np.float64)
        stats.basis_bytes += code.U[j, lo:hi].nbytes
        r -= a[j] * u
    return b, r


def insert_source_inplace(code, source, owner, layout, block_size=4096):
    """Same one-sweep/K+1 rule; reduction order may expose near-tie differences.

    The provider must not depend on U being overwritten during the final pass.
    A changed explicit provider token is rejected BEFORE any basis write.
    """
    rank, dim = code.U.shape
    if dim != layout.size:
        raise ValueError('layout mismatch')
    owner = np.array(owner, dtype=np.float64, copy=True)
    if owner.shape != (code.V.shape[1],):
        raise ValueError('owner shape mismatch')
    token = getattr(source, 'token', None)
    stats, a = ScanStats(), np.zeros(rank)
    for j in range(rank):
        dot, square = 0., 0.
        for _, lo, hi in blocks(layout, block_size):
            _, r = _residual(code, source, a, lo, hi, j, stats)
            u = code.U[j, lo:hi].astype(np.float64)
            stats.basis_bytes += code.U[j, lo:hi].nbytes
            dot += float(u @ r)
            square += float(u @ u)
        a[j] = dot / square if square else 0.

    count = len(layout.fields)
    u2, r2, split2 = np.zeros((rank, count)), np.zeros(count), np.zeros(count)
    for f, lo, hi in blocks(layout, block_size):
        b, r = _residual(code, source, a, lo, hi, rank, stats)
        expansion = np.zeros(hi - lo)
        for j in range(rank):
            u = code.U[j, lo:hi].astype(np.float64)
            stats.basis_bytes += code.U[j, lo:hi].nbytes
            u2[j, f] += float(u @ u)
            expansion += a[j] * u
        defect = b - (expansion + r)
        r2[f] += float(r @ r)
        split2[f] += float(defect @ defect)
    umax = np.zeros(rank)
    for j in range(rank):
        merged = code.V[j].astype(np.float64) + a[j] * owner
        umax[j] = np.max(np.abs(merged))
    owner_max = float(np.max(np.abs(owner)))
    costs = np.concatenate(([owner_max * math.sqrt(float(r2.sum()))],
                            umax * np.sqrt(u2.sum(1))))
    winner = 0
    for j in range(rank):
        if costs[j + 1] < costs[winner]:
            winner = j + 1
    drop = None if winner == 0 else winner - 1
    if getattr(source, 'token', None) != token:
        raise ValueError('source time/version changed during insertion')
    field_costs = owner_max * np.sqrt(r2) if drop is None else umax[drop] * np.sqrt(u2[drop])
    stored2 = u2.copy()
    rounding2 = np.zeros(count)
    if drop is not None:
        stored2[drop] = 0
        # Earlier written blocks do not affect the still-old current block.
        for f, lo, hi in blocks(layout, block_size):
            _, r = _residual(code, source, a, lo, hi, rank, stats)
            stored = r.astype(code.U.dtype)
            delta = r - stored.astype(np.float64)
            rounding2[f] += float(delta @ delta)
            stored64 = stored.astype(np.float64)
            stored2[drop, f] += float(stored64 @ stored64)
            code.U[drop, lo:hi] = stored
    storage_fields = np.zeros(count)
    storage = 0.
    for j in range(rank):
        merged = owner if j == drop else code.V[j].astype(np.float64) + a[j] * owner
        stored = merged.astype(code.V.dtype)
        dmax = float(np.max(np.abs(merged - stored.astype(np.float64))))
        if j == drop:
            weight = float(np.max(np.abs(merged)))
            storage += weight * math.sqrt(float(rounding2.sum()))
            storage_fields += weight * np.sqrt(rounding2)
        storage += dmax * math.sqrt(float(stored2[j].sum()))
        storage_fields += dmax * np.sqrt(stored2[j])
        code.V[j] = stored
    projection_error = owner_max * math.sqrt(float(split2.sum()))
    code.bills += [costs[winner], storage, projection_error]
    code.fields += field_costs + storage_fields + owner_max * np.sqrt(split2)
    return dict(drop=drop, cost=float(costs[winner]), candidate_costs=costs,
                coefficients=a, field_costs=field_costs,
                representation_error=storage, projection_residual=projection_error,
                source_passes=rank + 1 + int(drop is not None), stats=vars(stats),
                workspace=workspace_budget(rank, len(owner), count, block_size),
                source_token=token)


@dataclass(frozen=True)
class FieldLeg:
    field: str
    coefficient: float = 1.
    offset: int = 0                   # coordinate offset within source field


def field_row_gain(layout, rows):
    """||A||₂ <= sqrt(||G||₁ ||G||∞), from sparse field-norm rows."""
    column = np.ones(len(layout.fields))
    for target in rows:
        column[layout.indices[target]] -= 1.
    maximum_row = 1. if len(rows) < len(layout.fields) else 0.
    for legs in rows.values():
        maximum_row = max(maximum_row, sum(abs(leg.coefficient) for leg in legs))
        for leg in legs:
            column[layout.indices[leg.field]] += abs(leg.coefficient)
    return math.sqrt(maximum_row * float(column.max()))


def assign_fields_inplace(code, layout, rows, block_size=4096):
    """Sparse copy/mean maps; source fields must not be other destinations.

    Self-mean of a field at offset zero is safe. A coupled native step must
    first hold its small local inputs, not use this copy-only routine.
    """
    for target, legs in rows.items():
        target_slice = layout.slices[target]
        length = target_slice.stop - target_slice.start
        for leg in legs:
            source_slice = layout.slices[leg.field]
            if leg.offset < 0 or leg.offset + length > source_slice.stop - source_slice.start:
                raise ValueError('field projection lies outside source')
            if leg.field in rows and (leg.field != target or leg.offset != 0):
                raise ValueError('cross-target source needs a held local input')
    gain = field_row_gain(layout, rows)
    old_errors = {target: sum(abs(leg.coefficient) * code.fields[layout.indices[leg.field]]
                              for leg in legs) for target, legs in rows.items()}
    weights = np.array([np.max(np.abs(row)) for row in code.V])
    code.bills *= gain
    traversed = 0
    for target, legs in rows.items():
        sl = layout.slices[target]
        round2 = np.zeros(len(code.U))
        for lo in range(sl.start, sl.stop, block_size):
            hi = min(lo + block_size, sl.stop)
            offset = lo - sl.start
            for j in range(len(code.U)):
                value = np.zeros(hi - lo)
                for leg in legs:
                    src = layout.slices[leg.field].start + leg.offset + offset
                    u = code.U[j, src:src + hi - lo].astype(np.float64)
                    traversed += code.U[j, src:src + hi - lo].nbytes
                    value += leg.coefficient * u
                stored = value.astype(code.U.dtype)
                diff = value - stored.astype(np.float64)
                round2[j] += float(diff @ diff)
                code.U[j, lo:hi] = stored
        error = float(weights @ np.sqrt(round2))
        code.bills[1] += error
        code.fields[layout.indices[target]] = old_errors[target] + error
    return dict(gain=gain, touched_fields=len(rows), basis_bytes=traversed,
                max_block=block_size, field_metadata_scalars=len(layout.fields))


def begin_capture_inplace(code, records, layout, port, event, block_size=4096):
    if records.opened[port]:
        raise ValueError('capture already open')
    rows = {f'pending.{port}.{name}': [FieldLeg(f'native.{layout.ports[port]}.{name}')]
            for name in layout.cache_shapes}
    bill = assign_fields_inplace(code, layout, rows, block_size)
    records.opened[port], records.opening_events[port] = True, event
    records.captured[port] = False
    return bill


def capture_activation_inplace(code, records, layout, port, *, key=False,
                               source_index=None, block_size=4096):
    if not records.opened[port]:
        raise ValueError('no opening capture')
    index = layout.tokens - 1 if source_index is None else source_index
    target, offset = (f'pending.{port}.key', index * layout.width) if key else (f'pending.{port}.inputs', 0)
    bill = assign_fields_inplace(code, layout, {target: [FieldLeg('activation', offset=offset)]}, block_size)
    records.captured[port, 0 if key else 1] = True
    return bill


def commit_record_inplace(code, records, layout, port, slot, *, replace=False, block_size=4096):
    if not records.opened[port] or not records.captured[port].all():
        raise ValueError('incomplete opening/key/input capture')
    n = 0 if replace else int(records.counts[port, slot])
    rows = {f'record.{port}.{slot}.{name}': [FieldLeg(f'pending.{port}.{name}')]
            for name in layout.record_shapes if name != 'key'}
    rows[f'record.{port}.{slot}.key'] = [FieldLeg(f'record.{port}.{slot}.key', n / (n + 1)),
                                      FieldLeg(f'pending.{port}.key', 1 / (n + 1))]
    bill = assign_fields_inplace(code, layout, rows, block_size)
    records.counts[port, slot] = n + 1
    records.versions[port, slot] += 1
    records.opened[port] = False
    return bill


def capture_credit_blocked(code, readout, layout, budget=NumericalBudget(), *,
                           recent=None, same_reference=False, block_size=4096):
    stats = ScanStats()
    contraction = np.zeros(len(code.U))
    u2 = np.zeros(len(code.U))
    m2 = np.zeros(len(layout.fields))
    for f, lo, hi in blocks(layout, block_size):
        m = stats.read(readout, lo, hi)
        m2[f] += float(m @ m)
        for j in range(len(code.U)):
            u = code.U[j, lo:hi].astype(np.float64)
            stats.basis_bytes += code.U[j, lo:hi].nbytes
            contraction[j] += float(u @ m)
            u2[j] += float(u @ u)
    estimate, norm_bound = np.zeros(code.V.shape[1]), np.zeros(code.V.shape[1])
    for j in range(len(code.U)):
        row = code.V[j].astype(np.float64)
        estimate += contraction[j] * row
        norm_bound += math.sqrt(float(u2[j])) * np.abs(row)
    if recent is not None:
        estimate -= np.asarray(recent)
    recent_error = ErrorTerm(0.) if recent is None or same_reference else budget.recent_mismatch
    terms = (budget.state, budget.modulation, budget.contraction, recent_error)
    available = all(t.value is not None and t.provenance != 'unavailable' for t in terms)
    if available:
        m_norm = math.sqrt(float(m2.sum()))
        state_error = min(m_norm * code.radius, float(np.sqrt(m2) @ code.fields))
        error = (state_error + m_norm * budget.state.value + budget.modulation.value * norm_bound +
                 budget.contraction.value + recent_error.value)
        margin = np.abs(estimate) - error
    else:
        error, margin = np.full_like(estimate, np.nan), np.full_like(estimate, np.nan)
    mask = margin > 0
    return dict(estimate=estimate, error=error, margin=margin, margin_pass=mask,
                conditional_credit=np.where(mask, estimate, 0.),
                certified_credit=np.zeros_like(estimate), certified=False,
                scope='empirical' if any(t.provenance == 'empirical' for t in terms) else 'conditional',
                unavailable=not available, stats=vars(stats),
                workspace=workspace_budget(len(code.U), code.V.shape[1], len(layout.fields), block_size))
