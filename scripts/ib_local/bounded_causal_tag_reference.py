"""NumPy-only reference for fixed-budget eligibility; not installed in the body.

The real-algebra bounds are conditional. Float residuals and dtype-conversion
bills below are numerical diagnostics, not certified IEEE error enclosures.
No dense state-by-parameter eligibility or state Jacobian is retained here.
"""
from dataclasses import dataclass, field
import math

import numpy as np


@dataclass
class Layout:
    layers: int = 24
    width: int = 1024
    heads: int = 4
    head_dim: int = 256
    conv: int = 4
    tokens: int = 65
    activation_tokens: int = 65
    ports: tuple = (1, 12)
    capacity: int = 32                 # per port

    def __post_init__(self):
        self.fields, self.shapes, self.slices = [], {}, {}
        cursor = 0

        def add(name, shape):
            nonlocal cursor
            size = math.prod(shape)
            self.fields.append(name)
            self.shapes[name] = shape
            self.slices[name] = slice(cursor, cursor + size)
            cursor += size

        add('activation', (self.activation_tokens, self.width))
        for layer in range(self.layers):
            for name, shape in self.cache_shapes.items():
                add(f'native.{layer}.{name}', shape)
        for port in range(len(self.ports)):
            for slot in range(self.capacity):
                for name, shape in self.record_shapes.items():
                    add(f'record.{port}.{slot}.{name}', shape)
            for name, shape in self.record_shapes.items():
                add(f'pending.{port}.{name}', shape)
        self.size = cursor
        self.indices = {name: i for i, name in enumerate(self.fields)}

    @property
    def cache_shapes(self):
        return dict(s=(self.heads, self.head_dim, self.head_dim),
                    q=(self.width, self.conv), k=(self.width, self.conv),
                    v=(self.width, self.conv))

    @property
    def record_shapes(self):
        return dict(key=(self.width,), inputs=(self.tokens, self.width), **self.cache_shapes)

    def pack(self, values):
        result = np.empty(self.size, dtype=np.float64)
        for name in self.fields:
            value = np.asarray(values[name])
            if value.shape != self.shapes[name]:
                raise ValueError(f'{name}: expected {self.shapes[name]}, got {value.shape}')
            result[self.slices[name]] = value.reshape(-1)
        return result

    def unpack(self, vector):
        vector = np.asarray(vector)
        if vector.shape != (self.size,):
            raise ValueError('coordinate size mismatch')
        return {name: vector[self.slices[name]].reshape(self.shapes[name]) for name in self.fields}

    def budget(self, rank, owners, dtype=np.float32):
        item = np.dtype(dtype).itemsize
        # These are arrays, not total process RAM or device feasibility.
        return dict(coordinates=self.size, effect_bytes=rank * self.size * item,
                    owner_bytes=rank * owners * item,
                    ledger_bytes=(3 + len(self.fields)) * 8,
                    metadata_bytes=len(self.ports) * (self.capacity * 16 + 11),
                    source_payload_bytes=(self.size + owners + rank) * 8,
                    insertion_enumerated_workspace_bytes=(2 * rank * self.size + 2 * self.size +
                                                          3 * rank * owners + owners + rank) * 8,
                    scope='shape-only reference arrays; not peak RSS; excludes other temporaries/model/allocator')


@dataclass
class Code:
    U: np.ndarray                    # K x D, all fields share these K columns
    V: np.ndarray                    # K x P, fixed original parameter coordinates
    fields: np.ndarray               # aggregate bound for each field restriction
    bills: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # bills: compression, stored-factor conversion, local numerical residual

    @classmethod
    def empty(cls, rank, owners, layout, dtype=np.float64):
        return cls(np.zeros((rank, layout.size), dtype=dtype),
                   np.zeros((rank, owners), dtype=dtype), np.zeros(len(layout.fields)))

    @property
    def radius(self):
        return float(self.bills.sum())

    @property
    def nbytes(self):
        return self.U.nbytes + self.V.nbytes + self.fields.nbytes + self.bills.nbytes

    def copy(self):
        return Code(self.U.copy(), self.V.copy(), self.fields.copy(), self.bills.copy())


@dataclass
class InsertBill:
    drop: int | None                 # None discards the new residual
    cost: float
    field_costs: np.ndarray
    residual: np.ndarray             # transient, not retained by Code
    coefficients: np.ndarray
    discarded_effect: np.ndarray
    discarded_owner: np.ndarray
    representation_error: float
    projection_residual: float
    workspace_bytes: int


def project_source(U, source):
    U, residual = np.asarray(U, dtype=np.float64), np.array(source, dtype=np.float64, copy=True)
    coefficients = np.zeros(len(U))
    for j, u in enumerate(U):
        denominator = float(u @ u)
        c = float(u @ residual) / denominator if denominator != 0 else 0.
        coefficients[j] = c
        residual -= c * u
    return coefficients, residual


def _field_norms(layout, vector):
    return np.array([np.linalg.norm(vector[layout.slices[n]]) for n in layout.fields])


def _rounding_bill(layout, U, V, storedU, storedV):
    """Real factor identity evaluated in FP64; not a rigorous rounding bound."""
    du = U - storedU.astype(np.float64)
    dv = V - storedV.astype(np.float64)
    weights = np.max(np.abs(V), axis=1)
    dweights = np.max(np.abs(dv), axis=1)
    total = float(weights @ np.linalg.norm(du, axis=1) +
                  dweights @ np.linalg.norm(storedU.astype(np.float64), axis=1))
    fields = np.zeros(len(layout.fields))
    for i, name in enumerate(layout.fields):
        sl = layout.slices[name]
        fields[i] = (weights @ np.linalg.norm(du[:, sl], axis=1) +
                     dweights @ np.linalg.norm(storedU[:, sl].astype(np.float64), axis=1))
    return total, fields


def insert_source(code, source, owner, layout):
    U, V = code.U.astype(np.float64), code.V.astype(np.float64)
    source, owner = np.asarray(source, dtype=np.float64), np.asarray(owner, dtype=np.float64)
    if source.shape != (U.shape[1],) or owner.shape != (V.shape[1],):
        raise ValueError('source/owner shape mismatch')
    a, r = project_source(U, source)
    W = V + a[:, None] * owner[None, :]
    drop = None
    cost = float(np.max(np.abs(owner)) * np.linalg.norm(r))
    for j in range(len(U)):
        candidate = float(np.max(np.abs(W[j])) * np.linalg.norm(U[j]))
        if candidate < cost:          # strict scan; ties preserve current choice
            drop, cost = j, candidate
    discarded_effect = (r if drop is None else U[drop]).copy()
    discarded_owner = (owner if drop is None else W[drop]).copy()
    field_costs = np.max(np.abs(discarded_owner)) * _field_norms(layout, discarded_effect)
    newU, newV = U.copy(), W.copy()
    if drop is not None:
        newU[drop], newV[drop] = r, owner
    out = code.copy()
    out.U, out.V = newU.astype(code.U.dtype), newV.astype(code.V.dtype)
    storage, storage_fields = _rounding_bill(layout, newU, newV, out.U, out.V)
    split_error = source - (a @ U + r)
    numerical = float(np.max(np.abs(owner)) * np.linalg.norm(split_error))
    out.bills += [cost, storage, numerical]
    out.fields += field_costs + storage_fields + np.max(np.abs(owner)) * _field_norms(layout, split_error)
    workspace = sum(x.nbytes for x in (U, V, source, owner, a, r, W, newU, newV))
    return out, InsertBill(drop, cost, field_costs, r, a, discarded_effect, discarded_owner,
                           storage, numerical, workspace)


def transport(code, apply_linear, gain, layout, arithmetic_error=0., field_gains=None):
    """Caller supplies the source-free operator and its conditional gain.

    arithmetic_error is a separately supplied local bound/measurement, not
    inferred from the final uncompressed-oracle credit error.
    """
    if gain < 0 or arithmetic_error < 0:
        raise ValueError('negative error or gain')
    U = np.stack([apply_linear(u.astype(np.float64)) for u in code.U]) if len(code.U) else code.U.astype(np.float64)
    if U.shape != code.U.shape:
        raise ValueError('transport must preserve the declared coordinate space')
    out = code.copy()
    out.U = U.astype(code.U.dtype)
    storage, fields = _rounding_bill(layout, U, code.V.astype(np.float64), out.U, out.V)
    out.bills *= gain
    out.bills += [0., storage, arithmetic_error]
    if field_gains is None:
        out.fields[:] = gain * code.radius
    else:
        # Explicit caller-supplied nonnegative field-to-field gain matrix.
        field_gains = np.asarray(field_gains)
        if field_gains.shape != (len(layout.fields), len(layout.fields)) or np.any(field_gains < 0):
            raise ValueError('invalid field gains')
        out.fields = field_gains @ code.fields
    out.fields += fields + arithmetic_error
    return out


@dataclass
class Records:
    counts: np.ndarray
    versions: np.ndarray
    opened: np.ndarray
    opening_events: np.ndarray
    captured: np.ndarray              # pre-inlet key and revised inputs

    @classmethod
    def empty(cls, layout):
        shape = (len(layout.ports), layout.capacity)
        return cls(np.zeros(shape, dtype=np.int64), np.zeros(shape, dtype=np.int64),
                   np.zeros(len(layout.ports), dtype=bool),
                   np.full(len(layout.ports), -1, dtype=np.int64),
                   np.zeros((len(layout.ports), 2), dtype=bool))

    @property
    def nbytes(self):
        return sum(a.nbytes for a in (self.counts, self.versions, self.opened, self.opening_events, self.captured))

    def receipt(self, port, slot):
        return port, slot, int(self.versions[port, slot])

    def validate(self, receipt):
        port, slot, version = receipt
        if version != int(self.versions[port, slot]) or self.counts[port, slot] == 0:
            raise ValueError('stale or empty record receipt')


def _assign_fields(code, layout, assignments):
    """Each destination copies/averages named fields from the same old basis.

    Current callers overwrite disjoint destinations from preserved sources;
    gain 2 is a conservative real bound for these particular copy/mean maps.
    """
    out = code.copy()
    out.bills *= 2.
    for target, sources in assignments.items():
        sl = layout.slices[target]
        value = sum(weight * code.U[:, layout.slices[source]].astype(np.float64)
                    for source, weight in sources)
        out.U[:, sl] = value.astype(code.U.dtype)
        error = sum(abs(weight) * code.fields[layout.indices[source]] for source, weight in sources)
        rounding = float(np.max(np.abs(code.V), axis=1) @
                         np.linalg.norm(value - out.U[:, sl].astype(np.float64), axis=1))
        out.fields[layout.indices[target]] = error + rounding
        out.bills[1] += rounding
    return out


def begin_capture(code, records, layout, port, event):
    if records.opened[port]:
        raise ValueError('capture already open')
    assignments = {f'pending.{port}.{name}': [(f'native.{layout.ports[port]}.{name}', 1.)]
                   for name in layout.cache_shapes}
    out = _assign_fields(code, layout, assignments)
    records.opened[port], records.opening_events[port] = True, event
    records.captured[port] = False
    return out


def capture_activation(code, records, layout, port, *, key=False, source_index=None):
    """Call key capture at the pre-inlet state, inputs at the revised state.

    Both live in U, so later insertion/recompression also transforms pending.
    The opening boundary is untouched by either operation.
    """
    if not records.opened[port]:
        raise ValueError('no opening capture')
    out = code.copy()
    out.bills *= 2.
    activation = code.U[:, layout.slices['activation']].reshape(len(code.U), layout.activation_tokens, layout.width)
    if key:
        index = layout.tokens - 1 if source_index is None else source_index
        value, target = activation[:, index], f'pending.{port}.key'
    else:
        value, target = activation[:, :layout.tokens].reshape(len(code.U), -1), f'pending.{port}.inputs'
    out.U[:, layout.slices[target]] = value
    out.fields[layout.indices[target]] = code.fields[layout.indices['activation']]
    records.captured[port, 0 if key else 1] = True
    return out


def commit_record(code, records, layout, port, slot, *, replace=False):
    if not records.opened[port] or not records.captured[port].all():
        raise ValueError('incomplete opening/key/input capture')
    n = 0 if replace else int(records.counts[port, slot])
    assignments = {f'record.{port}.{slot}.{name}': [(f'pending.{port}.{name}', 1.)]
                   for name in layout.record_shapes if name != 'key'}
    assignments[f'record.{port}.{slot}.key'] = [(f'record.{port}.{slot}.key', n / (n + 1)),
                                             (f'pending.{port}.key', 1 / (n + 1))]
    out = _assign_fields(code, layout, assignments)
    records.counts[port, slot] = n + 1
    records.versions[port, slot] += 1
    records.opened[port] = False
    return out


@dataclass(frozen=True)
class ErrorTerm:
    value: float | None
    provenance: str = 'conditional'    # conditional, empirical, unavailable

    def __post_init__(self):
        if self.provenance not in ('conditional', 'empirical', 'unavailable'):
            raise ValueError('unknown error provenance')
        if self.value is not None and (not np.isfinite(self.value) or self.value < 0):
            raise ValueError('invalid error value')
        if self.value is None and self.provenance != 'unavailable':
            raise ValueError('missing error must be unavailable')


@dataclass(frozen=True)
class NumericalBudget:
    state: ErrorTerm = ErrorTerm(0.)
    modulation: ErrorTerm = ErrorTerm(0.)
    contraction: ErrorTerm = ErrorTerm(None, 'unavailable')
    recent_mismatch: ErrorTerm = ErrorTerm(None, 'unavailable')


def capture_credit(code, readout, budget=NumericalBudget(), recent=None, *, same_reference=False, layout=None):
    """Margins are conditional/empirical, never certified machine-float signs.

    recent is the SAME registered-source/window credit. Any mismatch requires
    its own budget; the caller must not subtract an unrelated total gradient.
    """
    readout = np.asarray(readout, dtype=np.float64)
    estimate = (code.U.astype(np.float64) @ readout) @ code.V.astype(np.float64)
    if recent is not None:
        estimate = estimate - np.asarray(recent)
    recent_error = ErrorTerm(0.) if recent is None or same_reference else budget.recent_mismatch
    terms = (budget.state, budget.modulation, budget.contraction, recent_error)
    available = all(t.value is not None and t.provenance != 'unavailable' for t in terms)
    norm_bound = np.linalg.norm(code.U.astype(np.float64), axis=1) @ np.abs(code.V.astype(np.float64))
    if available:
        state_error = np.linalg.norm(readout) * code.radius
        if layout is not None:
            state_error = min(state_error, float(_field_norms(layout, readout) @ code.fields))
        error = (state_error + np.linalg.norm(readout) * budget.state.value +
                 budget.modulation.value * norm_bound + budget.contraction.value +
                 recent_error.value)
        margin = np.abs(estimate) - error
    else:
        error, margin = np.full_like(estimate, np.nan), np.full_like(estimate, np.nan)
    mask = margin > 0                 # equality is deliberately unresolved
    return dict(estimate=estimate, error=error, margin=margin, margin_pass=mask,
                conditional_credit=np.where(mask, estimate, 0.),
                certified_credit=np.zeros_like(estimate), certified=False,
                scope='empirical' if any(t.provenance == 'empirical' for t in terms) else 'conditional',
                unavailable=not available)
