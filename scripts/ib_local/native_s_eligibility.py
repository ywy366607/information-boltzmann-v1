"""Conditional approximate native-S eligibility for code layers 1 and 12.

Only the local S->S leg and direct a/b/dt_bias/A_log gate sources are tracked.
Upper-cache, q/k/v/convolution, output, inter-layer and record/read paths are
omitted. This is neither full-joint nor an all-history derivative. FP64 key
normalization can differ from the native kernel. Error bills are diagnostics,
never an admission gate on raw credit. No model, optimizer or training imports.
"""
from copy import deepcopy
import math
import time

import numpy as np

from scripts.ib_local.bounded_causal_tag_reference import Code
from scripts.ib_local.bounded_causal_tag_blocked import (
    capture_credit_blocked, insert_source_inplace, workspace_budget,
)


WINDOW = 128
EPS = 1e-6


class _SLayout:
    def __init__(self, specs):
        self.fields, self.slices, self.shapes, self.head_fields = [], {}, {}, {}
        cursor = 0
        for layer, spec in specs.items():
            shape = (spec['key_dim'], spec['value_dim'])
            for head in range(spec['heads']):
                name = f'native.{layer}.s.{head}'
                self.head_fields[layer, head] = name
                self.fields.append(name)
                self.shapes[name] = shape
                self.slices[name] = slice(cursor, cursor + math.prod(shape))
                cursor += math.prod(shape)
        self.indices = {name: i for i, name in enumerate(self.fields)}
        self.size = cursor


class _HeadSource:
    """Stable local matrix, exposed as bounded sparse coordinate blocks."""
    def __init__(self, matrix, field_slice, token):
        self.matrix = matrix.reshape(-1)
        self.field_slice = field_slice
        self.token = token

    def __call__(self, start, stop):
        result = np.zeros(stop - start, dtype=np.float64)
        lo, hi = max(start, self.field_slice.start), min(stop, self.field_slice.stop)
        if lo < hi:
            result[lo - start:hi - start] = self.matrix[
                lo - self.field_slice.start:hi - self.field_slice.start]
        return result


def _array(value, shape, name, *, copy=True):
    if not isinstance(value, np.ndarray) or value.shape != shape:
        raise ValueError(f'{name}: expected CPU numpy array {shape}')
    if value.dtype.kind not in 'fi' or not np.isfinite(value).all():
        raise ValueError(f'{name}: expected finite real values')
    result = np.array(value, dtype=np.float32, copy=True) if copy else value
    if copy and not np.isfinite(result).all():
        raise ValueError(f'{name}: values exceed float32 capture range')
    return result


def _array_bytes(*values):
    """Count unique ndarray backing stores, not Python objects or RSS."""
    seen = set()
    def visit(value):
        if isinstance(value, np.ndarray):
            while isinstance(value.base, np.ndarray):
                value = value.base
            if id(value) not in seen:
                seen.add(id(value))
                return value.nbytes
        elif isinstance(value, dict):
            return sum(visit(v) for v in value.values())
        elif isinstance(value, (tuple, list)):
            return sum(visit(v) for v in value)
        return 0
    return sum(visit(v) for v in values)


class NativeSEligibility:
    """Shared fixed-rank code; capture now, contract OLD code, then advance.

    ``is_complete()`` is a method; ``nbytes`` is a property. Clock ``seen`` starts at the
    first explicitly opened boundary (migration does not backfill history).
    Subsequent windows must be contiguous; parameter versions are nondecreasing.
    Capture owns float32 copies of packets; physical reconstruction uses FP64.
    """
    scope = 'conditional approximate local native-S; uppercache/qkv/records omitted'

    def __init__(self, specs, rank=2, block_size=4096, max_bytes=64 * 2**20):
        if set(specs) != {1, 12}:
            raise ValueError('selected native-S layers must be exactly 1 and 12')
        self.specs = {}
        for layer in sorted(specs):
            spec = specs[layer]
            if set(spec) != {'heads', 'key_dim', 'value_dim', 'width'}:
                raise ValueError('spec requires heads, key_dim, value_dim, width')
            if any(not isinstance(v, (int, np.integer)) or v <= 0 for v in spec.values()):
                raise ValueError('spec dimensions must be positive integers')
            self.specs[layer] = {k: int(v) for k, v in spec.items()}
        if any(not isinstance(v, (int, np.integer)) or v <= 0
               for v in (rank, block_size, max_bytes)):
            raise ValueError('rank, block_size and max_bytes must be positive integers')
        self.rank, self.block_size, self.max_bytes = int(rank), int(block_size), int(max_bytes)
        self.layout = _SLayout(self.specs)
        self.parameter_shapes, self.parameter_slices = {}, {}
        cursor = 0
        for layer, spec in self.specs.items():
            h, w = spec['heads'], spec['width']
            for suffix, shape in (('a_proj.weight', (h, w)), ('b_proj.weight', (h, w)),
                                  ('dt_bias', (h,)), ('A_log', (h,))):
                name = f'layer{layer}.{suffix}'
                self.parameter_shapes[name] = shape
                self.parameter_slices[name] = slice(cursor, cursor + math.prod(shape))
                cursor += math.prod(shape)
        self.owners = cursor
        persistent = self.rank * (self.layout.size + cursor) * 4 + (3 + len(self.layout.fields)) * 8
        self._guard(persistent)
        self.code = Code.empty(self.rank, cursor, self.layout, dtype=np.float32)
        self.pending = None
        self.clock = dict(seen=None, parameter_version=None, windows=0, inserted_sources=0)
        self.last_credit = None

    @property
    def estimated_temporary_bytes(self):
        """Conservative explicit array allowance, excluding BLAS/Python/RSS."""
        scratch = workspace_budget(self.rank, self.owners, len(self.layout.fields), self.block_size)
        blocked = sum(v for v in scratch.values() if isinstance(v, int))
        head = max(s['key_dim'] * s['value_dim'] for s in self.specs.values())
        row = max(max(s['key_dim'], s['value_dim'], s['width']) for s in self.specs.values())
        # Local matrices, rank-at-a-time transport, owner/readout rows, physical
        # FP64 S copies, and a complete teaching-state copy allowance.
        return blocked + 12 * head * 8 + 6 * row * 8 + 2 * self.layout.size * 8

    def _guard(self, persistent):
        if persistent + self.estimated_temporary_bytes > self.max_bytes:
            raise MemoryError('native-S explicit array budget exceeds max_bytes (not RSS)')

    @property
    def nbytes(self):
        return _array_bytes(self.code.U, self.code.V, self.code.fields, self.code.bills, self.pending)

    def is_complete(self):
        return self.pending is not None and all(n == WINDOW for n in self.pending['counts'].values())

    def _states(self, values, *, copy=True):
        if set(values) != set(self.specs):
            raise ValueError('S state must contain both selected layers')
        states = {}
        for layer, spec in self.specs.items():
            shape = (spec['heads'], spec['key_dim'], spec['value_dim'])
            value = values[layer]
            if isinstance(value, np.ndarray) and value.shape == (1, *shape):
                value = value[0]
            states[layer] = _array(value, shape, f'layer{layer}.S', copy=copy)
        return states

    def open_window(self, start_seen, parameter_version, initial_s):
        if self.pending is not None:
            raise ValueError('window already open')
        if any(not isinstance(v, (int, np.integer)) or v < 0 for v in (start_seen, parameter_version)):
            raise ValueError('invalid window count/version')
        if self.clock['seen'] is not None and start_seen != self.clock['seen']:
            raise ValueError('window must start at the settled clock; no backfill')
        version = self.clock['parameter_version']
        if version is not None and parameter_version < version:
            raise ValueError('parameter version cannot go backwards')
        extra = self.layout.size * 4
        self._guard(self.nbytes + extra)
        initial = self._states(initial_s)
        self.pending = dict(start_seen=int(start_seen), parameter_version=int(parameter_version),
                            initial_s=initial, end_s=None, counts={l: 0 for l in self.specs},
                            chunks={l: [] for l in self.specs})

    def _packet(self, layer, packet):
        if layer not in self.specs:
            raise ValueError('unselected layer')
        required = {'x', 'k', 'v', 'g', 'beta', 'a_slope', 'Aexp', 'branch'}
        if set(packet) != required or packet['branch'] not in ('recurrent', 'chunk'):
            raise ValueError('invalid packet fields/normalization branch')
        x, spec = packet['x'], self.specs[layer]
        if not isinstance(x, np.ndarray) or x.ndim != 2 or not 0 < len(x) <= WINDOW:
            raise ValueError('packet length must be 1..128')
        t, h, k, v, w = len(x), spec['heads'], spec['key_dim'], spec['value_dim'], spec['width']
        shapes = dict(x=(t, w), k=(t, h, k), v=(t, h, v), g=(t, h),
                      beta=(t, h), a_slope=(t, h), Aexp=(h,))
        for name, shape in shapes.items():
            _array(packet[name], shape, name, copy=False)
        if (np.any(packet['g'] > 0) or np.any(packet['beta'] < 0) or np.any(packet['beta'] > 1)
                or np.any(packet['a_slope'] < 0) or np.any(packet['a_slope'] > 1)
                or np.any(packet['Aexp'] <= 0)):
            raise ValueError('invalid actual gate values')
        return shapes

    def add_chunk(self, layer, offset, packet):
        if self.pending is None:
            raise ValueError('no open window')
        shapes = self._packet(layer, packet)
        count = self.pending['counts'][layer]
        if not isinstance(offset, (int, np.integer)) or offset != count:
            raise ValueError('chunk offset must equal existing layer count')
        length = shapes['x'][0]
        if count + length > WINDOW:
            raise ValueError('capture exceeds 128 tokens per layer')
        self._guard(self.nbytes + sum(math.prod(s) * 4 for s in shapes.values()))
        captured = {name: _array(packet[name], shape, name) for name, shape in shapes.items()}
        captured.update(branch=packet['branch'], offset=int(offset))
        self.pending['chunks'][layer].append(captured)
        self.pending['counts'][layer] += length

    def set_end_s(self, values):
        if self.pending is None:
            raise ValueError('no open window')
        old = _array_bytes(self.pending['end_s'])
        self._guard(self.nbytes - old + self.layout.size * 4)
        self.pending['end_s'] = self._states(values)

    def raw_credit(self, teaching):
        """Contract incoming OLD code with caller's 10*CE S teaching, unscaled.

        The caller supplies the loss scale; no tolerance/margin mask is applied.
        Neither capture nor U/V/ledgers are changed by this operation.
        """
        states = self._states(teaching, copy=False)
        def readout(lo, hi):
            result = np.zeros(hi - lo, dtype=np.float64)
            for (layer, head), name in self.layout.head_fields.items():
                sl = self.layout.slices[name]
                start, stop = max(lo, sl.start), min(hi, sl.stop)
                if start < stop:
                    result[start - lo:stop - lo] = states[layer][head].reshape(-1)[
                        start - sl.start:stop - sl.start]
            return result
        result = capture_credit_blocked(self.code, readout, self.layout, block_size=self.block_size)
        estimate = result['estimate']  # ALWAYS raw, including unavailable/failed margins.
        gradients = {name: estimate[sl].astype(np.float32).reshape(self.parameter_shapes[name])
                     for name, sl in self.parameter_slices.items()}
        self.last_credit = dict(norm=float(np.linalg.norm(estimate)), scope=self.scope,
                                per_parameter={n: float(np.linalg.norm(g)) for n, g in gradients.items()})
        return gradients

    def _transport_head(self, name, k, alpha, beta):
        sl, index = self.layout.slices[name], self.layout.indices[name]
        gain = abs(alpha) * max(1., abs(1. - beta * float(k @ k)))
        # All other head fields are identities in this stage, not contractions.
        self.code.bills *= max(1., gain) if len(self.layout.fields) > 1 else gain
        self.code.fields[index] *= gain
        rounding = 0.
        shape = self.layout.shapes[name]
        for j in range(self.rank):
            old = self.code.U[j, sl].reshape(shape).astype(np.float64)
            value = alpha * (old - beta * np.outer(k, k @ old))
            stored = value.astype(np.float32)
            defect = np.linalg.norm(value - stored.astype(np.float64))
            rounding += float(np.max(np.abs(self.code.V[j]))) * float(defect)
            self.code.U[j, sl] = stored.reshape(-1)
        self.code.bills[1] += rounding
        self.code.fields[index] += rounding

    def _advance_head(self, layer, head, packet, tick, primal, owner):
        raw_k = packet['k'][tick, head].astype(np.float64)
        denominator = (np.linalg.norm(raw_k) + EPS if packet['branch'] == 'recurrent'
                       else np.sqrt(float(raw_k @ raw_k) + EPS))
        k = raw_k / denominator
        v = packet['v'][tick, head].astype(np.float64)
        g, beta = float(packet['g'][tick, head]), float(packet['beta'][tick, head])
        alpha = math.exp(g)
        s = primal[layer][head]
        r = s - beta * np.outer(k, k @ s)
        b = np.outer(k, v - k @ (alpha * s))
        c_z = -alpha * float(packet['Aexp'][head]) * float(packet['a_slope'][tick, head])
        name = self.layout.head_fields[layer, head]
        self._transport_head(name, k, alpha, beta)
        x = packet['x'][tick].astype(np.float64)
        prefix, width = f'layer{layer}.', self.specs[layer]['width']
        for kind, matrix in (('R', r), ('B', b)):
            owner.fill(0.)
            if kind == 'R':
                sl = self.parameter_slices[prefix + 'a_proj.weight']
                owner[sl.start + head * width:sl.start + (head + 1) * width] = c_z * x
                owner[self.parameter_slices[prefix + 'dt_bias'].start + head] = c_z
                owner[self.parameter_slices[prefix + 'A_log'].start + head] = alpha * g
            else:
                sl = self.parameter_slices[prefix + 'b_proj.weight']
                owner[sl.start + head * width:sl.start + (head + 1) * width] = beta * (1. - beta) * x
            token = (self.pending['start_seen'], layer, packet['offset'] + tick, head, kind)
            source = _HeadSource(matrix, self.layout.slices[name], token)
            self._insert_source(source, owner)
        # Sources above use the pre-transition physical S, not a changed basis.
        primal[layer][head] = alpha * s + beta * b

    def _insert_source(self, source, owner):
        """Shared insertion receipt; optional age bookkeeping changes no code."""
        return insert_source_inplace(self.code, source, owner, self.layout, self.block_size)

    def advance_window(self):
        """Call only after actual settlement/observation; no parameter updates."""
        if not self.is_complete():
            raise ValueError('cannot advance an incomplete 128-token window')
        self._guard(self.nbytes)
        started = time.perf_counter()
        primal = {layer: s.astype(np.float64) for layer, s in self.pending['initial_s'].items()}
        owner = np.zeros(self.owners, dtype=np.float64)
        indices = {layer: 0 for layer in self.specs}
        for tick in range(WINDOW):
            for layer, spec in self.specs.items():
                chunks = self.pending['chunks'][layer]
                while tick >= chunks[indices[layer]]['offset'] + len(chunks[indices[layer]]['x']):
                    indices[layer] += 1
                packet = chunks[indices[layer]]
                for head in range(spec['heads']):
                    self._advance_head(layer, head, packet, tick - packet['offset'], primal, owner)
        delta = None
        if self.pending['end_s'] is not None:
            delta = {}
            for layer, actual in self.pending['end_s'].items():
                difference = primal[layer] - actual
                delta[layer] = dict(l2=float(np.linalg.norm(difference)),
                                    max_abs=float(np.max(np.abs(difference))))
        inserted = 2 * WINDOW * sum(s['heads'] for s in self.specs.values())
        captured_bytes = self.nbytes
        self.clock.update(seen=self.pending['start_seen'] + WINDOW,
                          parameter_version=self.pending['parameter_version'],
                          windows=self.clock['windows'] + 1,
                          inserted_sources=self.clock['inserted_sources'] + inserted)
        self.pending = None
        return dict(processed=WINDOW, inserted_sources=inserted, reconstruction_delta=delta,
                    bytes=self.nbytes, captured_bytes=captured_bytes,
                    estimated_temporary_bytes=self.estimated_temporary_bytes,
                    estimated_peak_bytes=captured_bytes + self.estimated_temporary_bytes,
                    elapsed=time.perf_counter() - started, scope=self.scope)

    def state_dict(self):
        """Detached array copies; only the current <=128-token capture is saved."""
        return dict(schema=1, specs=deepcopy(self.specs), rank=self.rank, block_size=self.block_size,
                    max_bytes=self.max_bytes, parameter_slices=self.parameter_slices.copy(),
                    parameter_shapes=self.parameter_shapes.copy(),
                    code={n: getattr(self.code, n).copy() for n in ('U', 'V', 'fields', 'bills')},
                    pending=deepcopy(self.pending), clock=self.clock.copy(),
                    last_credit=deepcopy(self.last_credit))

    def load_state_dict(self, state):
        """Restore an identical producer, including an incomplete current window."""
        loaded_specs = {int(layer): spec for layer, spec in state['specs'].items()}
        if (state['schema'] != 1 or loaded_specs != self.specs or state['rank'] != self.rank
                or state['parameter_slices'] != self.parameter_slices
                or state['parameter_shapes'] != self.parameter_shapes):
            raise ValueError('native-S state specs/owner map mismatch')
        if state['block_size'] != self.block_size:
            raise ValueError('native-S block_size mismatch')
        arrays = state['code']
        for name in ('U', 'V', 'fields', 'bills'):
            value, expected = arrays[name], getattr(self.code, name)
            _array(value, expected.shape, name, copy=False)
            if value.dtype != expected.dtype or (name in ('fields', 'bills') and np.any(value < 0)):
                raise ValueError('invalid stored code dtype/ledger')
        clock = state['clock']
        if (set(clock) != set(self.clock)
                or any(not isinstance(clock[n], int) or clock[n] < 0
                       for n in ('windows', 'inserted_sources'))):
            raise ValueError('invalid eligibility clock')
        for name in ('seen', 'parameter_version'):
            if clock[name] is not None and (not isinstance(clock[name], int) or clock[name] < 0):
                raise ValueError('invalid eligibility clock boundary/version')
        if ((clock['seen'] is None) != (clock['parameter_version'] is None)
                or (clock['windows'] == 0) != (clock['seen'] is None)
                or clock['inserted_sources'] != clock['windows'] * 2 * WINDOW *
                sum(s['heads'] for s in self.specs.values())):
            raise ValueError('inconsistent eligibility clock')
        pending = state['pending']
        if pending is not None:
            pending = dict(pending)
            for name in ('initial_s', 'end_s', 'counts', 'chunks'):
                if pending[name] is not None:
                    pending[name] = {int(layer): value for layer, value in pending[name].items()}
            if any(not isinstance(pending[n], int) or pending[n] < 0
                   for n in ('start_seen', 'parameter_version')):
                raise ValueError('invalid pending boundary/version')
            if ((clock['seen'] is not None and pending['start_seen'] != clock['seen'])
                    or (clock['parameter_version'] is not None
                        and pending['parameter_version'] < clock['parameter_version'])):
                raise ValueError('pending clock/version mismatch')
            self._states(pending['initial_s'], copy=False)
            if pending['end_s'] is not None:
                self._states(pending['end_s'], copy=False)
            if set(pending['chunks']) != set(self.specs) or set(pending['counts']) != set(self.specs):
                raise ValueError('pending layers mismatch')
            for layer, chunks in pending['chunks'].items():
                count = 0
                if len(chunks) > WINDOW:
                    raise ValueError('too many pending chunks')
                for chunk in chunks:
                    packet = {k: v for k, v in chunk.items() if k != 'offset'}
                    shapes = self._packet(layer, packet)
                    if chunk['offset'] != count:
                        raise ValueError('pending chunk offset mismatch')
                    count += shapes['x'][0]
                if count != pending['counts'][layer] or count > WINDOW:
                    raise ValueError('pending count mismatch')
        self._guard(_array_bytes(arrays, pending))
        self.code = Code(**{n: arrays[n].copy() for n in ('U', 'V', 'fields', 'bills')})
        self.pending, self.clock = deepcopy(pending), clock.copy()
        self.last_credit = deepcopy(state['last_credit'])
