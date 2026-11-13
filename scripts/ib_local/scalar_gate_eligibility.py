"""Bounded per-head A_log eligibility for the held-input local GDN S leg.

This candidate deliberately selects only the eight A_log scalars in layers
1/12. Each head owns three independent FP64 matrices: older, previous,
current. It does NOT compress different heads into a shared rank-two basis.
For fixed actual k/v/beta/g packets, E transports as
    E+ = alpha * (E - beta * k @ (k.T @ E)),
and the newly captured A_log source is alpha*g*(S-beta*k@(k.T@S)).
The derivative holds input/gate ancestors fixed, except this direct scalar
source; qkv, convolution, inter-layer, records and optimizer paths are omitted.
This is not an all-network derivative or a claim of future learning benefit.

At settled boundary T, credit() contracts ONLY older sources, born strictly
before T-128. Call credit before advancing the next factual window. Advance
transports all cohorts through the same 128 actual tokens, injects sources
only into current, then older += previous; previous = current; current = 0.
No old states, historical graphs, weights or optimizer snapshots are stored.
For 2 layers x 4 heads x 256 x 256, resident cohorts take 12 MiB. There is
one FP64 physical-S reconstruction plus bounded head-local workspace during
advance; supplied packets and caller/model memory are outside this count.
"""

from copy import deepcopy
import math

import numpy as np


WINDOW = 128
EPS = 1e-6


class ScalarGateEligibility:
    """Eight selected scalar owners; independent finite state per head."""

    scope = 'held-input local S direct A_log sources; strict old sources only'

    def __init__(self, specs):
        if set(specs) != {1, 12}:
            raise ValueError('selected layers must be exactly 1 and 12')
        self.specs = {}
        for layer, spec in sorted(specs.items()):
            if set(spec) != {'heads', 'key_dim', 'value_dim', 'width'}:
                raise ValueError('invalid scalar eligibility spec fields')
            if any(type(v) is not int or v <= 0 for v in spec.values()):
                raise ValueError('positive integer dimensions required')
            self.specs[layer] = dict(spec)
        self.older, self.previous, self.current = (
            {layer: np.zeros(self._shape(layer), np.float64) for layer in self.specs}
            for _ in range(3))
        self.seen = None
        self.windows = 0

    def _shape(self, layer):
        spec = self.specs[layer]
        return spec['heads'], spec['key_dim'], spec['value_dim']

    @property
    def nbytes(self):
        return sum(v.nbytes for cohort in (self.older, self.previous, self.current)
                   for v in cohort.values())

    @staticmethod
    def _finite(value, shape, name):
        value = np.asarray(value)
        if (value.shape != shape or value.dtype.kind not in 'fi'
                or not np.isfinite(value).all()):
            raise ValueError(f'{name}: expected finite real array {shape}')
        return value

    def _states(self, values):
        if set(values) != set(self.specs):
            raise ValueError('both selected S layers required')
        result = {}
        for layer in self.specs:
            value = np.asarray(values[layer])
            shape = self._shape(layer)
            if value.shape == (1, *shape):
                value = value[0]
            result[layer] = self._finite(value, shape, f'layer{layer}.S')
        return result

    def _packets(self, chunks):
        if set(chunks) != set(self.specs):
            raise ValueError('both selected packet layers required')
        result = {}
        for layer, spec in self.specs.items():
            offset, prepared = 0, []
            for packet in chunks[layer]:
                fields = {'x', 'k', 'v', 'g', 'beta', 'a_slope', 'Aexp', 'branch'}
                if set(packet) not in (fields, fields | {'offset'}):
                    raise ValueError('invalid factual packet fields')
                if packet.get('offset', offset) != offset:
                    raise ValueError('factual chunks must be contiguous')
                if packet['branch'] not in ('recurrent', 'chunk'):
                    raise ValueError('unknown normalization branch')
                x = np.asarray(packet['x'])
                if x.ndim != 2 or not 0 < len(x) <= WINDOW:
                    raise ValueError('invalid packet length')
                n, h, k, v = len(x), spec['heads'], spec['key_dim'], spec['value_dim']
                shapes = dict(x=(n, spec['width']), k=(n,h,k), v=(n,h,v),
                              g=(n,h), beta=(n,h), a_slope=(n,h), Aexp=(h,))
                for name, shape in shapes.items():
                    self._finite(packet[name], shape, name)
                if (np.any(packet['g'] > 0) or np.any(packet['beta'] < 0)
                        or np.any(packet['beta'] > 1) or np.any(packet['a_slope'] < 0)
                        or np.any(packet['a_slope'] > 1) or np.any(packet['Aexp'] <= 0)):
                    raise ValueError('invalid factual gate values')
                prepared.append(packet)
                offset += n
            if offset != WINDOW:
                raise ValueError('advance requires exactly 128 factual tokens per layer')
            result[layer] = prepared
        return result

    def credit(self, teaching):
        """Return only strictly old direct A_log contributions, in FP32."""
        values = self._states(teaching)
        result = {}
        for layer in self.specs:
            value = np.einsum('hkv,hkv->h', self.older[layer],
                              values[layer].astype(np.float64))
            stored = value.astype(np.float32)
            if not np.isfinite(stored).all():
                raise FloatingPointError('nonfinite scalar historical credit')
            result[f'layer{layer}.A_log'] = stored
        return result

    def advance(self, initial_s, chunks, start_seen):
        """Advance from real physical S and already captured 128-token packets.

        Reads caller arrays, never mutates them. No parameter update occurs.
        ``seen`` may begin at an inherited boundary; unobserved earlier sources
        start empty and are never fabricated as old eligibility.
        """
        if type(start_seen) is not int or start_seen < 0:
            raise ValueError('nonnegative factual start required')
        if self.seen is not None and start_seen != self.seen:
            raise ValueError('factual windows must be contiguous')
        states, packets = self._states(initial_s), self._packets(chunks)
        primal = {layer: value.astype(np.float64, copy=True) for layer, value in states.items()}
        for layer, spec in self.specs.items():
            for packet in packets[layer]:
                for tick in range(len(packet['x'])):
                    for head in range(spec['heads']):
                        raw = np.asarray(packet['k'][tick, head], np.float64)
                        denominator = (np.linalg.norm(raw) + EPS
                            if packet['branch'] == 'recurrent'
                            else np.sqrt(float(raw @ raw) + EPS))
                        k = raw / denominator
                        v = np.asarray(packet['v'][tick, head], np.float64)
                        beta, g = float(packet['beta'][tick, head]), float(packet['g'][tick, head])
                        alpha = math.exp(g)
                        s = primal[layer][head]
                        residual_s = s - beta * np.outer(k, k @ s)
                        for cohort in (self.older, self.previous, self.current):
                            e = cohort[layer][head]
                            e[:] = alpha * (e - beta * np.outer(k, k @ e))
                        self.current[layer][head] += alpha * g * residual_s
                        primal[layer][head] = alpha * residual_s + beta * np.outer(k, v)
        for layer in self.specs:
            self.older[layer] += self.previous[layer]
            self.previous[layer][:] = self.current[layer]
            self.current[layer].fill(0.)
            if not all(np.isfinite(cohort[layer]).all()
                       for cohort in (self.older, self.previous, self.current)):
                raise FloatingPointError('nonfinite scalar eligibility')
        self.seen, self.windows = start_seen + WINDOW, self.windows + 1
        return dict(processed=WINDOW, seen=self.seen, windows=self.windows,
                    strict_old_source_exclusive_stop=self.seen-WINDOW,
                    persistent_bytes=self.nbytes, scope=self.scope)

    def state_dict(self):
        return dict(schema=1, specs=deepcopy(self.specs), seen=self.seen, windows=self.windows,
                    **{name: {l: v.copy() for l,v in getattr(self,name).items()}
                       for name in ('older','previous','current')})

    def load_state_dict(self, state):
        if state['schema'] != 1 or state['specs'] != self.specs:
            raise ValueError('scalar eligibility state/spec mismatch')
        seen, windows = state['seen'], state['windows']
        if (type(windows) is not int or windows < 0 or
                (seen is not None and (type(seen) is not int or seen < windows*WINDOW))
                or (seen is None) != (windows == 0)):
            raise ValueError('invalid scalar eligibility clock')
        loaded = {}
        for name in ('older','previous','current'):
            values = self._states(state[name])
            if any(v.dtype != np.float64 for v in values.values()):
                raise ValueError('eligibility state must be FP64')
            loaded[name] = {l: v.copy() for l,v in values.items()}
        if any(v.any() for v in loaded['current'].values()):
            raise ValueError('current sources must be settled before saving')
        if windows == 0 and any(v.any() for c in loaded.values() for v in c.values()):
            raise ValueError('cannot fabricate eligibility before factual capture')
        for name, values in loaded.items():
            setattr(self, name, values)
        self.seen, self.windows = seen, windows
