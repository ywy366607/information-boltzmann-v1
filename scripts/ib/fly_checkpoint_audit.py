"""Observe original and recomputed spike decisions within one backward.

Diagnostic only. Original voltage margins are kept on CPU, not in the
production CUDA graph. Hooks change no tensor or surrogate derivative.
"""

from contextlib import contextmanager

import numpy as np

import information_boltzmann.core.fly_bptt_learning as learning
from information_boltzmann.core.fly_reservoir import SpikeFn


class CheckpointSpikeAudit:
    def __init__(self, *, track_proxy=False):
        self.segments = []
        self.track_proxy = track_proxy
        self.active = None
        self.cpu_snapshot_bytes = 0
        self._installed = False

    @contextmanager
    def _context(self, index, mode):
        previous = self.active
        self.active = [index, mode, 0]
        try:
            yield
        finally:
            self.segments[index][mode + '_calls'] = self.active[2]
            self.active = previous

    def record(self, voltage):
        if self.active is None:
            return
        index, mode, tick = self.active
        self.active[2] += 1
        value = voltage.detach().cpu().numpy().copy()
        segment = self.segments[index]
        if mode == 'original':
            segment['original'].append(value)
            self.cpu_snapshot_bytes += value.nbytes
            return
        if tick >= len(segment['original']):
            raise AssertionError('Recompute contains an extra physical spike call')
        original = segment['original'][tick]
        if original.shape != value.shape:
            raise AssertionError('Recompute changed the spike-state shape')
        finite = bool(np.isfinite(value).all() and np.isfinite(original).all())
        flips = int(np.count_nonzero((original >= 0) != (value >= 0)))
        segment['comparisons'].append({
            'tick': tick,
            'elements': int(value.size),
            'spike_flips': flips,
            'voltage_bitwise_equal': bool(np.array_equal(value, original)),
            'voltage_max_abs_difference': float(np.max(np.abs(value - original))),
            'original_min_abs_threshold_margin': float(np.min(np.abs(original))),
            'recomputed_min_abs_threshold_margin': float(np.min(np.abs(value))),
            'finite': finite,
        })

    def record_proxy(self, voltage, width):
        if not self.track_proxy or self.active is None:
            return
        index, mode, next_tick = self.active
        tick = next_tick - 1
        value = (voltage.detach()/width.detach()).cpu().numpy().copy()
        segment = self.segments[index]
        if mode == 'original':
            segment['original_proxy'].append(value)
            self.cpu_snapshot_bytes += value.nbytes
            return
        original = segment['original_proxy'][tick]
        row = segment['comparisons'][-1]
        psi = lambda x: 1 / (1 + (np.pi*x.astype(np.float64))**2)
        row.update(proxy_finite=bool(np.isfinite(value).all() and np.isfinite(original).all()),
                   proxy_max_abs_difference=float(np.max(np.abs(value-original))),
                   proxy_derivative_max_abs_difference=float(np.max(np.abs(psi(value)-psi(original)))))

    def __enter__(self):
        if self._installed:
            raise RuntimeError('Audit already installed')
        self._checkpoint = learning.checkpoint
        self._spike_forward = SpikeFn.forward

        def checkpoint(function, *args, **kwargs):
            if 'context_fn' in kwargs:
                raise ValueError('Audit must not replace an existing checkpoint context')
            index = len(self.segments)
            self.segments.append({'original': [], 'original_proxy': [], 'comparisons': [],
                                  'original_calls': 0, 'recompute_calls': 0})
            kwargs['context_fn'] = lambda: (
                self._context(index, 'original'), self._context(index, 'recompute'))
            return self._checkpoint(function, *args, **kwargs)

        def spike_forward(ctx, voltage, *args):
            # Non-reentrant checkpoint may stop while saving the last needed
            # tensor. Observe before the original save, including that node.
            self.record(voltage)
            if args and args[0] is not None:
                self.record_proxy(voltage, args[0])
            return self._spike_forward(ctx, voltage, *args)

        learning.checkpoint = checkpoint
        SpikeFn.forward = staticmethod(spike_forward)
        self._installed = True
        return self

    def __exit__(self, *unused):
        learning.checkpoint = self._checkpoint
        SpikeFn.forward = staticmethod(self._spike_forward)
        self._installed = False

    def summary(self):
        rows = []
        for index, segment in enumerate(self.segments):
            comparisons = segment['comparisons']
            rows.append({
                'segment': index,
                'original_calls': segment['original_calls'],
                'recompute_calls': segment['recompute_calls'],
                'compared_calls': len(comparisons),
                'uncompared_original_calls': len(segment['original']) - len(comparisons),
                'spike_flips': sum(row['spike_flips'] for row in comparisons),
                'voltage_max_abs_difference': max(
                    (row['voltage_max_abs_difference'] for row in comparisons), default=0),
                'comparisons': comparisons,
            })
        comparisons = [row for segment in rows for row in segment['comparisons']]
        return {
            'scope': 'Same-backward original vs checkpoint recomputation; no independent-run comparison',
            'segments': rows,
            'original_spike_calls': sum(row['original_calls'] for row in rows),
            'compared_spike_calls': len(comparisons),
            'compared_neuron_decisions': sum(row['elements'] for row in comparisons),
            'total_spike_flips': sum(row['spike_flips'] for row in comparisons),
            'calls_with_spike_flips': sum(row['spike_flips'] > 0 for row in comparisons),
            'all_voltages_finite': all(row['finite'] for row in comparisons),
            'voltage_max_abs_difference': max(
                (row['voltage_max_abs_difference'] for row in comparisons), default=0),
            'proxy_compared_calls': sum('proxy_finite' in row for row in comparisons),
            'all_proxies_finite': all(row.get('proxy_finite', True) for row in comparisons),
            'proxy_max_abs_difference': max(
                (row.get('proxy_max_abs_difference', 0) for row in comparisons), default=0),
            'proxy_derivative_max_abs_difference': max(
                (row.get('proxy_derivative_max_abs_difference', 0) for row in comparisons), default=0),
            'original_min_abs_threshold_margin': min(
                (row['original_min_abs_threshold_margin'] for row in comparisons), default=None),
            'cpu_snapshot_bytes': self.cpu_snapshot_bytes,
            'extra_gpu_history_bytes': 0,
            'coverage_complete': bool(rows) and all(
                row['uncompared_original_calls'] == 0 for row in rows),
        }
