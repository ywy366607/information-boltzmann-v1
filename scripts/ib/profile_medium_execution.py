"""Warm production-path component spans on an immutable real OWT checkpoint.

Nested component spans are inclusive and must not be summed with outer phases.
CUDA-event spans include stream idle gaps; they are not exclusive kernel time.
No production state, cursor or checkpoint is committed.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch

from information_boltzmann.core.plastic_medium import PlasticMedium3D
from information_boltzmann.core.plastic_ports import PlasticMediumPorts3D
from information_boltzmann.core.state_checkpoint import _StateVJPCheckpoint
from information_boltzmann.core.shared_credit import _SharedCreditCheckpoint
from information_boltzmann.runtime import active_medium_training, training
from scripts.ib import audit_medium_segment_graph as audit


class ComponentSpans:
    def __init__(self):
        self.enabled = False
        self.role = 'main.forward'
        self.rows = []
        self.windows = []
        self.shared_vjp_index = None

    @contextmanager
    def span(self, name):
        if not self.enabled:
            yield
            return
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        started = time.perf_counter()
        begin.record()
        try:
            yield
        finally:
            end.record()
            self.rows.append((name, time.perf_counter() - started, begin, end))

    def wrap(self, owner, name, label, role=None):
        original = getattr(owner, name)
        def wrapped(*args, **kwargs):
            previous = self.role
            if role is not None:
                self.role = role(previous) if callable(role) else role
            try:
                with self.span(label(self.role) if callable(label) else label):
                    return original(*args, **kwargs)
            finally:
                self.role = previous
        setattr(owner, name, wrapped)
        return original

    def install(self):
        self.wrap(training, '_writer_auxiliary_gradients', 'auxiliary_complete', 'aux.forward')
        self.wrap(training, 'quiet_training_chunk', 'chunk_forward_including_auxiliary')
        # ActiveMediumTrainer imports this function directly.
        active_medium_training.quiet_training_chunk = training.quiet_training_chunk
        self.wrap(torch.Tensor, 'backward', 'main_backward', 'main.backward')
        self.wrap(PlasticMediumPorts3D, 'assimilate', lambda r: r + '.write')
        self.wrap(PlasticMedium3D, 'advance', lambda r: r + '.physical_step')
        self.wrap(PlasticMedium3D, 'field_rhs', lambda r: r + '.endpoint_rhs')
        self.wrap(PlasticMediumPorts3D, 'read', lambda r: r + '.read')
        self.wrap(_StateVJPCheckpoint, 'backward', lambda r: r + '.event_vjp',
                  lambda r: 'aux.backward' if r.startswith('aux.') else 'main.backward')
        # Preserve the staticmethod descriptor expected by autograd.Function.
        _StateVJPCheckpoint.backward = staticmethod(_StateVJPCheckpoint.backward)
        original_shared = _SharedCreditCheckpoint.backward
        def shared_backward(*args):
            previous = self.shared_vjp_index
            self.shared_vjp_index = 0
            try:
                with self.span('shared_event_replay_and_vjps'):
                    return original_shared(*args)
            finally:
                self.shared_vjp_index = previous
        _SharedCreditCheckpoint.backward = staticmethod(shared_backward)
        original_grad = torch.autograd.grad
        def grad(*args, **kwargs):
            if self.shared_vjp_index is None:
                return original_grad(*args, **kwargs)
            name = ('shared_auxiliary_vjp' if self.shared_vjp_index == 0 else 'shared_task_vjp')
            self.shared_vjp_index += 1
            with self.span(name):
                return original_grad(*args, **kwargs)
        torch.autograd.grad = grad
        self.wrap(PlasticMediumPorts3D, 'flush_deferred_credit', 'linear_weight_gradient_gemms')
        self.wrap(torch.optim.AdamW, 'step', 'optimizer_step')
        self.wrap(torch.optim.AdamW, 'zero_grad', 'gradient_clear')
        self.wrap(active_medium_training, 'stable_clip_grad_norm_', 'gradient_clip')
        self.wrap(active_medium_training.ActiveMediumTrainer, '_health_update_snapshot', 'health_snapshot')
        self.wrap(active_medium_training.ActiveMediumTrainer, '_health_update_norms', 'health_update_norms')
        original_context = active_medium_training.optimizer_state_on_host
        @contextmanager
        def staged(*args, **kwargs):
            manager = original_context(*args, **kwargs)
            with self.span('optimizer_offload'):
                manager.__enter__()
            try:
                yield
            finally:
                with self.span('optimizer_restore'):
                    manager.__exit__(*sys.exc_info())
        active_medium_training.optimizer_state_on_host = staged
        original_consume = active_medium_training.ActiveMediumTrainer.consume
        calls = 0
        def consume(learner, *args, **kwargs):
            nonlocal calls
            calls += 1
            self.enabled = calls > 1
            self.rows = []
            result = original_consume(learner, *args, **kwargs)
            torch.cuda.synchronize()
            if self.enabled:
                totals = defaultdict(lambda: {'calls': 0, 'host_seconds': 0., 'stream_seconds': 0.})
                for name, elapsed, start, end in self.rows:
                    totals[name]['calls'] += 1
                    totals[name]['host_seconds'] += elapsed
                    totals[name]['stream_seconds'] += start.elapsed_time(end) / 1000
                self.windows.append(dict(totals))
                print('COMPONENT_SPANS ' + json.dumps(dict(totals)), flush=True)
            self.enabled = False
            self.rows = []
            return result
        active_medium_training.ActiveMediumTrainer.consume = consume


def main():
    spans = ComponentSpans()
    spans.install()
    try:
        audit.main()
    finally:
        if '--report' in sys.argv:
            path = Path(sys.argv[sys.argv.index('--report') + 1])
            if path.exists():
                result = json.loads(path.read_text(encoding='utf-8'))
                result['component_spans'] = spans.windows
                result['span_semantics'] = 'Inclusive nested host/stream spans, instrumentation overhead; warm windows only'
                path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()

