"""Explicit opt-in S/P migration for the existing continuous navigation body.

Every actual objective captures the actual post-hook layer input. A bounded
replay computes the corresponding P kick before optimizer.step; imagined and
diagnostic forward passes cannot write P. Internal evolution happens once at
the existing decision boundary, before the whole turn takes its start state.
"""
from __future__ import annotations

import torch

from scripts.ib_local.bfss_gdn_capture import matrix_view, observed_momentum, quiet_packed
from scripts.ib_local.navigation_internal_rollout import InternalRolloutBody, InternalRolloutEngine


class MomentumBody(InternalRolloutBody):
    def objective(self, *args, **kwargs):
        if not getattr(self, '_momentum_active', False):
            return super().objective(*args, **kwargs)
        layers = self.body.base.layers
        captured = [None] * len(layers)
        handles = []
        for i, layer in enumerate(layers):
            def capture(module, inputs, output, index=i):
                x, before = inputs
                s, p, correction = observed_momentum(module, x.detach(), before.detach(),
                                                       self._momentum_current[index])
                actual = matrix_view(module, output[1].detach())
                error = float((s-actual).abs().max())
                scale = float(actual.abs().max())
                if error > 2e-5 + 2e-4*scale:
                    raise RuntimeError(f'S/P observed replay disagrees at layer {index}: {error}')
                captured[index] = (p, error, float(correction))
            handles.append(layer.register_forward_hook(capture))
        try:
            result = super().objective(*args, **kwargs)
        finally:
            for handle in handles:
                handle.remove()
        if any(item is None for item in captured):
            raise RuntimeError('S/P event capture missed a layer')
        self._momentum_captured = captured
        return result


class MomentumEngine(InternalRolloutEngine):
    def _advance(self, loss, features, diagnostics, next_state, actor=None, advantage=0.):
        captured = self.model._momentum_captured
        result = super()._advance(loss, features, diagnostics, next_state, actor, advantage)
        self.bfss_momenta = [item[0].detach() for item in captured]
        self.model._momentum_current = self.bfss_momenta
        self._bfss_event_errors.extend(item[1] for item in captured)
        self._bfss_corrections.extend(item[2] for item in captured)
        self.model._momentum_captured = None
        return result

    def step(self, directions, age_ablate=False):
        self._bfss_event_errors = []
        self._bfss_corrections = []
        # The control suppresses quiet motion only; it still follows the real
        # S/P event writes so a later decision never resumes with stale P.
        self.model._momentum_active = True
        self.model._momentum_current = self.bfss_momenta
        before = self.recurrent
        if self.bfss_enabled:
            self.recurrent, self.bfss_momenta = quiet_packed(
                self.model.body.base, self.recurrent, self.bfss_momenta,
                dt=self.bfss_dt, frequency=self.bfss_frequency,
                coupling=self.bfss_coupling, steps=self.bfss_steps)
            self.model._momentum_current = self.bfss_momenta
        distance = float((self.recurrent-before).norm())
        result = super().step(directions, age_ablate)
        result['bfss_dynamics'] = dict(enabled=self.bfss_enabled,
            internal_steps=self.bfss_steps if self.bfss_enabled else 0,
            state_change=distance, momentum_norms=[float(p.norm()) for p in self.bfss_momenta],
            event_replay_max_error=max(self._bfss_event_errors, default=0.),
            event_correction_norms=self._bfss_corrections)
        return result


def configure_momentum(engine, *, enabled=True, dt=.01, frequency=.8, coupling=.01, steps=1):
    """Migrate an existing decision-boundary individual without resetting S."""
    if type(engine) not in (InternalRolloutEngine, MomentumEngine):
        raise TypeError('S/P migration requires the existing internal rollout individual')
    if type(engine.model) not in (InternalRolloutBody, MomentumBody):
        raise TypeError('unsupported body')
    if dt < 0 or frequency <= 0 or coupling < 0 or type(steps) is not int or steps < 0:
        raise ValueError('invalid dynamics configuration')
    if not hasattr(engine, 'bfss_momenta'):
        base = engine.model.body.base
        engine.bfss_momenta = [torch.zeros_like(matrix_view(layer, engine.recurrent[i]))
                               for i, layer in enumerate(base.layers)]
    engine.__class__ = MomentumEngine
    engine.model.__class__ = MomentumBody
    engine.bfss_enabled = bool(enabled)
    engine.bfss_dt, engine.bfss_frequency = float(dt), float(frequency)
    engine.bfss_coupling, engine.bfss_steps = float(coupling), steps
    engine.bfss_schema = 1
    engine.model._momentum_active = False
    engine.model._momentum_current = engine.bfss_momenta
    return engine
