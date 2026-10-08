"""One physical event, one persistent predictive-observer update.

Truncation detaches graphs, never the values or timestamps of the individual.
This module deliberately owns a separate interface from the HX-0/HX-1 arms.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .fly_bptt_learning import (
    FlyBPTTLearner, FlyPhysicalState, advance_fly_input_event,
)


def motor_features(model, physical: FlyPhysicalState) -> torch.Tensor:
    """Read only the configured motor surface, before shared normalization."""
    membrane = physical.h
    if getattr(model, 'read_centering', False):
        membrane = membrane - physical.h_mean
    return model.output_read(membrane[:, model.read_indices])


def predict_streaming_next(model, physical, observer_state, token):
    """Label-free streaming inference; caller carries both returned states.

    Passing ``None`` as observer state is intentionally rejected: newborn
    initialization and resumed life must be explicit at the call site.
    """
    if observer_state is None:
        raise ValueError('Supply the continuing observer state explicitly')
    physical = advance_fly_input_event(
        model, physical, token, settle_ticks=0, writer_baseline_clock='physical')
    forecast, _, observer_state, _ = model.streaming_observer.step(
        physical, observer_state)
    features = motor_features(model, physical) + forecast
    return model.decoder(model.read_norm(features)), physical, observer_state


class FlyStreamingLearner(FlyBPTTLearner):
    """Joint truncated-BPTT training with a continuing graph observer."""

    def __init__(self, model, state, *, observer_state=None, **options):
        if getattr(model, 'streaming_observer', None) is None:
            raise ValueError('Attach the streaming observer before constructing learner')
        if any(getattr(model, name, None) is not None
               for name in ('graph_observer', 'latent_predictor')):
            raise ValueError('Streaming arm requires other forecast modules disabled')
        if getattr(model, 'use_read_gamma_trace', False):
            raise ValueError('Streaming arm has no Gamma observation filter')
        if options.get('settle_ticks', 0) != 0:
            raise ValueError('Streaming arm uses exactly one physical tick per token')
        options['settle_ticks'] = 0
        options['writer_baseline_clock'] = 'physical'
        names = options.pop('adam_names', None)
        if names is None:
            names = [
                'output_read.weight', 'read_norm.weight', 'decoder.weight',
                'decoder.bias', 'log_threshold', 'log_tau_m', 'log_beta',
                'log_tau_a', 'log_tau_s_e', 'log_tau_s_i', 'log_g_e', 'log_g_i',
                'topographic_writer.gate_linear.weight',
                'topographic_writer.gate_linear.bias',
            ]
        names = list(names)
        for name, _ in model.streaming_observer.named_parameters():
            name = 'streaming_observer.' + name
            if name not in names:
                names.append(name)
        super().__init__(model, state, adam_names=names, **options)
        self.observer_state = (observer_state if observer_state is not None
            else model.streaming_observer.initial_state(
                state.h.shape[0], state.h.device, state.h.dtype)).detached()
        self._observer_gradient_norms = {}
        self._observer_hooks = []
        for name, parameter in model.streaming_observer.named_parameters():
            def capture(gradient, parameter_name=name):
                self._observer_gradient_norms[parameter_name] = gradient.detach().norm()
                return gradient
            capture = torch.utils.hooks.unserializable_hook(capture)
            self._observer_hooks.append(parameter.register_hook(capture))

    def forward_window(self, input_ids, targets):
        if input_ids.shape[0] != 1:
            raise ValueError('This learner represents one continuous individual')
        model, physical = self.model, self.state
        observer = self.observer_state
        rates, thresholds = model.get_decay_rates(), model.get_thresholds()
        gains = model.get_conductance_gains()
        alif, stp = model.get_alif_params(), model.get_stp_params()
        features, losses, metric_values = [], [], {}
        for token in input_ids.unbind(1):
            physical = advance_fly_input_event(
                model, physical, token, settle_ticks=0,
                writer_baseline_clock='physical', base_rates=rates,
                thresholds=thresholds, conductance_gains=gains,
                alif_params=alif, stp_params=stp)
            forecast, auxiliary, observer, metrics = model.streaming_observer.step(
                physical, observer)
            features.append(motor_features(model, physical) + forecast)
            losses.append(auxiliary)
            for key, value in metrics.items():
                metric_values.setdefault(key, []).append(value)
        features = torch.cat(features, dim=0)
        logits = model.decoder(model.read_norm(features))
        scores = F.cross_entropy(logits, targets.flatten(), reduction='none')
        self.last_jepa_loss = torch.stack(losses).mean()
        self.last_jepa_metrics = {
            key: torch.stack(values).mean().detach()
            for key, values in metric_values.items()}
        self.observer_state = observer
        return scores, physical, features

    def observe(self, target_tokens):
        self._observer_gradient_norms.clear()
        scores, metrics = super().observe(target_tokens)
        metrics.update({key: float(value) for key, value
                        in self.last_jepa_metrics.items()})
        params = list(self.model.streaming_observer.parameters())
        # Parent has already cleared parameter gradients; auxiliary monitoring
        # uses recorded values, avoiding a second backward or state advance.
        metrics['observer_parameters'] = sum(p.numel() for p in params)
        metrics['observer_grad_norm'] = float(torch.linalg.vector_norm(
            torch.stack(list(self._observer_gradient_norms.values())))) if self._observer_gradient_norms else 0.0
        metrics['observer_gradient_rms'] = metrics['observer_grad_norm'] / max(
            1, metrics['observer_parameters'])**.5
        self.observer_state = self.observer_state.detached()
        return scores, metrics

    def state_dict(self):
        state = super().state_dict()
        state.update(streaming_observer=self.observer_state.state_dict(),
                     learning_route='continuous-predict-arrive-correct-v1')
        return state

    def restore_learning_state(self, saved):
        """Restore the complete learner after loading identical model weights."""
        if saved.get('learning_route') != 'continuous-predict-arrive-correct-v1':
            raise ValueError('Use a complete streaming-route checkpoint')
        device = self.state.h.device
        self.state = FlyPhysicalState(**{
            key: tuple(t.to(device) for t in value) if key == 'ring'
            else value.to(device) for key, value in saved['physical'].items()})
        state_type = type(self.observer_state)
        self.observer_state = state_type.from_state_dict(
            saved['streaming_observer']).to(device).detached()
        self.load_edge_signs(saved)
        self.optimizer.load_state_dict(saved['optimizer'])
        self.sgd.load_state_dict(saved['sgd'])
        for key in ('events', 'updates', 'physical_ticks', 'previous_token', 'ema'):
            setattr(self, key, saved[key])
        self.latent_window.copy_(saved['latent_window'].to(device))
