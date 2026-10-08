"""Joint language learning and bounded student-guided physical query replay.

Only the real input stream advances the individual's clock. All quiet teacher
queries run sequentially on detached copies at unchanged parameter versions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import torch
import torch.nn.functional as F

from .fly_bptt_learning import (
    FlyBPTTLearner, FlyPhysicalState, advance_fly_input_event,
    step_fly_physical_tick,
)
from .fly_streaming_learning import motor_features


ROUTE = 'fly-rtc-tick-dagger-v1'


def copy_physical(physical, device=None):
    payload = physical.detached().state_dict()
    return FlyPhysicalState(**{
        name: tuple(t.clone().to(device or t.device) for t in value) if name == 'ring'
        else value.clone().to(device or value.device) for name, value in payload.items()})


@dataclass
class RTCStudentState:
    history: torch.Tensor
    tick: int = 0
    query_ticks: int = 0
    speculative_ticks: int = 0
    replay: list = field(default_factory=list)

    def detached(self):
        return type(self)(self.history.detach(), self.tick, self.query_ticks,
                          self.speculative_ticks, self.replay)

    def state_dict(self):
        return dict(history=self.history, tick=self.tick,
                    query_ticks=self.query_ticks, speculative_ticks=self.speculative_ticks,
                    replay=self.replay)

    @classmethod
    def restore(cls, saved, device):
        if set(saved) != set(cls(torch.empty(0)).state_dict()):
            raise ValueError('Incomplete tick-student continuation')
        return cls(**{**saved, 'history': saved['history'].to(device)})


def initial_student_state(model, physical):
    student = model.rtc_student
    return RTCStudentState(student.initial_history(
        physical.h.shape[0], physical.h.device, physical.h.dtype))


def read_rtc_event(model, physical, continuing, token, *, options=None):
    """Common training/inference event path, with no target or quiet teacher.

    One actual input tick, then autonomous motor-response drafts. The readout
    mixes forecast horizons as task features; this does not equate a horizon k
    to token t+k, nor to a prediction of unknown future sensory stimuli.
    """
    if continuing is None:
        raise ValueError('Continuing tick-student state must be explicit')
    options = options or {}
    physical = advance_fly_input_event(model, physical, token, settle_ticks=0,
                                      writer_baseline_clock='input', **options)
    student = model.rtc_student
    observed = student.codec.encode(physical)
    history = torch.cat((observed[None], continuing.history[:-1]), dim=0)
    forecast, drafts, weights = student.issue(history)
    features = motor_features(model, physical) + forecast
    following = RTCStudentState(history, continuing.tick + 1,
        continuing.query_ticks, continuing.speculative_ticks + student.horizon,
        continuing.replay)
    return features, physical, following, drafts, weights


def predict_rtc_next(model, physical, continuing, token):
    features, physical, continuing, _, _ = read_rtc_event(
        model, physical, continuing, token)
    return model.decoder(model.read_norm(features)), physical, continuing


def physical_options(model):
    return dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds(),
                conductance_gains=model.get_conductance_gains(),
                alif_params=model.get_alif_params(), stp_params=model.get_stp_params())


@torch.no_grad()
def quiet_teacher_step(model, physical, options):
    return step_fly_physical_tick(model, physical, None, torch.zeros_like(physical.h),
                                 physical.baseline, options,
                                 base_rates=options['base_rates'])


class FlyRTCLearner(FlyBPTTLearner):
    def __init__(self, model, state, *, rtc_state=None, replay_capacity=2,
                 query_radius=0.1, **options):
        if getattr(model, 'rtc_student', None) is None:
            raise ValueError('Attach TickResponseStudent before constructing learner')
        if model.read_surface != 'output' or model.topographic_writer is None:
            raise ValueError('Sensory injection and motor-only readout required')
        sensory = set(model.topographic_writer.injection_index.detach().cpu().tolist())
        motor = set(model.read_indices.detach().cpu().tolist())
        if sensory & motor:
            raise ValueError('This route requires disjoint physical port surfaces')
        if any(getattr(model, name, None) is not None
               for name in ('graph_observer', 'latent_predictor', 'streaming_observer')):
            raise ValueError('Other observer routes must be disabled')
        if model.use_read_gamma_trace or options.get('settle_ticks', 0) != 0:
            raise ValueError('One actual tick, without Gamma or settle loops')
        if replay_capacity < 1 or not 0 <= query_radius <= 1:
            raise ValueError('Positive replay budget and radius in [0, 1] required')
        options['settle_ticks'] = 0
        options['writer_baseline_clock'] = 'input'
        names = list(options.pop('adam_names', None) or (
            'output_read.weight', 'read_norm.weight', 'decoder.weight', 'decoder.bias',
            'log_threshold', 'log_tau_m', 'log_beta', 'log_tau_a',
            'log_tau_s_e', 'log_tau_s_i', 'log_g_e', 'log_g_i',
            'topographic_writer.gate_linear.weight', 'topographic_writer.gate_linear.bias'))
        names.extend('rtc_student.' + name for name, _ in model.rtc_student.named_parameters())
        super().__init__(model, state, adam_names=names, **options)
        self.rtc_state = rtc_state or initial_student_state(model, state)
        self.rtc_state = self.rtc_state.detached()
        self.replay_capacity, self.query_radius = replay_capacity, query_radius
        self._gradient_norms, self._gradient_hooks = {}, []
        for name, parameter in model.rtc_student.named_parameters():
            def record(gradient, parameter_name=name):
                self._gradient_norms[parameter_name] = gradient.detach().norm()
                return gradient
            record = torch.utils.hooks.unserializable_hook(record)
            self._gradient_hooks.append(parameter.register_hook(record))

    def _physical_queries(self, physical, history, drafts, options):
        """One horizon-matched teacher path and one re-encoded query, per window."""
        student = self.model.rtc_student
        targets, motor_targets = [], []
        current = copy_physical(physical)
        with torch.no_grad():
            for _ in range(student.horizon):
                current = quiet_teacher_step(self.model, current, options)
                targets.append(student.codec.encode(current))
                motor_targets.append(motor_features(self.model, current))
        labels = torch.stack(targets)
        # Each draft[1+k] and label[k] shares origin, zero-drive control and k+1.
        loss_rollout = F.mse_loss(drafts[1:], labels)
        response_predictions = student.draft_features(
            drafts, motor_features(self.model, physical))
        loss_response = F.mse_loss(response_predictions[1:], torch.stack(motor_targets))
        discrepancy = drafts[-1].detach() - labels[-1]
        queried, work = student.perturb_copy(
            current, discrepancy, relative_radius=self.query_radius)
        # Carry actual prior quiet states, not fabricated zero/repeated history.
        teacher_history = history.detach().clone()
        for observed in labels:
            teacher_history = torch.cat((observed[None], teacher_history[:-1]), dim=0)
        query = dict(physical=copy_physical(queried, 'cpu').state_dict(),
                     older_history=teacher_history[1:].detach().cpu().clone(),
                     origin_tick=self.rtc_state.tick + student.horizon,
                     generated_update=self.updates, control='zero-sensory-drive')
        self.rtc_state.replay.append(query)
        self.rtc_state.replay = self.rtc_state.replay[-self.replay_capacity:]
        self.rtc_state.query_ticks += student.horizon
        # Round-robin old/new queries. Targets are re-labeled under current
        # physical weights; labels are never stale across optimizer updates.
        entry = self.rtc_state.replay[self.updates % len(self.rtc_state.replay)]
        device = physical.h.device
        source = FlyPhysicalState(**{
            name: tuple(t.to(device) for t in value) if name == 'ring'
            else value.to(device) for name, value in entry['physical'].items()})
        with torch.no_grad():
            encoded_source = student.codec.encode(source)
            actual_next = quiet_teacher_step(self.model, source, options)
            encoded_target = student.codec.encode(actual_next)
        query_history = torch.cat((encoded_source[None],
                                   entry['older_history'].to(device)), dim=0)
        prediction, _ = student.transition_step(query_history)
        loss_query = F.mse_loss(prediction, encoded_target)
        self.rtc_state.query_ticks += 1
        return loss_rollout + loss_query + loss_response, dict(
            rtc_rollout_mse=loss_rollout.detach(), rtc_query_mse=loss_query.detach(),
            rtc_motor_response_mse=loss_response.detach(),
            rtc_perturbation_work_proxy=work,
            rtc_replay_size=len(self.rtc_state.replay),
            rtc_replay_source_age_updates=self.updates - entry['generated_update'])

    def forward_window(self, input_ids, targets):
        if input_ids.shape[0] != 1:
            raise ValueError('One continuing individual required')
        model, physical, continuing = self.model, self.state, self.rtc_state
        options = physical_options(model)
        features, weights, arrival_losses = [], [], []
        for token in input_ids.unbind(1):
            # At issue time only this arriving token and the old full state
            # are available. Supervise its physical tick with the actual drive,
            # keeping this distinct from all zero-drive future forecasts.
            old_observed = model.rtc_student.codec.encode(physical)
            actual_history = torch.cat((old_observed[None], continuing.history[1:]), dim=0)
            drive, _ = model.topographic_writer.forward_with_state(
                model.embedding(token), physical.h, physical.baseline)
            forecast_arrival, _ = model.rtc_student.transition_step(
                actual_history, model.rtc_student.codec.encode_drive(drive))
            feature, physical, continuing, drafts, attention = read_rtc_event(
                model, physical, continuing, token, options=options)
            arrival_losses.append(F.mse_loss(
                forecast_arrival, continuing.history[0].detach()))
            features.append(feature)
            weights.append(attention)
        self.rtc_state = continuing
        features = torch.cat(features)
        logits = model.decoder(model.read_norm(features))
        scores = F.cross_entropy(logits, targets.flatten(), reduction='none')
        self.last_jepa_loss = None
        self.last_jepa_metrics = {}
        if self.lambda_jepa > 0:
            self.last_jepa_loss, self.last_jepa_metrics = self._physical_queries(
                physical, continuing.history, drafts, options)
            arrival_loss = torch.stack(arrival_losses).mean()
            self.last_jepa_loss = self.last_jepa_loss + arrival_loss
            self.last_jepa_metrics['rtc_real_input_arrival_mse'] = arrival_loss.detach()
        attention = torch.cat(weights).detach()
        self.last_jepa_metrics.update(
            rtc_mean_horizon=float((attention * torch.arange(
                model.rtc_student.horizon + 1, device=attention.device)).sum(-1).mean()),
            rtc_zero_horizon_mass=float(attention[:, 0].mean()))
        return scores, physical, features

    def observe(self, targets):
        self._gradient_norms.clear()
        scores, metrics = super().observe(targets)
        self.rtc_state = self.rtc_state.detached()
        metrics.update({name: float(value) for name, value in self.last_jepa_metrics.items()})
        metrics.update(rtc_query_ticks=self.rtc_state.query_ticks,
                       rtc_speculative_ticks=self.rtc_state.speculative_ticks,
                       rtc_live_ticks=self.rtc_state.tick)
        metrics['rtc_student_grad_norm'] = float(torch.linalg.vector_norm(
            torch.stack(list(self._gradient_norms.values())))) if self._gradient_norms else 0.0
        metrics['rtc_student_parameters'] = sum(
            p.numel() for p in self.model.rtc_student.parameters())
        return scores, metrics

    def state_dict(self):
        saved = super().state_dict()
        saved.update(learning_route=ROUTE, rtc_student=self.rtc_state.state_dict(),
                     replay_capacity=self.replay_capacity, query_radius=self.query_radius)
        return saved

    def restore_learning_state(self, saved):
        if (saved.get('learning_route') != ROUTE
                or saved.get('replay_capacity') != self.replay_capacity
                or saved.get('query_radius') != self.query_radius):
            raise ValueError('RTC route/query configuration mismatch')
        device = self.state.h.device
        payload = saved['physical']
        self.state = FlyPhysicalState(**{
            name: tuple(t.to(device) for t in value) if name == 'ring'
            else value.to(device) for name, value in payload.items()})
        self.rtc_state = RTCStudentState.restore(saved['rtc_student'], device).detached()
        self.load_edge_signs(saved)
        self.optimizer.load_state_dict(saved['optimizer'])
        self.sgd.load_state_dict(saved['sgd'])
        for name in ('events', 'updates', 'physical_ticks', 'previous_token', 'ema'):
            setattr(self, name, saved[name])
        self.latent_window.copy_(saved['latent_window'].to(device))
