"""Opt-in native categorical perception/action learning, not a shadow auditor.

The brain's actual categorical prediction supplies external evidence. Finite
native-expert information proposes wait/token actions; actual external response
evidence returns through the exact joint policy score to existing parameters.
Sensory control learns from real external CE through the physical input path.
No count novelty, fixed speech quota or fabricated DAN current is added.
"""
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from .fly_native_generative import NativeCategoricalWorld, _clone_state
from .fly_native_policy_credit import native_parameter_names, prepare_native_policy_backward


FORMAT = 'native-categorical-control-v1'


def _cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_cpu(v) for v in value)
    return value


class NativeGenerativeControl:
    def __init__(self, learner):
        self.learner, self.model = learner, learner.model
        if getattr(learner, 'mother_action_clock', None) != 'commit':
            raise ValueError('Native generative control requires the real commit event clock')
        if getattr(learner, 'eye_binary', False):
            raise ValueError('This controller uses the existing continuous sensory actuator')
        if getattr(learner, 'pending_speak_credit', None) is not None:
            raise ValueError('Drain the pending legacy head credit before changing its objective')
        prepare_native_policy_backward(self.model)
        if getattr(learner, 'native_generative_state', None) is None:
            path = getattr(learner, 'native_generative_reference', None)
            if path is None:
                raise ValueError('Record an explicit retained native checkpoint as the hypothesis reference')
            path = Path(path)
            saved = torch.load(path, mmap=True, map_location='cpu', weights_only=False)
            if saved.get('format') not in ('fly-bptt-v1', 'fly-native-reference-v1'):
                raise ValueError('Native reference must be a complete fly checkpoint')
            tensors = saved['model'] if saved['format'] == 'fly-bptt-v1' else saved['parameters']
            reference = {name: tensors[name] for name in native_parameter_names(self.model)}
            self.world_model = NativeCategoricalWorld(learner, reference, {
                'path': str(path.resolve()), 'trained': saved.get('bptt_train_tokens'),
                'origin': saved.get('provenance'),
                'size': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns})
            del saved
        else:
            self.world_model = NativeCategoricalWorld(learner)
        durable = self.world_model.state
        old = getattr(learner, 'consequence_state', {})
        if old.get('format') != FORMAT:
            if old.get('response') is not None or old.get('native_credit', {}).get('response_score') is not None:
                raise ValueError('Drain the real pending legacy response before changing its likelihood')
            # Preserve the old objective's traces/history as an explicit mechanism
            # archive; do not assign a new meaning to derivatives of old actions.
            durable['legacy_consequence'] = _cpu(old)
            clock = old.get('native_credit', {}).get('clock_ticks', 0)
            learner.consequence_state = {'format': FORMAT, 'native_credit': {
                'format': 'native-policy-score-v1', 'E': {}, 'clock_ticks': clock}}
        self.state = durable.setdefault('control', {
            'source': old.get('source', 'unknown'), 'in_child_turn': False,
            'trial_active': False, 'commands': 0, 'response': None,
            'clock_ticks': learner.consequence_state['native_credit'].get('clock_ticks', 0),
            'settlements': 0, 'cancelled': 0})
        self.native_events, self.timing_biases = [], []
        self.decisions, self.reports, self.plans = [], [], []
        self.decision_bounds = (0, 0)
        self.word_bias = None
        self.wait_value = 0.
        self.temperature = 1.
        self.enabled = False
        self.pending_interval = None

    def features(self, physical):
        # Interface marker only. Values use actual native physical rollouts,
        # never an external statistical projection of KC state.
        return physical.h.new_empty(0), None

    def input_ratio(self, source):
        return 0.

    def observe(self, *args, **kwargs):
        pass

    def begin_child_turn(self):
        if self.state['response'] is not None:
            raise ValueError('Finish the actual teacher response before opening another causal trial')
        if not self.state['trial_active']:
            self.state.update(trial_active=True, commands=0)
        self.state['in_child_turn'] = True

    def begin_interval(self, physical, token, voice, reward):
        if self.pending_interval is not None:
            raise ValueError('A native external prediction interval is already pending')
        self.pending_interval = (_clone_state(physical), int(token), bool(voice), bool(reward))

    @torch.no_grad()
    def prepare(self, context, physical, enabled=True, mother_turn=False, temperature=1.):
        from .fly_native_motor_port import read_action, action_logits
        self.enabled = bool(enabled)
        self.temperature = float(temperature)
        self.word_bias = None
        if not self.enabled:
            return
        if not self.state['trial_active']:
            self.begin_child_turn()
        logits = action_logits(self.model, read_action(self.model, physical)).float()[0]
        count = min(int(getattr(self.learner, 'native_generative_candidates', 4)), logits.numel())
        if count < 1:
            raise ValueError('Native planning needs a positive proposal budget')
        candidates = torch.topk(logits, count).indices.tolist()
        ticks = int(getattr(self.learner, 'native_generative_ticks', 0) or
                    max(1, round(self.learner.stream_pace_mean)))
        # The gain is the real current actuator setting, not inverse noise.
        gain = float(getattr(self, 'initial_eye', 1.))
        forecast = self.world_model.command_values(physical, candidates, ticks, eye_gain=gain)
        self.wait_value = float(forecast['wait'])
        self.word_bias = torch.full_like(logits, self.wait_value)
        self.word_bias[candidates] = forecast['words'].to(logits)
        self.plans.append({'candidate_ids': candidates, 'wait_information': self.wait_value,
                           'word_information': forecast['words'].cpu().tolist(), 'lookahead_ticks': ticks,
                           'unassessed_tail': 'retained_native_mass_neutral_reference',
                           'scope': 'finite_next_external_word_proposal', 'applied': True})

    @torch.no_grad()
    def timing_bias(self, physical, muscles):
        """Called after this exact tick's muscles/read, before its real sample."""
        if not self.enabled or self.word_bias is None:
            value = 0.
        else:
            from .fly_native_motor_port import read_action, action_logits
            logits = action_logits(self.model, read_action(self.model, physical)).float()[0]
            lp = F.log_softmax(logits / self.temperature, -1)
            value = float(torch.logsumexp(lp + self.word_bias, -1) - self.wait_value)
        self.timing_biases.append(value)
        return value

    def advance_ticks(self, ticks):
        self.native_events.append({'kind': 'clock', 'ticks': int(ticks)})
        self.state['clock_ticks'] += int(ticks)

    def choose(self, context, frames, first_commit, mother_turn=False):
        start, length = self.decision_bounds
        if self.word_bias is None:
            raise ValueError('A real decision needs its committed pre-action proposal')
        self.native_events.append({'kind': 'joint_timing', 'start': start, 'length': length,
                                   'first': first_commit, 'tail_ticks': 0,
                                   'word_value': self.word_bias.detach(), 'wait_value': self.wait_value,
                                   'temperature': self.temperature})
        self.decisions.append(first_commit is not None)
        self.content_tick = start + (first_commit if first_commit is not None else length - 1)

    def say(self, read, word, temperature):
        self.native_events.append({'kind': 'content', 'tick': self.content_tick,
                                   'word': int(word), 'temperature': float(temperature),
                                   'tail_ticks': 0, 'logit_bias': self.word_bias.detach()})
        self.state['commands'] += 1

    def end_response(self):
        response = self.state['response']
        if response is not None:
            prior_log = response['prior_log_weights']
            after_log = self.world_model.state['log_weights']
            after = after_log.exp()
            gain = float(torch.where(after > 0, after * (after_log - prior_log),
                                     torch.zeros_like(after)).sum().clamp_min(0))
            self.native_events.append({'kind': 'response_end', 'advantage': gain})
            self.state['settlements'] += 1
        if self.state['trial_active']:
            self.native_events.append({'kind': 'trial_end'})
        self.state.update(response=None, trial_active=False, commands=0)

    def arrive(self, physical, word, source, ticks, quiet_ranges, eyes, end):
        """Forecast from already received input; target enters only after prediction."""
        if self.pending_interval is None:
            raise ValueError('Missing native pre-observation interval snapshot')
        before, incoming, voice, reward = self.pending_interval
        if source == 'self_hearing':
            report = self.world_model.update(None, word, source)
        else:
            predicted = self.world_model.predict(before, incoming, int(ticks), voice=voice, reward=reward,
                                                  quiet_ranges=quiet_ranges, eye_tape=eyes)
            # This hypothesis is the actual learner, so its pre-target forecast
            # must match the realized endpoint under the recorded sensory tape.
            from .fly_bptt_learning import read_fly_motor
            with torch.no_grad():
                actual = F.log_softmax(self.model.decoder(self.model.read_norm(
                    read_fly_motor(self.model, physical))).float(), -1)[0]
                replay_error = float((predicted[0] - actual).abs().max())
            if replay_error > 2e-4:
                raise RuntimeError(f'Native evidence forecast differs from actual sensing: {replay_error}')
            causal = self.state['trial_active'] and (
                (source == 'mother_reply' and self.state['commands'] > 0) or
                (source == 'mother_call' and self.state['commands'] == 0))
            # A new unrelated source cannot be charged to the preceding reply.
            if self.state['trial_active'] and not causal:
                self.end_response()
                self.state['cancelled'] += 1
            if causal and self.state['response'] is None:
                self.state['response'] = {
                    'prior_log_weights': self.world_model.state['log_weights'].clone(), 'source': source}
                self.native_events.append({'kind': 'response_begin'})
            report = self.world_model.update(predicted, word, source)
            report['actual_forecast_logp_maxabs_error'] = replay_error
            report['credited_to_trial'] = causal
            if causal and end:
                self.end_response()
        self.pending_interval = None
        self.reports.append(report)

    def finish(self):
        if self.pending_interval is not None:
            raise ValueError('Do not save a partly evaluated native interval as complete')
        self.learner.last_epistemic = {
            'source': 'native_categorical_active_control', 'policy_actuated': True,
            'perception_learning': 'actual_external_ce_through_native_senses',
            'count_eye_reward': False, 'fixed_speech_kl': False,
            'hypothesis_weights': self.world_model.weights.tolist(),
            'external_updates': self.world_model.state['external_updates'],
            'decisions': len(self.decisions), 'commits': sum(self.decisions),
            'settlements': self.state['settlements'], 'plans': self.plans,
            'evidence': self.reports, 'finite_expert_assumption': True}
