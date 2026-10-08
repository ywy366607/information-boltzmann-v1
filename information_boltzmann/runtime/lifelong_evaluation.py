"""Live test-then-learn evaluation, shared by persistent online learners.

No model/eval switch, state reset, learning freeze or separate update rule.
The learner returns pre-update predictive NLL; this runner records it before
the optimizer acts. Optimizer cadence continues across every stream boundary.
"""
from __future__ import annotations

from collections import deque
import copy
import math
import sys

import torch


def block_curve(values, block_tokens):
    if block_tokens < 1:
        raise ValueError('Positive measurement block size required')
    return [{'end_token': start + len(values[start:start + block_tokens]),
             'mean_nll': sum(values[start:start + block_tokens]) / len(values[start:start + block_tokens])}
            for start in range(0, len(values), block_tokens)
            if len(values[start:start + block_tokens]) == block_tokens]


def sustained_crossing(blocks, threshold, hold_blocks):
    """Return confirmation time; one unusually easy token cannot imply recovery."""
    if hold_blocks < 1:
        raise ValueError('Positive measurement confirmation length required')
    consecutive = 0
    for block in blocks:
        consecutive = consecutive + 1 if block['mean_nll'] <= threshold else 0
        if consecutive >= hold_blocks:
            return block['end_token']
    return None


def adaptation_generalization_summary(values, *, block_tokens, hold_blocks,
                                     plateau_blocks=None, plateau_tolerance_nll=0.1,
                                     reference_nll=None):
    """Joint recovery speed and observed terminal quality on the SAME live B.

    Tolerance is declared measurement resolution (nats/token), not a physical
    parameter. A fixed trailing window must pass both trend and half-window
    drift checks. This establishes observed tail stability, not an asymptote.
    Canonical times require a stable tail and sustained recovery without later
    relapse. No model state, optimizer, learning rate or experience is changed.
    """
    values = [float(x) for x in values]
    if any(not math.isfinite(x) for x in values):
        raise ValueError('Finite predictive NLL required')
    if hold_blocks < 1 or block_tokens < 1:
        raise ValueError('Positive measurement block/confirmation required')
    tail_size = max(4, 2 * hold_blocks) if plateau_blocks is None else int(plateau_blocks)
    if tail_size < max(4, 2 * hold_blocks):
        raise ValueError('Plateau needs two halves and two confirmation windows')
    if not math.isfinite(plateau_tolerance_nll) or plateau_tolerance_nll <= 0:
        raise ValueError('Positive finite plateau measurement tolerance required')
    reference = None if reference_nll is None else [float(x) for x in reference_nll]
    if reference is not None and (len(reference) != len(values)
                                 or any(not math.isfinite(x) for x in reference)):
        raise ValueError('Reference must score the exact same finite targets')
    blocks = block_curve(values, block_tokens)
    result = {
        'protocol': 'live_adaptation_speed_quality_v2',
        'definition': 'generalization = recovery time AND adapted predictive NLL',
        'scored_tokens': len(values), 'block_tokens': block_tokens,
        'confirmation_blocks': hold_blocks, 'plateau_blocks': tail_size,
        'plateau_tolerance_nll': float(plateau_tolerance_nll),
        'status': 'insufficient_plateau_observation', 'plateau_status': 'insufficient_blocks',
        'plateau_nll': None, 'plateau_estimate_nll': None,
        'recovery_tokens': None, 'half_recovery_tokens': None,
        'adaptation_mean_nll': sum(values) / len(values) if values else None,
        'adaptation_cumulative_nll': sum(values),
        'comparison': 'same target order, exposure and optimizer budgets; lower time and NLL preferred',
    }
    mean = result['adaptation_mean_nll']
    result['appl_h'] = math.exp(mean) if mean is not None and mean <= math.log(sys.float_info.max) else None
    result['log_appl_h'] = mean
    result['log_gain_over_reference'] = None
    result['gain_ratio_over_reference'] = None
    if reference:
        log_gain = sum(reference) / len(reference) - mean
        result['reference_mean_nll'] = sum(reference) / len(reference)
        result['log_gain_over_reference'] = log_gain
        result['gain_ratio_over_reference'] = math.exp(log_gain) if log_gain <= math.log(sys.float_info.max) else None
    if len(blocks) < tail_size + 1:
        return result
    tail = blocks[-tail_size:]
    ys = [b['mean_nll'] for b in tail]
    late = sum(ys) / tail_size
    midpoint = (tail_size - 1) / 2
    slope = sum((i - midpoint) * (y - late) for i, y in enumerate(ys)) / sum(
        (i - midpoint) ** 2 for i in range(tail_size))
    split = tail_size // 2
    drift = sum(ys[split:]) / (tail_size - split) - sum(ys[:split]) / split
    trend = slope * (tail_size - 1)
    stable = abs(drift) <= plateau_tolerance_nll and abs(trend) <= plateau_tolerance_nll
    start = tail[0]['end_token'] - block_tokens
    end = tail[-1]['end_token']
    opening = blocks[0]['mean_nll']
    result.update(plateau_estimate_nll=late, initial_block_nll=opening,
                  plateau_interval_tokens=[start, end],
                  trailing_unblocked_tokens=len(values) - blocks[-1]['end_token'],
                  plateau_trend_nll=trend, plateau_half_window_drift_nll=drift,
                  plateau_block_std_nll=(sum((x - late) ** 2 for x in ys) / (tail_size - 1)) ** .5,
                  plateau_status='observed_stable_tail' if stable else 'still_changing',
                  status='observed_adaptation' if stable else 'plateau_unconfirmed')
    if reference is not None:
        baseline = sum(reference[start:end]) / (end - start)
        result.update(plateau_reference_nll=baseline,
                      plateau_estimate_gain_over_reference=baseline - late)
    if not stable:
        return result
    result['plateau_nll'] = late
    result['recovery_threshold_nll'] = late + plateau_tolerance_nll
    def final_sustained_crossing(threshold):
        # A temporary dip followed by a later failure cannot count as recovery.
        for i in range(len(blocks) - hold_blocks + 1):
            if all(b['mean_nll'] <= threshold for b in blocks[i:]):
                return blocks[i + hold_blocks - 1]['end_token']
        return None
    if all(b['mean_nll'] <= late + plateau_tolerance_nll for b in blocks):
        result.update(recovery_tokens=0, half_recovery_tokens=0,
                      status='already_at_observed_plateau')
    else:
        result['recovery_tokens'] = final_sustained_crossing(late + plateau_tolerance_nll)
        if opening - late > plateau_tolerance_nll:
            result['half_recovery_tokens'] = final_sustained_crossing((opening + late) / 2)
        if result['recovery_tokens'] is None:
            result['status'] = 'recovery_right_censored'
    if reference is not None and result.get('recovery_tokens') is not None:
        p_est = result['plateau_estimate_nll']
        p_ref = result.get('plateau_reference_nll', p_est)
        tau_val = float(result.get('recovery_tokens') if result.get('recovery_tokens') is not None else len(values))
        tau_ref = float(len(values))
        delta_L = p_est - p_ref
        denom = 2.0 + (tau_val / tau_ref) + math.exp(delta_L)
        result['ag'] = 2.0 / denom
        result['s_speed'] = 1.0 / (1.0 + tau_val / tau_ref)
        result['q_quality'] = 1.0 / (1.0 + math.exp(delta_L))
        result['tau_tokens'] = tau_val
        result['tau_ref_tokens'] = tau_ref
    return result


def recovery_summary(values, *, block_tokens, hold_blocks, plateau_blocks=None,
                     plateau_tolerance_nll=0.1, reference_nll=None):
    values = [float(x) for x in values]
    generalization = adaptation_generalization_summary(
        values, block_tokens=block_tokens, hold_blocks=hold_blocks,
        plateau_blocks=plateau_blocks, plateau_tolerance_nll=plateau_tolerance_nll,
        reference_nll=reference_nll)
    blocks = block_curve(values, block_tokens)
    result = {'generalization': generalization, 'legacy_half_recovery': True, 'tokens': len(values), 'block_tokens': block_tokens,
              'confirmation_blocks': hold_blocks, 'block_curve': blocks,
              'half_recovery_tokens': None, 'status': 'insufficient_blocks'}
    if len(blocks) < hold_blocks + 2:
        return result
    first = blocks[0]['mean_nll']
    late = sum(b['mean_nll'] for b in blocks[-hold_blocks:]) / hold_blocks
    gap = first - late
    result.update(initial_block_nll=first, observed_late_nll=late, observed_drop=gap)
    if gap <= 0:
        result['status'] = 'no_observed_improvement'
        return result
    threshold = (first + late) / 2
    crossing = sustained_crossing(blocks[1:], threshold, hold_blocks)
    result.update(half_recovery_tokens=crossing, threshold_nll=threshold,
                  status='observed_recovery' if crossing is not None else 'right_censored')
    if 'ag' in generalization:
        result.update(ag=generalization['ag'],
                      s_speed=generalization['s_speed'],
                      q_quality=generalization['q_quality'],
                      appl_h=generalization.get('appl_h'),
                      gain_ratio_over_reference=generalization.get('gain_ratio_over_reference'))
    return result


def savings_summary(initial, revisit, *, intervening_events, block_tokens, hold_blocks):
    if len(initial) != len(revisit) or not initial:
        raise ValueError('Match the exact initial/revisit targets and score length')
    a, b = sum(initial) / len(initial), sum(revisit) / len(revisit)
    opening = min(block_tokens, len(initial))
    result = {'scored_tokens_per_encounter': len(initial),
              'actual_intervening_events': intervening_events,
              'initial_nll': a, 'revisit_nll': b, 'nll_saving': a - b,
              'relative_nll_saving': (a - b) / a if a > 0 else None,
              'opening_nll_saving': (sum(initial[:opening]) - sum(revisit[:opening])) / opening,
              'initial_threshold_tokens': None, 'revisit_threshold_tokens': None,
              'relearning_acceleration': None}
    recovery = recovery_summary(initial, block_tokens=block_tokens, hold_blocks=hold_blocks)
    if 'threshold_nll' in recovery:
        # The SAME absolute criterion is used for both encounters.
        threshold = recovery['threshold_nll']
        t_a = sustained_crossing(block_curve(initial, block_tokens), threshold, hold_blocks)
        t_b = sustained_crossing(block_curve(revisit, block_tokens), threshold, hold_blocks)
        result.update(shared_threshold_nll=threshold, initial_threshold_tokens=t_a,
                      revisit_threshold_tokens=t_b,
                      relearning_acceleration=t_a / t_b if t_a is not None and t_b else None)
    return result


class LiveOnlineEvaluation:
    """Own scoring/optimizer cadence while the same individual keeps learning.

    `backward_event` must compute a causal prediction before any optimizer
    update. Targets enter only its loss. This is the contract of both existing
    3D online learners and their captured wrappers.
    """

    def __init__(self, learner, optimizer, *, tokens_per_update, window_tokens=128,
                 captured=None, gradient_clip=1.0, carry_token=None, health=None):
        if tokens_per_update < 1 or window_tokens < 1 or gradient_clip <= 0:
            raise ValueError('Positive optimizer/measurement budgets required')
        self.learner, self.optimizer, self.captured = learner, optimizer, captured
        self.tokens_per_update, self.window_tokens = int(tokens_per_update), int(window_tokens)
        self.gradient_clip = float(gradient_clip)
        self.health = health
        if health is not None and getattr(learner, 'health_capture', None) is not health.capture:
            raise ValueError('Attach health capture before constructing the event CUDA graph')
        self.pending = 0
        self.optimizer_updates = 0
        self.events = 0
        self.novel_events = 0
        self.cumulative_nll = 0.0
        self.novel_nll = 0.0
        self.recent = deque(maxlen=window_tokens)
        self.recent_events = deque(maxlen=window_tokens)
        self.carry_token = carry_token
        self.optimizer.zero_grad(set_to_none=captured is None)

    def step(self, observed, target, *, novel=True, phase='stream'):
        observed_id, target_id = int(observed.item()), int(target.item())
        if self.carry_token is not None and observed_id != self.carry_token:
            raise ValueError('Stream discontinuity: score the bridge token before changing context')
        with torch.enable_grad():
            if self.captured is None:
                metrics = self.learner.backward_event(observed, target, loss_scale=1 / self.tokens_per_update)
            else:
                metrics = self.captured.backward(observed, target)
        nll = float(metrics['token_nll'])
        if not math.isfinite(nll):
            raise FloatingPointError('Nonfinite pre-update predictive NLL')
        # Commit score BEFORE an optimizer update. Target is now observed and
        # may train future predictions; this prediction is never rescored.
        self.events += 1
        self.cumulative_nll += nll
        self.recent.append(nll)
        self.recent_events.append((observed_id, target_id, nll, bool(novel)))
        if novel:
            self.novel_events += 1
            self.novel_nll += nll
        self.carry_token = target_id
        if self.health is not None:
            self.health.record_event(nll, novel=novel, phase=phase)
        self.pending += 1
        if self.pending == self.tokens_per_update:
            parameters = tuple(self.learner.model.parameters())
            previous = tuple(p.detach().clone() for p in parameters) if self.health is not None else None
            norm = torch.nn.utils.clip_grad_norm_(self.learner.model.parameters(), self.gradient_clip,
                                                  error_if_nonfinite=True)
            self.optimizer.step()
            if previous is not None:
                with torch.no_grad():
                    update_norm = sum((p - old).square().sum() for p, old in zip(parameters, previous)).sqrt()
                    parameter_norm = sum(p.square().sum() for p in parameters).sqrt()
                self.health.record_update(norm, update_norm, parameter_norm)
            self.optimizer.zero_grad(set_to_none=self.captured is None)
            self.optimizer_updates += 1
            self.pending = 0
            metrics['gradient_norm_before_clip'] = norm.detach()
        return metrics

    def observe(self, target, *, novel=True, phase='stream'):
        if self.carry_token is None:
            raise ValueError('A real preceding observation is required for next-token scoring')
        observed = target.new_tensor([self.carry_token])
        return self.step(observed, target.reshape(1), novel=novel, phase=phase)

    def summary(self):
        return {'events': self.events, 'optimizer_updates': self.optimizer_updates,
                'pending_gradient_events': self.pending, 'novel_events': self.novel_events,
                'prequential_nll': self.cumulative_nll / self.events if self.events else None,
                'novel_prequential_nll': self.novel_nll / self.novel_events if self.novel_events else None,
                'window_nll': sum(self.recent) / len(self.recent) if self.recent else None}

    def revisit_after_change(self, a_sequence, initial_nll, b_targets, *, block_tokens, hold_blocks):
        """Finish a real A1->B->A2 trajectory on the live individual.

        A1 has ALREADY occurred; pass its recorded scores and exact observations.
        B's first target is scored from A's final observation. Returning to A
        adds one scored bridge to A[0], then replays the identical scored pairs.
        No optimizer flush or state/weight restoration occurs at phase edges.
        """
        if len(a_sequence) != len(initial_nll) + 1 or not initial_nll or len(b_targets) < 1:
            raise ValueError('Actual A1 scores, A observations and nonempty B required')
        if self.carry_token != int(a_sequence[-1]):
            raise ValueError('A1 must immediately precede B in this individual history')
        recorded = list(self.recent_events)[-len(initial_nll):]
        expected = [(int(left), int(right), float(score), True)
                    for left, right, score in zip(a_sequence[:-1], a_sequence[1:], initial_nll)]
        if recorded != expected:
            raise ValueError('A1 observations and scores must match the actual first-pass ledger')
        start_event, start_update = self.events, self.optimizer_updates
        health_before = self.health.summary() if self.health is not None else None
        b_curve = [float(self.observe(token, novel=True, phase='change_B')['token_nll']) for token in b_targets]
        health_after_b = self.health.summary() if self.health is not None else None
        bridge_nll = float(self.observe(a_sequence[0], novel=False, phase='return_bridge')['token_nll'])
        intervening = self.events - start_event
        revisit = [float(self.observe(token, novel=False, phase='revisit_A')['token_nll']) for token in a_sequence[1:]]
        result = {'protocol': 'live_prequential_A_B_A_v1',
                'learning_active': True, 'reset': False,
                'event_start': start_event, 'event_end': self.events,
                'optimizer_updates_during_protocol': self.optimizer_updates - start_update,
                'b_events': len(b_curve), 'return_bridge_nll': bridge_nll,
                'shock': recovery_summary(b_curve, block_tokens=block_tokens, hold_blocks=hold_blocks),
                'savings': savings_summary(initial_nll, revisit, intervening_events=intervening,
                                           block_tokens=block_tokens, hold_blocks=hold_blocks),
                'initial_curve': list(initial_nll), 'b_curve': b_curve, 'revisit_curve': revisit,
                'prequential': self.summary()}
        if self.health is not None:
            result['health'] = self.health.summary()
            result['health_before_change'] = health_before
            result['health_after_change'] = health_after_b
        return result

    def state_dict(self):
        values = self.summary()
        return {'version': 1, 'tokens_per_update': self.tokens_per_update,
                'window_tokens': self.window_tokens, 'gradient_clip': self.gradient_clip,
                'events': values['events'], 'optimizer_updates': values['optimizer_updates'],
                'pending': self.pending, 'novel_events': self.novel_events,
                'cumulative_nll': self.cumulative_nll, 'novel_nll': self.novel_nll,
                'recent': list(self.recent), 'carry_token': self.carry_token,
                'recent_events': list(self.recent_events),
                'health': self.health.state_dict() if self.health is not None else None,
                'pending_gradients': {n: None if p.grad is None else p.grad.detach().clone()
                                      for n, p in self.learner.model.named_parameters()}}

    def load_state_dict(self, saved):
        if (saved['version'] != 1 or saved['tokens_per_update'] != self.tokens_per_update
                or saved['window_tokens'] != self.window_tokens or saved['gradient_clip'] != self.gradient_clip
                or not 0 <= saved['pending'] < self.tokens_per_update):
            raise ValueError('Live evaluator continuation policy mismatch')
        for name in ('events', 'optimizer_updates', 'pending', 'novel_events',
                     'cumulative_nll', 'novel_nll', 'carry_token'):
            setattr(self, name, copy.deepcopy(saved[name]))
        self.recent = deque(saved['recent'], maxlen=self.window_tokens)
        self.recent_events = deque(saved.get('recent_events', []), maxlen=self.window_tokens)
        if saved.get('health') is not None:
            if self.health is None:
                raise ValueError('Resume requires the saved fourth-pillar auditor')
            self.health.load_state_dict(saved['health'])
        for name, parameter in self.learner.model.named_parameters():
            gradient = saved['pending_gradients'][name]
            if gradient is None:
                if parameter.grad is not None:
                    parameter.grad.zero_()
            elif parameter.grad is None:
                if self.captured is not None:
                    raise ValueError('Captured gradient layout differs from continuation')
                parameter.grad = gradient.to(parameter).clone()
            else:
                parameter.grad.copy_(gradient.to(parameter))
