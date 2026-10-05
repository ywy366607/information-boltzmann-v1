"""Bounded-memory accounting of every recorded first-pass training target.

This meter observes scores returned by the real learner. It holds no model,
performs no update, and begins at an explicit cursor boundary when installed.
Evaluation and replay traffic must not be submitted to this training meter.
"""
from __future__ import annotations

from collections import deque
import math


class FirstPassPredictionMeter:
    """Keep full totals and a bounded tail of optimizer-window aggregates."""

    def __init__(self, start_train_targets: int, start_train_cursor: int,
                 reference: dict, *, recent_windows: int = 128):
        if min(start_train_targets, start_train_cursor) < 0 or recent_windows < 1:
            raise ValueError('Nonnegative start counters and positive tail size required')
        self.start_train_targets = start_train_targets
        self.start_train_cursor = start_train_cursor
        self.reference = dict(reference)
        self.targets = 0
        self.nll_sum = 0.0
        self.reference_sum = 0.0
        self.recent = deque(maxlen=recent_windows)

    def record(self, scores, reference_scores) -> None:
        scores, reference_scores = list(scores), list(reference_scores)
        if not scores or len(scores) != len(reference_scores):
            raise ValueError('Each recorded target requires a model and reference score')
        if any(not math.isfinite(float(value)) or value < 0
               for value in (*scores, *reference_scores)):
            raise ValueError('Finite nonnegative negative-log-likelihood scores required')
        nll = math.fsum(scores)
        reference_nll = math.fsum(reference_scores)
        self.targets += len(scores)
        self.nll_sum = math.fsum((self.nll_sum, nll))
        self.reference_sum = math.fsum((self.reference_sum, reference_nll))
        self.recent.append((len(scores), nll, reference_nll))

    def verify_cursor(self, train_targets: int, train_cursor: int) -> None:
        if (train_targets != self.start_train_targets + self.targets
                or train_cursor != self.start_train_cursor + self.targets):
            raise ValueError('First-pass accounting must cover exactly the fresh training cursor')

    def summary(self) -> dict:
        tail_count = sum(row[0] for row in self.recent)
        tail_nll = math.fsum(row[1] for row in self.recent)
        tail_reference = math.fsum(row[2] for row in self.recent)
        return {
            'first_pass_measurement_start_bptt_tokens': self.start_train_targets,
            'first_pass_measurement_start_train_cursor': self.start_train_cursor,
            'first_pass_measured_train_targets': self.targets,
            'first_pass_train_nll': self.nll_sum / self.targets if self.targets else None,
            'first_pass_fixed_unigram_nll': self.reference_sum / self.targets if self.targets else None,
            'first_pass_gain_over_fixed_unigram': (
                (self.reference_sum - self.nll_sum) / self.targets if self.targets else None),
            'recent_first_pass_measured_targets': tail_count,
            'recent_first_pass_train_nll': tail_nll / tail_count if tail_count else None,
            'recent_first_pass_fixed_unigram_nll': tail_reference / tail_count if tail_count else None,
            'recent_first_pass_gain_over_fixed_unigram': (
                (tail_reference - tail_nll) / tail_count if tail_count else None),
        }

    def state_dict(self) -> dict:
        return {
            'format': 'first-pass-meter-v1',
            'start_train_targets': self.start_train_targets,
            'start_train_cursor': self.start_train_cursor,
            'reference': dict(self.reference),
            'targets': self.targets,
            'nll_sum': self.nll_sum,
            'reference_sum': self.reference_sum,
            'recent_windows': self.recent.maxlen,
            'recent': list(self.recent),
        }

    @classmethod
    def from_state_dict(cls, state: dict, reference: dict) -> FirstPassPredictionMeter:
        if state['format'] != 'first-pass-meter-v1' or state['reference'] != reference:
            raise ValueError('Continue the same meter format and fixed reference')
        meter = cls(state['start_train_targets'], state['start_train_cursor'], reference,
                    recent_windows=state['recent_windows'])
        meter.targets = state['targets']
        meter.nll_sum = state['nll_sum']
        meter.reference_sum = state['reference_sum']
        if meter.targets < 0 or any(not math.isfinite(value) or value < 0
                                  for value in (meter.nll_sum, meter.reference_sum)):
            raise ValueError('Invalid restored first-pass totals')
        if len(state['recent']) > meter.recent.maxlen:
            raise ValueError('Restored tail exceeds its registered capacity')
        for count, nll, reference_nll in state['recent']:
            if count < 1 or any(not math.isfinite(value) or value < 0
                               for value in (nll, reference_nll)):
                raise ValueError('Invalid restored first-pass tail')
            meter.recent.append((count, nll, reference_nll))
        if sum(row[0] for row in meter.recent) > meter.targets:
            raise ValueError('Restored tail exceeds total measured targets')
        return meter
