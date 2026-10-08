"""Timestamped forecast handoff, independent of wall-clock worker scheduling.

This is RTC-inspired scheduling for tick-response drafts, not the flow-policy
inpainting algorithm of the original RTC paper. A driver can submit drafts and
late teacher results from separate workers. No GPU concurrency is promised.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from threading import RLock
import torch


def synchronized(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return call


@dataclass(frozen=True)
class ForecastKey:
    origin_tick: int
    target_tick: int
    revision: int
    input_cutoff: int
    model_revision: int = 0


class TickForecastBuffer:
    """Bounded uncommitted suffix; executed outputs are immutable."""

    def __init__(self, horizon: int = 14):
        if horizon < 1:
            raise ValueError('Positive cache horizon required')
        self.horizon = horizon
        self._lock = RLock()
        self.committed_tick = -1
        self.input_cutoff = -1
        self.revision = 0
        self.model_revision = 0
        self.pending = {}
        self.issued = {}
        self.arrival_count = 0
        self.arrival_squared_error = 0.0

    @synchronized
    def new_input(self, tick: int):
        if tick <= self.input_cutoff:
            raise ValueError('Observation timestamps must increase')
        self.input_cutoff = tick
        self.revision += 1
        # Future plans assumed no new input. The issued ledger survives so
        # a genuine matched arrival can still be evaluated before correction.
        # An input admitted at origin t can influence responses after t. Keep
        # earlier causally valid outputs even if the consumer has not read them.
        self.pending = {target: value for target, value in self.pending.items()
                        if target <= tick}
        self.issued = {key: value for key, value in self.issued.items()
                       if key.target_tick >= tick - self.horizon}
        return self.revision

    @synchronized
    def revise_suffix(self, keep_through_tick: int):
        self.revision += 1
        self.pending = {target: value for target, value in self.pending.items()
                        if target <= keep_through_tick}
        return self.revision

    @synchronized
    def new_model(self, revision: int):
        if revision <= self.model_revision:
            raise ValueError('Model revisions must increase')
        self.model_revision = revision
        self.revision += 1
        self.pending.clear()
        self.issued.clear()

    @synchronized
    def publish(self, origin_tick, values, *, revision, input_cutoff,
                model_revision=None):
        model_revision = self.model_revision if model_revision is None else model_revision
        if (revision != self.revision or input_cutoff != self.input_cutoff
                or model_revision != self.model_revision):
            return False
        if origin_tick < input_cutoff or not 1 <= len(values) <= self.horizon:
            raise ValueError('Draft origin or forecast horizon is inconsistent')
        self.issued = {key: value for key, value in self.issued.items()
                       if key.target_tick >= origin_tick - self.horizon}
        for offset, value in enumerate(values, start=1):
            key = ForecastKey(origin_tick, origin_tick + offset, revision,
                              input_cutoff, model_revision)
            if key.target_tick > self.committed_tick:
                self.pending[key.target_tick] = (key, value.detach().clone())
                self.issued[key] = value.detach().clone()
        return True

    @synchronized
    def commit(self, target_tick):
        if target_tick != self.committed_tick + 1:
            raise ValueError('Commit physical output ticks consecutively')
        if target_tick not in self.pending:
            raise LookupError('No causally valid draft; request a fresh prediction')
        key, value = self.pending.pop(target_tick)
        self.committed_tick = target_tick
        return key, value.clone()

    @synchronized
    def arrive(self, key: ForecastKey, actual, *, same_input_history: bool):
        """Score an unchanged issued prediction before replacing future output.

        Real future stimuli and zero-drive counterfactual forecasts are distinct
        branches. Drivers must explicitly confirm the input-history contract.
        Unmatched arrivals can seed a new plan, but cannot validate this draft.
        """
        if not same_input_history or key.model_revision != self.model_revision:
            return None
        issued = self.issued.get(key)
        if issued is None:
            return None
        error = float((issued - actual.detach()).square().mean())
        self.arrival_count += 1
        self.arrival_squared_error += error
        candidate = self.pending.get(key.target_tick)
        if (candidate is not None
                and candidate[0].input_cutoff == key.input_cutoff
                and candidate[0].model_revision == key.model_revision):
            # A reforecast after arrival may have a newer plan revision while
            # representing the very same input history and physical target.
            self.pending[key.target_tick] = (candidate[0], actual.detach().clone())
        # A committed output stays committed. Late/stale results cannot replace
        # a newer revision or rewind an already-spoken token or physical action.
        self.issued.pop(key)
        return error

    @synchronized
    def state_dict(self):
        return dict(horizon=self.horizon, committed_tick=self.committed_tick,
                    input_cutoff=self.input_cutoff, revision=self.revision,
                    model_revision=self.model_revision,
                    pending=self.pending, issued=self.issued,
                    arrival_count=self.arrival_count,
                    arrival_squared_error=self.arrival_squared_error)

    @classmethod
    def from_state_dict(cls, saved):
        expected = set(cls().state_dict())
        if set(saved) != expected:
            raise ValueError('Incomplete RTC execution continuation')
        obj = cls(saved['horizon'])
        for name, value in saved.items():
            setattr(obj, name, value)
        return obj


class FlyRTCExecutor:
    """Background physical validation of drafts during zero-drive execution.

    Called after a real input event supplies a complete physical snapshot and
    observed latent history. New stimuli invalidate the pending suffix; the
    caller processes that input and stages a fresh plan. Physical verification
    never advances the live individual. Output can be consumed while this
    worker is running. Whether GPU kernels overlap is backend-dependent.

    Stage/commit/update calls belong to one execution thread. Parameter changes
    go through update_parameters, which waits for teacher queries and prevents
    weights from changing mid-query. No deepcopy of the large parameter set.
    """

    def __init__(self, model, *, origin_tick=0):
        from concurrent.futures import ThreadPoolExecutor
        if model.training:
            raise ValueError('Use evaluation mode for asynchronous response execution')
        self.model = model
        self.buffer = TickForecastBuffer(model.rtc_student.horizon)
        self.buffer.committed_tick = origin_tick
        self._teacher_lock = RLock()
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix='fly-rtc')
        self._future = None

    def stage(self, physical, history, origin_tick, *, input_cutoff):
        """Stage a fast forecast, return immediately with a future audit handle."""
        from information_boltzmann.core.fly_rtc_learning import (
            copy_physical, physical_options, quiet_teacher_step, motor_features)
        if input_cutoff > self.buffer.input_cutoff:
            self.buffer.new_input(input_cutoff)
        if self._future is not None and not self._future.done():
            # Keep worker backlog bounded. The latest fast suffix still wins;
            # a verification job is launched again once the prior one finishes.
            verify = False
        else:
            verify = True
        revision, model_revision = self.buffer.revision, self.buffer.model_revision
        with torch.no_grad():
            student = self.model.rtc_student
            drafts, _ = student.rollout(history)
            features = student.draft_features(drafts, motor_features(self.model, physical))
        accepted = self.buffer.publish(origin_tick, features[1:], revision=revision,
            input_cutoff=input_cutoff, model_revision=model_revision)
        if not accepted or not verify:
            return None
        snapshot = copy_physical(physical)
        if snapshot.h.is_cuda:
            # Ensure cross-thread consumers see complete immutable inputs.
            torch.cuda.current_stream(snapshot.h.device).synchronize()

        def validate():
            errors = []
            with self._teacher_lock, torch.no_grad():
                if model_revision != self.buffer.model_revision:
                    return errors
                options = physical_options(self.model)
                current = snapshot
                for offset in range(1, student.horizon + 1):
                    current = quiet_teacher_step(self.model, current, options)
                    key = ForecastKey(origin_tick, origin_tick + offset, revision,
                                      input_cutoff, model_revision)
                    errors.append(self.buffer.arrive(key, motor_features(self.model, current),
                                                    same_input_history=True))
            return errors

        self._future = self._worker.submit(validate)
        return self._future

    @torch.no_grad()
    def output(self, target_tick):
        key, features = self.buffer.commit(target_tick)
        return self.model.decoder(self.model.read_norm(features)), key

    def update_parameters(self, update):
        """Serialize learning with physics; invalidate plans after weight change."""
        with self._teacher_lock:
            result = update()
            if next(self.model.parameters()).is_cuda:
                torch.cuda.synchronize(next(self.model.parameters()).device)
            self.buffer.new_model(self.buffer.model_revision + 1)
            return result

    def state_dict(self):
        # Quiesce verification, not the live physical individual. Queued
        # execution values and original issued forecasts survive checkpointing.
        if self._future is not None:
            self._future.result()
        return self.buffer.state_dict()

    def restore_execution(self, saved):
        if self._future is not None:
            self._future.result()
        if saved['horizon'] != self.model.rtc_student.horizon:
            raise ValueError('Execution horizon mismatch')
        self.buffer = TickForecastBuffer.from_state_dict(saved)

    def close(self):
        self._worker.shutdown(wait=True, cancel_futures=True)
