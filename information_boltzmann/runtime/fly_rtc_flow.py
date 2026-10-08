"""Continuous sensory inbox, speculative response and delayed physical arrival.

Driving currents are chosen once from the most recent arrived physical state.
Both student and teacher consume those same timestamped currents. This permits
new input without waiting for a physical tick, while preserving a causal,
auditable input scenario. GPU overlap is not assumed by this Python runtime.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import RLock
import torch

from information_boltzmann.core.fly_rtc_learning import (
    copy_physical, initial_student_state, motor_features,
    physical_options, step_fly_physical_tick,
)
from .fly_rtc import ForecastKey, TickForecastBuffer


class FlyRTCFlow:
    """One execution thread plus one sequential physical worker.

    A bounded H-tick inbox supplies backpressure if the teacher falls farther
    behind than the registered forecast budget. Parameter updates wait for all
    real ticks to arrive; the complete life can then be saved or continued.
    This runtime is an inference interface, not an independent learning rule.
    """

    def __init__(self, model, physical, *, student_state=None):
        if model.training:
            raise ValueError('Asynchronous flow requires evaluation mode')
        self.model = model
        continuing = student_state or initial_student_state(model, physical)
        self.tick = continuing.tick
        self.history = continuing.history.detach().clone()
        # A resumed nonempty history is supplied explicitly; newborn encoding
        # supplies its present observation without fabricating older pulses.
        self.history[0] = model.rtc_student.codec.encode(physical).detach()
        self.anchor_history = self.history.clone()
        self.anchor_tick = self.tick
        self.anchor_physical = copy_physical(physical)
        self._physical = copy_physical(physical)
        self.writer_baseline = physical.baseline.detach().clone()
        self.pending = []
        self.buffer = TickForecastBuffer(model.rtc_student.horizon)
        self.buffer.committed_tick = self.tick
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix='fly-live-ticks')
        self._parameter_lock = RLock()

    @torch.no_grad()
    def poll(self):
        """Assimilate completed real ticks, then replay outstanding known drives."""
        advanced = False
        while self.pending and self.pending[0]['future'].done():
            entry = self.pending.pop(0)
            physical, observed, actual_features = entry['future'].result()
            self.anchor_physical = physical
            self.anchor_history = torch.cat(
                (observed[None], self.anchor_history[:-1]), dim=0)
            self.anchor_tick = entry['tick']
            self.buffer.arrive(entry['key'], actual_features, same_input_history=True)
            advanced = True
        if advanced:
            self.history = self.anchor_history.clone()
            for entry in self.pending:
                _, self.history = self.model.rtc_student.transition_step(
                    self.history, entry['encoded_drive'])
            self._refresh_suffix()
        return self.anchor_tick

    @torch.no_grad()
    def _refresh_suffix(self):
        student = self.model.rtc_student
        drafts, _ = student.rollout(self.history)
        root = self.anchor_history[0, :, student.codec.graph.motor_regions].flatten(1)
        motor = drafts[:, :, student.codec.graph.motor_regions].flatten(2)
        features = motor_features(self.model, self.anchor_physical)[None] + student.motor_adapter(
            motor - root[None])
        if self.anchor_tick == self.tick:
            # Every input has genuinely arrived. Preserve these actual output
            # slots; regenerate only responses strictly after current tick.
            origin, values = self.tick, features[1:]
        else:
            # Known but not-yet-arrived input responses include current tick.
            origin, values = self.tick - 1, features[:self.buffer.horizon]
        revision = self.buffer.revise_suffix(origin)
        self.buffer.publish(origin, values, revision=revision,
                            input_cutoff=self.buffer.input_cutoff,
                            model_revision=self.buffer.model_revision)

    @torch.no_grad()
    def submit(self, token=None):
        """Enqueue one real input/quiet tick and issue a revised response plan.

        token=None is an actual elapsed zero-drive tick, not a hidden settling
        loop. Every queued tick advances the real individual exactly once.
        """
        self.poll()
        if len(self.pending) >= self.buffer.horizon:
            raise BufferError('Physical worker exceeded forecast budget; apply backpressure')
        student, model = self.model.rtc_student, self.model
        origin = self.tick
        if token is not None:
            self.buffer.new_input(origin)
            drive, baseline = model.topographic_writer.forward_with_state(
                model.embedding(token), self.anchor_physical.h, self.writer_baseline)
        else:
            drive = torch.zeros_like(self.anchor_physical.h)
            baseline = self.writer_baseline
        # This selected source is immutable and shared by teacher and student.
        drive, baseline = drive.detach().clone(), baseline.detach().clone()
        self.writer_baseline = baseline
        encoded_drive = student.codec.encode_drive(drive)
        _, self.history = student.transition_step(self.history, encoded_drive)
        self.tick += 1
        drafts, _ = student.rollout(self.history)
        root_motor = self.anchor_history[0, :, student.codec.graph.motor_regions].flatten(1)
        forecast_motor = drafts[:, :, student.codec.graph.motor_regions].flatten(2)
        features = motor_features(model, self.anchor_physical)[None] + student.motor_adapter(
            forecast_motor - root_motor[None])
        revision, model_revision = self.buffer.revision, self.buffer.model_revision
        cutoff = self.buffer.input_cutoff
        # First slot is this tick's input-conditioned response, followed by
        # zero-drive future slots. Physical labels validate only their own slot.
        self.buffer.publish(origin, features[:self.buffer.horizon], revision=revision,
                            input_cutoff=cutoff, model_revision=model_revision)
        key = ForecastKey(origin, self.tick, revision, cutoff, model_revision)
        if drive.is_cuda:
            torch.cuda.current_stream(drive.device).synchronize()

        def advance():
            with self._parameter_lock, torch.no_grad():
                options = physical_options(model)
                self._physical = step_fly_physical_tick(
                    model, self._physical, None, drive, baseline, options,
                    base_rates=options['base_rates'])
                arrived = copy_physical(self._physical)
                observed = student.codec.encode(arrived)
                actual = motor_features(model, arrived)
                if arrived.h.is_cuda:
                    torch.cuda.current_stream(arrived.h.device).synchronize()
                return arrived, observed, actual

        self.pending.append(dict(tick=self.tick, key=key, drive=drive,
                                 baseline=baseline, encoded_drive=encoded_drive,
                                 future=self._worker.submit(advance)))
        return self.tick

    @torch.no_grad()
    def output(self, tick):
        self.poll()
        key, features = self.buffer.commit(tick)
        return self.model.decoder(self.model.read_norm(features)), key

    def synchronize(self):
        for entry in self.pending:
            entry['future'].result()
        self.poll()

    def update_parameters(self, update):
        if self.buffer.committed_tick < self.tick:
            raise RuntimeError('Commit elapsed outputs before changing model parameters')
        self.synchronize()
        with self._parameter_lock:
            result = update()
            if next(self.model.parameters()).is_cuda:
                torch.cuda.synchronize(next(self.model.parameters()).device)
            self.buffer.new_model(self.buffer.model_revision + 1)
            return result

    def state_dict(self):
        # Complete pending real work before serialization; no value resets.
        self.synchronize()
        return dict(physical=self.anchor_physical.state_dict(),
                    history=self.anchor_history, tick=self.tick,
                    writer_baseline=self.writer_baseline, execution=self.buffer.state_dict())

    @classmethod
    def restore(cls, model, saved):
        from information_boltzmann.core.fly_rtc_learning import RTCStudentState, FlyPhysicalState
        device = next(model.parameters()).device
        physical = FlyPhysicalState(**{
            name: tuple(t.to(device) for t in value) if name == 'ring'
            else value.to(device) for name, value in saved['physical'].items()})
        continuing = RTCStudentState(saved['history'].to(device), saved['tick'])
        obj = cls(model, physical, student_state=continuing)
        obj.writer_baseline = saved['writer_baseline'].to(device)
        execution = saved['execution']
        execution = {**execution,
            'pending': {tick: (key, value.to(device))
                        for tick, (key, value) in execution['pending'].items()},
            'issued': {key: value.to(device) for key, value in execution['issued'].items()}}
        obj.buffer = TickForecastBuffer.from_state_dict(execution)
        return obj

    def close(self):
        try:
            self.synchronize()
        finally:
            self._worker.shutdown(wait=True, cancel_futures=True)
