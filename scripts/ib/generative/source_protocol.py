"""Explicit external clock/drive controls; these are not brain mechanisms."""
from dataclasses import dataclass, asdict

import numpy as np

from scripts.ib.generative.native_learning_stream import advance_byte, external_step_count


@dataclass
class SourceProtocol:
    mode: str
    held_voltage: list[float] | None = None
    intervals: int = 0
    text_exposures: int = 0

    def validate(self, model):
        if self.mode not in ('text_drive', 'held_source', 'source_disconnected'):
            raise ValueError('An explicitly supported boundary protocol is required')
        if (not isinstance(self.intervals, int) or not isinstance(self.text_exposures, int)
                or not 0 <= self.text_exposures <= self.intervals):
            raise ValueError('Nonnegative actual interval/exposure counters required')
        if (self.mode == 'text_drive' and self.intervals != self.text_exposures
                or self.mode != 'text_drive' and self.text_exposures != 0):
            raise ValueError('Protocol counters must describe actual text exposure')
        if self.mode == 'held_source':
            held = np.asarray(self.held_voltage, float)
            if held.shape != model.port_nodes.shape or not np.isfinite(held).all() or np.any(np.abs(held)>100):
                raise ValueError('One finite declared source voltage per world port required')
        elif self.held_voltage is not None:
            raise ValueError('Only a held source owns a constant voltage vector')

    def advance(self, model, session, target=None, *, dt_ms=2.):
        self.validate(model)
        steps = external_step_count(dt_ms)
        if self.mode == 'text_drive':
            if not isinstance(target, (int, np.integer)) or not 0 <= target < 256:
                raise ValueError('A real stream byte is required')
            row = advance_byte(model, session, int(target), 'fresh_E', dt_ms=dt_ms)
            self.text_exposures += 1
        else:
            if target is not None:
                raise ValueError('A physical boundary control must not claim unexposed text')
            before = model.t
            voltage = np.asarray(self.held_voltage, float) if self.mode == 'held_source' else None
            for _ in range(steps):
                model.step(dt_ms, voltage)
            if voltage is not None:
                old = np.asarray(session['source_only_voltage'], float)
                session['source_only_voltage'] = (voltage+(old-voltage)*np.exp(
                    -4*model.source_conductance/model.world_capacitance)).tolist()
            row = dict(before_ms=before, after_ms=model.t, source=self.mode,
                       target=None, scores=None, pre_arrival_logits=None)
        self.intervals += 1
        row.update(source_protocol=self.mode, external_interval=self.intervals,
                   actual_new_text_exposures=self.text_exposures)
        return row

    def state_dict(self):
        return asdict(self)
