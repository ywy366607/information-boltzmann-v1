"""Opt-in read of the selector's first age moment, in dispatch ticks.

The ordinary eligibility is borrowed from the actor, never replaced. The
observer is process-local; a restored prefix without ages is marked unknown.
"""
from dataclasses import dataclass

import torch


@dataclass
class ActionAge:
    tick: int
    sources: int
    eligibility: torch.Tensor
    moment: torch.Tensor | None


class SelectorAge:
    def __init__(self, engine):
        self.tick = engine.autonomous_event_clock
        self.moment = None if engine.autonomous_choices else self.zero(engine)
        self.action = self.previous = None

    @staticmethod
    def zero(engine):
        return torch.zeros_like(engine.autonomous_eligibility[0], dtype=torch.float64)

    def before_insert(self, engine):
        tick = engine.autonomous_event_clock + 1
        if self.moment is not None:
            self.moment.add_(engine.autonomous_eligibility[0], alpha=tick-self.tick)
        self.tick = tick

    def take(self, engine):
        tick = engine.autonomous_event_clock
        eligibility, = engine.autonomous_eligibility
        if self.moment is not None:
            self.moment.add_(eligibility, alpha=tick-self.tick)
        self.previous = self.action
        self.action = ActionAge(tick, engine.autonomous_choices, eligibility, self.moment)
        self.tick, self.moment = tick, self.zero(engine)

    def delivery(self, engine, factor):
        tick = engine.autonomous_event_clock + 1
        action, self.previous = self.previous, None
        record = dict(clock='autonomous_dispatch_tick', applied_event=tick,
                      parameter='binding_choice.weight', advantage=float(factor))
        if action is None:
            return dict(record, complete=False, reason='action predates observation')
        record.update(action_boundary=action.tick, source_count=action.sources)
        if action.moment is None:
            return dict(record, complete=False, reason='selector sources predate observation')
        age = action.moment + (tick-action.tick)*action.eligibility.double()
        return dict(record, complete=True, eligibility=action.eligibility.flatten().tolist(),
                    age_moment=age.flatten().tolist())


def attach(engine):
    from scripts.ib_local.run_navigation_residual_wood import _DERIVED
    _DERIVED.add('_selector_age_observer')
    observer = engine._selector_age_observer = SelectorAge(engine)
    return observer
