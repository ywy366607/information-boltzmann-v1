"""Read-only call-tree timing during a complete original life continuation."""
from contextlib import contextmanager
from functools import wraps
import time
import sys
from pathlib import Path


class Profile:
    def __init__(self):
        self.started = time.perf_counter()
        self.stack, self.stats = [], {}

    def wrap(self, label, function):
        @wraps(function)
        def call(*args, **kwargs):
            caller = None
            if label == 'body.predict_observed':
                frame = sys._getframe(1)
                caller = f'{Path(frame.f_code.co_filename).name}:{frame.f_lineno}:{frame.f_code.co_name}'
                del frame
            entry = [time.perf_counter(), 0.]
            self.stack.append(entry)
            try:
                return function(*args, **kwargs)
            finally:
                elapsed = time.perf_counter()-entry[0]
                self.stack.pop()
                stat = self.stats.setdefault(label, dict(calls=0, inclusive_seconds=0., exclusive_seconds=0.))
                stat['calls'] += 1
                stat['inclusive_seconds'] += elapsed
                stat['exclusive_seconds'] += elapsed-entry[1]
                if caller is not None:
                    by_caller = stat.setdefault('callers', {})
                    item = by_caller.setdefault(caller, dict(calls=0, seconds=0.))
                    item['calls'] += 1
                    item['seconds'] += elapsed
                if self.stack:
                    self.stack[-1][1] += elapsed
        return call

    def snapshot(self):
        import torch
        return dict(seconds=time.perf_counter()-self.started, functions=self.stats,
                    cuda_peak_allocated_bytes=(torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0),
                    cuda_peak_reserved_bytes=(torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0),
                    timing='host wall time including waits performed by original code; no per-call CUDA fences')


@contextmanager
def instrument(engine, *, enabled):
    if not enabled:
        yield None
        return
    from scripts.ib_local import (prospective_decision, prospective_self_execution,
        epistemic_value_reference, joint_response, joint_successor, known_utterance_runtime,
        finite_use_value, native_word_batch, native_residual_quadrature, native_amortized_decision,
        latent_action_planner, navigation_momentum_wood, causal_relation_program,
        navigation_generative_cycle)
    targets = [(prospective_decision, 'value'), (prospective_self_execution, 'execute'),
        (epistemic_value_reference, 'expected'), (epistemic_value_reference, 'batched_context_scores'),
        (joint_response, 'hypotheses'), (joint_successor, 'quadrature'),
        (known_utterance_runtime, 'known_film'), (finite_use_value, 'expected'),
        (native_word_batch, 'endpoints'), (native_word_batch, 'table_payload'),
        (native_residual_quadrature, 'value'), (native_residual_quadrature, 'cheap_proxy'),
        (native_amortized_decision, 'value'), (native_amortized_decision, 'predict'),
        (native_amortized_decision, 'recipe'), (native_amortized_decision, 'settle'),
        (latent_action_planner, 'plan'), (latent_action_planner, 'predict'),
        (latent_action_planner, 'teach'), (latent_action_planner, 'settle'),
        (latent_action_planner, 'flush_teachers'),
        (latent_action_planner, 'capture_world'),
        (navigation_momentum_wood, 'decision_momentum_read'),
        (causal_relation_program, 'compose'),
        (navigation_generative_cycle, 'question_choices'),
        (prospective_decision, 'program_actions'),
        (latent_action_planner, '_context_packet'),
        (engine, '_policy_grad'), (engine, '_advance')]
    profile = Profile()
    originals = [(owner, name, getattr(owner, name)) for owner, name in targets]
    try:
        for owner, name, original in originals:
            label = (owner.__name__.rsplit('.', 1)[-1] if hasattr(owner, '__name__') else 'engine')+'.'+name
            setattr(owner, name, profile.wrap(label, original))
        yield profile
    finally:
        for owner, name, original in reversed(originals):
            setattr(owner, name, original)
