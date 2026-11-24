"""Persistent conditional plans and cooperative events in the complete learner.

The existing conditional bank carries raw plans across real transitions. Reads
re-encode those conditions with current parameters; native references and world
labels still enter the bank only at the original real settlement. A stackful
continuation lets the same external transaction yield at observation arrival
and after each complete recurrent learning update, on the same GPU thread.
"""
from __future__ import annotations

import copy
from contextvars import copy_context
import hashlib
import time

import torch


def enabled(engine):
    return getattr(engine, 'persistent_rhythm_schema', None) == 1


def configure(engine, *, interleave=True):
    if getattr(engine, 'native_amortization', {}).get('correction') != 'decision_budget':
        raise ValueError('persistent rhythm requires the existing conditional banks and two-reference budget')
    engine.persistent_rhythm_schema = 1
    engine.persistent_rhythm_interleave = bool(interleave)
    if not hasattr(engine, 'rhythm_state'):
        engine.rhythm_state = dict(schema=1, clock=0, external=0, reads=0,
            reencoded=0, cache_hits=0, plan_revision=0, plans={}, last_events=[],
            within_learning_choices=0, within_learning_quiet=0)
    for record in engine.rhythm_state['plans'].values():
        record['packet']['raw'] = record['packet']['raw'].detach().cpu()
        record['stamp'] = None
    engine._rhythm_runtime = None
    from scripts.ib_local import run_navigation_residual_wood as legacy
    legacy._DERIVED.add('_rhythm_runtime')


@torch.no_grad()
def read_plans(engine, states, sources, *, carried=None):
    """Read persistent conditional decision programs in today's coordinates.

    The same recipe/targets as native_amortized_decision are used. The plan
    identity binds raw roles, relation version and unknown; values are freshly
    computed, rather than reusing an old tensor after learning or quiet motion.
    """
    from scripts.ib_local import native_amortized_decision as amort
    packets = amort.recipes_batched(engine, states, sources, carried=carried)
    state = engine.rhythm_state
    observed_clock = int(engine.model.memory.clock)
    stamp = (engine.native_value_bank['revision'], observed_clock,
             tuple(p._version for p in engine.model.parameters()))
    identities = [hashlib.sha256(p['raw'].numpy().tobytes()+
        repr((p['scalars'],p.get('relation'),p.get('unknown'))).encode()).hexdigest() for p in packets]
    found = {k:state['plans'][k] for k in identities
             if k in state['plans'] and state['plans'][k].get('stamp') == stamp}
    missing = [i for i,k in enumerate(identities) if k not in found]
    if missing:
        values, bases, surprises = amort.predict(engine, [packets[i] for i in missing])
        for i, (v, b, n) in zip(missing, torch.stack((values, bases, surprises), -1).tolist()):
            found[identities[i]] = dict(value=v, prior=b, novelty=n)
    predicted = engine.model.encoder.weight.new_tensor([found[k]['value'] for k in identities])
    cases = []
    for i,(packet,identity) in enumerate(zip(packets,identities)):
        result = found[identity]
        state['reads'] += 1
        state['reencoded'] += int(i in missing)
        state['cache_hits'] += int(i not in missing)
        state['plans'].pop(identity, None)
        state['plans'][identity] = dict(packet=packet, **{k:result[k] for k in ('value','prior','novelty')},
            stamp=stamp,
            bank_revision=engine.native_value_bank['revision'],
            observed_clock=observed_clock, read_clock=state['clock'])
        cap = engine.model.causal_relations.evidence_budget
        while len(state['plans']) > cap:
            del state['plans'][next(iter(state['plans']))]
        cases.append([dict(novelty=[result['novelty']], persistent_plan=identity,
            bank_revision=engine.native_value_bank['revision'])])
    state['plan_revision'] = engine.native_value_bank['revision']
    return predicted, cases, len(packets)


@torch.no_grad()
def internal_utility(engine, state, source, action, forecast):
    """Actual local U plus the shared learned conditional continuation port."""
    from scripts.ib_local.prospective_decision import boundary, preview_bind
    from scripts.ib_local.causal_use_value import calibrated, queries
    from scripts.ib_local.finite_use_value import expected
    path = forecast.get('path', forecast)
    raws, states = [], []
    for i in range(len(source)):
        raw, _ = boundary(engine, state, source[i:i+1], action[i:i+1],
            path['mother']['mean'][i:i+1], path['decision']['mean'][i:i+1])
        selected, _ = preview_bind(engine.model, raw, overlay=state)
        raws.append(raw); states.append(selected)
    future, _, _ = read_plans(engine, states, raws, carried=state)
    local = calibrated(engine, expected(engine, state, source, action,
                                        path['mother'], queries(engine, state)))
    return 10*local + engine.gamma*future


def _tick(engine, phase, **fields):
    state = engine.rhythm_state
    if state['clock'] >= 2**63-1:
        raise OverflowError('rhythm event clock exhausted')
    state['clock'] += 1
    event = dict(clock=state['clock'], phase=phase,
        observed_clock=int(engine.model.memory.clock),
        plan_revision=engine.native_value_bank['revision'], **fields)
    state['last_events'].append(event)
    del state['last_events'][:-32]
    return event


def _poll_inside(engine):
    # A planning task may progress between two actual learning updates. Its
    # diagnostics must not replace the pending action's pre-world prediction.
    from scripts.ib_local.autonomous_cycle import poll
    names = ('epistemic_value_last', 'successor_distribution_last',
             'prospective_decision_last', 'joint_internal_receipt')
    missing = object()
    saved = {name: getattr(engine, name, missing) for name in names}
    before = engine.joint_internal_ticks
    observed = (int(engine.model.memory.clock), int(engine.model.memory.episodes))
    versions = tuple(p._version for p in engine.model.parameters())
    try:
        receipt = poll(engine)
    finally:
        for name, value in saved.items():
            if value is missing:
                if hasattr(engine, name): delattr(engine, name)
            else: setattr(engine, name, value)
    engine.rhythm_state['within_learning_choices'] += 1
    engine.rhythm_state['within_learning_quiet'] += engine.joint_internal_ticks-before
    if observed != (int(engine.model.memory.clock), int(engine.model.memory.episodes)):
        raise RuntimeError('quiet handoff changed the observed-memory clock')
    if versions != tuple(p._version for p in engine.model.parameters()):
        raise RuntimeError('quiet handoff performed a parameter update')
    return receipt


class ExternalTask:
    """Same-thread continuation of the ORIGINAL complete external step.

    The runnable Python stack is transient. Checkpoints commit at completed
    external transactions; persisted plans, clocks, pending credit and RNG
    retain the ordinary complete-body restore contract.
    """
    def __init__(self, engine, execute):
        from greenlet import greenlet, getcurrent
        self.engine, self.parent = engine, getcurrent()
        self.phase, self.allow_poll = None, False
        self.offered = False
        self.advances = 0
        self.started = time.perf_counter()
        self.owner_grad = torch.is_grad_enabled()
        self.worker = greenlet(lambda: self._run(execute), parent=self.parent)
        self.worker.gr_context = copy_context()

    def yield_at(self, phase, *, allow_poll=False, **data):
        self.phase, self.allow_poll, self.offered = phase, allow_poll, False
        receipt = _tick(self.engine, phase, **data)
        grad = torch.is_grad_enabled()
        torch.set_grad_enabled(self.owner_grad)
        try:
            self.parent.switch(receipt)
        finally:
            torch.set_grad_enabled(grad)

    def _run(self, execute):
        engine = self.engine
        advance, take = engine._advance, engine.world.take
        missing = object()
        own_advance = vars(engine).get('_advance', missing)
        own_take = vars(engine.world).get('take', missing)
        def observed_take(*args, **kwargs):
            result = take(*args, **kwargs)
            self.yield_at('world_arrived', received_bytes=len(result[0])+len(result[1]),
                          new_world_evidence=True)
            return result
        def completed_advance(*args, **kwargs):
            result = advance(*args, **kwargs)
            # All backward/optimizer/recurrent/P capture work has returned.
            # The next objective has not started. Quiet affects its true input.
            self.advances += 1
            self.yield_at('response_updated' if self.advances == 1 else 'continuation_updated',
                          allow_poll=True, new_world_evidence=False)
            return result
        engine.world.take, engine._advance = observed_take, completed_advance
        try:
            return execute(engine)
        finally:
            for owner, name, original in ((engine, '_advance', own_advance),
                                          (engine.world, 'take', own_take)):
                if original is missing:
                    delattr(owner, name)
                else:
                    setattr(owner, name, original)

    def resume(self):
        e = self.engine
        if self.allow_poll and not self.offered and not e.autonomous_ready:
            self.offered = True
            from scripts.ib_local import phase_coupled_rhythm
            if phase_coupled_rhythm.allow_mid_poll(e):
                receipt = _poll_inside(e)
                event = _tick(e, 'coupled_'+receipt['kind'], after=self.phase,
                              new_world_evidence=False)
                e.autonomous_event_clock += 1
                return dict(autonomous_event=dict(receipt, event=e.autonomous_event_clock,
                    within_external=True), rhythm_event=event, updates=0,
                    external_bytes=0, self_bytes=0, internal_ticks=e.joint_internal_ticks)
        try:
            result = self.worker.switch()
        except BaseException:
            e._rhythm_runtime = None
            raise
        if self.worker.dead:
            e._rhythm_runtime = None
            e.rhythm_state['external'] += 1
            _tick(e, 'settled', new_world_evidence=False)
            result['persistent_rhythm'] = diagnostics(e)
            result['persistent_rhythm']['transaction_seconds'] = time.perf_counter()-self.started
            return result
        e.autonomous_event_clock += 1
        return dict(autonomous_event=dict(kind='handoff', event=e.autonomous_event_clock,
            phase=self.phase, new_world_evidence=result['new_world_evidence']),
            rhythm_event=result, updates=0, external_bytes=0, self_bytes=0,
            internal_ticks=e.joint_internal_ticks)


def dispatch(engine, execute):
    task = engine._rhythm_runtime
    if task is not None:
        return task.resume()
    if engine.persistent_rhythm_interleave and engine.autonomous_ready:
        task = engine._rhythm_runtime = ExternalTask(engine, execute)
        return task.resume()
    row = execute(engine)
    if row['autonomous_event']['kind'] == 'external':
        engine.rhythm_state['external'] += 1
        _tick(engine, 'settled', new_world_evidence=False)
        row['persistent_rhythm'] = diagnostics(engine)
    return row


def diagnostics(engine):
    state = engine.rhythm_state
    return {key: copy.deepcopy(state[key]) for key in
        ('clock', 'external', 'reads', 'reencoded', 'cache_hits', 'plan_revision',
         'within_learning_choices', 'within_learning_quiet', 'last_events')} | dict(
            plans=len(state['plans']), interleave=engine.persistent_rhythm_interleave,
            pending_external=engine._rhythm_runtime is not None,
            plan_semantics='raw conditional programs re-encoded now; labels committed on real settlement')
