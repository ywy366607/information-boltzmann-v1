"""Input-free events using the body's existing selector and delayed policy credit.

No new controller, thought token, purpose or reward is installed. One poll is
one bounded scheduling event; choosing internal never forces a world call when
a quota expires. Score eligibility is accumulated in fixed-size head tensors,
so an arbitrarily long quiet interval does not retain its autograd history.
"""
from __future__ import annotations

import copy
import math

import torch


def configure(engine):
    if not hasattr(engine, '_sample_internal_choice'):
        raise TypeError('autonomous events require the complete generative body')
    if getattr(engine, 'autonomous_events_schema', None) == 1:
        return
    engine.autonomous_events_schema = 1
    engine.autonomous_event_clock = 0
    engine.autonomous_ready = False
    engine.autonomous_choices = 0
    engine.autonomous_internal_steps = 0
    engine.autonomous_logp = 0.0
    engine.autonomous_last_admission = None
    engine.autonomous_eligibility = [torch.zeros_like(p) for p in engine.model.binding_choice.parameters()]
    engine.autonomous_action_eligibility = None
    engine.autonomous_action_receipt = None


def poll(engine):
    """Select one internal step or release to the world, without observing it."""
    if engine.autonomous_ready:
        raise RuntimeError('a released boundary must be dispatched before another internal choice')
    engine._protect_relations()
    old_context = getattr(engine.model, '_question_context', None)
    engine.model._question_context = None
    try:
        selected, choice, admission = engine._sample_internal_choice()
    finally:
        engine.model._question_context = old_context
    # Exactly the existing detached-feature selector score, before any update.
    # This derivative stays eligible until the real action's delayed outcome.
    with torch.enable_grad():
        logits = engine.model.binding_choice(choice['features']).flatten()+choice['prior']
        score = logits.log_softmax(0)[selected]
        if abs(float(score.detach())-choice['sampled_logp']) > 1e-5:
            raise RuntimeError('autonomous selector replay differs')
        grads = torch.autograd.grad(score, tuple(engine.model.binding_choice.parameters()))
    proposed = [old+g.detach() for old, g in zip(engine.autonomous_eligibility, grads)]
    if not all(bool(torch.isfinite(g).all()) for g in proposed):
        raise FloatingPointError('autonomous eligibility overflow; no world event committed')
    logp = engine.autonomous_logp+choice['sampled_logp']
    if not math.isfinite(logp):
        raise FloatingPointError('autonomous log probability overflow')
    internal = engine.continue_internal(1) if selected else None
    observer = getattr(engine, '_selector_age_observer', None)
    if observer is not None:
        observer.before_insert(engine)
    engine.autonomous_eligibility = proposed
    engine.autonomous_choices += 1
    engine.autonomous_logp = logp
    engine.autonomous_internal_steps += int(bool(selected))
    engine.autonomous_last_admission = admission
    engine.autonomous_ready = not bool(selected)
    return dict(kind='internal' if selected else 'release', admission=admission,
        internal=internal, choices=engine.autonomous_choices,
        internal_steps=engine.autonomous_internal_steps,
        eligible_logp=logp, eligibility_norm=float(torch.sqrt(sum(g.square().sum() for g in proposed))),
        new_world_evidence=False, external_bytes=0, updates=0)


def take_boundary(engine):
    """Carry the entire preceding internal decision path into actual fire."""
    if not engine.autonomous_ready:
        raise RuntimeError('autonomous body must release itself before external step')
    observer = getattr(engine, '_selector_age_observer', None)
    if observer is not None:
        observer.take(engine)
    engine.autonomous_action_eligibility = engine.autonomous_eligibility
    engine.autonomous_action_receipt = dict(choices=engine.autonomous_choices,
        internal_steps=engine.autonomous_internal_steps, sampled_logp=engine.autonomous_logp,
        credit='bounded accumulated selector score -> same real delayed actor factor',
        forced_release=False)
    engine._internal_admission = [copy.deepcopy(engine.autonomous_last_admission)]
    engine._internal_budget_hit = False
    engine.autonomous_eligibility = [torch.zeros_like(p) for p in engine.model.binding_choice.parameters()]
    engine.autonomous_choices = engine.autonomous_internal_steps = 0
    engine.autonomous_logp = 0.0
    engine.autonomous_ready = False


def add_boundary_credit(engine, gradients):
    eligibility = engine.autonomous_action_eligibility
    if eligibility is None:
        raise RuntimeError('missing autonomous participation for the actual action')
    indices = {id(p): i for i, p in enumerate(engine.params)}
    for p, g in zip(engine.model.binding_choice.parameters(), eligibility):
        i = indices[id(p)]
        gradients[i] = g.detach().clone() if gradients[i] is None else gradients[i]+g.detach()


def dispatch(engine):
    from scripts.ib_local import persistent_rhythm
    if persistent_rhythm.enabled(engine):
        return persistent_rhythm.dispatch(engine, _dispatch)
    return _dispatch(engine)


def _dispatch(engine):
    """One event: either no observation, or self-released real interaction."""
    if engine.autonomous_event_clock >= 2**63-1:
        raise OverflowError('autonomous event clock exhausted')
    if engine.autonomous_ready:
        adaptive = getattr(engine.model.body.base, '_adaptive_quiet', None)
        before_theta = adaptive.theta.detach().clone() if adaptive is not None else None
        row = engine.step(torch.randn(engine.model.encoder.out_features, 32, device=engine.device))
        if adaptive is not None:
            row['adaptive_dynamics'] = adaptive.diagnostics()
            row['adaptive_dynamics']['per_layer_parameter_step'] = (
                adaptive.theta.detach()-before_theta).norm(dim=-1).cpu().tolist()
            parts = adaptive.theta_parts()
            if len(parts) == 1:
                observer = engine.optimizer.state.get(parts[0], {})
                gate = observer.get('budget_gate')
                row['adaptive_dynamics']['budget_write_fraction'] = (
                    None if gate is None else gate.detach().cpu().tolist())
                row['adaptive_dynamics']['budget_scope'] = 'one existing optimizer observer for the dynamics tensor'
            else:
                gates = {}
                for role_name, part in zip(adaptive.theta_part_names(), parts):
                    gate = engine.optimizer.state.get(part, {}).get('budget_gate')
                    gates[role_name] = None if gate is None else gate.detach().cpu().tolist()
                row['adaptive_dynamics']['budget_write_fraction'] = gates
                row['adaptive_dynamics']['budget_scope'] = 'per-role optimizer observers for split dynamics tensors'
        row['autonomous_event'] = dict(kind='external', new_world_evidence=True)
    else:
        memory = engine.model.memory
        def boundary():
            return (int(memory.clock), int(memory.episodes),
                    tuple(getattr(engine.world, name, None) for name in ('cursor', 'anchor', 'offset', 'moves')))
        before = boundary()
        receipt = poll(engine)
        if boundary() != before:
            raise RuntimeError('input-free event advanced an observed memory/world clock')
        receipt['boundary_clocks_unchanged'] = True
        row = dict(autonomous_event=receipt, updates=0, external_bytes=0, self_bytes=0,
                   internal_ticks=engine.joint_internal_ticks)
    engine.autonomous_event_clock += 1
    row['autonomous_event']['event'] = engine.autonomous_event_clock
    return row
