"""Arrived language use revises the existing bounded retention evidence.

Cut one versioned identity from the actual co-evolving workset and deep read,
then replay the same finite observation from its same physical start. This is
current read-use credit, not original-write return or a new actor reward.
"""
from contextlib import contextmanager, nullcontext
import time

import torch
from torch.nn import functional as F


MODES = ('off', 'observe', 'apply')


def configure(engine, mode):
    if mode not in MODES:
        raise ValueError('unknown native retention feedback mode')
    m = engine.model.memory
    if mode != 'off' and (not hasattr(engine.model.body, 'coevolution') or
            not m.consolidation_enabled or m.consolidation_rule != 'utility'):
        raise ValueError('native feedback requires the existing co-evolving utility body')
    engine.native_retention_feedback_mode = mode


@contextmanager
def without_identity(model, slot, version):
    """Remove use, not stored content; never refill the fixed read candidates."""
    memory, field = model.memory, model.body.coevolution
    previous = getattr(memory, '_retention_excluded', None)
    valid = field.valid
    memory._retention_excluded = (slot, version)
    # Replace the buffer reference, not its storage: the real backward may
    # still hold the original mask. Each workspace evolves independently.
    field.valid = valid & ~((field.slots_id == slot) & (field.versions == version))
    try:
        yield
    finally:
        field.valid = valid
        if previous is None:
            del memory._retention_excluded
        else:
            memory._retention_excluded = previous


def language_loss(logits, targets, mask=None):
    error = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction='none')
    if mask is None:
        return 10*error.mean()
    weight = mask.reshape(-1).to(error.dtype)
    return 10*(error*weight).sum()/weight.sum().clamp_min(1)


@torch.no_grad()
def measure(engine, ids, targets, start, logp, diagnostics, mask=None):
    mode = getattr(engine, 'native_retention_feedback_mode', 'off')
    if mode == 'off':
        return
    from scripts.ib_local import phase_coupled_rhythm
    if not phase_coupled_rhythm.allow_swr_measure(engine):
        return
    model, m = engine.model, engine.model.memory
    field = model.body.coevolution
    candidates = [(int(s), int(v)) for s,v,ok in
        zip(field.slots_id, field.versions, field.valid) if bool(ok) and
        bool(m.occupied[s]) and int(m.read_credit_version[s]) == int(v)]
    candidates = list(dict.fromkeys(candidates))
    if ids.is_cuda:
        torch.cuda.synchronize(ids.device)
    began = time.perf_counter()
    observations = []
    # Forward bridge is the same physical input as MomentumBody.objective;
    # this replay has no gradient receipt and never writes S/P or a world.
    if hasattr(model, 'head_coupling'):
        from scripts.ib_local.adaptive_hamiltonian import bridge_state
        start = bridge_state(model, start)
    devices = [ids.device] if ids.is_cuda else []
    # Parameters and stored facts are fixed for this withdrawal panel. Only
    # validity and version masks change; those are applied after projection.
    scope = nullcontext()
    decision = getattr(engine, 'decision', None)
    if ids.shape[1] > 1 and hasattr(decision, '_decode_context'):
        decision._refresh_projections()
        scope = decision._decode_context()
    with torch.random.fork_rng(devices=devices), scope:
        width = getattr(engine, 'native_replay_batch_size', 1)
        reuse = getattr(engine, 'native_replay_reuse_actual', False)
        if reuse:
            # world_learning_cycle calls measure immediately after objective,
            # before _advance or any write. The actual logits are this exact
            # body/input/question map, already evaluated with live derivatives.
            logits = diagnostics['logits'].detach()
            error = None
        elif width > 1:
            from scripts.ib_local.native_replay_batch import logits as replay_logits
            cuts = replay_logits(engine, ids, start, logp, [(-1,-1), *candidates], batch_size=width)
            logits = next(cuts)
        else:
            logits = model.predict(ids, start, logp)[0]
        if not reuse:
            error = float((logits-diagnostics['logits'].detach()).abs().max())
            torch.testing.assert_close(logits, diagnostics['logits'].detach(), rtol=2e-4, atol=2e-5)
        factual = float(language_loss(logits, targets, mask))
        def serial():
            for slot, version in candidates:
                with without_identity(model, slot, version):
                    cut = model.predict(ids, start, logp)[0]
                yield cut
        if width > 1 and reuse:
            from scripts.ib_local.native_replay_batch import logits as replay_logits
            cuts = replay_logits(engine, ids, start, logp, candidates, batch_size=width)
        elif width <= 1:
            cuts = serial()
        for (slot,version), cut in zip(candidates, cuts):
            without = float(language_loss(cut, targets, mask))
            observations.append(dict(slot=slot, version=version, benefit=without-factual,
                age_bytes=int(m.clock-m.last_write[slot])))
    if ids.is_cuda:
        torch.cuda.synchronize(ids.device)
    diagnostics['retention_feedback'] = dict(mode=mode, observations=observations,
        factual_language_loss=factual, replay_logit_max_error=error,
        seconds=time.perf_counter()-began, scope='arrived external language, fixed coevolving workset',
        candidates=len(candidates), units='10*nats/byte', settled=False)
    if reuse:
        diagnostics['retention_feedback']['factual_source'] = 'actual objective logits before learning update'


def settle(engine, diagnostics):
    receipt = diagnostics.get('retention_feedback')
    if receipt is None:
        return
    if receipt['settled']:
        raise RuntimeError('native retention feedback already settled')
    from scripts.ib_local.experience_consolidation import bounded_preference, settle_retention_utility
    m = engine.model.memory
    before = bounded_preference(m)
    observations = receipt['observations']
    if receipt['mode'] == 'apply':
        result = settle_retention_utility(m, observations)
    else:
        result = dict(records=0, stale=0, diagnostic_only=True)
    after = bounded_preference(m)
    for item in observations:
        slot = item['slot']
        item.update(preference_before=float(before[slot]), preference_after=float(after[slot]))
    receipt.update(result, settled=True)
