"""One source-aware observation/learning/continuation transition.

The real stream is unchanged: caregiver bytes, background response, background
continuation. Each observed source is split into a learning prefix and a still
unlearned suffix. Score the suffix before/after learning that prefix, then learn
the suffix as part of the same life. Neither imagined text nor a source label
is a reward. The score is the existing joint predictive improvement, applied
to every observed source instead of only the background.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from scripts.ib_local.navigation_live_witness import frozen_table_logp
from scripts.ib_local.navigation_settlement import settlement_code
from scripts.ib_local.navigation_gain_units import td_gain_target


@dataclass(frozen=True)
class StreamPiece:
    source: str
    start: int
    end: int
    probe_start: int | None = None
    probe_end: int | None = None


def observed_pieces(response, continuation, info):
    """Partition only actual observations, preserving every byte and its order."""
    data = np.concatenate((response, continuation)).astype(np.uint8, copy=False)
    mother = int(info.get('mother_bytes', 0)) if info.get('source') == 'mother' else 0
    if not 0 <= mother <= len(response):
        raise ValueError('invalid actual caregiver boundary')
    ranges = [('mother', 0, mother), ('ambient', mother, len(data))]
    pieces = []
    for source, start, end in ranges:
        if start == end:
            continue
        if end-start == 1:
            pieces.append(StreamPiece(source, start, end))
        else:
            middle = start+(end-start)//2
            pieces.extend((StreamPiece(source, start, middle, middle, end),
                           StreamPiece(source, middle, end)))
    if not pieces or pieces[0].start != 0 or pieces[-1].end != len(data):
        raise ValueError('actual observation stream is incomplete')
    return data, pieces


def settle_sources(readings):
    """Product-of-source geometric score, reported in nats/byte per source.

    Equal source roles, rather than a manually tuned mother/background weight.
    Within each source all scored suffix bytes contribute once. A singleton
    source has no future sample and is explicitly unavailable, never zero gain.
    """
    valid = [r for r in readings if r['bytes'] > 0]
    if not valid or not all(np.isfinite(r['pre']) and np.isfinite(r['post']) for r in valid):
        raise ValueError('finite, genuinely scored future observations required')
    return float(np.mean([r['pre']-r['post'] for r in valid]))


def _predict_score(engine, raw, state, logp):
    ids = torch.as_tensor(raw[:-1].astype(np.int64), device=engine.device)[None]
    target = torch.as_tensor(raw[1:].astype(np.int64), device=engine.device)[None]
    with torch.no_grad():
        logits, _, _ = engine.model.predict(ids, state.detach(), logp)
        return float(F.cross_entropy(logits.flatten(0, 1), target.flatten()))


def world_cycle_step(engine, directions, age_ablate=False):
    """Replace the primitive event; all existing outer learning/credit hooks run."""
    if age_ablate:
        raise ValueError('age ablation must use the declared source-aware stream contract')
    memory = engine.model.memory
    witness_pre = engine.witness_nll()
    source_time = int(memory.clock)
    utterance, record = engine._utter()
    own = len(utterance)
    probabilities = torch.tensor(record['choice_logits']).softmax(-1)
    value_now = float((probabilities*torch.tensor(record['values'])).sum())
    response, heldout, info = engine.world.take(utterance, engine.response_bytes, engine.holdout_bytes)
    external, pieces = observed_pieces(response, heldout, info)
    timeline, heard_probs, errors, norms, credits, readings = [], [], [], [], [], []
    actor, actor_norm, policy_error, td = None, 0., 0., None
    delayed, advantage = None, 0.
    pending = engine.pending
    first_diag, outcome_diag = None, {}
    utterance_probs = np.asarray(record['probs'], np.float32).reshape(-1, 256)
    for index, piece in enumerate(pieces):
        chunk = external[piece.start:piece.end]
        action_tail = utterance[-1] if index == 0 and own else pending
        # The frozen pre-update table is queried on the future suffix BEFORE
        # any byte of that suffix is committed to the live table.
        probe_raw, old_logp = None, None
        if piece.probe_start is not None:
            suffix = external[piece.probe_start:piece.probe_end]
            probe_raw = np.concatenate((chunk[-1:], suffix))
            prefix = np.concatenate((np.array([action_tail], np.uint8), chunk, suffix[:-1]))
            old_logp = torch.as_tensor(frozen_table_logp(engine.table, prefix)[len(chunk):],
                                      device=engine.device)[None]
        _, chunk_probs, _ = engine.table.run_credit(
            np.concatenate((np.array([action_tail], np.uint8), chunk)), max_window=len(chunk))
        heard_probs.append(chunk_probs)
        if index == 0:
            raw = np.concatenate((np.array([pending], np.uint8),
                                  np.frombuffer(utterance, np.uint8), chunk))
            probs = np.concatenate((utterance_probs, chunk_probs))
        else:
            raw, probs = np.concatenate((np.array([pending], np.uint8), chunk)), chunk_probs
        ids, target, logp = engine._inputs(raw, probs)
        mask = torch.ones_like(target, dtype=torch.float32)
        if index == 0:
            mask[:, :own] = 0
        loss, next_state, features, diag = engine.model.objective(
            ids, target, engine.recurrent, logp, directions,
            score_mask=mask if index == 0 else None)
        timeline.append(features.detach())
        ce = F.cross_entropy(diag['logits'][0].detach(), target[0], reduction='none').cpu().numpy()
        errors.append(ce[own:] if index == 0 else ce)
        if index == 0:
            first_diag = diag
            if engine.mode != 'silence':
                actor, replay = engine._policy_grad(target, own, record)
                from scripts.ib_local.navigation_direct_read import replay_error
                policy_error = replay_error(replay, record)
                if policy_error > 1e-3:
                    raise RuntimeError(f'sampled world-cycle policy replay differs: {policy_error}')
                actor_norm = float(torch.stack([g.norm() for g in actor if g is not None]).norm())
            if engine.prev is not None:
                same = engine.prev.get('settlement_kind', 'witness') == engine.settlement_kind
                td = engine._settle_previous(td_gain_target(engine, engine.prev['g'], value_now, same))
                if not same:
                    td['objective_boundary'] = True
                delayed, advantage = td.pop('actor'), td['advantage']
        reading = None
        if probe_raw is not None:
            # q conditions speech only; the unlearned suffix is world input.
            q = getattr(engine.model, '_question_context', None)
            engine.model._question_context = None
            try:
                before = _predict_score(engine, probe_raw, next_state, old_logp)
            finally:
                engine.model._question_context = q
            new_logp = torch.as_tensor(frozen_table_logp(engine.table, prefix)[len(chunk):],
                                      device=engine.device)[None]
            reading = dict(source=piece.source, bytes=len(probe_raw)-1, pre=before)
        if index == len(pieces)-1:
            all_features = torch.cat(timeline)
            short_at = min(own+memory.short, len(all_features)-1)
            long_at = min(memory.long, len(all_features)-1)
            source, action, long = all_features[0], all_features[own], all_features[long_at]
            engine._capture_mother_response_boundary(all_features, own, len(external), info)
            outcome_loss, outcome_diag = engine._outcome_loss(source, action, long)
            loss = loss+engine.outcome_weight*outcome_loss
        norm = engine._advance(loss, features, diag, next_state,
            actor=delayed if index == len(pieces)-1 else None,
            advantage=advantage if index == len(pieces)-1 else 0.)
        norms.append(norm)
        credits.append(diag.get('read_credit'))
        pending = int(chunk[-1])
        if reading is not None:
            reading['post'] = _predict_score(engine, probe_raw, engine.recurrent, new_logp)
            readings.append(reading)
    if delayed is not None:
        engine.delayed_applied += 1
    engine.pending = pending
    witness_post = engine.witness_nll()
    g = settle_sources(readings)
    slot, merged = memory.commit_experience(source, action, all_features[short_at], long,
        engine.model.encoder, utterance, g, source_time)
    if hasattr(engine, 'settlement_versions'):
        code = settlement_code(engine.settlement_kind)
        old = int(engine.settlement_versions[slot])
        engine.settlement_versions[slot] = code if not merged or old == code else 2
    memory.option[slot] = record['choice']
    memory.feeling[slot] = record['feeling']
    engine.prev = dict(g=g, slot=slot, actor=actor,
        inputs=record['critic_inputs'][record['choice']].detach(),
        value=float(record['values'][record['choice']]), choice=record['choice'],
        settlement_kind=engine.settlement_kind)
    engine.intero_raw = np.array([float(np.mean(np.concatenate(errors))), 100*g,
                                 float(first_diag['latent'])])
    engine.intero_stats.update(engine.intero_raw)
    nll = np.concatenate(errors)
    mother = int(info.get('mother_bytes', 0)) if info.get('source') == 'mother' else 0
    try:
        utterance.decode('utf-8'); utf8 = True
    except UnicodeDecodeError:
        utf8 = False
    # Legacy names remain explicit aliases for one source, never presented as
    # the new aggregate. The full source ledger is the authoritative score.
    ambient = next((v for v in readings if v['source'] == 'ambient'), None)
    pre = ambient['pre'] if ambient else None
    post = ambient['post'] if ambient else None
    return dict(mode=engine.mode, choice=('silence','generate','reuse')[record['choice']],
        choice_probability=probabilities.tolist(), reused_slot=record['reused_slot'],
        utterance_hex=utterance.hex(), utf8_valid=utf8, spoke=bool(own),
        truncated=record['truncated'], external_bytes=len(external), self_bytes=own,
        world=info, response_nll=float(nll[:len(response)].mean()),
        mother_response_nll=float(nll[:mother].mean()) if mother else None,
        ambient_response_nll=float(nll[mother:len(response)].mean()) if mother<len(response) else None,
        heldout_pre_nll=pre, heldout_post_nll=post,
        progress=pre-post if ambient else None, fresh=bool(info['holdout_position_fresh']),
        joint_pre_nll=pre, joint_post_nll=post, joint_progress=pre-post if ambient else None,
        witness_pre=witness_pre,witness_post=witness_post,witness_g=witness_pre-witness_post,
        settlement_kind=engine.settlement_kind, settlement=g, metal_g=g, usable_signal=g,
        witnesses=len(engine.w_ids),witness_admitted=engine.w_admitted,admitted=False,
        td_previous=td,value_now=value_now,delayed_actor_applied=engine.delayed_applied,
        actor_norm=actor_norm,policy_replay_max_error=policy_error,
        short_loss=float(first_diag['latent']),long_loss=float(first_diag['latent_long']),**outcome_diag,
        episode_slot=slot,episode_merged=merged,age_component_ablation=None,
        gradient_norms=norms,updates=len(pieces)+int(td is not None and engine.mode != 'silence'),
        read_credit=dict(response=credits[0],heldout=credits[-1]),
        habit_logits=record['habit_logits'],imagination_offset=record['offset'],
        imagined_value=record['values'],precision=engine.precision(),
        critic_rms=None if engine.critic_error is None else float(np.sqrt(engine.critic_error)),
        interoception=engine.intero_raw.tolist(),
        world_learning=dict(schema=1,readings=readings,settlement=g,
            source_roles=[p.source for p in pieces],training_pieces=len(pieces),
            external_bytes_once=len(external),counterfactual_scope='same actual future suffix before/after prefix learning',
            meaning='equal-role mean predictive gain; actual stream and observations retained'))
