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


_INTERACTIVE_SOURCES = ('mother', 'parent', 'tool', 'object')


def observed_pieces(response, continuation, info):
    """Partition only actual observations, preserving every byte and its order."""
    data = np.concatenate((response, continuation)).astype(np.uint8, copy=False)
    src = info.get('source')
    is_interactive = src in _INTERACTIVE_SOURCES
    mother = int(info.get('mother_bytes', 0)) if is_interactive else 0
    if not 0 <= mother <= len(response):
        raise ValueError('invalid actual caregiver boundary')
    fg_role = str(src) if is_interactive else 'mother'
    ranges = [(fg_role, 0, mother), ('ambient', mother, len(data))]
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


@torch.no_grad()
def _select_and_condition_target(engine, utterance: bytes, record: dict) -> dict | None:
    """Pre-world target-conditioned action selection: a = (target_id, utterance).

    Runs AFTER `_utter()` drafts `utterance` and BEFORE `causal_use_value.begin()`
    and `world.take()`, so:
      1. Wood's causal model evaluates `p(response | body, target_i, utterance)`
         for every in-scene object `i` via `a_tilde = a(u) + delta(k_i)`.
      2. Fast gaze habit + slow causal consequence prior jointly select `target_id`.
      3. `engine._internal_action[selected]` and the frozen causal query reflect
         the chosen target BEFORE `begin_use` freezes expectations.
    """
    from scripts.ib_local import gaze_pointer as _gp
    if not _gp.enabled(engine):
        return None
    prep = getattr(engine.world, 'prepare_gaze_candidates', None)
    if not callable(prep):
        return None
    candidates, meta = prep(utterance)
    if not candidates:
        return None

    gp = engine.model.gaze_pointer
    selected = int(record['choice'])
    for cid, intro in candidates:
        gp.ensure_object(engine, cid, intro)

    causal_values = None
    observe_causal_values = None
    state = getattr(engine, '_program_question_state', None)
    base_actions = getattr(engine, '_internal_action', None)
    base_src = getattr(engine, '_internal_source', None)
    has_wood = (
        state is not None
        and torch.is_tensor(base_actions)
        and torch.is_tensor(base_src)
        and selected < len(base_actions)
        and len(candidates) >= 2
    )
    has_cond_weights = (
        float(gp.target_to_action.weight.detach().abs().max().item()) > 0.0
        or float(gp._dual().observe_to_action.weight.detach().abs().max().item()) > 0.0
    )
    if has_wood and has_cond_weights:
        try:
            from scripts.ib_local.causal_relation_program import mechanism
            from scripts.ib_local.causal_use_value import queries as use_queries
            from scripts.ib_local.finite_use_value import expected as finite_expected
            from scripts.ib_local.learnable_precision_scales import effective_ridge

            model = engine.model
            ridge = effective_ridge(model, engine.relation_ridge)
            raw_act = base_actions[selected:selected + 1]
            facts = use_queries(engine, state)
            # Single vectorized batched evaluation for BOTH `out` and `in` candidates (O(1) calls!).
            m_cands = len(candidates)
            src_rows = base_src.reshape(1, -1).expand(2 * m_cands, -1)
            out_acts = [gp.condition_action(raw_act, cid, observe_id=None) for cid, _ in candidates]
            in_acts = [gp.condition_action(raw_act, None, observe_id=cid) for cid, _ in candidates]
            cond_acts = torch.cat(out_acts + in_acts, dim=0)
            dist = mechanism(model, state, src_rows, cond_acts, ridge=ridge)
            if getattr(engine, 'causal_use_arm', None) == 'C2' and facts:
                vals = finite_expected(engine, state, src_rows, cond_acts, dist, facts)
                causal_values = {cid: float(vals[i].item()) * 10.0 for i, (cid, _) in enumerate(candidates)}
                observe_causal_values = {cid: float(vals[m_cands + i].item()) * 10.0 for i, (cid, _) in enumerate(candidates)}
            else:
                causal_values = {cid: 0.0 for cid, _ in candidates}
                observe_causal_values = {cid: 0.0 for cid, _ in candidates}
        except Exception:
            causal_values = None
            observe_causal_values = None

    gaze_dec = gp.select_interaction(
        engine,
        candidates=candidates,
        utterance_bytes=utterance,
        causal_values=causal_values,
        observe_causal_values=observe_causal_values,
        stumble_ratio=float(meta.get('stumble_ratio', 1.0)),
        nll_norm=float(meta.get('nll_norm', 0.0)),
        last_speaker=meta.get('last_speaker'),
        awaiting_from=meta.get('awaiting_from'),
    )
    chosen_id = gaze_dec['target_id']
    chosen_send = gaze_dec.get('send_target', chosen_id)
    chosen_obs = gaze_dec.get('observe_target', chosen_id)

    # Propagate chosen `(out, in)` condition into `_internal_action[selected]` and mother forecast
    # BEFORE `causal_use_value.begin()` freezes the pre-world expectation.
    if torch.is_tensor(base_actions) and selected < len(base_actions):
        delta = gp.target_offset(chosen_send, observe_id=chosen_obs, like=base_actions[selected])
        if float(delta.abs().max().item()) > 0.0:
            engine._internal_action = base_actions.clone()
            engine._internal_action[selected] = base_actions[selected] + delta
            forecasts = getattr(engine, '_program_forecasts', None)
            if isinstance(forecasts, dict) and state is not None and torch.is_tensor(base_src):
                try:
                    from scripts.ib_local.causal_relation_program import mechanism
                    from scripts.ib_local.learnable_precision_scales import effective_ridge
                    ridge = effective_ridge(engine.model, engine.relation_ridge)
                    src_row = base_src.reshape(1, -1)
                    d_new = mechanism(engine.model, state, src_row, engine._internal_action, ridge=ridge)
                    forecasts['path']['mother'] = d_new
                except Exception:
                    pass
    return gaze_dec


def _complete_self_evolution_beat(
    *,
    engine,
    utterance: bytes,
    record: dict,
    probabilities: torch.Tensor,
    value_now: float,
    witness_pre: float,
    source_time: int,
    info: dict,
    option,
) -> dict:
    """Execute a genuine zero-external-pull internal self-evolution / self-talk beat (`in == 'none'`).

    Preserves continuous biological self-evolution AND every outer engine wrapper invariant:
      1. Policy replay (`_policy_grad`) & delayed TD settlement run FIRST against the exact
         pre-turn state (`engine.recurrent`, `engine.bfss_momenta`) so `variable_expression`
         and `autonomous_cycle` boundary credit match bit-for-bit.
      2. If MengLi spoke (`u != b''`, pure self-talk / inner monologue), her 32-layer GDN
         hears her own bytes under `no_grad()` (zero LM cross-entropy self-training, zero Fast Table writes).
      3. Hamiltonian BFSS `(S, P)` quiet physics evolves her recurrent state and momenta.
      4. `_outcome_loss` + `hippocampal_anchor_loss` consolidate real historical memories and
         keep `engine._wood_source/_wood_action/_wood_replay` valid for all outer Wood wrappers.
      5. The self-evolution episode commits to episodic memory (`slot, merged`), and honest
         `witness_pre - witness_post` gain settles credit on `GazeObjectPointer`.
    """
    from scripts.ib_local import bounded_reading_option, gaze_pointer as _gp
    from scripts.ib_local.alif_executive import _step_quiet_physics, hippocampal_anchor_loss

    memory = engine.model.memory
    own = len(utterance)
    pending = int(engine.pending)
    utterance_probs = np.asarray(record['probs'], np.float32).reshape(-1, 256)

    # 1. Policy replay & delayed TD settlement FIRST (strictly before mutating recurrent/BFSS state!).
    if own > 0:
        raw_self = np.concatenate((np.array([pending], np.uint8), np.frombuffer(utterance, np.uint8)))
        ids_s, target_s, logp_s = engine._inputs(raw_self, utterance_probs)
    else:
        raw_dummy = np.array([pending, pending], np.uint8)
        prob_dummy = np.full((1, 256), 1.0 / 256.0, dtype=np.float32)
        ids_s, target_s, logp_s = engine._inputs(raw_dummy, prob_dummy)

    actor, actor_norm, policy_error, td = None, 0.0, 0.0, None
    delayed, advantage = None, 0.0
    if engine.mode != 'silence':
        actor, replay = engine._policy_grad(target_s, own, record)
        from scripts.ib_local.navigation_direct_read import replay_error
        policy_error = replay_error(replay, record)
        if policy_error > 1e-3:
            raise RuntimeError(f'sampled self-evolution policy replay differs: {policy_error}')
        actor_norm = float(torch.stack([g.norm() for g in actor if g is not None]).norm())

    if engine.prev is not None:
        same = engine.prev.get('settlement_kind', 'witness') == engine.settlement_kind
        td = engine._settle_previous(td_gain_target(engine, engine.prev['g'], value_now, same))
        if not same:
            td['objective_boundary'] = True
        delayed, advantage = td.pop('actor'), td['advantage']

    # 2. Self-hearing (when MengLi spoke `u != b''`) + continuous Hamiltonian BFSS `(S, P)` self-evolution.
    self_speech_nll = None
    if own > 0:
        from scripts.ib_local.adaptive_hamiltonian import bridge_state
        from scripts.ib_local.language_action import peek
        from scripts.ib_local.question_language_port import observation_scope
        last_b = int(utterance[-1])
        p_last = peek(engine.table, last_b)
        raw_full = np.concatenate((raw_self, np.array([last_b], np.uint8)))
        probs_full = np.concatenate((utterance_probs, np.asarray(p_last, np.float32).reshape(1, 256)), axis=0)
        ids_full = torch.as_tensor(raw_full[:-1].astype(np.int64), device=engine.device)[None]
        logp_full = torch.as_tensor(np.log(np.maximum(probs_full, 1e-6)), device=engine.device)[None]
        start_state = bridge_state(engine.model, engine.recurrent.detach())
        with torch.no_grad():
            logits_self, next_state, _, h_self, _ = engine.model.predict_observed(ids_s, start_state, logp_s)
            self_speech_nll = float(F.cross_entropy(logits_self[0].float(), target_s[0]).item())
            # Exact post-utterance action endpoint (`all_features[own]`, before BFSS quiet physics),
            # while keeping `engine.recurrent = next_state` strictly before `engine.pending = utterance[-1]`.
            with observation_scope(engine.model, own, own + 1):
                _, _, _, h_full, _ = engine.model.predict_observed(ids_full, start_state, logp_full, decode=False)
        engine.recurrent = next_state.detach()
        engine.pending = last_b
        feat_for_anchor = h_self.detach()
        raw_action_for_gaze = h_full[own].detach().clone().reshape(-1)
    else:
        base_src = getattr(engine, '_expression_decision_source', None)
        if not torch.is_tensor(base_src):
            base_src = getattr(engine, '_internal_source', None)
        if torch.is_tensor(base_src):
            feat_for_anchor = base_src.detach().reshape(1, -1)
        else:
            feat_for_anchor = torch.zeros(1, engine.model.encoder.in_features, device=engine.device)
        raw_action_for_gaze = feat_for_anchor[-1].detach().clone()

    bfss_change = _step_quiet_physics(engine)

    # 3. Outcome & hippocampal consolidation (populates `_wood_source/_action/_replay` cleanly),
    #    bracketed by a strictly disjoint held-out consolidation probe on real historical transitions.
    if getattr(engine, 'variable_expression_schema', None) == 1 and torch.is_tensor(
        getattr(engine, '_expression_decision_source', None)
    ):
        source = engine._expression_decision_source.reshape(-1).detach().clone()
    else:
        source = feat_for_anchor[0].detach().clone()
    draft_raw_action_l2 = None
    drafts = getattr(engine, '_draft_actions', None)
    if drafts and record['choice'] < len(drafts):
        draft_raw_action_l2 = float((drafts[record['choice']] - raw_action_for_gaze).norm().item())
    action = raw_action_for_gaze
    actual_send_id = info.get('send_target_id') or info.get('target_object_id')
    if _gp.enabled(engine):
        action = engine.model.gaze_pointer.condition_action(action, actual_send_id, observe_id='none')
    long_vec = raw_action_for_gaze.detach().clone()

    # Sample a strictly disjoint held-out slice of real internal transitions BEFORE updating weights.
    # Excluded from BOTH `engine._outcome_loss` (InternalRolloutEngine) AND `hippocampal_anchor_loss`.
    holdout_slots = None
    enc_tgt_pre = None
    self_cons_pre = None
    self_cons_post = None
    self_cons_fixed_pre = None
    self_cons_fixed_post = None
    self_cons_coord_shift = None
    self_cons_norm_pre = None
    self_cons_norm_post = None
    self_cons_norm_g = 0.0
    self_cons_g = 0.0
    self_cons_samples = 0
    if hasattr(memory, 'internal_valid') and hasattr(engine, 'replay_rng'):
        from scripts.ib_local.navigation_internal_rollout import next_boundary_prediction
        valid_int = (
            memory.occupied & memory.internal_valid & (memory.count == 1)
            & (memory.internal_stamp == memory.last_write)
        )
        pending_int = getattr(engine, 'internal_pending', None)
        if isinstance(pending_int, dict) and 'slot' in pending_int:
            valid_int = valid_int.clone()
            valid_int[int(pending_int['slot'])] = False
        all_int_slots = torch.nonzero(valid_int).flatten()
        if len(all_int_slots) >= 24:
            n_hold = min(16, len(all_int_slots) // 4)
            picks = engine.replay_rng.choice(len(all_int_slots), n_hold, replace=False)
            holdout_slots = all_int_slots[torch.as_tensor(picks, device=all_int_slots.device)]
            self_cons_samples = int(len(holdout_slots))
            with torch.no_grad():
                h_src = memory.source[holdout_slots]
                h_act = memory.action[holdout_slots]
                h_nxt = memory.internal_next[holdout_slots]
                pred_pre = next_boundary_prediction(engine.model, h_src, h_act)
                enc_tgt_pre = engine.model.encoder(h_nxt).detach()
                self_cons_pre = float(F.mse_loss(pred_pre, enc_tgt_pre).item())
                self_cons_fixed_pre = self_cons_pre
                self_cons_norm_pre = float(
                    F.mse_loss(F.normalize(pred_pre, dim=-1, eps=1e-6), F.normalize(enc_tgt_pre, dim=-1, eps=1e-6)).item()
                )

    engine.model._wood_nll = torch.zeros(2, device=engine.device)
    engine.model._near_outcome = long_vec
    engine._mother_boundary_target = None
    engine._exclude_internal_slots = holdout_slots
    try:
        outcome_loss, outcome_diag = engine._outcome_loss(source, action, long_vec)
        anchor_loss, anchor_diag = hippocampal_anchor_loss(
            engine, feat_for_anchor, exclude_internal_slots=holdout_slots,
        )
    finally:
        engine._exclude_internal_slots = None
    has_anchor = (
        anchor_diag['wood_replay_samples']
        + anchor_diag['internal_transition_samples']
        + anchor_diag['causal_program_replay']
    ) > 0 and anchor_loss.requires_grad
    total_loss = float(getattr(engine, 'outcome_weight', 0.1)) * outcome_loss
    if has_anchor:
        total_loss = total_loss + float(getattr(engine, 'outcome_weight', 0.1)) * anchor_loss

    updates_done = 0
    norms: list[float] = []
    if total_loss.requires_grad or delayed is not None:
        engine.optimizer.zero_grad(set_to_none=True)
        if total_loss.requires_grad:
            total_loss.backward()
        if delayed is not None:
            for p, g in zip(engine.params, delayed):
                if g is not None:
                    term = (-float(advantage)) * g.to(p.device, dtype=p.dtype)
                    p.grad = term if p.grad is None else (p.grad + term)
            engine.delayed_applied += 1
        grad_sq = 0.0
        for group in engine.groups:
            live = [p for p in group if p.grad is not None]
            if live:
                gn = float(torch.nn.utils.clip_grad_norm_(live, 1.0, foreach=True))
                grad_sq += gn * gn
        engine.optimizer.step()
        norms.append(float(grad_sq ** 0.5))
        updates_done = 1

    if holdout_slots is not None and self_cons_pre is not None and enc_tgt_pre is not None:
        from scripts.ib_local.navigation_internal_rollout import next_boundary_prediction
        with torch.no_grad():
            h_src = memory.source[holdout_slots]
            h_act = memory.action[holdout_slots]
            h_nxt = memory.internal_next[holdout_slots]
            pred_post = next_boundary_prediction(engine.model, h_src, h_act)
            enc_tgt_post = engine.model.encoder(h_nxt).detach()
            self_cons_post = float(F.mse_loss(pred_post, enc_tgt_post).item())
            self_cons_fixed_post = float(F.mse_loss(pred_post, enc_tgt_pre).item())
            self_cons_coord_shift = float(F.mse_loss(enc_tgt_post, enc_tgt_pre).item())
            self_cons_norm_post = float(
                F.mse_loss(F.normalize(pred_post, dim=-1, eps=1e-6), F.normalize(enc_tgt_post, dim=-1, eps=1e-6)).item()
            )
        self_cons_norm_g = float(self_cons_norm_pre - self_cons_norm_post)
        rel_imp = self_cons_norm_g / max(float(self_cons_norm_pre), 0.05)
        self_cons_g = float(max(-0.05, min(0.05, 0.05 * rel_imp)))

    # 4. Advance episodic memory clock cleanly over this internal self-evolution horizon.
    #    Keep `self_cons_g` strictly observation-only by default (`self_cons_credit_mode='observe'`)
    #    so internal reward / credit assignment is NEVER silently altered.
    from scripts.ib_local.causal_use_value import settle as settle_use
    horizon = max(1, int(memory.long))
    if len(feat_for_anchor) >= horizon:
        evo_feats = feat_for_anchor[-horizon:].detach()
    else:
        pad_n = horizon - len(feat_for_anchor)
        evo_feats = torch.cat((feat_for_anchor.detach(), long_vec[None].expand(pad_n, -1)), dim=0)
    memory.commit(evo_feats, engine.model.encoder)

    witness_post = engine.witness_nll()
    witness_g = float(witness_pre - witness_post)
    credit_mode = str(getattr(engine, 'self_cons_credit_mode', 'observe'))
    has_witness = len(getattr(engine, 'w_ids', ())) > 0
    if has_witness:
        g = witness_g
        ev_avail = True
    elif credit_mode == 'apply' and self_cons_samples > 0:
        g = float(self_cons_g)
        ev_avail = True
    else:
        g = witness_g
        ev_avail = False
    if getattr(engine, 'causal_use_arm', None) == 'C2':
        slot, merged = memory.commit_experience(
            source, action, long_vec, long_vec, engine.model.encoder, utterance, 0.0, source_time,
        )
        from scripts.ib_local.identified_event_credit import capture
        capture(
            engine,
            source=source,
            action=action,
            short=long_vec,
            long=long_vec,
            words=utterance,
            source_time=source_time,
            slot=slot,
            merged=merged,
        )
        _, use_receipt = settle_use(engine, 0.0, slot=slot, merged=merged)
        if not merged:
            memory.progress[slot] = g
    else:
        _, use_receipt = settle_use(engine, 0.0)
        slot, merged = memory.commit_experience(
            source, action, long_vec, long_vec, engine.model.encoder, utterance, g, source_time,
        )
    if hasattr(engine, 'settlement_versions'):
        code = settlement_code(engine.settlement_kind)
        old = int(engine.settlement_versions[slot])
        engine.settlement_versions[slot] = code if not merged or old == code else 2
    memory.option[slot] = record['choice']
    memory.feeling[slot] = record['feeling']

    # Update bodily interoception honestly for this internal self-evolution beat.
    intero_surprise = (
        float(self_speech_nll)
        if self_speech_nll is not None
        else float(self_cons_norm_post if self_cons_norm_post is not None else outcome_diag.get('outcome_error', 0.0))
    )
    engine.intero_raw = np.array([
        intero_surprise,
        100.0 * float(g),
        float(outcome_diag.get('outcome_error', 0.0)),
    ], dtype=np.float64)
    engine.intero_stats.update(engine.intero_raw)

    option_receipt = bounded_reading_option.finish(engine, option)
    if _gp.enabled(engine):
        engine._gaze_boundary_context = dict(
            model=engine.model,
            source=source.detach().clone(),
            raw_action=raw_action_for_gaze,
            actual_target_id=actual_send_id,
            actual_observe_id='none',
            boundary_target=None,
            evidence_available=ev_avail,
        )

    engine.prev = dict(
        g=g,
        slot=slot,
        actor=actor,
        inputs=record['critic_inputs'][record['choice']].detach(),
        value=float(record['values'][record['choice']]),
        choice=record['choice'],
        settlement_kind=engine.settlement_kind,
    )
    try:
        utterance.decode('utf-8')
        utf8 = True
    except UnicodeDecodeError:
        utf8 = False

    return dict(
        mode=engine.mode,
        choice=('silence', 'generate', 'reuse')[record['choice']],
        choice_probability=probabilities.tolist(),
        reused_slot=record['reused_slot'],
        utterance_hex=utterance.hex(),
        utf8_valid=utf8,
        spoke=bool(own),
        truncated=record['truncated'],
        external_bytes=0,
        self_bytes=own,
        world=info,
        response_nll=0.0,
        self_speech_nll=self_speech_nll,
        draft_raw_action_l2=draft_raw_action_l2,
        mother_response_nll=None,
        ambient_response_nll=None,
        heldout_pre_nll=None,
        heldout_post_nll=None,
        progress=None,
        fresh=True,
        joint_pre_nll=None,
        joint_post_nll=None,
        joint_progress=None,
        witness_pre=witness_pre,
        witness_post=witness_post,
        witness_g=witness_g,
        settlement_kind=engine.settlement_kind,
        settlement=g,
        metal_g=g,
        usable_signal=g,
        evidence_available=ev_avail,
        target_object_id=info.get('target_object_id'),
        send_target_id=info.get('send_target_id'),
        observe_target_id='none',
        interaction_mode=info.get('interaction_mode'),
        witnesses=len(engine.w_ids),
        witness_admitted=engine.w_admitted,
        admitted=False,
        td_previous=td,
        value_now=value_now,
        delayed_actor_applied=engine.delayed_applied,
        actor_norm=actor_norm,
        policy_replay_max_error=policy_error,
        short_loss=0.0,
        long_loss=0.0,
        **outcome_diag,
        episode_slot=slot,
        episode_merged=merged,
        age_component_ablation=None,
        gradient_norms=norms,
        updates=updates_done,
        read_credit=dict(response=None, heldout=None),
        habit_logits=record['habit_logits'],
        imagination_offset=record['offset'],
        imagined_value=record['values'],
        precision=engine.precision(),
        critic_rms=None if engine.critic_error is None else float(np.sqrt(engine.critic_error)),
        interoception=engine.intero_raw.tolist(),
        causal_use=use_receipt,
        self_evolution=dict(
            bfss_state_change=float(bfss_change),
            self_heard_bytes=int(own),
            self_speech_nll=self_speech_nll,
            self_cons_credit_mode=credit_mode,
            self_cons_pre=self_cons_pre,
            self_cons_post=self_cons_post,
            self_cons_fixed_pre=self_cons_fixed_pre,
            self_cons_fixed_post=self_cons_fixed_post,
            self_cons_coord_shift=self_cons_coord_shift,
            self_cons_norm_pre=self_cons_norm_pre,
            self_cons_norm_post=self_cons_norm_post,
            self_cons_norm_g=float(self_cons_norm_g),
            self_cons_g=float(self_cons_g),
            self_cons_samples=int(self_cons_samples),
            **anchor_diag,
        ),
        **({'bounded_reading_option': option_receipt} if option_receipt is not None else {}),
        world_learning=dict(
            schema=1,
            readings=[],
            settlement=0.0,
            source_roles=['self_evolution'],
            training_pieces=0,
            external_bytes_once=0,
            counterfactual_scope='zero-external-pull internal consolidation',
            meaning='endogenous BFSS + hippocampal replay without pulling external streams',
        ),
    )


def world_cycle_step(engine, directions, age_ablate=False):
    """Replace the primitive event; all existing outer learning/credit hooks run."""
    if age_ablate:
        raise ValueError('age ablation must use the declared source-aware stream contract')
    memory = engine.model.memory
    engine._identified_event_fact = None
    engine._gaze_boundary_context = None
    witness_pre = engine.witness_nll()
    source_time = int(memory.clock)
    utterance, record = engine._utter()
    own = len(utterance)
    probabilities = torch.tensor(record['choice_logits']).softmax(-1)
    value_now = float((probabilities*torch.tensor(record['values'])).sum())
    planned_target = _select_and_condition_target(engine, utterance, record)
    from scripts.ib_local.causal_use_value import begin as begin_use, settle as settle_use
    begin_use(engine, record)
    from scripts.ib_local import bounded_reading_option
    option = bounded_reading_option.begin(engine)
    if planned_target is not None:
        response, heldout, info = engine.world.take(
            utterance, engine.response_bytes, engine.holdout_bytes, planned_target=planned_target,
        )
    else:
        response, heldout, info = engine.world.take(utterance, engine.response_bytes, engine.holdout_bytes)
    if info.get('self_evolution_beat'):
        return _complete_self_evolution_beat(
            engine=engine,
            utterance=utterance,
            record=record,
            probabilities=probabilities,
            value_now=value_now,
            witness_pre=witness_pre,
            source_time=source_time,
            info=info,
            option=option,
        )
    external, pieces = observed_pieces(response, heldout, info)
    root_piece_count = len(pieces)
    block_end, block_info, block_error_start = len(external), info, 0
    timeline, heard_probs, errors, norms, credits, readings = [], [], [], [], [], []
    actor, actor_norm, policy_error, td = None, 0., 0., None
    delayed, advantage = None, 0.
    age_delivery = None
    pending = engine.pending
    first_diag, outcome_diag = None, {}
    utterance_probs = np.asarray(record['probs'], np.float32).reshape(-1, 256)
    raw_spans = info.get('envelope_mask_spans')
    if isinstance(raw_spans, (list, tuple)) and raw_spans:
        env_mask_spans = [(int(s), int(e)) for s, e in raw_spans if int(e) > int(s)]
    else:
        env_mask_end = max(0, int(info.get('envelope_prefix_bytes', 0)))
        env_mask_spans = [(0, env_mask_end)] if env_mask_end > 0 else []
    self_speech_nll = None
    draft_raw_action_l2 = None
    for index, piece in enumerate(pieces):
        chunk = external[piece.start:piece.end]
        action_tail = utterance[-1] if index == 0 and own else pending
        # Compute leading masked identity-header bytes in this piece (covers both piece 0 `妈妈：`
        # and the exact piece boundary where `书页：` resumes after a speaker/tool finishes!).
        masked_in_piece = 0
        for sp_s, sp_e in env_mask_spans:
            if sp_s <= piece.start < sp_e:
                masked_in_piece = max(masked_in_piece, min(piece.end, sp_e) - piece.start)
        # The frozen pre-update table is queried on the future suffix BEFORE
        # any byte of that suffix is committed to the live table.
        probe_raw, old_logp = None, None
        if piece.probe_start is not None:
            suffix = external[piece.probe_start:piece.probe_end]
            probe_raw = np.concatenate((chunk[-1:], suffix))
            prefix = np.concatenate((np.array([action_tail], np.uint8), chunk, suffix[:-1]))
            old_logp = torch.as_tensor(frozen_table_logp(engine.table, prefix)[len(chunk):],
                                      device=engine.device)[None]
        if masked_in_piece > 0:
            # Identity-card / speaker-header prefix bytes are READ-ONLY for the Fast Table
            # (GDN still sees them in full, while Fast Table never memorizes role prefixes).
            # `frozen_table_logp(table, seq)` returns 1 row per byte in `seq` (predicting the next byte),
            # so `[action_tail, *chunk[:masked_in_piece - 1]]` yields exact shape `(masked_in_piece, 256)`.
            head_seq = np.concatenate((np.array([action_tail], np.uint8), chunk[:masked_in_piece - 1]))
            head_probs = np.exp(frozen_table_logp(engine.table, head_seq)).astype(np.float32, copy=False)
            if masked_in_piece < len(chunk):
                _, tail_probs, _ = engine.table.run_credit(
                    np.concatenate((chunk[masked_in_piece - 1:masked_in_piece], chunk[masked_in_piece:])),
                    max_window=len(chunk) - masked_in_piece,
                )
                chunk_probs = np.concatenate((head_probs, tail_probs), axis=0)
            else:
                chunk_probs = head_probs
        else:
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
            skip_n = own + (masked_in_piece if (own + masked_in_piece) < target.shape[1] else 0)
            mask[:, :skip_n] = 0
            # Finish the independent draft gradient before building the live
            # text graph. Both still read the same pre-update parameters.
            if engine.mode != 'silence':
                actor, replay = engine._policy_grad(target, own, record)
                from scripts.ib_local.navigation_direct_read import replay_error
                policy_error = replay_error(replay, record)
                if policy_error > 1e-3:
                    raise RuntimeError(f'sampled world-cycle policy replay differs: {policy_error}')
                actor_norm = float(torch.stack([g.norm() for g in actor if g is not None]).norm())
        else:
            skip_n = masked_in_piece if masked_in_piece < target.shape[1] else 0
            if skip_n > 0:
                mask[:, :skip_n] = 0
        from scripts.ib_local.question_language_port import observation_scope
        from scripts.ib_local.native_retention_feedback import measure as measure_retention
        # Scope only the actual first prefix. Speculative drafts and their
        # policy replay run outside this scope, under the original speech q.
        if index == 0:
            with observation_scope(engine.model, own, ids.shape[1]):
                loss, next_state, features, diag = engine.model.objective(
                    ids, target, engine.recurrent, logp, directions, score_mask=mask)
                measure_retention(engine, ids, target, engine.recurrent, logp, diag, mask)
        else:
            if skip_n > 0:
                loss, next_state, features, diag = engine.model.objective(
                    ids, target, engine.recurrent, logp, directions, score_mask=mask)
                measure_retention(engine, ids, target, engine.recurrent, logp, diag, mask)
            else:
                loss, next_state, features, diag = engine.model.objective(
                    ids, target, engine.recurrent, logp, directions)
                measure_retention(engine, ids, target, engine.recurrent, logp, diag)
        timeline.append(features.detach())
        ce = F.cross_entropy(diag['logits'][0].detach(), target[0], reduction='none').cpu().numpy()
        if index == 0 and own > 0:
            self_speech_nll = float(ce[:own].mean())
        errors.append(ce[skip_n:] if skip_n < len(ce) else ce)
        if index == 0:
            first_diag = diag
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
        if index == root_piece_count-1:
            all_features = torch.cat(timeline)
            short_at = min(own+memory.short, len(all_features)-1)
            long_at = min(memory.long, len(all_features)-1)
            source, action, long = all_features[0], all_features[own], all_features[long_at]
            if getattr(engine,'variable_expression_schema',None)==1:
                source=engine._expression_decision_source.reshape(-1)
            raw_action_for_gaze = action.detach().clone()
            drafts = getattr(engine, '_draft_actions', None)
            if drafts and record['choice'] < len(drafts):
                draft_raw_action_l2 = float((drafts[record['choice']] - raw_action_for_gaze).norm().item())
            actual_target_id = info.get('send_target_id') or info.get('target_object_id')
            actual_observe_id = info.get('observe_target_id')
            from scripts.ib_local import gaze_pointer as _gp
            if _gp.enabled(engine) and (actual_target_id is not None or actual_observe_id is not None):
                action = engine.model.gaze_pointer.condition_action(
                    action, actual_target_id, observe_id=actual_observe_id,
                )
            engine._capture_mother_response_boundary(all_features, own, len(external), info)
            outcome_loss, outcome_diag = engine._outcome_loss(source, action, long)
            loss = loss+engine.outcome_weight*outcome_loss
            observer = getattr(engine, '_selector_age_observer', None)
            if observer is not None and delayed is not None:
                age_delivery = observer.delivery(engine, td.get('advantage_head', advantage))
        elif option is not None and index >= root_piece_count:
            anchor_scale = float(engine.reading_option_spec.get('hippocampal_anchor_scale', 0.0))
            if anchor_scale > 0.0:
                from scripts.ib_local.alif_executive import hippocampal_anchor_loss
                anchor_loss, anchor_diag = hippocampal_anchor_loss(engine, features)
                if (anchor_diag['wood_replay_samples'] + anchor_diag['internal_transition_samples'] +
                        anchor_diag['causal_program_replay']) > 0:
                    loss = loss + (engine.outcome_weight * anchor_scale) * anchor_loss
                option.setdefault('hippocampal_anchors', []).append(anchor_diag)
        norm = engine._advance(loss, features, diag, next_state,
            actor=delayed if index == root_piece_count-1 else None,
            advantage=advantage if index == root_piece_count-1 else 0.)
        norms.append(norm)
        credits.append(diag.get('read_credit'))
        pending = int(chunk[-1])
        if reading is not None:
            reading['post'] = _predict_score(engine, probe_raw, engine.recurrent, new_logp)
            readings.append(reading)
        if option is not None and piece.end == block_end:
            block_nll = float(np.concatenate(errors[block_error_start:]).mean())
            more = bounded_reading_option.completed_block(engine, option, block_nll, block_info)
            if more:
                # Fast contextual retrieval has its own 960B cadence. Holding
                # an action does not pin the eight working fact anchors.
                if getattr(engine, 'past_present_coevolution_schema', None) == 1 or hasattr(engine.model.body, 'coevolution'):
                    from scripts.ib_local.past_present_coevolution import refresh_workset
                    option.setdefault('workset_refreshes', []).append(refresh_workset(engine))
                # The just-finished block has committed learning and S/P before
                # the world supplies another actual contiguous reading block.
                r, h, next_info = engine.world.take(b'', engine.response_bytes, engine.holdout_bytes)
                data, extra = observed_pieces(r, h, next_info)
                if len(data) != 960 or next_info['source'] != 'text':
                    raise ValueError('continued reading must deliver exactly 960 natural bytes')
                offset = len(external)
                if int(next_info['start']) != int(block_info['start'])+960:
                    raise ValueError('continued reading changed the actual world cursor')
                # The root reply target stays at its actual first 480B port;
                # source accounting covers every additional real byte.
                info = dict(info)
                if 'text_source_bytes' in info:
                    info['text_source_bytes'] = {k: v+next_info['text_source_bytes'][k]
                                                for k, v in info['text_source_bytes'].items()}
                    info['text_holdout_source_bytes'] = {k: v+next_info['text_source_bytes'][k]
                        for k, v in info['text_holdout_source_bytes'].items()}
                info['holdout_previously_seen_bytes'] += (
                    next_info['response_previously_seen_bytes']+next_info['holdout_previously_seen_bytes'])
                info['holdout_position_fresh'] = info['holdout_previously_seen_bytes'] == 0
                external = np.concatenate((external, data))
                pieces.extend(StreamPiece(p.source, offset+p.start, offset+p.end,
                    None if p.probe_start is None else offset+p.probe_start,
                    None if p.probe_end is None else offset+p.probe_end) for p in extra)
                block_end, block_info, block_error_start = len(external), next_info, len(errors)
    if delayed is not None:
        engine.delayed_applied += 1
    engine.pending = pending
    witness_post = engine.witness_nll()
    text_gain = settle_sources(readings)
    if getattr(engine, 'causal_use_arm', None) == 'C2':
        # Allocation depends on observed content and existing retention credit,
        # not on this event's reward. Know the actual individual/merged event
        # before scoring U, then publish its reward before any downstream read.
        slot, merged = memory.commit_experience(source, action, all_features[short_at], long,
            engine.model.encoder, utterance, 0., source_time)
        from scripts.ib_local.identified_event_credit import capture
        capture(engine, source=source, action=action, short=all_features[short_at], long=long,
                words=utterance, source_time=source_time, slot=slot, merged=merged)
        g, use_receipt = settle_use(engine, text_gain, slot=slot, merged=merged)
        if not merged:
            memory.progress[slot] = g
    else:
        g, use_receipt = settle_use(engine, text_gain)
        slot, merged = memory.commit_experience(source, action, all_features[short_at], long,
            engine.model.encoder, utterance, g, source_time)
    if hasattr(engine, 'settlement_versions'):
        code = settlement_code(engine.settlement_kind)
        old = int(engine.settlement_versions[slot])
        engine.settlement_versions[slot] = code if not merged or old == code else 2
    evidence_available = bool(
        use_receipt is None
        or (
            bool(use_receipt.get('target_available', False))
            and (
                'finite_U' not in use_receipt
                or bool((use_receipt.get('finite_U') or {}).get('committed', False))
            )
        )
    )
    from scripts.ib_local import gaze_pointer as _gp
    if _gp.enabled(engine):
        engine._gaze_boundary_context = dict(
            model=engine.model,
            source=source.detach().clone(),
            raw_action=raw_action_for_gaze,
            actual_target_id=info.get('send_target_id') or info.get('target_object_id'),
            actual_observe_id=info.get('observe_target_id'),
            boundary_target=(
                None if getattr(engine, '_mother_boundary_target', None) is None
                else engine._mother_boundary_target.detach().clone()
            ),
            evidence_available=evidence_available,
        )
    memory.option[slot] = record['choice']
    memory.feeling[slot] = record['feeling']
    engine.prev = dict(g=g, slot=slot, actor=actor,
        inputs=record['critic_inputs'][record['choice']].detach(),
        value=float(record['values'][record['choice']]), choice=record['choice'],
        settlement_kind=engine.settlement_kind)
    option_receipt = bounded_reading_option.finish(engine, option)
    if option_receipt is not None:
        engine.prev.update(duration_blocks=option_receipt['blocks'], raw_terminal_gain=g)
        engine.prev['g'] = g*option_receipt['terminal_gain_discount']
    engine.intero_raw = np.array([float(np.mean(np.concatenate(errors))), 100*g,
                                 float(first_diag['latent'])])
    engine.intero_stats.update(engine.intero_raw)
    nll = np.concatenate(errors)
    mother = int(info.get('mother_bytes', 0)) if info.get('source') in _INTERACTIVE_SOURCES else 0
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
        self_speech_nll=self_speech_nll,
        draft_raw_action_l2=draft_raw_action_l2,
        mother_response_nll=float(nll[:mother].mean()) if mother else None,
        ambient_response_nll=float(nll[mother:len(response)].mean()) if mother<len(response) else None,
        heldout_pre_nll=pre, heldout_post_nll=post,
        progress=pre-post if ambient else None, fresh=bool(info['holdout_position_fresh']),
        joint_pre_nll=pre, joint_post_nll=post, joint_progress=pre-post if ambient else None,
        witness_pre=witness_pre,witness_post=witness_post,witness_g=witness_pre-witness_post,
        settlement_kind=engine.settlement_kind, settlement=g, metal_g=g, usable_signal=g,
        evidence_available=evidence_available,
        target_object_id=info.get('target_object_id'),
        send_target_id=info.get('send_target_id'),
        observe_target_id=info.get('observe_target_id'),
        interaction_mode=info.get('interaction_mode'),
        witnesses=len(engine.w_ids),witness_admitted=engine.w_admitted,admitted=False,
        td_previous=td,value_now=value_now,delayed_actor_applied=engine.delayed_applied,
        actor_norm=actor_norm,policy_replay_max_error=policy_error,
        **({'selector_age_delivery': age_delivery} if age_delivery is not None else {}),
        short_loss=float(first_diag['latent']),long_loss=float(first_diag['latent_long']),**outcome_diag,
        episode_slot=slot,episode_merged=merged,age_component_ablation=None,
        gradient_norms=norms,updates=len(pieces)+int(td is not None and engine.mode != 'silence'),
        read_credit=dict(response=credits[0],heldout=credits[-1]),
        habit_logits=record['habit_logits'],imagination_offset=record['offset'],
        imagined_value=record['values'],precision=engine.precision(),
        critic_rms=None if engine.critic_error is None else float(np.sqrt(engine.critic_error)),
        interoception=engine.intero_raw.tolist(),
        causal_use=use_receipt,
        **({'bounded_reading_option': option_receipt} if option_receipt is not None else {}),
        world_learning=dict(schema=1,readings=readings,settlement=text_gain,
            source_roles=[p.source for p in pieces],training_pieces=len(pieces),
            external_bytes_once=len(external),counterfactual_scope='same actual future suffix before/after prefix learning',
            meaning='equal-role mean predictive gain; actual stream and observations retained'))
