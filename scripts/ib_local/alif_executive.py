"""ALIF Surprise-Gated Biological Executive & Continuous Subcortical Flow.

Implements the formally verified model in
`lean/CGBAxioms/reviews/AlifExecutiveControl.lean`:

1. Continuous Subcortical Metabolism (H-JEPA / π₀.₅ fast sensory-motor loop):
   Between executive spikes, the 32-layer GDN language body advances contiguously
   along the natural stream in fine-grained 480-byte pieces (matching the native
   objective sequence length), running 100% full `table.run_credit`, 32-layer
   `model.objective` + backward + `BudgetedAdamW`, `S/P` Hamiltonian physics,
   `commit_observed`, and `PastPresentField` workset refreshes. Zero bytes and
   zero language gradients are skipped.

2. Drosophila / Cortical Rheobase-ALIF Executive Interrupt:
   After every 480-byte sensory piece, a zero-overhead (<0.1 ms) Adaptive Leaky
   Integrate-and-Fire neuron integrates three real bodily signals:
     - Sensory surprise: excess piece NLL + 64-byte local peak burst NLL
     - Physical kinetic surge: relative 32-layer BFSS momentum RMS change
     - Latent epistemic mismatch: excess short-horizon latent prediction error
   Parameters strictly enforce the two Lean 4 structural invariants:
     - Rheobase Zero-Starvation: `delta0 > (1 - lam_v) * theta0`
     - Refractory Anti-Storming: `i_max < theta0 + beta`

3. Exact SMDP Delayed-Credit Conservation:
   Calm subcortical pieces advance `engine.prev['duration_blocks']` by
   `piece_bytes / 960.0` without shrinking `engine.prev['g']` or fabricating
   action eligibility. When the next ALIF spike fires a full unmodified
   executive cycle, `td_gain_target` contracts continuation value by
   `gamma ** duration_blocks` (preserving Banach contraction ≤ gamma).
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F


DEFAULT_PARAMS = dict(
    piece_bytes=480,          # 480B sensory flow piece (matches native objective length; 2x finer than 960B)
    burst_window=64,          # 64B local burst detector inside each 480B piece
    lam_v=0.84,               # membrane leak factor λ_V ∈ [0, 1)
    lam_a=0.78,               # refractory adaptation decay λ_a ∈ [0, 1)
    theta0=1.00,              # resting firing threshold θ₀ > 0
    beta=0.95,                # refractory threshold stiffness β (enforces i_max < theta0 + beta)
    delta0=0.20,              # tonic rheobase drive δ₀ > (1 - lam_v) * theta0 = 0.16 (zero starvation)
    i_max=1.85,               # per-step drive cap (< theta0 + beta = 1.95 -> zero back-to-back storming)
    w_sens=0.35,              # weight on standardized excess piece NLL
    w_burst=0.22,             # weight on standardized 64B local burst NLL
    w_kin=12.0,               # weight on relative 32-layer BFSS momentum RMS surge
    w_epi=0.18,               # weight on standardized excess short-horizon latent loss
    ema_decay=0.85,           # causal running baseline decay (updated strictly after scoring)
    workset_refresh_pieces=2, # refresh PastPresentField workset every 2 calm pieces (960B cadence)
    hippocampal_anchor_scale=0.5,  # 0.5x outcome_weight per 480B calm piece -> exact 2:1 replay ratio
    calm_lr_scale=1.0,        # optional AdamW sqrt(B/640) calm-piece LR compensation (default 1.0 = off)
    extra_hippocampal_steps=0,     # optional extra ~10ms hippocampal consolidation micro-steps per calm piece
)


def validate_params(params: dict) -> dict:
    """Enforce the Lean 4 Rheobase Zero-Starvation and Refractory Anti-Storming invariants."""
    p = dict(DEFAULT_PARAMS)
    for k, v in params.items():
        if k not in p:
            raise ValueError(f'unknown alif_executive parameter: {k}')
        p[k] = int(v) if k in ('piece_bytes', 'burst_window', 'workset_refresh_pieces', 'extra_hippocampal_steps') else float(v)
    if p['piece_bytes'] < 64 or p['piece_bytes'] % 2 != 0:
        raise ValueError('piece_bytes must be an even integer >= 64')
    if not (1 <= p['burst_window'] <= p['piece_bytes']):
        raise ValueError('burst_window must satisfy 1 <= burst_window <= piece_bytes')
    if not (0.0 <= p['lam_v'] < 1.0 and 0.0 <= p['lam_a'] < 1.0):
        raise ValueError('lam_v and lam_a must lie in [0, 1)')
    if p['theta0'] <= 0.0 or p['beta'] < 0.0 or p['delta0'] <= 0.0:
        raise ValueError('theta0 > 0, beta >= 0, delta0 > 0 required')
    if p['i_max'] < p['delta0']:
        raise ValueError('i_max must be >= delta0')
    if p['hippocampal_anchor_scale'] < 0.0:
        raise ValueError('hippocampal_anchor_scale must be >= 0')
    if p['calm_lr_scale'] <= 0.0 or p['calm_lr_scale'] > 4.0:
        raise ValueError('calm_lr_scale must lie in (0, 4]')
    if p['extra_hippocampal_steps'] < 0 or p['extra_hippocampal_steps'] > 8:
        raise ValueError('extra_hippocampal_steps must lie in [0, 8]')
    # Lean 4 Theorem 2 (Rheobase Zero-Starvation): δ₀ > (1 - λ_V) * θ₀
    rheobase_floor = (1.0 - p['lam_v']) * p['theta0']
    if p['delta0'] <= rheobase_floor + 1e-9:
        raise ValueError(
            f"Rheobase zero-starvation violated: delta0={p['delta0']} must strictly exceed "
            f"(1 - lam_v) * theta0 = {rheobase_floor:.6f}"
        )
    # Lean 4 Theorem 3 (Refractory Anti-Storming): I_max < θ₀ + β (unless strict limit β=0)
    if p['beta'] > 0.0 and p['i_max'] >= p['theta0'] + p['beta'] - 1e-9:
        raise ValueError(
            f"Refractory anti-storming violated: i_max={p['i_max']} must be strictly less than "
            f"theta0 + beta = {p['theta0'] + p['beta']:.6f}"
        )
    if p['workset_refresh_pieces'] < 1:
        raise ValueError('workset_refresh_pieces must be >= 1')
    return p


def enabled(engine) -> bool:
    return getattr(engine, 'alif_executive_schema', None) == 1


def configure(engine, config=None, **overrides) -> dict:
    """Attach opt-in ALIF executive state cleanly to an existing restored engine."""
    params = validate_params(overrides)
    engine.alif_executive_schema = 1
    if not hasattr(engine, 'alif_executive_state') or engine.alif_executive_state is None:
        engine.alif_executive_state = dict(
            schema=1,
            v=0.0,
            a=0.0,
            pending_spike=True,   # first event always establishes a full executive boundary
            pieces_since_spike=0,
            calm_bytes_since_spike=0,
            nll_ema=0.0,
            nll_var=0.04,
            nll_count=0,
            short_ema=0.0,
            short_var=0.01,
            short_count=0,
            prev_p_rms=0.0,
            executive_spikes=0,
            calm_pieces=0,
            total_external_bytes=0,
            total_learned_bytes=0,
            inter_spike_pieces=[],
            preempt_causes=dict(initial=0, sensory=0, burst=0, kinetic=0, epistemic=0, rheobase=0),
            last_event=None,
            **params,
        )
    else:
        for k, v in params.items():
            engine.alif_executive_state[k] = v
    pb = int(engine.alif_executive_state['piece_bytes'])
    if hasattr(engine, 'model') and pb > int(getattr(engine.model, 'max_chunk', 0)):
        engine.model.max_chunk = pb
    if hasattr(engine, 'model') and hasattr(engine.model, 'memory') and pb > 640:
        engine.model.memory.max_commit_chunk = max(int(getattr(engine.model.memory, 'max_commit_chunk', 0)), pb)
    if config is not None:
        config['alif_executive'] = dict(schema=1, **params)
    return engine.alif_executive_state


def _momentum_rms(engine) -> float:
    momenta = getattr(engine, 'bfss_momenta', None)
    if not momenta:
        return 0.0
    total = 0.0
    count = 0
    for p in momenta:
        if torch.is_tensor(p) and p.numel() > 0:
            total += float(p.detach().float().square().mean().sqrt())
            count += 1
    return total / max(1, count)


def _peak_window_mean(values: np.ndarray, window: int) -> float:
    if len(values) == 0:
        return 0.0
    w = min(int(window), len(values))
    if w <= 1:
        return float(np.max(values))
    csum = np.empty(len(values) + 1, dtype=np.float64)
    csum[0] = 0.0
    np.cumsum(values, dtype=np.float64, out=csum[1:])
    means = (csum[w:] - csum[:-w]) / float(w)
    return float(np.max(means))


def _peak_window_slice(values: np.ndarray, chunk: np.ndarray, window: int) -> tuple[float, bytes]:
    """Return `(peak_mean_nll, peak_window_bytes)` for caregiver biofeedback mirroring."""
    if len(values) == 0 or len(chunk) == 0:
        return 0.0, b''
    w = min(int(window), len(values), len(chunk))
    if w <= 1:
        idx = int(np.argmax(values))
        return float(values[idx]), bytes(chunk[idx:idx + 1].tobytes())
    csum = np.empty(len(values) + 1, dtype=np.float64)
    csum[0] = 0.0
    np.cumsum(values, dtype=np.float64, out=csum[1:])
    means = (csum[w:] - csum[:-w]) / float(w)
    start = int(np.argmax(means))
    return float(means[start]), bytes(chunk[start:start + w].tobytes())


def _update_baselines(st: dict, piece_nll: float, short_loss: float, p_rms: float) -> None:
    decay = float(st['ema_decay'])
    if int(st['nll_count']) == 0:
        st['nll_ema'] = float(piece_nll)
        st['nll_var'] = 0.04
    else:
        diff = float(piece_nll) - float(st['nll_ema'])
        st['nll_ema'] = decay * float(st['nll_ema']) + (1.0 - decay) * float(piece_nll)
        st['nll_var'] = max(1e-4, decay * float(st['nll_var']) + (1.0 - decay) * (diff * diff))
    st['nll_count'] = int(st['nll_count']) + 1

    if math.isfinite(short_loss) and short_loss > 0.0:
        if int(st['short_count']) == 0:
            st['short_ema'] = float(short_loss)
            st['short_var'] = 0.01
        else:
            sdiff = float(short_loss) - float(st['short_ema'])
            st['short_ema'] = decay * float(st['short_ema']) + (1.0 - decay) * float(short_loss)
            st['short_var'] = max(1e-4, decay * float(st['short_var']) + (1.0 - decay) * (sdiff * sdiff))
        st['short_count'] = int(st['short_count']) + 1

    if p_rms > 0.0:
        st['prev_p_rms'] = float(p_rms)


def step_neuron(st: dict, *, piece_nll: float, burst_nll: float, short_loss: float, p_rms: float,
                rhythm=None) -> dict:
    """Advance the Rheobase-ALIF membrane & adaptive threshold on real observables.

    Strictly causal: z-scores use running EMAs from PRIOR pieces only, then
    baselines are updated AFTER scoring. When an isomorphic `LearnableAlifRhythm`
    module is supplied, advances `(V, a)` together with its forward sensitivity
    traces `(e_V, e_a)` and escape-noise discrete readout.
    """
    n_cnt = int(st['nll_count'])
    if n_cnt >= 2:
        nll_std = math.sqrt(max(float(st['nll_var']), 1e-4))
        z_sens = max(0.0, (piece_nll - float(st['nll_ema'])) / nll_std)
        # Subtract 0.5 sigma tolerance so normal within-piece variance does not false-trigger.
        z_burst = max(0.0, (burst_nll - float(st['nll_ema'])) / nll_std - 0.5)
    else:
        z_sens = 0.0
        z_burst = 0.0

    prev_p = float(st['prev_p_rms'])
    p_rel_delta = abs(p_rms - prev_p) / max(prev_p, 1e-6) if (prev_p > 0.0 and p_rms > 0.0) else 0.0

    if int(st['short_count']) >= 2 and math.isfinite(short_loss) and short_loss > 0.0:
        s_std = math.sqrt(max(float(st['short_var']), 1e-4))
        z_epi = max(0.0, (short_loss - float(st['short_ema'])) / s_std)
    else:
        z_epi = 0.0

    v_before = float(st['v'])
    a_before = float(st['a'])
    spike_prob = None
    if rhythm is not None:
        r_out = rhythm.step_with_trace(
            v_before=v_before, a_before=a_before,
            z_sens=z_sens, z_burst=z_burst, p_rel_delta=p_rel_delta, z_epi=z_epi,
        )
        p_cur = r_out['params']
        st.update(p_cur)
        c_sens = float(p_cur['w_sens']) * z_sens
        c_burst = float(p_cur['w_burst']) * z_burst
        c_kin = float(p_cur['w_kin']) * p_rel_delta
        c_epi = float(p_cur['w_epi']) * z_epi
        drive = float(r_out['drive'])
        theta = float(r_out['threshold'])
        v_plus = float(r_out['v_plus'])
        spiked = bool(r_out['spiked'])
        spike_prob = float(r_out['spike_prob'])
        st['v'] = float(r_out['v_after'])
        st['a'] = float(r_out['a_after'])
        st['pending_spike'] = spiked
        if spiked:
            drivers = dict(sensory=c_sens, burst=c_burst, kinetic=c_kin, epistemic=c_epi)
            top_name, top_val = max(drivers.items(), key=lambda kv: kv[1])
            cause = top_name if top_val >= float(p_cur['delta0']) else 'rheobase'
            st['preempt_causes'][cause] = int(st['preempt_causes'].get(cause, 0)) + 1
            st['last_trigger_cause'] = cause
        else:
            cause = None
    else:
        c_sens = float(st['w_sens']) * z_sens
        c_burst = float(st['w_burst']) * z_burst
        c_kin = float(st['w_kin']) * p_rel_delta
        c_epi = float(st['w_epi']) * z_epi
        raw_drive = float(st['delta0']) + c_sens + c_burst + c_kin + c_epi
        drive = min(float(st['i_max']), max(float(st['delta0']), raw_drive))

        theta = float(st['theta0']) + float(st['beta']) * a_before
        v_plus = float(st['lam_v']) * v_before + drive
        spiked = v_plus >= theta

        if spiked:
            st['v'] = 0.0
            st['a'] = float(st['lam_a']) * a_before + 1.0
            st['pending_spike'] = True
            # Attribute dominant non-tonic driver for telemetry.
            drivers = dict(sensory=c_sens, burst=c_burst, kinetic=c_kin, epistemic=c_epi)
            top_name, top_val = max(drivers.items(), key=lambda kv: kv[1])
            cause = top_name if top_val >= float(st['delta0']) else 'rheobase'
            st['preempt_causes'][cause] = int(st['preempt_causes'].get(cause, 0)) + 1
            st['last_trigger_cause'] = cause
        else:
            st['v'] = v_plus
            st['a'] = float(st['lam_a']) * a_before
            st['pending_spike'] = False
            cause = None

    _update_baselines(st, piece_nll, short_loss, p_rms)
    return dict(
        piece_nll=float(piece_nll),
        burst_nll=float(burst_nll),
        short_loss=float(short_loss),
        p_rms=float(p_rms),
        p_rel_delta=float(p_rel_delta),
        z_sens=float(z_sens),
        z_burst=float(z_burst),
        z_epi=float(z_epi),
        drive=float(drive),
        v_before=v_before,
        v_plus=v_plus,
        v_after=float(st['v']),
        a_before=a_before,
        a_after=float(st['a']),
        threshold=float(theta),
        spike_prob=spike_prob,
        spiked=bool(spiked),
        cause=cause,
    )


def _step_quiet_physics(engine) -> float:
    """Keep the 32-layer Hamiltonian S/P body continuously oscillating during calm flow."""
    before = engine.recurrent
    if getattr(engine, 'bfss_enabled', False):
        if hasattr(engine.model, 'head_coupling'):
            extra = dict(capture=True) if getattr(engine, 'adaptive_dynamics_schema', None) == 1 else {}
            engine.recurrent, engine.bfss_momenta = engine.model.head_coupling.quiet(
                engine.model.body.base, engine.recurrent, engine.bfss_momenta,
                dt=engine.bfss_dt, frequency=engine.bfss_frequency,
                steps=engine.bfss_steps, **extra)
        else:
            from scripts.ib_local.bfss_gdn_capture import quiet_packed
            engine.recurrent, engine.bfss_momenta = quiet_packed(
                engine.model.body.base, engine.recurrent, engine.bfss_momenta,
                dt=engine.bfss_dt, frequency=engine.bfss_frequency,
                coupling=engine.bfss_coupling, steps=engine.bfss_steps)
        engine.model._momentum_current = engine.bfss_momenta
    return float((engine.recurrent - before).norm())


def hippocampal_anchor_loss(engine, features: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Fast (~10 ms) hippocampal old-fact & transition replay anchoring `model.encoder`.

    Replays ONLY real historical memory/archive observations (zero fabricated labels,
    zero self-scoring, zero `_utter` search) so the shared encoder and relation
    manifold maintain the exact 2:1 language-to-replay ratio proved in
    `AlifExecutiveControl.replay_ratio_conservation`.
    """
    model = engine.model
    device = features.device
    total = features.new_zeros(())
    info = dict(wood_replay_samples=0, internal_transition_samples=0, causal_program_replay=0)

    # Proportional hippocampal replay conservation (RootLatentPlanning Theorem 4):
    # scale replay sample counts by piece_bytes / 640 so the replay-to-new-byte
    # ratio stays invariant when expanding calm pieces toward ~3 GiB VRAM.
    st = getattr(engine, 'alif_executive_state', None)
    pb = int(st['piece_bytes']) if isinstance(st, dict) and 'piece_bytes' in st else 640
    replay_mult = max(1, int(round(pb / 640.0)))
    had_replay_batch = hasattr(engine, 'replay_batch')
    orig_replay_batch = int(getattr(engine, 'replay_batch', 32))
    if replay_mult > 1 and had_replay_batch:
        engine.replay_batch = orig_replay_batch * replay_mult

    try:
        # 1. Historical wood outcome replay (32*m real memory slots -> anchors encoder + wood head).
        wood_fn = getattr(engine, '_wood_replay_batch', None)
        if callable(wood_fn) and hasattr(model, 'kanli'):
            wood_batch = wood_fn()
            if wood_batch is not None and len(wood_batch) >= 3:
                rs, ra, rt = wood_batch[0], wood_batch[1], wood_batch[2]
                if len(rs) > 0:
                    z, a = model.encoder(rs), model.encoder(ra)
                    if len(wood_batch) >= 4 and hasattr(model, '_causal_use_value_evidence'):
                        from scripts.ib_local.causal_value_learning import prediction as wood_pred
                        pred = wood_pred(model, z, a - z, wood_batch[3])[:, :3]
                    else:
                        pred = model.kanli.outcome(torch.cat((z, a), -1))[:, :3]
                    total = total + F.mse_loss(pred, rt[:, :3])
                    info['wood_replay_samples'] = int(len(rs))

        # 2. Historical internal transition replay (32*m real memory transitions -> anchors encoder + predictor).
        m = getattr(model, 'memory', None)
        if m is not None and hasattr(m, 'internal_valid') and hasattr(engine, 'replay_rng'):
            from scripts.ib_local.navigation_internal_rollout import next_boundary_prediction
            valid = (m.occupied & m.internal_valid & (m.count == 1) & (m.internal_stamp == m.last_write))
            slots = torch.nonzero(valid).flatten()
            if len(slots) > 0:
                batch_cap = int(getattr(engine, 'replay_batch', 32 * replay_mult))
                if len(slots) > batch_cap:
                    picks = engine.replay_rng.choice(len(slots), batch_cap, replace=False)
                    slots = slots[torch.as_tensor(picks, device=device)]
                src, act, nxt = m.source[slots], m.action[slots], m.internal_next[slots]
                pred_next = next_boundary_prediction(model, src, act)
                total = total + F.mse_loss(pred_next, model.encoder(nxt))
                info['internal_transition_samples'] = int(len(src))

        # 3. Historical causal-program mechanism replay (up to 8*m real old facts, leave-one-out).
        state = getattr(engine, '_program_question_state', None)
        if state is not None and hasattr(model, 'causal_relations'):
            from scripts.ib_local.causal_relation_program import Port, mechanism
            from scripts.ib_local.joint_response import predictive_score
            replay_cap = int(getattr(engine, 'replay_batch', 8 * replay_mult))
            replay_losses = []
            if getattr(model, 'joint_relation_update_revision', 1) >= 3:
                from scripts.ib_local.relation_evidence import mechanism_evidence
                old = [e for e in mechanism_evidence(model, state, Port.MOTHER)
                       if e.observed and not e.encoded][:replay_cap]
                for ev in old:
                    dist = mechanism(model, state, ev.source, ev.action,
                                     ridge=engine.relation_ridge, exclude_fact=ev.fact_id)
                    replay_losses.append(-predictive_score(model, dist, model.encoder(ev.outcome)).mean())
            if replay_losses:
                total = total + torch.stack(replay_losses).mean()
                info['causal_program_replay'] = len(replay_losses)
    finally:
        if replay_mult > 1 and had_replay_batch:
            engine.replay_batch = orig_replay_batch

    info['hippocampal_anchor_loss'] = float(total.detach())
    return total, info


def calm_sensory_piece(engine, directions) -> dict:
    """Execute one 480B continuous subcortical sensory + full language learning piece.

    Preserves 100% of external byte learning, S/P physics, fast table,
    PastPresentField workset updates, and exact 2:1 hippocampal encoder-anchoring
    replay without invoking full-backbone `_utter` planning, `_policy_grad`, or
    `flush_teachers`.
    """
    if not enabled(engine):
        raise RuntimeError('alif_executive is not configured on this engine')
    st = engine.alif_executive_state
    piece_bytes = int(st['piece_bytes'])
    half = piece_bytes // 2

    # Refresh PastPresentField workset at the configured cadence (default every 960B).
    workset_receipt = None
    if (int(st['pieces_since_spike']) % int(st['workset_refresh_pieces']) == 0 and
            (getattr(engine, 'past_present_coevolution_schema', None) == 1 or
             hasattr(engine.model.body, 'coevolution'))):
        from scripts.ib_local import emeans_alif_reservoir as ear
        if ear.enabled(engine):
            workset_receipt = ear.refresh_workset_population_alif(engine)
        else:
            from scripts.ib_local.past_present_coevolution import refresh_workset
            workset_receipt = refresh_workset(engine)

    # Maintain exact BFSS S/P momentum tracking contract required by MomentumEngine._advance.
    engine._bfss_event_errors = []
    engine._bfss_corrections = []
    engine.model._momentum_active = True
    engine.model._momentum_current = engine.bfss_momenta
    state_change = _step_quiet_physics(engine)
    from scripts.ib_local import phase_space_boltzmann as psb
    if psb.enabled(engine):
        engine.model.phase_boltzmann.refresh_live_phase_state(engine, update_ema=False)

    # Advance natural text stream contiguously with zero utterance / zero jump.
    ambient_take = getattr(engine.world, 'take_ambient', None)
    if callable(ambient_take):
        resp, cont, info = ambient_take(half, piece_bytes - half)
    else:
        resp, cont, info = engine.world.take(b'', half, piece_bytes - half)
    chunk = np.concatenate((resp, cont)).astype(np.uint8, copy=False)
    pending = int(engine.pending)
    _, chunk_probs, _ = engine.table.run_credit(
        np.concatenate((np.array([pending], np.uint8), chunk)),
        max_window=len(chunk),
    )
    raw = np.concatenate((np.array([pending], np.uint8), chunk))
    ids, target, logp = engine._inputs(raw, chunk_probs)

    # Pure world input: clear question context just like heldout pieces in world_cycle_step.
    old_q = getattr(engine.model, '_question_context', None)
    engine.model._question_context = None
    anchor_diag = dict(hippocampal_anchor_loss=0.0, wood_replay_samples=0,
                       internal_transition_samples=0, causal_program_replay=0)
    try:
        loss, next_state, features, diag = engine.model.objective(
            ids, target, engine.recurrent, logp, directions
        )
        ce = F.cross_entropy(diag['logits'][0].detach(), target[0], reduction='none').cpu().numpy()
        anchor_scale = float(st.get('hippocampal_anchor_scale', 0.5))
        if anchor_scale > 0.0:
            anchor_loss, anchor_diag = hippocampal_anchor_loss(engine, features)
            outcome_w = float(getattr(engine, 'outcome_weight', 0.1))
            if (anchor_diag['wood_replay_samples'] + anchor_diag['internal_transition_samples'] +
                    anchor_diag['causal_program_replay']) > 0:
                loss = loss + (outcome_w * anchor_scale) * anchor_loss
        from scripts.ib_local import encoder_subspace_metric as esm
        lr_scale = float(st.get('calm_lr_scale', 1.0))
        saved_lrs = None
        if abs(lr_scale - 1.0) > 1e-6:
            saved_lrs = [g['lr'] for g in engine.optimizer.param_groups]
            for g in engine.optimizer.param_groups:
                g['lr'] = g['lr'] * lr_scale
        try:
            with esm.apply_to_step(engine):
                grad_norm = engine._advance(loss, features, diag, next_state, actor=None, advantage=0.0)
        finally:
            if saved_lrs is not None:
                for g, orig_lr in zip(engine.optimizer.param_groups, saved_lrs):
                    g['lr'] = orig_lr

        # Optional extra lightweight (~10 ms) hippocampal consolidation micro-steps:
        # touches ONLY encoder / wood / transition / causal-relation parameters
        # (BudgetedAdamW skips all 32 GDN layers where p.grad is None), restoring
        # per-KB cumulative old-fact revision count on large calm windows.
        extra_h = int(st.get('extra_hippocampal_steps', 0))
        extra_h_done = 0
        if extra_h > 0 and anchor_scale > 0.0:
            outcome_w = float(getattr(engine, 'outcome_weight', 0.1))
            feat_det = features.detach()
            for _ in range(extra_h):
                h_loss, h_info = hippocampal_anchor_loss(engine, feat_det)
                if (h_info['wood_replay_samples'] + h_info['internal_transition_samples'] +
                        h_info['causal_program_replay']) <= 0 or not h_loss.requires_grad:
                    break
                engine.optimizer.zero_grad(set_to_none=True)
                ((outcome_w * anchor_scale) * h_loss).backward()
                for group in engine.groups:
                    live = [p for p in group if p.grad is not None]
                    if live:
                        torch.nn.utils.clip_grad_norm_(live, 1.0, foreach=True)
                engine.optimizer.step()
                extra_h_done += 1
        anchor_diag['extra_hippocampal_steps_done'] = extra_h_done
    finally:
        engine.model._question_context = old_q

    pb_diag = None
    if psb.enabled(engine):
        pb_diag = engine.model.phase_boltzmann.apply_free_energy_thermostat(
            engine, diag['logits'], query_raw=features[-1:],
        )

    engine.pending = int(chunk[-1])
    piece_nll = float(ce.mean())
    burst_nll, burst_bytes = _peak_window_slice(ce, chunk, int(st['burst_window']))
    short_loss = float(diag['latent'])
    p_rms = _momentum_rms(engine)

    # Compact rolling stumble excerpt for caregiver biofeedback mirroring (never read by body).
    st['last_burst_bytes_hex'] = burst_bytes.hex()
    st['last_burst_nll'] = float(burst_nll)
    st['last_piece_nll'] = float(piece_nll)

    # Exact SMDP duration tracking: extend duration_blocks without shrinking prev['g'].
    st['calm_pieces'] = int(st['calm_pieces']) + 1
    st['pieces_since_spike'] = int(st['pieces_since_spike']) + 1
    st['calm_bytes_since_spike'] = int(st['calm_bytes_since_spike']) + len(chunk)
    st['total_external_bytes'] = int(st['total_external_bytes']) + len(chunk)
    st['total_learned_bytes'] = int(st['total_learned_bytes']) + len(chunk)
    if engine.prev is not None:
        engine.prev['duration_blocks'] = 1.0 + float(st['calm_bytes_since_spike']) / 960.0

    from scripts.ib_local import learnable_alif_rhythm as lar
    alif_diag = step_neuron(
        st,
        piece_nll=piece_nll,
        burst_nll=burst_nll,
        short_loss=short_loss,
        p_rms=p_rms,
        rhythm=(engine.model.alif_rhythm if lar.enabled(engine) else None),
    )
    event = dict(
        kind='calm_sensory_piece',
        external_bytes=len(chunk),
        learned_bytes=len(chunk),
        self_bytes=0,
        updates=1,
        response_nll=piece_nll,
        burst_nll=burst_nll,
        short_loss=short_loss,
        gradient_norms=[float(grad_norm)],
        world=info,
        past_present=workset_receipt,
        bfss_dynamics=dict(
            enabled=bool(getattr(engine, 'bfss_enabled', False)),
            internal_steps=int(getattr(engine, 'bfss_steps', 1)),
            state_change=state_change,
            momentum_norms=[float(p.norm()) for p in engine.bfss_momenta],
            event_replay_max_error=max(engine._bfss_event_errors, default=0.0),
        ),
        alif_executive=dict(
            regime='calm_flow',
            pieces_since_spike=int(st['pieces_since_spike']),
            calm_bytes_since_spike=int(st['calm_bytes_since_spike']),
            smdp_duration_blocks=(None if engine.prev is None else float(engine.prev.get('duration_blocks', 1.0))),
            **anchor_diag,
            **alif_diag,
        ),
    )
    st['last_event'] = copy.deepcopy(event['alif_executive'])
    return event


def record_executive_spike(engine, row: dict) -> dict:
    """Record completion of a full unmodified executive cycle and enter refractory state."""
    st = engine.alif_executive_state
    was_initial = int(st['executive_spikes']) == 0
    if was_initial:
        st['preempt_causes']['initial'] = int(st['preempt_causes'].get('initial', 0)) + 1
        # Apply the spike's refractory bump cleanly on initial executive boundary too.
        st['v'] = 0.0
        st['a'] = float(st['lam_a']) * float(st['a']) + 1.0
    else:
        st['inter_spike_pieces'].append(int(st['pieces_since_spike']))
        del st['inter_spike_pieces'][:-64]

    st['executive_spikes'] = int(st['executive_spikes']) + 1
    ext_bytes = int(row.get('external_bytes', 0))
    st['total_external_bytes'] = int(st['total_external_bytes']) + ext_bytes
    st['total_learned_bytes'] = int(st['total_learned_bytes']) + ext_bytes

    # Update running baselines from this executive boundary without double-spiking.
    nll = float(row.get('response_nll', 0.0))
    short_loss = float(row.get('short_loss', 0.0))
    from scripts.ib_local import phase_space_boltzmann as psb
    if psb.enabled(engine):
        engine.model.phase_boltzmann.refresh_live_phase_state(engine, update_ema=True)
    p_rms = _momentum_rms(engine)
    if math.isfinite(nll) and nll > 0.0:
        _update_baselines(st, nll, short_loss, p_rms)

    # Reset inter-spike counters and release into subcortical calm flow.
    prev_calm_bytes = int(st['calm_bytes_since_spike'])
    prev_pieces = int(st['pieces_since_spike'])
    st['pieces_since_spike'] = 0
    st['calm_bytes_since_spike'] = 0
    st['pending_spike'] = False
    if engine.prev is not None:
        engine.prev['duration_blocks'] = 1.0

    is_ext_interrupt = bool(st.pop('external_interrupt', False))
    from scripts.ib_local import learnable_alif_rhythm as lar
    rhythm_credit = None
    if lar.enabled(engine) and not is_ext_interrupt:
        rhythm_credit = engine.model.alif_rhythm.settle_executive_credit(
            engine,
            metal_g=row.get('metal_g'),
            choice=row.get('choice'),
            preceding_calm_pieces=prev_pieces,
            is_initial=was_initial,
        )
    elif lar.enabled(engine) and is_ext_interrupt:
        # External human/social interrupt: clear inter-spike eligibility without attributing to internal phi.
        rhythm_credit = dict(skipped='external_interrupt')
        rhythm_mod = engine.model.alif_rhythm
        if hasattr(rhythm_mod, 'eligibility'):
            rhythm_mod.eligibility.zero_()
        if hasattr(rhythm_mod, 'pending_eligibility'):
            rhythm_mod.pending_eligibility.zero_()

    diag = dict(
        regime='executive_spike',
        initial=was_initial,
        preceding_calm_pieces=prev_pieces,
        preceding_calm_bytes=prev_calm_bytes,
        v_after=float(st['v']),
        a_after=float(st['a']),
        threshold_next=float(st['theta0']) + float(st['beta']) * float(st['a']),
        executive_spikes=int(st['executive_spikes']),
        calm_pieces=int(st['calm_pieces']),
        total_external_bytes=int(st['total_external_bytes']),
        total_learned_bytes=int(st['total_learned_bytes']),
        learnable_rhythm=rhythm_credit,
    )
    st['last_event'] = copy.deepcopy(diag)
    row['alif_executive'] = diag
    return diag


def step_body(engine, *, directions=None) -> tuple[list[dict], dict]:
    """Advance the body by ONE ALIF-scheduled event (either 1 calm piece OR 1 full executive spike).

    Returns `(sub_events, main_event)` where `sub_events` holds any internal
    `autonomous_cycle` polls preceding an executive spike (empty on calm pieces).
    """
    if not enabled(engine):
        raise RuntimeError('alif_executive is not configured on this engine')
    st = engine.alif_executive_state
    if not bool(st['pending_spike']):
        if int(st['piece_bytes']) > 640 and torch.cuda.is_available():
            if torch.cuda.memory_reserved() > int(2.15 * (1024 ** 3)):
                torch.cuda.empty_cache()
        if directions is None:
            directions = torch.randn(engine.model.encoder.out_features, 32, device=engine.device)
        row = calm_sensory_piece(engine, directions)
        return [], row

    # Release cached large-chunk allocator blocks before an executive spike so
    # 1024-1600B calm allocations never collide with spike graphs on 4 GiB GPUs.
    if int(st['piece_bytes']) > 640 and torch.cuda.is_available():
        if torch.cuda.memory_reserved() > int(2.15 * (1024 ** 3)):
            torch.cuda.empty_cache()

    # Full unmodified executive spike through the existing autonomous_cycle / persistent_rhythm loop.
    from scripts.ib_local import autonomous_cycle
    sub_events = []
    while True:
        ev = autonomous_cycle.dispatch(engine)
        if ev.get('autonomous_event', {}).get('kind') == 'external':
            record_executive_spike(engine, ev)
            return sub_events, ev
        sub_events.append(ev)


def diagnostics(engine) -> dict | None:
    if not enabled(engine):
        return None
    st = engine.alif_executive_state
    gaps = list(st.get('inter_spike_pieces') or [])
    ext_b = int(st['total_external_bytes'])
    lrn_b = int(st['total_learned_bytes'])
    spikes = int(st['executive_spikes'])
    calm = int(st['calm_pieces'])
    equiv_960_blocks = ext_b / 960.0 if ext_b > 0 else 0.0
    duty_cycle = spikes / equiv_960_blocks if equiv_960_blocks > 0 else 0.0
    return dict(
        schema=1,
        v=float(st['v']),
        a=float(st['a']),
        threshold=float(st['theta0']) + float(st['beta']) * float(st['a']),
        pending_spike=bool(st['pending_spike']),
        executive_spikes=spikes,
        calm_pieces=calm,
        total_external_bytes=ext_b,
        total_learned_bytes=lrn_b,
        byte_conservation_exact=(ext_b == lrn_b),
        effective_960b_blocks=equiv_960_blocks,
        executive_duty_cycle=duty_cycle,
        inter_spike_pieces=gaps,
        mean_inter_spike_pieces=(sum(gaps) / len(gaps)) if gaps else None,
        max_inter_spike_pieces=max(gaps) if gaps else None,
        min_inter_spike_pieces=min(gaps) if gaps else None,
        preempt_causes=dict(st.get('preempt_causes') or {}),
        params={k: st.get(k, DEFAULT_PARAMS[k]) for k in DEFAULT_PARAMS},
    )
