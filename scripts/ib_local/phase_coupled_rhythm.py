"""Event-driven brainwave phase-coupled multi-rate rhythm for the continuing body.

Replaces forced lock-step meta-cognition on every 960-byte cycle with a
biologically grounded dual-band oscillatory gate driven by real bodily
observables (BFSS momentum kinetic activity, prediction-error surprise, and
self-chosen internal steps) rather than a fixed `step % N` counter:

- Online Gamma / Habit Flow (calm stream, low surprise, Theta trough):
  * H-JEPA theme persistence: holds the active typed question (`unknown_id`)
    across coherent steps within a bounded horizon instead of re-running full
    multi-candidate `question_choices` search.
  * Focused speech search: drafts 1 concrete utterance (`proposal_budget = 1`,
    exact singleton REINFORCE identity) instead of 4 competing branches.
  * Fast amortized valuation: runs one `compose` + exact local `finite_use_value`
    + 1-point `persistent_rhythm.read_plans`, deferring heavy 2-teacher native
    rollouts (`flush_teachers`) to Theta deliberation windows.
  * Uninterrupted absorption: skips redundant mid-transaction `_poll_inside`
    interrupts while keeping the exact between-cycle `autonomous_cycle.poll`.
  * Deferred SWR consolidation: accumulates hippocampal consolidation pressure
    instead of running 8-branch withdrawal replay on every calm half-chunk.

- Theta Deliberation + Hippocampal SWR Consolidation Window (triggered with
  0-step latency by surprise preemption or self-chosen internal steps, or
  within bounded latency by Theta phase wrap / SWR pressure threshold):
  * Full `question_choices` topic revision (H-JEPA goal update).
  * Full multi-hypothesis speech search (`joint_proposal_budget`) & `imagine`.
  * Full 2-teacher native rollout (`flush_teachers`) and mid-transaction polling.
  * Full batched withdrawal replay (`native_retention_feedback.measure`).

100% of real language absorption, 32-layer GDN `objective` + backward +
`BudgetedAdamW`, `S/P` physics, fast table, `causal_use_value` settlement,
`shared_unknown` commit, and delayed REINFORCE/TD credit execute on every cycle.
"""
from __future__ import annotations

import copy
import math

import torch


TWO_PI = 2.0 * math.pi

DEFAULT_PARAMS = dict(
    omega_min=TWO_PI / 5.0,       # guaranteed Theta wrap within <= 5 calm cycles
    omega_body=0.35,              # BFSS kinetic coupling to Theta phase velocity
    omega_err=0.55,               # prediction-error coupling to Theta phase velocity
    alpha_min=0.34,               # guaranteed SWR consolidation within <= 3 calm cycles
    alpha_err=0.35,               # surprise boost to SWR pressure
    swr_coupling_threshold=0.45,  # Theta-SWR cross-frequency coupling threshold
    surprise_threshold=1.25,      # standardized surprise (z-score) preemption threshold
    surprise_ema_decay=0.80,      # EMA decay for running response_nll baseline
    internal_preempt_steps=2,     # boundary internal steps required for preemption
    max_theme_hold=4,             # bounded H-JEPA theme-hold horizon
)


def enabled(engine) -> bool:
    return getattr(engine, 'phase_coupled_rhythm_schema', None) == 1


def configure(engine, config=None, **overrides):
    params = dict(DEFAULT_PARAMS)
    for k, v in overrides.items():
        if k not in params:
            raise ValueError(f'unknown phase_coupled_rhythm parameter: {k}')
        params[k] = int(v) if k in ('max_theme_hold', 'internal_preempt_steps') else float(v)
    if params['omega_min'] <= 0 or params['alpha_min'] <= 0:
        raise ValueError('strictly positive phase and SWR floors are required for bounded liveness')
    if params['max_theme_hold'] < 0 or params['internal_preempt_steps'] < 1:
        raise ValueError('max_theme_hold >= 0 and internal_preempt_steps >= 1 required')
    engine.phase_coupled_rhythm_schema = 1
    if not hasattr(engine, 'phase_rhythm_state') or engine.phase_rhythm_state is None:
        engine.phase_rhythm_state = dict(
            schema=1,
            cycles=0,
            theta_phase=0.0,
            swr_pressure=0.0,
            pending_theta_wrap=False,
            pending_surprise_preempt=False,
            last_within_quiet_total=int(getattr(engine, 'rhythm_state', {}).get('within_learning_quiet', 0)),
            last_within_quiet_delta=0,
            surprise_ema=0.0,
            surprise_var=0.25,
            surprise_count=0,
            active_theme=None,
            gamma_cycles=0,
            theta_cycles=0,
            swr_fires=0,
            surprise_preemptions=0,
            internal_preemptions=0,
            theme_holds=0,
            theme_revisions=0,
            poll_skips=0,
            teacher_skips=0,
            swr_skips=0,
            events=[],
            **params,
        )
    else:
        for k, v in params.items():
            engine.phase_rhythm_state.setdefault(k, v)
    engine._phase_rhythm_active_gate = None
    from scripts.ib_local import run_navigation_residual_wood as legacy
    legacy._DERIVED.add('_phase_rhythm_active_gate')
    if config is not None:
        config['phase_coupled_rhythm'] = dict(schema=1, **params)
    return engine.phase_rhythm_state


def active_gate(engine):
    if not enabled(engine):
        return None
    return getattr(engine, '_phase_rhythm_active_gate', None)


def is_theta_open(engine) -> bool:
    gate = active_gate(engine)
    return True if gate is None else bool(gate['theta_open'])


def begin_cycle(engine):
    """Evaluate coupled oscillator + surprise + internal-step gate for this cycle."""
    if not enabled(engine):
        return None
    st = engine.phase_rhythm_state
    receipt = getattr(engine, 'autonomous_action_receipt', None) or {}
    internal_steps = int(receipt.get('internal_steps', 0))
    prev_mid_quiet = int(st.get('last_within_quiet_delta', 0))
    boundary_internal_steps = max(0, internal_steps - prev_mid_quiet)
    internal_threshold = int(st.get('internal_preempt_steps', 2))
    first_cycle = st['cycles'] == 0
    surprise_exceeded = bool(st.get('pending_surprise_preempt', False))
    theta_wrapped = bool(st.get('pending_theta_wrap', False))
    swr_pressure_reached = float(st['swr_pressure']) >= 1.0 - 1e-12
    swr_coupled_ready = float(st['swr_pressure']) >= float(st['swr_coupling_threshold'])

    internal_preempt = boundary_internal_steps >= internal_threshold
    theta_open = first_cycle or surprise_exceeded or internal_preempt or theta_wrapped
    swr_open = surprise_exceeded or swr_pressure_reached or (theta_open and swr_coupled_ready)

    st['pending_theta_wrap'] = False
    st['pending_surprise_preempt'] = False
    if theta_open:
        st['theta_cycles'] += 1
    else:
        st['gamma_cycles'] += 1
    if surprise_exceeded:
        st['surprise_preemptions'] += 1
    if internal_preempt:
        st['internal_preemptions'] += 1

    base_budget = int(getattr(engine, 'joint_proposal_budget', 4))
    within_quiet_before = int(getattr(engine, 'rhythm_state', {}).get('within_learning_quiet', 0))
    gate = dict(
        regime='theta' if theta_open else 'gamma',
        theta_open=theta_open,
        swr_open=swr_open,
        allow_mid_poll=theta_open,
        proposal_budget=base_budget if theta_open else 1,
        first_cycle=first_cycle,
        surprise_exceeded=surprise_exceeded,
        internal_steps=internal_steps,
        boundary_internal_steps=boundary_internal_steps,
        theta_wrapped=theta_wrapped,
        within_quiet_before=within_quiet_before,
        swr_pressure_before=float(st['swr_pressure']),
        theta_phase_before=float(st['theta_phase']),
        theme_held=False,
        teacher_skipped=False,
        swr_skipped=0,
        swr_measured=0,
        mid_poll_skipped=0,
    )
    engine._phase_rhythm_active_gate = gate
    return gate


def proposal_budget(engine, default_budget: int) -> int:
    gate = active_gate(engine)
    if gate is None:
        return default_budget
    return int(gate['proposal_budget'])


def allow_mid_poll(engine) -> bool:
    gate = active_gate(engine)
    if gate is None:
        return True
    if gate['allow_mid_poll']:
        return True
    gate['mid_poll_skipped'] += 1
    engine.phase_rhythm_state['poll_skips'] += 1
    return False


def allow_teacher_flush(engine) -> bool:
    gate = active_gate(engine)
    if gate is None:
        return True
    if gate['theta_open']:
        return True
    gate['teacher_skipped'] = True
    engine.phase_rhythm_state['teacher_skips'] += 1
    return False


def allow_swr_measure(engine) -> bool:
    gate = active_gate(engine)
    if gate is None:
        return True
    if gate['swr_open']:
        gate['swr_measured'] += 1
        return True
    gate['swr_skipped'] += 1
    engine.phase_rhythm_state['swr_skips'] += 1
    return False


@torch.no_grad()
def try_hold_theme(engine, state, h):
    """H-JEPA theme persistence: hold the active typed question during Gamma flow."""
    gate = active_gate(engine)
    if gate is None or gate['theta_open']:
        return None
    st = engine.phase_rhythm_state
    theme = st.get('active_theme')
    if theme is None:
        return None
    if int(theme['held_cycles']) >= int(st['max_theme_hold']):
        return None
    if getattr(engine, 'shared_unknown_schema', None) is None:
        return None
    archive = getattr(engine.model, '_causal_archive', None)
    if archive is None:
        return None
    from scripts.ib_local import shared_unknown_bindings as shared
    q = shared._row(archive, theme['unknown_id'])
    if q is None:
        return None
    # Keep the exact pre-question physical boundary for capture_world().
    if getattr(engine, 'native_amortization', {}).get('schema') == 1:
        from scripts.ib_local.native_amortized_decision import capture_actual
        capture_actual(engine, state, h.reshape(1, -1))
    from scripts.ib_local.causal_relation_program import Port, infer_role
    clock = int(engine.model.memory.clock)
    with archive.db:
        archive.db.execute('UPDATE unknown_bindings SET touched=? WHERE id=?', (clock, q['identity']))
    if getattr(engine.model, 'joint_relation_update_schema', None) == 1:
        from scripts.ib_local.joint_relation_update import register_use
        register_use(engine, state, q)
    bound_state = shared._with_constraints(archive, state, q)
    engine._shared_question_receipt = dict(
        identity=q['identity'],
        version=q['version'],
        role=theme['role'],
        recalled=True,
        relation_identity=state.identity,
        signature=theme['signature'],
    )
    role = Port(theme['role'])
    s = h.new_tensor(theme['source']).reshape(1, -1)
    a = h.new_tensor(theme['action']).reshape(1, -1)
    if role == Port.MOTHER:
        # Refresh predicted target in today's encoder coordinates.
        y = infer_role(engine.model, bound_state, open_role=Port.MOTHER,
                       condition=s, action=a, ridge=engine.relation_ridge)['mean']
    elif theme.get('target_raw') is not None:
        y = engine.model.encoder(h.new_tensor(theme['target_raw']).reshape(1, -1))
    else:
        y = h.new_tensor(theme['target']).reshape(1, -1)
    theme['held_cycles'] = int(theme['held_cycles']) + 1
    st['theme_holds'] += 1
    gate['theme_held'] = True
    return dict(
        state=bound_state,
        role=role,
        source=s,
        action=a,
        target=y,
        unknown_id=q['identity'],
        version=q['version'],
        anchor=theme['anchor'],
        signature=theme['signature'],
        held_cycles=theme['held_cycles'],
    )


def record_theme(engine, state, role, s, a, y, *, unknown_id=None, anchor='', signature='', target_raw=None):
    """Store the newly selected H-JEPA typed question theme (CPU-only scalars/lists)."""
    if not enabled(engine):
        return
    st = engine.phase_rhythm_state
    st['theme_revisions'] += 1
    if unknown_id is None:
        st['active_theme'] = None
        return
    st['active_theme'] = dict(
        unknown_id=str(unknown_id),
        relation_identity=int(state.identity),
        role=role.value if hasattr(role, 'value') else str(role),
        source=s.detach().cpu().reshape(-1).tolist(),
        action=a.detach().cpu().reshape(-1).tolist(),
        target=y.detach().cpu().reshape(-1).tolist(),
        target_raw=None if target_raw is None else target_raw.detach().cpu().reshape(-1).tolist(),
        anchor=str(anchor),
        signature=str(signature),
        opened_cycle=int(st['cycles']),
        held_cycles=0,
    )


@torch.no_grad()
def gamma_relation_plan(engine, state, source, actions, momentum):
    """Fast Gamma-flow valuation: 1 `compose` + 1-point amortized `read_plans`.

    Preserves the exact contract expected by `CausalProgramEngine._relation_value`,
    `causal_use_value.utility` (exact local `finite_use_value.expected`),
    `causal_use_value.begin/settle`, `_outcome_loss`, and `shared_unknown.commit`.
    """
    from scripts.ib_local.causal_relation_program import compose
    from scripts.ib_local import persistent_rhythm
    model = engine.model
    path = compose(model, state, source, actions, ridge=engine.relation_ridge,
                   reward_valid=engine.reward_label_valid(wood=True), momentum=momentum)
    if persistent_rhythm.enabled(engine):
        from scripts.ib_local.prospective_decision import boundary, preview_bind
        raws, states = [], []
        for i in range(len(source)):
            raw, _ = boundary(engine, state, source[i:i+1], actions[i:i+1],
                              path['mother']['mean'][i:i+1], path['decision']['mean'][i:i+1])
            selected, _ = preview_bind(model, raw, overlay=state)
            raws.append(raw)
            states.append(selected)
        future, cases, candidate_count = persistent_rhythm.read_plans(
            engine, states, raws, carried=state)
        if getattr(engine, 'epistemic_value_schema', 0) >= 4:
            engine.successor_distribution_last = dict(
                schema=5,
                budget=1,
                branches=1,
                weights=[1.0],
                values=[future.tolist()],
                expected=[float(future.mean())],
                df=float(path['decision']['df']),
                scale_mean=float(path['decision']['scale2'].mean()),
                joint=None,
                native_indices=[],
                native_values=[],
                correction_weights=[],
                native_count=0,
                allocated_draws=0,
                proposal_weights=[1.0],
                bank_revision=engine.latent_action_bank['revision'] if hasattr(engine, 'latent_action_bank') else 0,
                planned_candidates=candidate_count,
                action_plans=cases,
                semantics='gamma fast 1-point persistent_rhythm.read_plans; teachers deferred to Theta window',
            )
    else:
        future = path['value']
    dist = path['mother']
    conditional_information = 0.5 * dist['mean'].shape[-1] * torch.log1p(dist['leverage'])
    return dict(
        path=path,
        hypotheses=dist['mean'][:, None],
        expected_after_value=future,
        future_delta=future - path['value'],
        information=conditional_information,
        quadrature_information=torch.zeros_like(conditional_information),
        shadow_state_shift=torch.zeros_like(conditional_information),
        epistemic_after_value=future,
        successor_routes=[],
    )


def _body_kinetic_activity(engine) -> float:
    """Normalized RMS momentum across the 32 BFSS layers."""
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


def end_cycle(engine, row):
    """Advance the coupled Theta oscillator and SWR pressure from real observations."""
    if not enabled(engine):
        return None
    st = engine.phase_rhythm_state
    gate = getattr(engine, '_phase_rhythm_active_gate', None)
    if gate is None:
        gate = dict(regime='theta', theta_open=True, swr_open=True,
                    swr_pressure_before=float(st['swr_pressure']),
                    theta_phase_before=float(st['theta_phase']))

    err = float(row.get('response_nll', 0.0))
    if not math.isfinite(err):
        err = 0.0
    count = int(st['surprise_count'])
    if count >= 2:
        std = math.sqrt(max(float(st['surprise_var']), 1e-4))
        z = max(0.0, (err - float(st['surprise_ema'])) / std)
    else:
        z = 0.0

    # Update running surprise baseline AFTER computing this step's z-score.
    decay = float(st['surprise_ema_decay'])
    if count == 0:
        st['surprise_ema'] = err
        st['surprise_var'] = 0.25
    else:
        diff = err - float(st['surprise_ema'])
        st['surprise_ema'] = decay * float(st['surprise_ema']) + (1.0 - decay) * err
        st['surprise_var'] = max(1e-4, decay * float(st['surprise_var']) + (1.0 - decay) * (diff * diff))
    st['surprise_count'] = count + 1

    kinetic = _body_kinetic_activity(engine)
    delta_phi = (float(st['omega_min'])
                 + float(st['omega_body']) * math.tanh(kinetic)
                 + float(st['omega_err']) * math.tanh(z))
    new_phi = float(st['theta_phase']) + delta_phi
    wrapped = new_phi >= TWO_PI
    st['theta_phase'] = new_phi % TWO_PI if wrapped else new_phi
    st['pending_theta_wrap'] = bool(wrapped)

    if gate.get('swr_open', False):
        st['swr_pressure'] = 0.0
        st['swr_fires'] += 1
    else:
        delta_swr = float(st['alpha_min']) + float(st['alpha_err']) * math.tanh(z)
        st['swr_pressure'] = min(1.0, float(st['swr_pressure']) + delta_swr)

    surprise_preempt = z > float(st['surprise_threshold'])
    st['pending_surprise_preempt'] = bool(surprise_preempt)
    within_quiet_now = int(getattr(engine, 'rhythm_state', {}).get('within_learning_quiet', 0))
    st['last_within_quiet_delta'] = max(0, within_quiet_now - int(gate.get('within_quiet_before', within_quiet_now)))
    st['last_within_quiet_total'] = within_quiet_now
    st['cycles'] += 1

    event = dict(
        cycle=int(st['cycles']),
        regime=gate['regime'],
        theta_open=bool(gate['theta_open']),
        swr_open=bool(gate['swr_open']),
        theme_held=bool(gate.get('theme_held', False)),
        proposal_budget=int(gate.get('proposal_budget', getattr(engine, 'joint_proposal_budget', 4))),
        response_nll=err,
        surprise_z=z,
        kinetic=kinetic,
        delta_phi=delta_phi,
        theta_phase=float(st['theta_phase']),
        theta_wrapped_next=bool(wrapped),
        swr_pressure=float(st['swr_pressure']),
        surprise_preempt_next=bool(surprise_preempt),
        swr_measured=int(gate.get('swr_measured', 0)),
        swr_skipped=int(gate.get('swr_skipped', 0)),
        mid_poll_skipped=int(gate.get('mid_poll_skipped', 0)),
        teacher_skipped=bool(gate.get('teacher_skipped', False)),
    )
    st['events'].append(event)
    del st['events'][:-32]
    engine._phase_rhythm_active_gate = None
    diag = diagnostics(engine)
    row['phase_coupled_rhythm'] = diag
    return diag


def diagnostics(engine):
    if not enabled(engine):
        return None
    st = engine.phase_rhythm_state
    return dict(
        schema=1,
        cycles=int(st['cycles']),
        theta_phase=float(st['theta_phase']),
        swr_pressure=float(st['swr_pressure']),
        surprise_ema=float(st['surprise_ema']),
        surprise_var=float(st['surprise_var']),
        gamma_cycles=int(st['gamma_cycles']),
        theta_cycles=int(st['theta_cycles']),
        swr_fires=int(st['swr_fires']),
        surprise_preemptions=int(st['surprise_preemptions']),
        internal_preemptions=int(st['internal_preemptions']),
        theme_holds=int(st['theme_holds']),
        theme_revisions=int(st['theme_revisions']),
        poll_skips=int(st['poll_skips']),
        teacher_skips=int(st['teacher_skips']),
        swr_skips=int(st['swr_skips']),
        active_unknown=None if st.get('active_theme') is None else st['active_theme']['unknown_id'],
        last_event=copy.deepcopy(st['events'][-1]) if st['events'] else None,
    )
