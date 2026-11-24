"""Root Latent-Intent Comparison + Single-Utterance Generation & Rank-Flip Sparse Teacher.

Implements the formally verified planning acceleration proved in
`lean/CGBAxioms/reviews/RootLatentPlanning.lean` and audited in
`docs/information_boltzmann/CGB_PLANNING_EXECUTION_COST_AUDIT_20261124.md`:

1. **Root Latent-Intent Comparison + Single-Utterance Generation (LeWM / H-JEPA)**:
   Before freezing `_question_turn_context` in `QuestionContextEngine._utter()`,
   compares up to `K = joint_proposal_budget` real candidate intents in latent
   space via one batched `latent_action_planner.predict()` + `imagine()` +
   `causal_use_value.utility()` pass (~2 ms). On confident routine spikes,
   steers `_question_context` with the winning latent intent via the existing
   `question_reactivation` port and generates ONLY 1 concrete utterance (cutting
   both `ve.film` and `_policy_grad` by ~75% with zero replay drift, Theorem 5).
   When the student bank is cold, novelty is high, top-2 margin is narrow, or on
   periodic calibration ticks (Theorem 2), escalates to full 4-draft native
   search so full search remains the ground-truth teacher and calibrator.

2. **Rank-Flip-Gated Sparse Native Teacher Allocation (Theorem 1)**:
   In `latent_action_planner._flush_teachers()`, compares the certified error
   bound `2 * gamma * max_priority * ema_abs_error` against the top-2 root
   action margin `V_(1) - V_(2)`. When teacher error cannot flip the root action
   ranking (outside periodic calibration ticks), allocates 0 teachers and skips
   heavy `pse.execute` endpoint rollouts completely.

Default OFF; enabled explicitly via `--root-latent-planning` / `configure()`.
"""
from __future__ import annotations

import copy
import math
import torch

DEFAULT_PARAMS = dict(
    full_search_every=8,      # Theorem 2: guaranteed full 4-draft calibration every <= 8 spikes
    teacher_cal_every=4,      # Theorem 2: guaranteed native teacher calibration every <= 4 spikes
    novelty_threshold=0.75,   # escalate to full 4-draft search when student novelty exceeds this
    margin_threshold=0.001,   # escalate to full 4-draft search when latent top-2 margin is near-tied
    intent_steer_scale=0.25,  # bounded steering weight into existing question_reactivation port
    min_bank_records=4,       # require >= 4 bank records before switching off full 4-draft search
    teacher_error_init=0.15,  # conservative initial causal EMA for |native - predicted|
    teacher_ema_decay=0.80,   # causal EMA decay for observed teacher prediction errors
)


def enabled(engine) -> bool:
    return getattr(engine, 'root_latent_planning_schema', None) == 1


def configure(engine, config=None, **overrides) -> dict:
    params = dict(DEFAULT_PARAMS)
    for k, v in overrides.items():
        if k not in params:
            raise ValueError(f'unknown root_latent_planning parameter: {k}')
        if k in ('full_search_every', 'teacher_cal_every', 'min_bank_records'):
            params[k] = int(v)
        else:
            params[k] = float(v)
    if params['full_search_every'] < 1 or params['teacher_cal_every'] < 1:
        raise ValueError('calibration strides must be >= 1 for bounded liveness')
    if params['intent_steer_scale'] < 0.0 or params['intent_steer_scale'] > 1.0:
        raise ValueError('intent_steer_scale must lie in [0, 1]')

    engine.root_latent_planning_schema = 1
    st = getattr(engine, 'root_latent_planning_state', None)
    if st is None:
        st = dict(
            schema=1,
            decisions=0,
            latent_single_count=0,
            full_search_count=0,
            escalation_reasons=dict(
                cold_bank=0, periodic_calibration=0, high_novelty=0,
                narrow_margin=0, fallback=0,
            ),
            teacher_decisions=0,
            teacher_checks_seen=0,
            teacher_budget_0=0,
            teacher_budget_1=0,
            teacher_budget_2=0,
            teacher_abs_error_ema=float(params['teacher_error_init']),
            last_plan_summary=None,
            **params,
        )
        engine.root_latent_planning_state = st
    else:
        for k, v in params.items():
            st.setdefault(k, v)
        st.setdefault('escalation_reasons', dict(
            cold_bank=0, periodic_calibration=0, high_novelty=0,
            narrow_margin=0, fallback=0,
        ))
    engine._root_latent_plan = None
    from scripts.ib_local import run_navigation_residual_wood as legacy
    legacy._DERIVED.add('_root_latent_plan')
    if config is not None:
        config['root_latent_planning'] = dict(schema=1, **params)
    return st


@torch.no_grad()
def plan_root_intent(engine, h: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
    """Compare K candidate intents in latent space BEFORE freezing `_question_context`.

    Theorem 5 guarantees zero structural replay drift between `film`, `habit`,
    `_choice_policy_term`, and `proposal_score` because any steering shift is
    folded into `context` before `QuestionContextEngine._utter()` freezes it.
    """
    if not enabled(engine):
        return context
    st = engine.root_latent_planning_state
    engine._root_latent_plan = None
    budget = int(getattr(engine, 'joint_proposal_budget', 1))
    if budget <= 1:
        return context

    from scripts.ib_local import latent_action_planner as lap
    state = getattr(engine, '_program_question_state', None)
    gq = getattr(engine, '_generative_query', None)
    if state is None or gq is None or not lap.enabled(engine):
        st['decisions'] = int(st['decisions']) + 1
        st['full_search_count'] = int(st['full_search_count']) + 1
        st['escalation_reasons']['fallback'] = int(st['escalation_reasons'].get('fallback', 0)) + 1
        return context

    from scripts.ib_local.causal_relation_program import Port, imagine
    from scripts.ib_local.causal_use_value import utility
    from scripts.ib_local.finite_choice_contract import choose
    from scripts.ib_local.navigation_gain_units import wood_choice_logits
    from scripts.ib_local.navigation_momentum_wood import decision_momentum_read
    from scripts.ib_local.preview_runtime import preview_scope
    from scripts.ib_local.prospective_decision import program_actions

    model = engine.model
    pre = h[None]
    proposal = dict(gq, role=Port(gq['role']))
    decision_idx = int(st['decisions'])
    st['decisions'] = decision_idx + 1

    with preview_scope(model):
        nominal, slots, _ = program_actions(engine, pre, proposal)
        p_vec = decision_momentum_read(model, engine.bfss_momenta, pre)
        cand_list = []
        if len(nominal) > 0:
            cand_list.append(nominal[:budget])
        if sum(len(c) for c in cand_list) < 2:
            # Supplement with current boundary and live BFSS momentum-directed intent.
            mom_step = pre + model.momentum_to_boundary(p_vec)
            cand_list.append(torch.cat((pre, mom_step), 0))
        candidates = torch.cat(cand_list, 0)[:budget]
        k = len(candidates)

        packs = [lap.packet(engine, state, pre, proposal, candidates[j:j+1], 1) for j in range(k)]
        estimated, novelty = lap.predict(engine, packs)
        width = pre.shape[-1]
        hat_actions = pre + estimated[:, width:2 * width]
        source = pre.expand_as(hat_actions)
        momentum = p_vec.expand(k, -1)
        forecast = imagine(
            model, state, source, hat_actions,
            ridge=engine.relation_ridge,
            reward_valid=engine.reward_label_valid(wood=True),
            momentum=momentum,
        )
        selected_value = utility(engine, state, source, hat_actions, forecast)
        logits = wood_choice_logits(engine, selected_value)
        logp = logits.log_softmax(0)
        selected_idx = choose(logp.exp(), stage='proposal')

    if k >= 2:
        top2 = selected_value.topk(2).values
        top_margin = float(top2[0] - top2[1])
    else:
        top_margin = float('inf')
    sel_novelty = float(novelty[selected_idx])
    max_novelty = float(novelty.max())
    bank_n = len(getattr(engine, 'latent_action_bank', {}).get('records', []))

    # Calibrated Teacher-Escalation Gate (Theorems 2 & 3).
    escalate_reason = None
    if bank_n < int(st['min_bank_records']):
        escalate_reason = 'cold_bank'
    elif decision_idx % int(st['full_search_every']) == 0:
        escalate_reason = 'periodic_calibration'
    elif sel_novelty > float(st['novelty_threshold']):
        escalate_reason = 'high_novelty'
    elif top_margin < float(st['margin_threshold']):
        escalate_reason = 'narrow_margin'

    escalated = escalate_reason is not None
    steered = False
    out_context = context
    if escalated:
        st['full_search_count'] = int(st['full_search_count']) + 1
        st['escalation_reasons'][escalate_reason] = int(st['escalation_reasons'].get(escalate_reason, 0)) + 1
    else:
        st['latent_single_count'] = int(st['latent_single_count']) + 1
        steer_scale = float(st['intent_steer_scale'])
        if steer_scale > 0.0 and hasattr(model, 'question_reactivation'):
            win_action = hat_actions[selected_idx:selected_idx + 1]
            z_delta = model.encoder(win_action) - model.encoder(pre)
            steer = steer_scale * model.question_reactivation(z_delta)
            out_context = context + steer
            steered = True

    plan_info = dict(
        escalated=escalated,
        reason=('latent_single' if not escalated else escalate_reason),
        effective_budget=(budget if escalated else 1),
        candidate_count=k,
        selected_intent=int(selected_idx),
        selected_logp=float(logp[selected_idx]),
        distribution=logp.exp().cpu().tolist(),
        predicted_after_value=forecast['expected_after_value'].cpu().tolist(),
        selection_value=selected_value.cpu().tolist(),
        novelty=novelty.cpu().tolist(),
        selected_novelty=sel_novelty,
        max_novelty=max_novelty,
        top_margin=top_margin,
        steered=steered,
        relation_identity=state.identity,
        relation_version=state.version,
        program_slots=list(slots[:k]) if slots else [],
    )
    engine._root_latent_plan = plan_info
    st['last_plan_summary'] = dict(
        reason=plan_info['reason'],
        effective_budget=plan_info['effective_budget'],
        top_margin=top_margin,
        selected_novelty=sel_novelty,
        steered=steered,
    )
    return out_context


def proposal_budget(engine, default_budget: int) -> int:
    """Return 1 on confident latent-planned spikes, else `default_budget`."""
    if not enabled(engine):
        return int(default_budget)
    plan = getattr(engine, '_root_latent_plan', None)
    if plan is not None and not plan.get('escalated', True):
        return 1
    return int(default_budget)


def is_latent_single_active(engine) -> bool:
    if not enabled(engine):
        return False
    plan = getattr(engine, '_root_latent_plan', None)
    return plan is not None and not plan.get('escalated', True)


def build_single_selection_receipt(engine) -> dict:
    """Build `_joint_selection` receipt for a latent-steered single utterance."""
    plan = engine._root_latent_plan
    return dict(
        selected=0,
        logp=0.0,
        distribution=[1.0],
        predicted_after_value=[plan['predicted_after_value'][plan['selected_intent']]],
        selection_value=[plan['selection_value'][plan['selected_intent']]],
        relation_identity=plan['relation_identity'],
        relation_version=plan['relation_version'],
        budget=1,
        regime='root_latent_single',
        latent_intent_selected=plan['selected_intent'],
        latent_intent_logp=plan['selected_logp'],
        latent_intent_distribution=plan['distribution'],
        latent_intent_values=plan['selection_value'],
        latent_intent_margin=plan['top_margin'],
        latent_intent_novelty=plan['selected_novelty'],
        steered=plan['steered'],
    )


def teacher_budget(engine, menu: list[dict], decision_values, default_budget: int = 2) -> int:
    """Rank-Flip-Gated Sparse Native Teacher Allocation (Theorems 1 & 2).

    Computes the certified future-value error bound `2 * gamma * max_priority * ema_err`
    against the top-2 root action margin in wood units. Returns 0 when teacher
    error cannot flip the root action ranking (outside periodic calibration).
    """
    if not enabled(engine):
        return int(default_budget)
    st = engine.root_latent_planning_state
    t_idx = int(st['teacher_decisions'])
    st['teacher_decisions'] = t_idx + 1

    if not menu or default_budget <= 0:
        st['teacher_budget_0'] = int(st['teacher_budget_0']) + 1
        return 0

    # Theorem 2: guaranteed bounded periodic calibration & cold-start calibration.
    if int(st['teacher_checks_seen']) < 2 or (t_idx % int(st['teacher_cal_every']) == 0):
        b = min(int(default_budget), 2)
        if b >= 2:
            st['teacher_budget_2'] = int(st['teacher_budget_2']) + 1
        elif b == 1:
            st['teacher_budget_1'] = int(st['teacher_budget_1']) + 1
        else:
            st['teacher_budget_0'] = int(st['teacher_budget_0']) + 1
        return b

    # Compute top-2 root action margin in wood units (`decision_values` is 10*U).
    delta_root = 0.0
    if decision_values is not None and torch.is_tensor(decision_values):
        vals = decision_values.detach().reshape(-1).float()
        roots = getattr(engine, '_competing_action_roots', None)
        if roots:
            avail_opts = sorted({opt for opts in roots.values() for opt in opts if opt < len(vals)})
        else:
            avail_opts = [i for i in range(len(vals)) if math.isfinite(float(vals[i])) and float(vals[i]) > -1e3]
        if len(avail_opts) >= 2:
            sub = vals[avail_opts] / 10.0
            top2 = sub.topk(2).values
            delta_root = max(0.0, float(top2[0] - top2[1]))
        elif len(avail_opts) == 1:
            # Only 1 root action is available -> rank flip is impossible!
            delta_root = float('inf')

    max_priority = max((float(r.get('priority', 0.0)) for r in menu), default=0.0)
    ema_err = max(1e-4, float(st['teacher_abs_error_ema']))
    err_bound = 2.0 * float(getattr(engine, 'gamma', 0.9)) * max_priority * ema_err

    # Theorem 1 (Rank-Flip Sufficiency): strict margin certifies rank invariance.
    if err_bound < delta_root:
        st['teacher_budget_0'] = int(st['teacher_budget_0']) + 1
        return 0
    if err_bound < 2.0 * delta_root and default_budget >= 1:
        st['teacher_budget_1'] = int(st['teacher_budget_1']) + 1
        return 1
    b = min(int(default_budget), 2)
    if b >= 2:
        st['teacher_budget_2'] = int(st['teacher_budget_2']) + 1
    elif b == 1:
        st['teacher_budget_1'] = int(st['teacher_budget_1']) + 1
    else:
        st['teacher_budget_0'] = int(st['teacher_budget_0']) + 1
    return b


def observe_teacher_checks(engine, checks: list[dict]) -> None:
    """Update causal EMA of observed |native - predicted| teacher errors."""
    if not enabled(engine) or not checks:
        return
    st = engine.root_latent_planning_state
    decay = float(st['teacher_ema_decay'])
    for row in checks:
        err = abs(float(row.get('error', 0.0)))
        if math.isfinite(err):
            st['teacher_abs_error_ema'] = decay * float(st['teacher_abs_error_ema']) + (1.0 - decay) * err
            st['teacher_checks_seen'] = int(st['teacher_checks_seen']) + 1


def diagnostics(engine) -> dict | None:
    if not enabled(engine):
        return None
    st = engine.root_latent_planning_state
    dec = max(1, int(st['decisions']))
    tdec = max(1, int(st['teacher_decisions']))
    return dict(
        schema=1,
        decisions=int(st['decisions']),
        latent_single_count=int(st['latent_single_count']),
        full_search_count=int(st['full_search_count']),
        latent_single_rate=float(st['latent_single_count']) / float(dec),
        escalation_reasons=copy.deepcopy(st['escalation_reasons']),
        teacher_decisions=int(st['teacher_decisions']),
        teacher_checks_seen=int(st['teacher_checks_seen']),
        teacher_budget_0=int(st['teacher_budget_0']),
        teacher_budget_1=int(st['teacher_budget_1']),
        teacher_budget_2=int(st['teacher_budget_2']),
        teacher_skip_rate=float(st['teacher_budget_0']) / float(tdec),
        teacher_abs_error_ema=float(st['teacher_abs_error_ema']),
        last_plan_summary=copy.deepcopy(st.get('last_plan_summary')),
        params=dict(
            full_search_every=int(st['full_search_every']),
            teacher_cal_every=int(st['teacher_cal_every']),
            novelty_threshold=float(st['novelty_threshold']),
            margin_threshold=float(st['margin_threshold']),
            intent_steer_scale=float(st['intent_steer_scale']),
            min_bank_records=int(st['min_bank_records']),
        ),
    )
