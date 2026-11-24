"""Unit tests for event-driven brainwave phase-coupled multi-rate rhythm."""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from scripts.ib_local import phase_coupled_rhythm as pcr


def _fake_engine(momenta_scale=0.5):
    return SimpleNamespace(
        joint_proposal_budget=4,
        autonomous_action_receipt={'internal_steps': 0},
        bfss_momenta=[torch.full((4, 8), float(momenta_scale)) for _ in range(4)],
    )


def test_disabled_engine_keeps_strict_defaults():
    eng = _fake_engine()
    assert not pcr.enabled(eng)
    assert pcr.is_theta_open(eng)
    assert pcr.proposal_budget(eng, 4) == 4
    assert pcr.allow_mid_poll(eng)
    assert pcr.allow_teacher_flush(eng)
    assert pcr.allow_swr_measure(eng)
    assert pcr.try_hold_theme(eng, None, None) is None


def test_first_cycle_opens_theta_and_subsequent_calm_enters_gamma():
    eng = _fake_engine(momenta_scale=0.1)
    cfg = {}
    pcr.configure(eng, cfg)
    assert pcr.enabled(eng)
    assert cfg['phase_coupled_rhythm']['schema'] == 1

    # Cycle 0: first_cycle -> Theta open
    g0 = pcr.begin_cycle(eng)
    assert g0['regime'] == 'theta'
    assert g0['theta_open'] is True
    assert g0['proposal_budget'] == 4
    assert pcr.allow_mid_poll(eng) is True
    assert pcr.allow_teacher_flush(eng) is True
    row0 = {'response_nll': 2.5}
    pcr.end_cycle(eng, row0)
    assert row0['phase_coupled_rhythm']['theta_cycles'] == 1

    # Cycle 1: calm stream, no wrap yet -> Gamma flow
    g1 = pcr.begin_cycle(eng)
    assert g1['regime'] == 'gamma'
    assert g1['theta_open'] is False
    assert g1['proposal_budget'] == 1
    assert pcr.allow_mid_poll(eng) is False
    assert pcr.allow_teacher_flush(eng) is False
    assert pcr.allow_swr_measure(eng) is False
    row1 = {'response_nll': 2.5}
    pcr.end_cycle(eng, row1)
    assert eng.phase_rhythm_state['gamma_cycles'] == 1
    assert eng.phase_rhythm_state['poll_skips'] == 1
    assert eng.phase_rhythm_state['teacher_skips'] == 1
    assert eng.phase_rhythm_state['swr_skips'] == 1


def test_surprise_preempts_theta_and_swr_immediately():
    eng = _fake_engine(momenta_scale=0.0)
    pcr.configure(eng)
    # Build stable low-variance baseline over 3 cycles
    for _ in range(3):
        pcr.begin_cycle(eng)
        pcr.end_cycle(eng, {'response_nll': 2.0})
    # Inject sudden large surprise spike
    pcr.begin_cycle(eng)
    pcr.end_cycle(eng, {'response_nll': 5.0})
    assert eng.phase_rhythm_state['pending_surprise_preempt'] is True

    # Next cycle MUST preempt immediately into Theta + SWR
    gate = pcr.begin_cycle(eng)
    assert gate['theta_open'] is True
    assert gate['swr_open'] is True
    assert gate['surprise_exceeded'] is True
    assert gate['proposal_budget'] == 4
    assert pcr.allow_swr_measure(eng) is True
    pcr.end_cycle(eng, {'response_nll': 2.0})
    # SWR firing resets pressure to 0
    assert eng.phase_rhythm_state['swr_pressure'] == 0.0


def test_internal_steps_preempt_theta_immediately():
    eng = _fake_engine(momenta_scale=0.0)
    pcr.configure(eng)
    pcr.begin_cycle(eng)
    pcr.end_cycle(eng, {'response_nll': 2.0})

    # 1 casual coin-flip step does NOT preempt; >=2 boundary internal steps DO preempt
    eng.autonomous_action_receipt = {'internal_steps': 1}
    g_calm = pcr.begin_cycle(eng)
    assert g_calm['theta_open'] is False
    pcr.end_cycle(eng, {'response_nll': 2.0})

    eng.autonomous_action_receipt = {'internal_steps': 2}
    gate = pcr.begin_cycle(eng)
    assert gate['theta_open'] is True
    assert gate['boundary_internal_steps'] == 2
    assert gate['proposal_budget'] == 4
    assert eng.phase_rhythm_state['internal_preemptions'] == 1


def test_bounded_liveness_theta_and_swr_never_starve():
    eng = _fake_engine(momenta_scale=0.0)
    pcr.configure(eng, omega_min=pcr.TWO_PI / 5.0, alpha_min=0.34)
    theta_gaps, swr_gaps = [], []
    since_theta, since_swr = 0, 0
    for _ in range(25):
        gate = pcr.begin_cycle(eng)
        if gate['theta_open']:
            theta_gaps.append(since_theta)
            since_theta = 0
        else:
            since_theta += 1
        if gate['swr_open']:
            swr_gaps.append(since_swr)
            since_swr = 0
        else:
            since_swr += 1
        pcr.end_cycle(eng, {'response_nll': 2.2})
    assert max(theta_gaps) <= 5
    assert max(swr_gaps) <= 3
    assert eng.phase_rhythm_state['theta_cycles'] >= 5
    assert eng.phase_rhythm_state['swr_fires'] >= 6


def test_invalid_floors_rejected():
    eng = _fake_engine()
    with pytest.raises(ValueError):
        pcr.configure(eng, omega_min=0.0)
    with pytest.raises(ValueError):
        pcr.configure(eng, alpha_min=-0.1)


def test_configure_preserves_restored_state():
    eng = _fake_engine()
    pcr.configure(eng)
    pcr.begin_cycle(eng)
    pcr.end_cycle(eng, {'response_nll': 2.4})
    saved_phase = eng.phase_rhythm_state['theta_phase']
    saved_cycles = eng.phase_rhythm_state['cycles']

    # Re-calling configure on a restored engine preserves carried oscillator state
    pcr.configure(eng)
    assert math.isclose(eng.phase_rhythm_state['theta_phase'], saved_phase)
    assert eng.phase_rhythm_state['cycles'] == saved_cycles
