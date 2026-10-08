import math
import numpy as np
import pytest

from information_boltzmann.runtime.lifelong_evaluation import (
    adaptation_generalization_summary,
    recovery_summary,
)


def test_ag_metric_already_at_plateau_no_shock():
    """A model with zero shock is already at plateau (tau=0), yielding S=1.0 and AG=2/3 when quality=ref."""
    H = 256
    ref = [7.5] * H
    vals = [7.5] * H

    res = adaptation_generalization_summary(vals, block_tokens=16, hold_blocks=2, reference_nll=ref)
    assert 'ag' in res
    assert np.isclose(res['tau_tokens'], 0.0)
    assert np.isclose(res['s_speed'], 1.0)
    assert np.isclose(res['q_quality'], 0.5)
    assert np.isclose(res['ag'], 2.0 / 3.0, atol=1e-4)


def test_ag_metric_fast_recovery_deep_plateau():
    """Fast recovery (tau small) and deep plateau (delta_L < 0) yields high AG."""
    H = 256
    ref = [8.0] * H
    # Initial shock for 1 block (16 tokens), then deep plateau at 6.0
    vals = [9.5] * 16 + [6.0] * (H - 16)

    res = adaptation_generalization_summary(vals, block_tokens=16, hold_blocks=2, reference_nll=ref)
    assert 'ag' in res
    assert res['s_speed'] > 0.80
    assert res['q_quality'] > 0.80  # exp(-2.0) = 0.135 -> Q = 1/1.135 = 0.88
    assert res['ag'] > 0.80


def test_ag_metric_slow_recovery_penalized():
    """Slow recovery (tau >= 192) drags down S and total AG compared to fast recovery."""
    H = 256
    ref = [8.0] * H
    # Fast recovery
    vals_fast = [9.5] * 16 + [6.0] * (H - 16)
    res_fast = adaptation_generalization_summary(vals_fast, block_tokens=16, hold_blocks=2, reference_nll=ref)

    # Slow recovery: shock for 12 blocks (192 tokens), only late 4 blocks drop to 6.0
    vals_slow = [9.5] * 192 + [6.0] * 64
    res_slow = adaptation_generalization_summary(vals_slow, block_tokens=16, hold_blocks=2, reference_nll=ref)

    assert res_slow['tau_tokens'] > res_fast['tau_tokens']
    assert res_slow['s_speed'] < res_fast['s_speed']
    assert res_slow['ag'] < res_fast['ag']


def test_recovery_summary_bubbles_up_ag():
    """Verifies recovery_summary surfaces AG and APPL metrics at top level."""
    H = 256
    ref = [7.8] * H
    vals = [8.5] * 16 + [7.0] * (H - 16)

    rec = recovery_summary(vals, block_tokens=16, hold_blocks=2, reference_nll=ref)
    assert 'ag' in rec
    assert 's_speed' in rec
    assert 'q_quality' in rec
    assert 'appl_h' in rec
