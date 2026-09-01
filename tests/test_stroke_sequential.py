"""Stroke-sequential canvas generation: decomposition and x_init contract."""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fine_grain.capability_tasks import _scene
from fine_grain.omni_model import DualStreamOmni
from scripts.train_stroke_sequential import (
    K_MAX,
    digit_schedule,
    make_bank,
    partial_scene,
    sub_strokes,
)


def test_substroke_reassembly_reproduces_grid_mask():
    """The K-step cumulative mask must equal the registered digit mask."""
    for digit in OCR_DIGITS_STROKES:
        for place in ("top_left", "middle_center", "bottom_right"):
            _, full = _scene(digit, "red", place, 16)
            _, cum = partial_scene(digit, "red", 16, place, K_MAX - 1)
            union = ((cum + full) > 0).float().sum().clamp_min(1)
            iou = float((cum * full).sum() / union)
            assert iou == 1.0, (digit, place, iou)


def test_step_schedule_is_monotone_with_noop_tail():
    for digit in OCR_DIGITS_STROKES:
        schedule = digit_schedule(digit)
        active = sum(1 for strokes in schedule if strokes)
        assert 1 <= active == len(sub_strokes(digit)) <= K_MAX
        for k in range(K_MAX - 1):
            a = partial_scene(digit, "red", 16, "top_left", k)[1]
            b = partial_scene(digit, "red", 16, "top_left", k + 1)[1]
            assert float(b.sum()) >= float(a.sum())
        # Trailing steps are no-ops: final two partial masks are identical
        # whenever the digit finished before the schedule end.
        if active < K_MAX - 1:
            late = partial_scene(digit, "red", 16, "top_left", K_MAX - 2)[1]
            last = partial_scene(digit, "red", 16, "top_left", K_MAX - 1)[1]
            assert torch.equal(late, last)


def test_encode_x_init_identity_and_step_condition():
    torch.manual_seed(0)
    model = _tiny_omni().eval()
    x = torch.randn(2, 64, 32)
    with torch.no_grad():
        identity = model.mot_stack.encode_X(
            torch.zeros(2, 3, 8, 8), x_init=x,
        )
    assert torch.equal(identity, x)
    t = torch.tensor([0.0, 0.5])
    with torch.no_grad():
        stepped = model.mot_stack.encode_X(
            torch.zeros(2, 3, 8, 8), t=t, x_init=x,
        )
    expected = x + model.mot_stack.t_coord(
        t.reshape(-1, 1, 1).expand(2, 64, 1),
    )
    assert torch.allclose(stepped, expected, atol=1e-6, rtol=1e-6)


def test_canvas_persistence_changes_generation():
    """Sequential writes must see the accumulated canvas, not just the prompt."""
    torch.manual_seed(1)
    model = _tiny_omni().eval()
    prompts = ["Draw digit 7 with a thin red stroke at top left"]
    zeros = torch.zeros(1, 3, 8, 8)
    with torch.no_grad():
        base = model(zeros, prompts, pi_x=1.0, t=torch.zeros(1))["belief_mu"]
        perturbed_canvas = base.clone()
        perturbed_canvas[:, :8] += 0.5
        continued = model(
            zeros, prompts, pi_x=1.0, t=torch.ones(1),
            x_init=perturbed_canvas,
        )["belief_mu"]
        plain_continued = model(
            zeros, prompts, pi_x=1.0, t=torch.ones(1), x_init=base,
        )["belief_mu"]
    assert not torch.allclose(continued, plain_continued, atol=1e-4)


def test_sequential_loss_backprops_to_step_condition():
    torch.manual_seed(2)
    model = _tiny_omni()
    model.set_optimization_phase("generation_write")
    assert model.mot_stack.t_coord.weight.requires_grad
    bank = make_bank(8)[:4]
    from scripts.train_stroke_sequential import sequential_loss

    loss, _ = sequential_loss(model, bank, torch.device("cpu"), 8)
    assert bool(torch.isfinite(loss))
    loss.backward()
    grad = model.mot_stack.t_coord.weight.grad
    assert grad is not None and float(grad.abs().sum()) > 0.0


OCR_DIGITS_STROKES = tuple(str(d) for d in range(10))


def test_step_conditioned_prior_is_identity_at_init_and_responds_after():
    """The completion prior must be step-blind at init, step-aware once trained."""
    torch.manual_seed(5)
    model = _tiny_omni(prior_step_condition=True).eval()
    gate = model.mot_stack.layers[0].surprise_gate
    assert gate.prior_step_proj is not None
    assert float(gate.prior_step_proj.weight.abs().sum()) == 0.0
    S = torch.randn(2, 8, 32)
    H = torch.randn(2, 6, 32)
    with torch.no_grad():
        mu_none = gate.prior_predictive(S, H)["mu_p"]
        mu_t0 = gate.prior_predictive(S, H, t=torch.zeros(2))["mu_p"]
        mu_t1 = gate.prior_predictive(S, H, t=torch.ones(2))["mu_p"]
    assert torch.allclose(mu_none, mu_t0)
    assert torch.allclose(mu_t0, mu_t1)
    with torch.no_grad():
        gate.prior_step_proj.weight.fill_(0.1)
        gate.prior_step_proj.bias.fill_(0.05)
        mu_t1b = gate.prior_predictive(S, H, t=torch.ones(2))["mu_p"]
        mu_t0b = gate.prior_predictive(S, H, t=torch.zeros(2))["mu_p"]
    assert not torch.allclose(mu_t0b, mu_t1b)
    # Default off: no parameter, so champion checkpoints stay loadable.
    plain = _tiny_omni().eval()
    assert plain.mot_stack.layers[0].surprise_gate.prior_step_proj is None


def _tiny_omni(**overrides):
    kwargs = dict(
        d_model=32, n_slices=8, n_layers=1, n_heads=4, res=8,
        surprise_mode="v1_bayes", s_update="raw",
        prior_loss_coef=0.1, sigreg_coef=0.0,
        use_stiefel=False, deslice_topk=0,
        use_null_slice=False, use_residual_read=False,
        gate_on="u", deslice_write="increment", gate_h_local=False,
        vfe_coef=0.1, prior_write=1.0, prior_write_by_t=False,
        pixel_loss_mode="balanced_bce",
        spatial_prompt_vocab=True, capability_vocab=True,
        use_modal_precision=True, use_target_time=True,
        use_target_time_adaln=False, use_horizon_tokens=False,
        gate_action_by_horizon=True, history_size=2, action_dim=2,
        use_action_adaln=False, use_action_tokens=False,
        use_action_rel_bias=False, use_action_transport=False,
        use_action_slice_transition=False, use_active_gdn2=False,
        use_goal_adaln=False, seg_classes=2, seg_loss_coef=1.0,
        s0_acc_coef=0.0,
    )
    kwargs.update(overrides)
    return DualStreamOmni(**kwargs)
