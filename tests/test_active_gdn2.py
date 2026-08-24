import numpy as np
import torch

from fine_grain.active_gdn2 import ActiveInferenceGDN2
from fine_grain.capability_tasks import CAPABILITY_CASES, collate_capability, capability_sample
from tests.test_capability_closure import _model


def test_zero_observation_precision_is_exact_no_write():
    torch.manual_seed(0)
    memory = ActiveInferenceGDN2(d_model=8, n_slots=4, res=8)
    state = memory.initial_state(2, torch.device("cpu"), torch.float32)
    tokens = torch.randn(2, 4, 8)
    centers = memory.atlas.unsqueeze(0).expand(2, -1, -1)
    updated = memory.assimilate(state, tokens, centers, torch.zeros(2))
    assert torch.equal(updated.natural, state.natural)
    assert torch.equal(updated.precision, state.precision)


def test_dynamic_prior_releases_precision_without_moving_mean():
    torch.manual_seed(1)
    memory = ActiveInferenceGDN2(d_model=8, n_slots=4, res=8)
    state = memory.initial_state(1, torch.device("cpu"), torch.float32)
    tokens = torch.randn(1, 4, 8)
    centers = memory.atlas.unsqueeze(0)
    state = memory.assimilate(state, tokens, centers, torch.ones(1))
    old_mean = state.natural / state.precision
    predicted = memory.dynamic_prior(
        state, torch.ones(1), action=torch.tensor([[1.0, 0.0]]),
        action_precision=torch.ones(1),
    )
    new_mean = predicted.natural / predicted.precision
    assert torch.allclose(new_mean, old_mean, atol=1e-6)
    assert torch.all(predicted.precision < state.precision)


def test_action_changes_source_address_not_transient_slice_identity():
    torch.manual_seed(2)
    memory = ActiveInferenceGDN2(d_model=4, n_slots=9, res=9)
    state = memory.initial_state(1, torch.device("cpu"), torch.float32)
    tokens = torch.randn(1, 9, 4)
    centers = memory.atlas.unsqueeze(0)
    state = memory.assimilate(state, tokens, centers, torch.ones(1))
    _, _, still = memory.query(
        state, tokens, centers, horizon=torch.ones(1),
        action=torch.zeros(1, 2), action_precision=torch.ones(1),
    )
    moved, _, active = memory.query(
        state, tokens, centers, horizon=torch.ones(1),
        action=torch.tensor([[1.0, 0.0]]), action_precision=torch.ones(1),
    )
    assert not torch.equal(active["source_centers"], still["source_centers"])
    assert torch.allclose(
        active["displacement_yx"][:, 1],
        torch.full((1,), 2.0 / 3.0), atol=1e-5,
    )
    moved.square().mean().backward()
    assert memory.action_scale_raw.grad is not None
    assert float(memory.action_scale_raw.grad.abs()) > 0.0


def test_prior_action_landing_is_exact_identity_at_initialization_and_trainable():
    torch.manual_seed(20)
    memory = ActiveInferenceGDN2(d_model=8, n_slots=4, res=8)
    current = torch.randn(2, 4, 8)
    prior = torch.randn_like(current)
    landed = memory.apply_prior_action(current, prior)
    assert torch.equal(landed, current)
    landed.square().mean().backward()
    grad = memory.prior_channel_gain.grad
    assert grad is not None and float(grad.abs().sum()) > 0.0


def test_persistent_atlas_deslice_is_spatial_and_content_independent():
    memory = ActiveInferenceGDN2(d_model=8, n_slots=9, res=9)
    axis = torch.linspace(-1.0, 1.0, 9)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    points = torch.stack([yy, xx], dim=-1).reshape(1, -1, 2)
    weights = memory.point_to_atlas_weights(points)
    assert weights.shape == (1, 81, 9)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(1, 81), atol=1e-6)
    # A persistent address map must resolve space; uniform assignments would
    # recreate the rejected transient-Deslice bottleneck.
    assert float(weights.detach().std(dim=1).mean()) > 0.05
    shifted = memory.point_to_atlas_weights(points.flip(1))
    assert torch.equal(shifted, weights.flip(1))


def test_predict_step_transports_belief_before_observation_correction():
    memory = ActiveInferenceGDN2(d_model=2, n_slots=9, res=9)
    state = memory.initial_state(1, torch.device("cpu"), torch.float32)
    # Put a sharp identifiable belief at the atlas center.
    center = int(torch.argmin(memory.atlas.square().sum(dim=-1)))
    state.precision.fill_(100.0)
    state.natural.zero_()
    state.natural[0, center, 0] = 100.0
    moved = memory.transport_state(state, torch.tensor([[0.0, 2.0 / 3.0]]))
    mean = moved.natural / moved.precision
    peak = int(mean[0, :, 0].argmax())
    assert float(memory.atlas[peak, 1]) > float(memory.atlas[center, 1])


def test_unified_future_vfe_trains_causal_memory_in_same_slice_chart():
    torch.manual_seed(3)
    model = _model(
        action_tokens=True,
        action_rel_bias=True,
        transition_gate=True,
        target_time_adaln=False,
        transition_loss_coef=0.5,
        use_active_gdn2=True,
        causal_memory_loss_coef=0.25,
    )
    rng = np.random.default_rng(3)
    batch = collate_capability([
        capability_sample(rng, 16, case, digit="3", color="red", place="middle_center")
        for case in CAPABILITY_CASES
    ])
    zeros = torch.zeros(len(CAPABILITY_CASES))
    batch["t"] = zeros
    out = model.forward_with_future_posterior(
        batch["image"], batch["prompt"], batch["target_rgb"], t=zeros,
        image_precision=batch["image_precision"],
        text_precision=batch["text_precision"],
        target_time=batch["target_time"],
        history_images=batch["history_images"],
        history_precision=batch["history_precision"],
        action=batch["action"], action_precision=batch["action_precision"],
    )
    assert out["causal_prior_mu"].shape == (4, 8, 32)
    assert out["transition_q_causal_mu"].shape == (4, 8, 32)
    loss, meta = model.omni_loss(out, batch, torch.device("cpu"))
    assert torch.isfinite(loss)
    assert meta["causal_transition_kl"] >= 0.0
    loss.backward()
    grad = model.mot_stack.active_gdn2.content_address.weight.grad
    assert grad is not None and torch.isfinite(grad).all()


def test_present_boundary_keeps_active_memory_write_path_exactly_closed():
    torch.manual_seed(4)
    model = _model(
        action_tokens=True,
        action_rel_bias=True,
        transition_gate=True,
        use_active_gdn2=True,
    ).eval()
    image = torch.rand(2, 3, 16, 16)
    with torch.no_grad():
        model(
            image, ["Reconstruct current frame"] * 2,
            image_precision=torch.ones(2), text_precision=torch.ones(2),
            target_time=torch.zeros(2), action=torch.tensor([[1.0, 0.0]]).expand(2, -1),
            action_precision=torch.ones(2),
        )
    gate = model.mot_stack.layers[0].last_causal_gate
    assert gate is not None
    assert torch.equal(gate, torch.zeros_like(gate))


def test_semantic_generation_does_not_open_physical_prior_gate():
    torch.manual_seed(41)
    model = _model(use_active_gdn2=True).eval()
    with torch.no_grad():
        model(
            torch.zeros(1, 3, 16, 16),
            ["Draw digit 3 with a thin red stroke at middle center"],
            image_precision=torch.zeros(1), text_precision=torch.ones(1),
            target_time=torch.zeros(1), action=torch.zeros(1, 2),
            action_precision=torch.zeros(1),
        )
    gate = model.mot_stack.layers[0].last_causal_gate
    assert gate is not None
    assert torch.equal(gate, torch.zeros_like(gate))


def test_enabling_active_memory_does_not_advance_global_rng_stream():
    torch.manual_seed(99)
    plain = _model(use_active_gdn2=False)
    plain_tail = plain.seg_head[-1].weight.detach().clone()
    torch.manual_seed(99)
    active = _model(use_active_gdn2=True)
    active_tail = active.seg_head[-1].weight.detach().clone()
    assert torch.equal(plain_tail, active_tail)
