import numpy as np
import torch

from fine_grain.capability_tasks import (
    CAPABILITY_CASES,
    CAPABILITY_TASK_IDS,
    capability_sample,
    collate_capability,
    make_capability_batch,
    make_counterfactual_future_batch,
)
from fine_grain.omni_tasks import grid_digit_mask
from fine_grain.omni_model import DualStreamOmni
from scripts.train_northstar_capabilities import (
    _checkpoint_score,
    paired_action_likelihood,
)


def _model(
    res=16,
    target_time_adaln=False,
    transition_loss_coef=0.0,
    goal_adaln=False,
    action_tokens=False,
    task_tokens=False,
    horizon_tokens=False,
    action_rel_bias=False,
    action_transport=False,
    action_slice_transition=False,
    transition_gate=False,
    use_active_gdn2=False,
    causal_memory_loss_coef=0.0,
    deep_visual_likelihood_coef=0.0,
):
    return DualStreamOmni(
        d_model=32,
        n_slices=8,
        n_layers=2,
        n_heads=4,
        res=res,
        surprise_mode="baseline",
        s_update="raw",
        use_stiefel=False,
        deslice_topk=0,
        prior_write=0.0,
        vfe_coef=0.0,
        pixel_loss_mode="balanced_bce",
        spatial_prompt_vocab=True,
        capability_vocab=True,
        use_modal_precision=True,
        use_target_time=True,
        use_target_time_adaln=target_time_adaln,
        use_horizon_tokens=horizon_tokens,
        gate_action_by_horizon=transition_gate,
        history_size=2,
        action_dim=2,
        use_action_adaln=not action_tokens,
        use_action_tokens=action_tokens,
        use_task_tokens=task_tokens,
        n_task_tokens=len(CAPABILITY_CASES),
        use_action_rel_bias=action_rel_bias,
        use_action_transport=action_transport,
        use_action_slice_transition=action_slice_transition,
        use_active_gdn2=use_active_gdn2,
        use_goal_adaln=goal_adaln,
        seg_classes=2,
        deep_visual_likelihood_coef=deep_visual_likelihood_coef,
        transition_loss_coef=transition_loss_coef,
        causal_memory_loss_coef=causal_memory_loss_coef,
        s0_acc_coef=0.0,
    ).eval()


def test_missing_text_precision_removes_lexical_evidence():
    torch.manual_seed(0)
    model = _model()
    image = torch.rand(1, 3, 16, 16)
    with torch.no_grad():
        a = model(image, ["Draw digit 1"], text_precision=torch.zeros(1))
        b = model(image, ["Draw digit 2"], text_precision=torch.zeros(1))
    assert torch.equal(a["logits"], b["logits"])
    assert torch.equal(a["rgb"], b["rgb"])


def test_task_token_is_global_intention_in_shared_h_workspace():
    torch.manual_seed(0)
    model = _model(task_tokens=True)
    image = torch.zeros(2, 3, 16, 16)
    prompt = ["Draw digit 7"] * 2
    out = model(image, prompt, task_id=torch.tensor([
        CAPABILITY_TASK_IDS["text_to_both"],
        CAPABILITY_TASK_IDS["image_text_edit"],
    ]))
    assert not torch.equal(out["X"][0], out["X"][1])
    out["X"].sum().backward()
    assert model.mot_stack.task_embed.weight.grad is not None
    assert model.mot_stack.task_embed.weight.grad.abs().sum() > 0


def test_capability_batch_carries_declared_task_ids():
    rng = np.random.default_rng(9)
    samples = [capability_sample(rng, 16, case) for case in CAPABILITY_CASES]
    batch = collate_capability(samples)
    assert batch["task_id"].tolist() == [CAPABILITY_TASK_IDS[c] for c in CAPABILITY_CASES]


def test_deep_visual_likelihood_reuses_layer_fields_and_rgb_head():
    torch.manual_seed(0)
    model = _model(deep_visual_likelihood_coef=0.5)
    rng = np.random.default_rng(4)
    samples = [capability_sample(rng, 16, "text_to_both") for _ in range(3)]
    batch = collate_capability(samples)
    batch["t"] = torch.zeros(len(samples))
    out = model(
        batch["image"], batch["prompt"], t=batch["t"],
        image_precision=batch["image_precision"],
        text_precision=batch["text_precision"],
        target_time=batch["target_time"],
        history_images=batch["history_images"],
        history_precision=batch["history_precision"],
        action=batch["action"], action_precision=batch["action_precision"],
    )
    assert len(out["X_steps"]) == 2
    loss, meta = model.omni_loss(out, batch, torch.device("cpu"))
    assert meta["n_deep_visual_steps"] == 1
    assert meta["deep_visual_likelihood"] > 0
    loss.backward()
    assert model.mot_stack.layers[0].mot.Wv_t.weight.grad is not None


def test_normalized_glyph_course_preserves_legacy_16_and_relative_address():
    legacy = grid_digit_mask("7", 16, "top_left")
    normalized = grid_digit_mask(
        "7", 16, "top_left", box=6, normalized_layout=True,
    )
    assert torch.equal(legacy, normalized)
    small = grid_digit_mask(
        "7", 16, "middle_center", box=6, stroke_px=1, normalized_layout=True,
    )
    large = grid_digit_mask(
        "7", 64, "middle_center", box=24, stroke_px=4, normalized_layout=True,
    )
    sy, sx = torch.nonzero(small[0], as_tuple=True)
    ly, lx = torch.nonzero(large[0], as_tuple=True)
    small_center = torch.stack([sy.float().mean() / 15.0, sx.float().mean() / 15.0])
    large_center = torch.stack([ly.float().mean() / 63.0, lx.float().mean() / 63.0])
    # Rasterized 7 has asymmetric 1px endpoints, so exact pixel centroids do
    # not scale perfectly; the named-address drift must stay sub-cell.
    assert torch.allclose(small_center, large_center, atol=0.04)


def test_mesh_aware_local_dilation_changes_no_parameter_shapes():
    from fine_grain.native_mot import NativeMoTStack

    model = NativeMoTStack(
        d_llm=32, d_x=32, d=32, res=32, n_slices=8, n_layers=1,
        n_heads=4, local_dilation=2,
    )
    assert model.stem_local[0].dilation == (2, 2)
    assert model.layers[0].local.dw.dilation == (2, 2)
    legacy = NativeMoTStack(
        d_llm=32, d_x=32, d=32, res=16, n_slices=8, n_layers=1,
        n_heads=4,
    )
    assert model.stem_local[0].weight.shape == legacy.stem_local[0].weight.shape
    assert model.layers[0].local.dw.weight.shape == legacy.layers[0].local.dw.weight.shape


def test_missing_image_precision_removes_rgb_evidence():
    torch.manual_seed(1)
    model = _model()
    a_img = torch.rand(1, 3, 16, 16)
    b_img = torch.rand(1, 3, 16, 16)
    with torch.no_grad():
        a = model(a_img, ["Draw digit 3"], image_precision=torch.zeros(1))
        b = model(b_img, ["Draw digit 3"], image_precision=torch.zeros(1))
    assert torch.equal(a["logits"], b["logits"])
    assert torch.equal(a["rgb"], b["rgb"])


def test_only_future_port_observes_history():
    rng = np.random.default_rng(29)
    for case in CAPABILITY_CASES[:3]:
        sample = capability_sample(rng, 16, case)
        assert torch.count_nonzero(sample["history_precision"]) == 0
    future = capability_sample(rng, 16, "image_to_future")
    assert torch.all(future["history_precision"] == 1)


def test_one_graph_produces_text_rgb_and_segmentation_losses():
    torch.manual_seed(2)
    model = _model()
    model.train()
    batch = make_capability_batch(
        np.random.default_rng(2), batch=4, res=16, cases=CAPABILITY_CASES,
    )
    out = model(
        batch["image"],
        batch["prompt"],
        t=torch.zeros(4),
        image_precision=batch["image_precision"],
        text_precision=batch["text_precision"],
        target_time=batch["target_time"],
    )
    assert out["logits"].shape[0] == 4
    assert out["rgb"].shape == (4, 3, 16, 16)
    assert out["seg_logits"].shape == (4, 2, 16, 16)
    batch["t"] = torch.zeros(4)
    loss, meta = model.omni_loss(out, batch, torch.device("cpu"))
    assert torch.isfinite(loss)
    assert meta["n_text"] == meta["n_pix"] == meta["n_seg"] == 4
    loss.backward()
    assert model.seg_head[-1].weight.grad is not None


def test_target_time_adaln_is_identity_at_present_and_learns_for_future():
    torch.manual_seed(3)
    plain = _model(target_time_adaln=False)
    temporal = _model(target_time_adaln=True)
    temporal.load_state_dict(plain.state_dict())
    image = torch.rand(2, 3, 16, 16)
    prompts = ["Read current frame", "Read current frame"]
    present = torch.zeros(2)
    with torch.no_grad():
        base = plain(image, prompts, target_time=present)
        conditioned = temporal(image, prompts, target_time=present)
    assert torch.equal(base["rgb"], conditioned["rgb"])
    assert torch.equal(base["logits"], conditioned["logits"])

    temporal.train()
    future = temporal(image, prompts, target_time=torch.ones(2))
    loss = future["rgb"].square().mean() + future["logits"].square().mean()
    loss.backward()
    grad = temporal.mot_stack.layers[0].ada_x.net[-1].weight.grad
    assert grad is not None
    assert float(grad.abs().sum()) > 0.0


def test_future_observation_adds_full_resolution_transition_kl():
    torch.manual_seed(4)
    model = _model(
        target_time_adaln=True,
        transition_loss_coef=0.5,
    )
    rng = np.random.default_rng(4)
    batch = collate_capability([
        capability_sample(rng, 16, case, digit="4", color="red", place="middle_center")
        for case in CAPABILITY_CASES
    ])
    zeros = torch.zeros(len(CAPABILITY_CASES))
    batch["t"] = zeros
    out = model.forward_with_future_posterior(
        batch["image"],
        batch["prompt"],
        batch["target_rgb"],
        t=zeros,
        image_precision=batch["image_precision"],
        text_precision=batch["text_precision"],
        target_time=batch["target_time"],
        history_images=batch["history_images"],
        history_precision=batch["history_precision"],
        action=batch["action"],
        action_precision=batch["action_precision"],
    )
    loss, meta = model.omni_loss(out, batch, torch.device("cpu"))
    assert torch.isfinite(loss)
    assert meta["n_transition"] == 1
    assert meta["transition_kl"] > 0.0
    assert meta["transition_q_obs"] > 0.0
    assert out["transition_q_rgb"].shape == batch["target_rgb"].shape
    loss.backward()
    grad = model.belief_logvar_head[-1].weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert float(grad.abs().sum()) > 0.0

    action_grad = model.mot_stack.action_to_x.weight.grad
    assert action_grad is not None
    assert float(action_grad.abs().sum()) > 0.0


def test_eulerian_history_order_and_action_precision_are_live():
    torch.manual_seed(5)
    model = _model(target_time_adaln=True)
    image = torch.rand(1, 3, 16, 16)
    history = torch.stack([torch.zeros_like(image[0]), image[0]], dim=0).unsqueeze(0)
    reverse = history.flip(1)
    with torch.no_grad():
        x_fwd = model.mot_stack.encode_X(
            image, history_images=history, history_precision=torch.ones(1, 2),
        )
        x_rev = model.mot_stack.encode_X(
            image, history_images=reverse, history_precision=torch.ones(1, 2),
        )
    assert not torch.equal(x_fwd, x_rev)
    actions = torch.tensor([[[-1.0, 0.0], [1.0, 0.0]]])
    a_fwd = model.mot_stack._action_embedding(
        actions, torch.ones(1, 2), 1, image.device, image.dtype,
    )
    a_rev = model.mot_stack._action_embedding(
        actions.flip(1), torch.ones(1, 2), 1, image.device, image.dtype,
    )
    assert not torch.equal(a_fwd, a_rev)

    model.train()
    out = model(
        image,
        [""],
        target_time=torch.ones(1),
        history_images=history,
        history_precision=torch.ones(1, 2),
        action=torch.tensor([[1.0, 0.0]]),
        action_precision=torch.ones(1),
    )
    out["rgb"].mean().backward()
    grad = model.mot_stack.layers[0].ada_x.net[-1].weight.grad
    assert grad is not None
    assert float(grad.abs().sum()) > 0.0

    model.eval()
    with torch.no_grad():
        left = model(
            image, [""], action=torch.tensor([[-1.0, 0.0]]),
            action_precision=torch.zeros(1),
        )
        right = model(
            image, [""], action=torch.tensor([[1.0, 0.0]]),
            action_precision=torch.zeros(1),
        )
    assert torch.equal(left["rgb"], right["rgb"])


def test_action_token_is_slice_visible_and_zero_precision_is_exactly_masked():
    torch.manual_seed(55)
    model = _model(action_tokens=True)
    image = torch.rand(2, 3, 16, 16)
    prompts = ["Predict next frame", "Predict next frame"]

    model.eval()
    with torch.no_grad():
        zero_left = model(
            image, prompts,
            action=torch.tensor([[-1.0, 0.0], [-1.0, 0.0]]),
            action_precision=torch.zeros(2),
        )
        zero_right = model(
            image, prompts,
            action=torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
            action_precision=torch.zeros(2),
        )
    assert torch.equal(zero_left["rgb"], zero_right["rgb"])
    assert zero_left["logits"].shape == zero_right["logits"].shape

    model.train()
    active = model(
        image, prompts,
        action=torch.tensor([[-1.0, 0.0], [1.0, 0.0]]),
        action_precision=torch.ones(2),
    )
    active["rgb"].square().mean().backward()
    token_grad = model.mot_stack.action_to_h.weight.grad
    assert token_grad is not None and float(token_grad.abs().sum()) > 0.0
    assert model.mot_stack.action_to_x.weight.grad is None


def test_horizon_token_keeps_tau_out_of_the_persistent_state_chart():
    torch.manual_seed(56)
    model = _model(action_tokens=True, horizon_tokens=True)
    image = torch.rand(2, 3, 16, 16)
    prompts = ["Predict next frame", "Predict next frame"]

    model.eval()
    with torch.no_grad():
        absent = model(image, prompts, target_time=None)
        present_zero = model(image, prompts, target_time=torch.zeros(2))
    assert torch.equal(absent["belief_mu"], present_zero["belief_mu"])

    model.train()
    future = model(image, prompts, target_time=torch.ones(2))
    future["rgb"].square().mean().backward()
    grad = model.mot_stack.horizon_to_h.weight.grad
    assert grad is not None and float(grad.abs().sum()) > 0.0
    assert model.mot_stack.target_time_coord is None


def test_action_relative_slice_bias_is_precision_gated_and_trainable():
    torch.manual_seed(57)
    model = _model(action_tokens=True, action_rel_bias=True)
    image = torch.rand(2, 3, 16, 16)
    prompts = ["Predict next frame", "Predict next frame"]

    model.eval()
    with torch.no_grad():
        left = model(
            image, prompts, action=torch.tensor([[-1.0, 0.0]]).expand(2, -1),
            action_precision=torch.zeros(2),
        )
        right = model(
            image, prompts, action=torch.tensor([[1.0, 0.0]]).expand(2, -1),
            action_precision=torch.zeros(2),
        )
    assert torch.equal(left["belief_mu"], right["belief_mu"])

    model.train()
    active = model(
        image, prompts,
        action=torch.tensor([[-1.0, 0.0], [1.0, 0.0]]),
        action_precision=torch.ones(2),
    )
    active["rgb"].square().mean().backward()
    grad = model.mot_stack.layers[0].action_rel_mlp[-1].weight.grad
    assert grad is not None and float(grad.abs().sum()) > 0.0


def test_action_relative_geometry_converts_dx_dy_to_grid_y_x():
    torch.manual_seed(570)
    model = _model(
        action_tokens=True, action_rel_bias=True, transition_gate=True,
    ).eval()
    captured = []
    handle = model.mot_stack.layers[0].action_rel_mlp[0].register_forward_pre_hook(
        lambda _module, inputs: captured.append(inputs[0].detach().clone()),
    )
    with torch.no_grad():
        model(
            torch.rand(1, 3, 16, 16), ["Predict next frame"],
            target_time=torch.ones(1),
            action=torch.tensor([[1.0, 0.0]]),
            action_precision=torch.ones(1),
        )
    handle.remove()
    # Relation features are [relative_yx, action_yx, residual_yx].
    assert captured
    assert torch.equal(captured[0][..., 2], torch.zeros_like(captured[0][..., 2]))
    assert torch.equal(captured[0][..., 3], torch.ones_like(captured[0][..., 3]))


def test_action_transport_is_identity_when_absent_and_trainable_when_executed():
    torch.manual_seed(571)
    model = _model(
        action_tokens=True, action_rel_bias=True, action_transport=True,
        transition_gate=True,
    )
    image = torch.rand(2, 3, 16, 16)
    prompts = ["Predict next frame", "Predict next frame"]
    model.eval()
    with torch.no_grad():
        present_left = model(
            image, prompts, target_time=torch.zeros(2),
            action=torch.tensor([[-1.0, 0.0]]).expand(2, -1),
            action_precision=torch.ones(2),
        )
        present_right = model(
            image, prompts, target_time=torch.zeros(2),
            action=torch.tensor([[1.0, 0.0]]).expand(2, -1),
            action_precision=torch.ones(2),
        )
    assert torch.equal(present_left["belief_mu"], present_right["belief_mu"])
    layer = model.mot_stack.layers[0]
    w = torch.softmax(torch.randn(2, 16 * 16, layer.M), dim=-1)
    noop = layer._transport_write_weights(w, torch.zeros(2, 2), torch.ones(2))
    missing = layer._transport_write_weights(
        w, torch.tensor([[1.0, 0.0]]).expand(2, -1), torch.zeros(2),
    )
    assert torch.equal(noop, w)
    assert torch.equal(missing, w)

    model.train()
    future = model(
        image, prompts, target_time=torch.ones(2),
        action=torch.tensor([[-1.0, 0.0], [1.0, 0.0]]),
        action_precision=torch.ones(2),
    )
    future["rgb"].square().mean().backward()
    grad = model.mot_stack.layers[0].action_transport.weight.grad
    assert grad is not None and torch.isfinite(grad).all()
    assert float(grad.abs().sum()) > 0.0


def test_action_slice_transition_is_normalized_exactly_gated_and_applied_once():
    torch.manual_seed(572)
    model = _model(
        action_tokens=True, action_rel_bias=True, action_slice_transition=True,
        transition_gate=True,
    )
    first, second = model.mot_stack.layers
    assert first.action_transition_mlp is not None
    assert second.action_transition_mlp is None
    S = torch.randn(2, first.M, 32)
    w = torch.softmax(torch.randn(2, 16 * 16, first.M), dim=-1)
    noop = first._action_transition_prior(
        S, w, torch.zeros(2, 2), torch.ones(2), enabled=True,
    )
    missing = first._action_transition_prior(
        S, w, torch.tensor([[1.0, 0.0]]).expand(2, -1),
        torch.zeros(2), enabled=True,
    )
    assert torch.equal(noop, S)
    assert torch.equal(missing, S)

    active = first._action_transition_prior(
        S, w, torch.tensor([[-1.0, 0.0], [1.0, 0.0]]),
        torch.ones(2), enabled=True,
    )
    transition = first.last_action_transition
    assert active.shape == S.shape
    assert transition is not None
    assert torch.allclose(transition.sum(dim=1), torch.ones(2, first.M), atol=1e-6)
    identity = torch.eye(first.M).unsqueeze(0)
    assert float((transition - identity).abs().max()) > 0.05

    calls = []
    handle = first.action_transition_mlp.register_forward_hook(
        lambda *_args: calls.append(1),
    )
    image = torch.rand(2, 3, 16, 16)
    out = model(
        image, ["Predict next frame"] * 2, target_time=torch.ones(2),
        action=torch.tensor([[-1.0, 0.0], [1.0, 0.0]]),
        action_precision=torch.ones(2),
    )
    handle.remove()
    assert len(calls) == 1
    out["rgb"].square().mean().backward()
    grad = first.action_transition_mlp[-1].weight.grad
    assert grad is not None and torch.isfinite(grad).all()
    assert float(grad.abs().sum()) > 0.0


def test_horizon_gate_separates_proposed_action_from_executed_transition():
    torch.manual_seed(58)
    model = _model(
        action_tokens=True, action_rel_bias=True, transition_gate=True,
    ).eval()
    image = torch.rand(1, 3, 16, 16)
    prompt = ["Predict next frame"]
    with torch.no_grad():
        present_left = model(
            image, prompt, target_time=torch.zeros(1),
            action=torch.tensor([[-1.0, 0.0]]), action_precision=torch.ones(1),
        )
        present_right = model(
            image, prompt, target_time=torch.zeros(1),
            action=torch.tensor([[1.0, 0.0]]), action_precision=torch.ones(1),
        )
        future_left = model(
            image, prompt, target_time=torch.ones(1),
            action=torch.tensor([[-1.0, 0.0]]), action_precision=torch.ones(1),
        )
        future_right = model(
            image, prompt, target_time=torch.ones(1),
            action=torch.tensor([[1.0, 0.0]]), action_precision=torch.ones(1),
        )
    assert torch.equal(present_left["belief_mu"], present_right["belief_mu"])
    assert not torch.equal(future_left["belief_mu"], future_right["belief_mu"])
    assert model.mot_stack.target_time_coord is None


def test_zero_terminal_modality_precision_removes_likelihood():
    torch.manual_seed(6)
    model = _model()
    sample = capability_sample(
        np.random.default_rng(6), 16, "image_to_current",
        digit="2", color="green", place="middle_center",
    )
    batch = collate_capability([sample])
    batch["need_pix"] = [False]
    batch["need_seg"] = [False]
    batch["target_text_precision"].fill_(1.0)
    batch["target_text_precision"].zero_()
    out = model(
        batch["image"], batch["prompt"],
        image_precision=batch["image_precision"],
        text_precision=batch["text_precision"],
        target_time=batch["target_time"],
    )
    loss, meta = model.omni_loss(out, batch, torch.device("cpu"))
    assert float(loss.detach()) == 0.0
    assert meta["ce"] == 0.0

    batch["target_text_precision"].fill_(1.0)
    full, _ = model.omni_loss(out, batch, torch.device("cpu"))
    batch["target_text_precision"].fill_(0.5)
    half, _ = model.omni_loss(out, batch, torch.device("cpu"))
    assert torch.allclose(half, 0.5 * full)


def test_checkpoint_rejects_action_that_harms_either_future_dense_metric():
    result = {
        "text_to_both": {"score": 0.90},
        "image_to_current": {"score": 0.80},
        "image_text_edit": {"score": 0.85},
        "image_to_future": {"paired_iou": 0.30, "seg_iou": 0.40},
    }
    _, _, admitted = _checkpoint_score(
        result, {"paired_iou": 0.20, "seg_iou": 0.41},
    )
    assert not admitted

    _, _, admitted = _checkpoint_score(
        result, {"paired_iou": 0.30, "seg_iou": 0.30},
    )
    assert not admitted

    _, _, admitted = _checkpoint_score(
        result, {"paired_iou": 0.20, "seg_iou": 0.30},
    )
    assert admitted


def test_counterfactual_future_group_holds_state_history_and_velocity_fixed():
    batch = make_counterfactual_future_batch(
        np.random.default_rng(70), groups=2, res=16,
    )
    assert batch["image"].shape[0] == 10
    for group_id in (0, 1):
        ids = torch.nonzero(
            batch["counterfactual_group"] == group_id, as_tuple=False,
        ).flatten()
        assert ids.numel() == 5
        first = int(ids[0])
        assert torch.equal(
            batch["image"].index_select(0, ids),
            batch["image"][first].expand(ids.numel(), -1, -1, -1),
        )
        assert torch.equal(
            batch["history_images"].index_select(0, ids),
            batch["history_images"][first].expand(ids.numel(), -1, -1, -1, -1),
        )
        assert torch.equal(
            batch["velocity"].index_select(0, ids),
            batch["velocity"][first].expand(ids.numel(), -1),
        )
        assert torch.unique(batch["action"].index_select(0, ids), dim=0).shape[0] == 5
        assert len({batch["target_place"][int(i)] for i in ids}) == 5


def test_paired_action_likelihood_prefers_matched_counterfactual_targets():
    batch = make_counterfactual_future_batch(
        np.random.default_rng(71), groups=1, res=16,
    )
    rgb = batch["target_rgb"].clone().requires_grad_()
    one_hot = torch.nn.functional.one_hot(batch["target_seg"], 2).permute(0, 3, 1, 2)
    seg_logits = (one_hot.float() * 8.0 - (1.0 - one_hot.float()) * 8.0).requires_grad_()
    matched, meta = paired_action_likelihood(
        {"rgb": rgb, "seg_logits": seg_logits}, batch,
    )
    shuffled, shuffled_meta = paired_action_likelihood(
        {"rgb": rgb.roll(1, 0), "seg_logits": seg_logits.roll(1, 0)}, batch,
    )
    assert matched < shuffled
    assert meta["action_energy_gap"] > 0.0
    assert shuffled_meta["action_energy_gap"] < meta["action_energy_gap"]
    matched.backward()
    assert rgb.grad is not None and torch.isfinite(rgb.grad).all()
    assert seg_logits.grad is not None and torch.isfinite(seg_logits.grad).all()


def test_checkpoint_can_require_nontrivial_action_effect_size():
    result = {
        "text_to_both": {"score": 0.90},
        "image_to_current": {"score": 0.80},
        "image_text_edit": {"score": 0.85},
        "image_to_future": {"paired_iou": 0.306, "seg_iou": 0.406},
    }
    _, _, admitted = _checkpoint_score(
        result,
        {"paired_iou": 0.300, "seg_iou": 0.400},
        action_min_gain=0.005,
    )
    assert admitted
    result["image_to_future"]["paired_iou"] = 0.305
    _, _, admitted = _checkpoint_score(
        result,
        {"paired_iou": 0.300, "seg_iou": 0.400},
        action_min_gain=0.005,
    )
    assert not admitted

    result["image_to_future"]["paired_iou"] = 0.306
    _, _, admitted = _checkpoint_score(
        result,
        {"paired_iou": 0.300, "seg_iou": 0.400},
        action_min_gain=0.005,
        transition_ablated={"paired_iou": 0.306, "seg_iou": 0.405},
    )
    assert not admitted
