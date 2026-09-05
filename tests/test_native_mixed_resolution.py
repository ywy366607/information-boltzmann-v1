import pytest
import torch

from fine_grain.omni_model import DualStreamOmni
from fine_grain.native_mot import square_field_side


def model_kwargs(res=16):
    return dict(d_model=16, n_slices=4, n_layers=2, n_heads=2, res=res,
                surprise_mode="v1_bayes", s_update="raw", use_stiefel=False,
                use_modal_precision=True, use_target_time=True, use_task_tokens=True,
                control_prefix_attention=True, gaussian_head_layout="per_head",
                history_size=2, action_dim=2, use_action_tokens=True,
                use_action_rel_bias=True, gate_action_by_horizon=True,
                use_active_gdn2=True, active_gdn2_initial_trust=0.1,
                prior_write=1.0, seg_classes=2, pixel_loss_mode="gaussian_nll")


def test_two_native_sizes_share_parameters_and_accumulate_gradients():
    torch.set_num_threads(4)
    torch.manual_seed(27)
    model = DualStreamOmni(**model_kwargs())
    parameter_ids = [id(p) for p in model.parameters()]
    state_shapes = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    objectives = []
    for res in (64, 256):
        image = torch.rand(1, 3, res, res, requires_grad=True)
        out = model(image, ["Reconstruct"], image_precision=torch.ones(1),
                    text_precision=torch.zeros(1), target_time=torch.zeros(1),
                    task_id=torch.tensor([1]))
        assert out["X"].shape == (1, res * res, 16)
        assert out["rgb"].shape == (1, 3, res, res)
        assert out["seg_logits"].shape == (1, 2, res, res)
        objectives.append((out["rgb"].square().mean(), image))
    # Keep both graphs live: no global res mutation can corrupt the first.
    sum(loss for loss, _ in objectives).backward()
    assert all(image.grad is not None and torch.isfinite(image.grad).all()
               for _, image in objectives)
    assert model.res == model.mot_stack.res == 16
    assert parameter_ids == [id(p) for p in model.parameters()]
    assert state_shapes == {k: tuple(v.shape) for k, v in model.state_dict().items()}


def test_dynamic_call_matches_same_weights_instantiated_at_target_size():
    torch.manual_seed(12)
    model = DualStreamOmni(**model_kwargs(8)).eval()
    fixed = DualStreamOmni(**model_kwargs(16)).eval()
    fixed.load_state_dict(model.state_dict(), strict=True)
    image = torch.rand(1, 3, 16, 16)
    history = torch.rand(1, 2, 3, 16, 16)
    kwargs = dict(image_precision=torch.ones(1), text_precision=torch.zeros(1),
                  target_time=torch.ones(1), history_images=history,
                  history_precision=torch.ones(1, 2), action=torch.ones(1, 2),
                  action_precision=torch.ones(1), task_id=torch.tensor([3]))
    with torch.no_grad():
        dynamic_out = model(image, [""], **kwargs)
        fixed_out = fixed(image, [""], **kwargs)
    for key in ("rgb", "seg_logits", "belief_mu", "belief_logvar"):
        torch.testing.assert_close(dynamic_out[key], fixed_out[key], rtol=0, atol=0)


def test_square_field_validation_is_explicit():
    assert square_field_side(65536) == 256
    with pytest.raises(ValueError, match="square"):
        square_field_side(123)
