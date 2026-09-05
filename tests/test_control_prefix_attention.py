import torch
import pytest

from fine_grain.native_mot import NativeMoTBlock
from fine_grain.omni_model import DualStreamOmni
from fine_grain.bayesian_surprise import BayesianSurpriseGate


def test_control_prefix_visible_without_multilayer_answer_leakage():
    torch.manual_seed(4)
    blocks = [NativeMoTBlock(d=16, n_heads=4) for _ in range(3)]
    s, h = torch.randn(1, 4, 16), torch.randn(1, 5, 16)
    mask = torch.ones(1, 5, dtype=torch.bool)
    prompt = torch.tensor([[True, True, False, False, True]])
    control = torch.tensor([[False, False, False, False, True]])

    def run(h0):
        sx, hx = s, h0
        for block in blocks:
            sx, hx, _ = block(sx, hx, text_mask=mask, prompt_mask=prompt, condition_mask=control)
        return sx, hx

    sx, hx = run(h)
    answer_changed = h.clone()
    answer_changed[:, 3] += 20 * torch.randn(16)
    sa, ha = run(answer_changed)
    torch.testing.assert_close(sx, sa, rtol=0, atol=0)
    torch.testing.assert_close(hx[:, [0, 1, 2, 4]], ha[:, [0, 1, 2, 4]], rtol=0, atol=0)
    mode_changed = h.clone()
    mode_changed[:, 4] += torch.randn(16)
    _, hm = run(mode_changed)
    assert not torch.allclose(hx[:, 0], hm[:, 0])
    assert blocks[0].last_at[:, :, 0, -1].min() > 0


def test_masked_control_cannot_affect_any_live_output():
    torch.manual_seed(5)
    b = NativeMoTBlock(d=16, n_heads=4)
    s, h = torch.randn(1, 4, 16), torch.randn(1, 3, 16)
    mask = torch.tensor([[True, True, False]])
    control = torch.tensor([[False, False, True]])
    a = b(s, h, text_mask=mask, condition_mask=control)
    h[:, -1] += 100
    c = b(s, h, text_mask=mask, condition_mask=control)
    torch.testing.assert_close(a[0], c[0])
    torch.testing.assert_close(a[1][:, :2], c[1][:, :2])


def test_sink_is_exact_zero_value_option_and_has_gradients():
    torch.manual_seed(6)
    b = NativeMoTBlock(d=16, n_heads=4)
    b.enable_attention_sink(1.0)
    logits = torch.randn(2, 4, 3, 7, requires_grad=True)
    weights, importance = b._sink_attention(logits, b.sink_v)
    explicit = torch.softmax(torch.cat([logits, b.sink_v[None, :, None, None].expand(2, -1, 3, 1)], -1), -1)
    torch.testing.assert_close(weights, explicit[..., :-1])
    assert (weights.sum(-1) < 1).all()
    importance.sum().backward()
    assert b.sink_v.grad.abs().sum() > 0
    assert logits.grad.abs().sum() > 0


def test_sink_config_reload_and_disabled_parameter_compatibility():
    config = dict(d_model=16, n_slices=4, n_heads=4, n_layers=2, res=8,
                  use_task_tokens=True, control_prefix_attention=True, use_attention_sink=True)
    a, b = DualStreamOmni(**config), DualStreamOmni(**config)
    b.load_state_dict(a.state_dict(), strict=True)
    assert hasattr(a.mot_stack.layers[0].mot, "sink_v")
    old = NativeMoTBlock(d=16, n_heads=4)
    assert not any("sink" in key for key in old.state_dict())


def test_gaussian_per_head_mean_and_variance_preserve_channel_identity():
    class KnownGaussian(torch.nn.Module):
        def forward(self, x):
            return torch.cat([x, 2 * x], dim=-1)

    gate = BayesianSurpriseGate(16, n_heads=4, mode="v1_bayes", gaussian_head_layout="per_head")
    x = torch.arange(16.0).reshape(1, 1, 16).requires_grad_()
    mu, lv = gate._head_mlp(x, KnownGaussian()).chunk(2, -1)
    torch.testing.assert_close(mu, x)
    torch.testing.assert_close(lv, 2 * x)
    for target in (mu, lv):
        grad = torch.autograd.grad(target.sum(), x, retain_graph=True)[0]
        assert (grad.reshape(4, 4).abs().sum(-1) > 0).all()
    gate.gaussian_head_layout = "legacy"
    old_mu, old_lv = gate._head_mlp(x, KnownGaussian()).chunk(2, -1)
    grad_mu = torch.autograd.grad(old_mu.sum(), x, retain_graph=True)[0]
    grad_lv = torch.autograd.grad(old_lv.sum(), x)[0]
    assert torch.count_nonzero(grad_mu[..., 8:]) == 0
    assert torch.count_nonzero(grad_lv[..., :8]) == 0


def test_weight_loaders_reject_silent_gaussian_layout_changes(tmp_path):
    from scripts.train_northstar_capabilities import load_compatible
    config = dict(d_model=16, n_slices=4, n_heads=4, n_layers=2, res=8)
    old = DualStreamOmni(**config)
    path = tmp_path / "legacy.pt"
    torch.save({"state_dict": old.state_dict(), "config": config}, path)
    fixed = DualStreamOmni(**config, gaussian_head_layout="per_head")
    with pytest.raises(ValueError, match="layout mismatch"):
        load_compatible(fixed, path)
    with pytest.raises(ValueError, match="layout mismatch"):
        fixed.load_visual_champion(path)
