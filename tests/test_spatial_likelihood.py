import pytest
import torch

from fine_grain.spatial_likelihood import spatial_nll_mean, stratified_rgb_weights


@pytest.mark.parametrize("side", [64, 256])
def test_three_strata_equal_mass_and_sparse_gradient_not_diluted(side):
    target = torch.zeros(1, 3, side, side, requires_grad=True)
    foreground = torch.zeros(1, side, side, dtype=torch.bool)
    foreground[:, side//2, side//2-5:side//2+5] = True
    weights = stratified_rgb_weights(target, foreground)
    assert weights.shape == (1, 1, side, side) and not weights.requires_grad
    torch.testing.assert_close(weights.mean(), torch.tensor(1.0))
    expanded = torch.nn.functional.max_pool2d(foreground[:, None].float(), 5, 1, 2).bool()
    strata = [foreground[:, None], expanded & ~foreground[:, None], ~expanded]
    for mask in strata:
        torch.testing.assert_close(weights[mask].sum() / side**2, torch.tensor(1/3))
    nll = torch.ones_like(target, requires_grad=True)
    spatial_nll_mean(nll, weights).sum().backward()
    torch.testing.assert_close(nll.grad[foreground[:, None].expand_as(nll)].sum(), torch.tensor(1/3))


def test_constant_target_uniform_default_exact_and_no_target_gradient():
    target = torch.zeros(2, 3, 8, 8, requires_grad=True)
    weights = stratified_rgb_weights(target)
    assert weights.eq(1).all() and not weights.requires_grad
    nll = torch.randn_like(target)
    torch.testing.assert_close(spatial_nll_mean(nll), nll.flatten(1).mean(1), rtol=0, atol=0)
    torch.testing.assert_close(spatial_nll_mean(nll, weights), spatial_nll_mean(nll), rtol=0, atol=0)


def test_natural_edges_without_segmentation_mask_and_invalid_weights():
    target = torch.zeros(1, 3, 16, 16)
    target[..., 8:] = 1
    weights = stratified_rgb_weights(target)
    assert weights[..., 7].mean() > weights[..., 0].mean()
    with pytest.raises(ValueError, match="mean one"):
        spatial_nll_mean(target, weights * 2)
    with pytest.raises(ValueError, match="not be learned"):
        spatial_nll_mean(target, weights.requires_grad_())
    with pytest.raises(ValueError, match="finite"):
        spatial_nll_mean(target, torch.full((1, 1, 16, 16), float("nan")))


def test_omni_measure_changes_only_rgb_likelihood_not_forward_or_segmentation():
    from fine_grain.omni_model import DualStreamOmni
    from test_native_mixed_resolution import model_kwargs

    torch.manual_seed(8)
    model = DualStreamOmni(**model_kwargs(8), prior_loss_coef=0.0, vfe_coef=0.0)
    image = torch.zeros(1, 3, 8, 8)
    gold = torch.zeros_like(image)
    gold[:, 0, 4, 3:5] = 1
    mask = gold[:, 0].long()
    out = model(image, ["Draw digit 1"], image_precision=torch.zeros(1),
                text_precision=torch.ones(1), target_time=torch.zeros(1), task_id=torch.zeros(1, dtype=torch.long))
    batch = dict(need_text=[False], need_pix=[True], need_seg=[True], target_rgb=gold,
                 target_seg=mask, target_time=torch.zeros(1), target_text_precision=torch.zeros(1),
                 target_image_precision=torch.ones(1), target_seg_precision=torch.ones(1))
    rgb_before = out["rgb"].detach().clone()
    base_loss, base_meta = model.omni_loss(out, batch, torch.device("cpu"))
    weighted_loss, weighted_meta = model.omni_loss(
        out, {**batch, "rgb_likelihood_weight": stratified_rgb_weights(gold, mask)}, torch.device("cpu"))
    assert base_meta["seg"] == weighted_meta["seg"]
    assert abs(base_meta["pix"] - weighted_meta["pix"]) > 1e-4
    torch.testing.assert_close(weighted_loss - base_loss,
                               torch.tensor(weighted_meta["pix"] - base_meta["pix"]), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(out["rgb"], rgb_before, rtol=0, atol=0)
    weighted_loss.backward()
    assert model.pix_head[-1].weight.grad is not None
