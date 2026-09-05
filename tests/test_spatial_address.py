"""Unit tests for LanguageAddressPrior, multi_freq_coords, and address losses."""
import pytest
import torch
import torch.nn as nn
from fine_grain.spatial_address import (
    LanguageAddressPrior,
    compute_address_diversity_loss,
    compute_address_kl_loss,
    multi_freq_coords,
    compute_address_compactness_loss,
)


def test_multi_freq_coords():
    R = 16
    n_freq = 4
    device = torch.device("cpu")
    coords_phi = multi_freq_coords(R, device, n_freq=n_freq)
    N = R * R
    expected_dim = 2 + 4 * n_freq  # 2 + 16 = 18
    assert coords_phi.shape == (1, N, expected_dim)
    # Check linear coords are in [-1, 1]
    assert coords_phi[..., 0].min() >= -1.0 and coords_phi[..., 0].max() <= 1.0
    assert coords_phi[..., 1].min() >= -1.0 and coords_phi[..., 1].max() <= 1.0
    # Check sines/cosines are in [-1, 1]
    assert coords_phi[..., 2:].min() >= -1.0 and coords_phi[..., 2:].max() <= 1.0


def test_language_address_prior_sensitivity():
    """Verify that different language prompts produce DIFFERENT address routing."""
    torch.manual_seed(42)
    B, T, d = 2, 8, 64
    M = 16
    R = 16
    N = R * R
    prior_mod = LanguageAddressPrior(d_model=d, n_slices=M, n_freq=4, hidden_dim=32)

    # Prompt A and Prompt B
    H_a = torch.randn(1, T, d)
    H_b = torch.randn(1, T, d)

    logits_a, w_a = prior_mod(H_a, R)
    logits_b, w_b = prior_mod(H_b, R)

    assert w_a.shape == (1, N, M)
    assert w_b.shape == (1, N, M)

    # Sum of probabilities over slices must be 1.0 at every pixel
    assert torch.allclose(w_a.sum(dim=-1), torch.ones(1, N), atol=1e-5)
    assert torch.allclose(w_b.sum(dim=-1), torch.ones(1, N), atol=1e-5)

    # W(H_a) and W(H_b) must NOT be identical (Prompt sensitivity > 0)
    diff = float(torch.norm(w_a - w_b).item())
    assert diff > 0.1, f"Expected prompt sensitivity, got diff={diff:.6f}"


def test_address_kl_loss():
    B, N, M = 2, 64, 8
    q = torch.softmax(torch.randn(B, N, M), dim=-1)
    p = torch.softmax(torch.randn(B, N, M), dim=-1)

    # 1. Self KL is 0
    kl_self = compute_address_kl_loss(q, q)
    assert abs(float(kl_self.item())) < 1e-4

    # 2. Distinct KL is positive
    kl_diff = compute_address_kl_loss(q, p)
    assert float(kl_diff.item()) > 0.0

    # 3. Gradient flows back to p
    p_var = p.clone().detach().requires_grad_(True)
    kl_grad = compute_address_kl_loss(q, p_var)
    kl_grad.backward()
    assert p_var.grad is not None
    assert p_var.grad.abs().sum() > 0


def test_address_diversity_loss():
    B, N, M = 2, 128, 8
    # 1. Collapsed assignment (all pixels go to slot 0)
    w_collapsed = torch.zeros(B, N, M)
    w_collapsed[:, :, 0] = 0.99
    w_collapsed[:, :, 1:] = 0.01 / (M - 1)

    # 2. Diverse assignment
    w_diverse = torch.softmax(torch.randn(B, N, M), dim=-1)

    div_collapsed = compute_address_diversity_loss(w_collapsed)
    div_diverse = compute_address_diversity_loss(w_diverse)

    # Diverse assignment should have significantly higher slot entropy and mutual information
    assert div_diverse["slot_entropy"] > div_collapsed["slot_entropy"]


def test_address_compactness_prefers_localized_assignments_and_has_grad():
    # Four quadrant slots are compact; a uniform point-to-slot map is not.
    side, slots = 8, 4
    yy, xx = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
    slot = (yy.ge(side // 2).long() * 2 + xx.ge(side // 2).long()).reshape(-1)
    local = torch.nn.functional.one_hot(slot, slots).float().unsqueeze(0)
    uniform = torch.full_like(local, 1.0 / slots)
    assert compute_address_compactness_loss(local)["address_compactness_loss"] < compute_address_compactness_loss(uniform)["address_compactness_loss"]
    variable = uniform.clone().requires_grad_(True)
    compute_address_compactness_loss(variable)["address_compactness_loss"].backward()
    assert variable.grad is not None and variable.grad.abs().sum() > 0


def test_gamma_eight_preserves_diffuse_half_precision_write_mass():
    """(1/64)^8 underflows in fp16 unless Deslice normalizes in fp32."""
    from fine_grain.native_mot import DesliceWrite
    module = DesliceWrite(8, 8, write_sharpening=True)
    module.write_gamma_raw.data.fill_(torch.log(torch.tensor(8.0)))
    diffuse = torch.full((1, 16, 64), 1.0 / 64.0, dtype=torch.float16)
    written = module._write_w(diffuse)
    assert torch.allclose(written.sum(dim=-1), torch.ones(1, 16, dtype=torch.float16), atol=2e-3)
    assert torch.isfinite(written).all()


def test_native_mot_lang_address_end_to_end():
    """Verify that NativeMoT with use_lang_address=True breaks prompt-blindness on blank canvas."""
    from scripts.run_v0_surprise_eval import DualStreamVQAModel

    torch.manual_seed(42)
    res = 16
    d_model = 64
    n_slices = 16
    n_layers = 2

    model = DualStreamVQAModel(
        d_model=d_model, n_slices=n_slices, n_layers=n_layers, res=res,
        surprise_mode="v1_bayes", use_lang_address=True, lang_address_freq=4,
    )
    model.eval()

    blank_img = torch.zeros(1, 3, res, res)
    prompts = [
        "What digit is drawn with thin stroke ?",
        "How many corners does red polyline have ?",
    ]

    captured_w = []
    old_read = model.mot_stack.layers[0].read.forward
    def hook_read(*args, **kwargs):
        S, w = old_read(*args, **kwargs)
        captured_w.append(w)
        return S, w
    model.mot_stack.layers[0].read.forward = hook_read

    # Run prompt A on blank
    captured_w.clear()
    with torch.no_grad():
        _ = model(blank_img, [prompts[0]], image_precision=0.0)
    w_a = captured_w[0][0]  # [N, M]

    # Run prompt B on blank
    captured_w.clear()
    with torch.no_grad():
        _ = model(blank_img, [prompts[1]], image_precision=0.0)
    w_b = captured_w[0][0]  # [N, M]

    diff = float(torch.norm(w_a - w_b).item())
    rel_diff = diff / float(torch.norm(w_a).item() + 1e-8)

    print(f"End-to-end Blank Canvas Divergence: diff={diff:.5f}, rel={rel_diff*100:.2f}%")
    # Prompt-blindness is broken: diff must be distinctly positive! (previously strictly 0.000000)
    assert diff > 0.01, f"Expected prompt-aware address routing, got diff={diff:.6f}"
