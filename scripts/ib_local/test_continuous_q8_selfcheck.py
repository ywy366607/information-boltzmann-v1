"""Self-check suite for Continuous Adaptive Velocity Heads on S^2 and Validation Protocol."""
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(script_dir))
if sys.path and os.path.abspath(sys.path[0]) == script_dir:
    sys.path.pop(0)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import math
import torch
import torch.nn.functional as F

from scripts.ib_local.cbim_torus3d import (
    d3q_velocities,
    MicroStepDirection,
    MicroStepClock,
    VelocityCayleyTransport3D,
    CBIMTorus3D,
)
from scripts.ib_local.train_cbim_malecns_internal_time import TruncatedInternalTimeGraphTrainer


def test_microstep_direction():
    print("=== Test 1: MicroStepDirection & S^2 Geometry ===")
    d, heads = 128, 8
    mod = MicroStepDirection(d=d, heads=heads).cuda()
    field = torch.randn(2, 4, 4, 4, d, device="cuda")
    token_embed = torch.randn(2, d, device="cuda")

    n_dir = mod(field, token_embed)  # [2, 8, 3]
    assert n_dir.shape == (2, heads, 3), f"Wrong shape: {n_dir.shape}"

    # 1. Check unit norm on S^2
    norms = n_dir.norm(p=2, dim=-1)
    norm_err = (norms - 1.0).abs().max().item()
    print(f"Max S^2 unit norm error: {norm_err:.2e}")
    assert norm_err < 1e-5, f"Norm error too high: {norm_err}"

    # 2. Check initial displacement from D3Q8 baseline (should be ~0 at init)
    base = mod.base_dirs[None].expand(2, -1, -1)
    cos = (n_dir * base).sum(dim=-1)
    disp_deg = torch.rad2deg(torch.acos(cos.clamp(-1.0, 1.0))).mean().item()
    print(f"Initial displacement from D3Q8: {disp_deg:.4f} degrees")
    assert disp_deg < 0.01, f"Initial displacement should be ~0, got {disp_deg}"

    # 3. Check pairwise separation of 8 heads (should be ~98.21 deg for D3Q8)
    M = torch.bmm(n_dir, n_dir.transpose(1, 2))
    off_diag_cos = (M.sum(dim=(-1, -2)) - heads) / (heads * (heads - 1))
    sep_deg = torch.rad2deg(torch.acos(off_diag_cos.clamp(-1.0, 1.0))).mean().item()
    print(f"Initial pairwise separation between 8 heads: {sep_deg:.4f} degrees")
    assert abs(sep_deg - 98.21) < 1.0, f"Unexpected separation: {sep_deg}"

    # 4. Check backward pass & gradients
    loss = (n_dir * torch.randn_like(n_dir)).sum()
    loss.backward()
    for name, p in mod.named_parameters():
        assert p.grad is not None, f"Missing grad for {name}"
        assert torch.isfinite(p.grad).all(), f"Non-finite grad for {name}"
    torch.cuda.synchronize()
    print("Test 1 PASSED.\n")


def test_velocity_transport():
    print("=== Test 2: VelocityCayleyTransport3D with S^2 Directions ===")
    shape = (4, 4, 4)
    velocities, content_dim = 8, 16
    d = velocities * content_dim
    transport = VelocityCayleyTransport3D(shape=shape, velocities=velocities, content_dim=content_dim).cuda()

    field = torch.randn(1, 4, 4, 4, d, device="cuda")
    init_energy = field.square().sum().item()

    # Random directions on S^2
    n_dir = F.normalize(torch.randn(1, velocities, 3, device="cuda"), p=2, dim=-1)
    dt = torch.tensor([[1.25]], device="cuda")

    mult, omega = transport.multiplier(delta_tau=dt, direction=n_dir)
    assert mult.shape == (1, 4, 4, 4, d), f"Wrong mult shape: {mult.shape}"

    out = transport.apply_multiplier(field, mult)
    out_energy = out.square().sum().item()
    energy_rel_diff = abs(out_energy - init_energy) / init_energy
    print(f"Transport unitary energy conservation relative error: {energy_rel_diff:.2e}")
    assert energy_rel_diff < 1e-5, f"Energy conservation failed: {energy_rel_diff}"
    torch.cuda.synchronize()
    print("Test 2 PASSED.\n")


def test_full_model_and_cudagraph():
    print("=== Test 3: CBIMTorus3D with Continuous Velocities & CUDA Graph Capture ===")
    model = CBIMTorus3D(
        shape=(4, 4, 4),
        velocities=8,
        content_dim=16,
        readout_type="dynamic_linear",
        write_type="w2_impedance",
        micro_steps=3,
        adaptive_clock=True,
        continuous_velocities=True,
        relative_address=False,
        v2_coordinate_components=True,
    ).cuda()

    print(f"Architecture: {model.architecture}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.3f}M")

    # Forward pass test
    ids = torch.randint(0, 1000, (1, 16), device="cuda")
    targets = torch.randint(0, 1000, (1, 16), device="cuda")
    with torch.no_grad():
        loss, state, diag = model(ids, targets)

    print(f"Forward loss: {loss.item():.4f}")
    print("Diagnostics captured:")
    for k in ["dir_disp_deg", "dir_pairwise_sep_deg", "dir_change_micro_deg", "alpha_1", "alpha_2", "alpha_3", "t_packet"]:
        if k in diag:
            val = diag[k].item() if isinstance(diag[k], torch.Tensor) else diag[k]
            print(f"  {k}: {val:.4f}")
            assert math.isfinite(val), f"Non-finite diagnostic: {k}={val}"

    torch.cuda.synchronize()
    # CUDA Graph capture and replay test
    print("\nTesting TruncatedInternalTimeGraphTrainer with CUDAGraph (tokens=64, chunk=16)...")
    runner = TruncatedInternalTimeGraphTrainer(model, tokens=64, chunk_tokens=16, lr=3e-4)
    for step in range(3):
        chunk_ids = torch.randint(0, 1000, (1, 64), device="cuda")
        chunk_targets = torch.randint(0, 1000, (1, 64), device="cuda")
        step_loss, step_state, step_diag = runner.step(chunk_ids, chunk_targets)
        print(f"  Step {step+1}: Loss = {step_loss:.4f}, GradNorm = {runner.grad_norm.item():.4f}")
        assert math.isfinite(step_loss), f"Non-finite loss at step {step+1}"
        assert math.isfinite(runner.grad_norm.item()), f"Non-finite grad norm at step {step+1}"
    print("Test 3 PASSED.\n")


def test_validation_protocol():
    print("=== Test 4: Validation Protocol (Mature Stream vs Cold Vacuum) ===")
    model = CBIMTorus3D(
        shape=(4, 4, 4),
        velocities=8,
        content_dim=16,
        readout_type="dynamic_linear",
        write_type="w2_impedance",
        micro_steps=3,
        adaptive_clock=True,
        continuous_velocities=True,
        relative_address=False,
        v2_coordinate_components=True,
    ).cuda()

    # 1. Cold vacuum state
    cold_state = model.initial_state(1, "cuda")
    ids = torch.randint(0, 1000, (1, 16), device="cuda")
    targets = torch.randint(0, 1000, (1, 16), device="cuda")
    with torch.no_grad():
        cold_loss, mature_state, _ = model(ids, targets, cold_state)

    # 2. Inherit mature state (step > 0)
    with torch.no_grad():
        mature_loss, next_state, _ = model(ids, targets, mature_state.detach().clone())

    print(f"Cold vacuum initial loss: {cold_loss.item():.4f}")
    print(f"Mature stream next loss: {mature_loss.item():.4f}")
    print("State inheritance verified without error.")
    print("Test 4 PASSED.\n")


if __name__ == "__main__":
    test_microstep_direction()
    test_velocity_transport()
    test_full_model_and_cudagraph()
    test_validation_protocol()
    print("ALL SELF-CHECK TESTS PASSED SUCCESSFULLY!")
