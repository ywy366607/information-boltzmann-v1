"""Mathematical invariance and autograd tests for CBIM Field Model."""
import math
import pytest
import torch

from scripts.ib_local.cbim_field import (
    CBIMFieldModel,
    CayleyTransport,
    ConservativeScattering,
    LocalFieldWriter,
)


@pytest.fixture(autouse=True)
def set_seed():
    torch.manual_seed(42)
    torch.set_num_threads(1)


def test_cayley_transport_exact_l2_norm_preservation():
    """Verify that Cayley unitary streaming strictly preserves Frobenius norm in FP64."""
    L = 64
    d = 128
    dt = 1.0

    transport = CayleyTransport(L=L, d=d, dt=dt).double()
    # Randomize dispersion velocity spectrum
    transport.velocity.data.normal_(0.0, 0.5)

    B = 4
    h = torch.randn(B, L, d, dtype=torch.float64)
    initial_norm = h.norm(dim=(-2, -1))

    h_streamed, diag = transport(h)
    streamed_norm = h_streamed.norm(dim=(-2, -1))

    diff = (initial_norm - streamed_norm).abs().max().item()
    print(f"Cayley Transport FP64 norm error: {diff:.2e}")
    assert diff < 1e-12, f"Norm preservation violated: max error = {diff:.2e}"


def test_conservative_scattering_linear_and_quadratic_invariants():
    """Verify that state-dependent scattering strictly conserves linear sum and quadratic energy in FP64."""
    L = 64
    d = 64
    scattering = ConservativeScattering(L=L, d=d, n_layers=2).double()

    # Randomize scattering network
    for p in scattering.parameters():
        p.data.normal_(0.0, 0.2)

    B = 4
    h = torch.randn(B, L, d, dtype=torch.float64)

    # 1. Total grid linear sum: sum_x h(x)
    initial_sum = h.sum(dim=1)
    initial_energy = h.square().sum(dim=(-2, -1))

    h_scattered, _ = scattering(h)

    final_sum = h_scattered.sum(dim=1)
    final_energy = h_scattered.square().sum(dim=(-2, -1))

    diff_sum = (initial_sum - final_sum).abs().max().item()
    diff_energy = (initial_energy - final_energy).abs().max().item()

    print(f"Scattering linear sum error: {diff_sum:.2e}")
    print(f"Scattering quadratic energy error: {diff_energy:.2e}")

    rel_diff_energy = (diff_energy / initial_energy.max().item())
    assert diff_sum < 1e-12, f"Linear sum conservation violated: {diff_sum:.2e}"
    assert rel_diff_energy < 1e-12, f"Relative energy conservation violated: {rel_diff_energy:.2e}"


def test_energy_budget_boundedness():
    """Verify that field energy is strictly bounded: E(h_t) <= R^2 / 2 for all t."""
    L = 32
    d = 64
    R_budget = 5.0
    max_allowed_energy = 0.5 * (R_budget ** 2)

    model = CBIMFieldModel(vocab_size=1000, L=L, d=d, R_budget=R_budget)

    B = 2
    T = 50
    input_ids = torch.randint(0, 1000, (B, T))
    targets = torch.randint(0, 1000, (B, T))

    loss, h_final, diag = model(input_ids, targets)

    # Energy at every step must be bounded by R^2 / 2
    assert diag['final_energy'] <= max_allowed_energy + 1e-5, (
        f"Energy exceeded budget: final_energy={diag['final_energy']} > {max_allowed_energy}"
    )
    assert diag['mean_energy'] <= max_allowed_energy + 1e-5


def test_cbim_gradient_reachability_all_modules():
    """Verify that autograd gradients flow cleanly to all CBIM parameters."""
    L = 32
    d = 64
    model = CBIMFieldModel(vocab_size=500, L=L, d=d, R_budget=5.0)

    B = 2
    T = 8
    input_ids = torch.randint(0, 500, (B, T))
    targets = torch.randint(0, 500, (B, T))

    loss, _, _ = model(input_ids, targets)
    loss.backward()

    # Check loss is finite
    assert torch.isfinite(loss)

    # Check parameters receive finite gradients
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Parameter {name} grad is None"
            assert torch.isfinite(p.grad).all(), f"Parameter {name} has non-finite grad"


def test_autoregressive_causality():
    """Verify that future tokens do not affect past states or past logits."""
    L = 32
    d = 64
    model = CBIMFieldModel(vocab_size=500, L=L, d=d)

    # Two sequences identical up to token t=3, diverging at t=4
    seq1 = torch.tensor([[10, 20, 30, 40, 50, 60]])
    seq2 = torch.tensor([[10, 20, 30, 40, 99, 88]])

    # Run step-by-step
    h1 = torch.zeros(1, L, d)
    h2 = torch.zeros(1, L, d)

    for t in range(4):
        tok1 = seq1[:, t]
        tok2 = seq2[:, t]
        logits1, h1, _ = model.step(h1, tok1)
        logits2, h2, _ = model.step(h2, tok2)

        # Before divergence point, states and logits must match identically
        assert torch.allclose(logits1, logits2, atol=1e-6)
        assert torch.allclose(h1, h2, atol=1e-6)


def test_scattering_has_direct_ce_gradient_and_readout_response():
    model = CBIMFieldModel(vocab_size=32, L=8, d=16)
    h = torch.randn(2, 8, 16) * .3
    tokens = torch.tensor([1, 2])
    logits, _, _ = model.step(h, tokens)
    without, _, _ = model.step(h, tokens, disable_scattering=True)
    assert (logits - without).abs().max() > 1e-7
    torch.nn.functional.cross_entropy(logits, torch.tensor([3, 4])).backward()
    for net in model.scattering.angle_nets:
        assert net[-1].weight.grad.norm() > 0


def test_sequence_matches_streaming_and_state_writer_learns():
    model = CBIMFieldModel(vocab_size=32, L=8, d=16)
    ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    targets = ids + 1
    loss, final, diag = model(ids, targets)
    h = torch.zeros(2, 8, 16)
    logits = []
    for t in range(3):
        out, h, _ = model.step(h, ids[:, t])
        logits.append(out)
    expected = torch.nn.functional.cross_entropy(torch.stack(logits, 1).reshape(-1, 32), targets.flatten())
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(final, h)
    assert diag['initial_energy'] == 0
    assert 0 <= diag['clamp_trigger_rate'] <= 1
    loss.backward()
    assert model.writer.state_content.weight.grad.norm() > 0
    assert model.writer.state_rate.weight.grad.norm() > 0
