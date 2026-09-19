"""Unit tests for M03 WriteOperator."""
import pytest
import torch

from scripts.ib_local.write_operator import WriteOperator


@pytest.fixture(autouse=True)
def set_seed():
    torch.manual_seed(42)
    torch.set_num_threads(1)


def test_write_operator_identity_at_initialization():
    """Verify that zero-initialized final layers produce exact identity mapping."""
    dim = 4
    hidden_dim = 128
    n_particles = 256
    op = WriteOperator(dim=dim, hidden_dim=hidden_dim)

    x = torch.randn(n_particles, dim)
    v = torch.randn(n_particles, dim)
    tok_emb = torch.randn(hidden_dim)

    x_new, v_new, diag = op(x, v, tok_emb, diagnostics=True)

    assert torch.equal(x_new, x), "x must be identical to input at initialization"
    assert torch.equal(v_new, v), "v must be identical to input at initialization"
    assert diag['delta_kinetic_energy'] == 0.0
    assert diag['delta_potential_energy'] == 0.0
    assert diag['layer0_diag']['max_abs_s'] == 0.0
    assert diag['layer0_diag']['max_abs_t'] == 0.0
    assert diag['layer1_diag']['max_abs_s'] == 0.0
    assert diag['layer1_diag']['max_abs_t'] == 0.0


def test_write_operator_invertibility():
    """Verify exact numerical inverse: inverse(forward(x, v)) == (x, v)."""
    dim = 4
    hidden_dim = 128
    n_particles = 64
    op = WriteOperator(dim=dim, hidden_dim=hidden_dim).double()

    # Randomly initialize weights away from zero to test non-identity transformation
    for p in op.parameters():
        p.data.normal_(0.0, 0.1)

    x = torch.randn(n_particles, dim, dtype=torch.float64)
    v = torch.randn(n_particles, dim, dtype=torch.float64)
    tok_emb = torch.randn(hidden_dim, dtype=torch.float64)

    # Forward
    x_new, v_new, _ = op(x, v, tok_emb, inverse=False)
    # Ensure transformation is nontrivial
    assert not torch.allclose(x_new, x, atol=1e-5)
    assert not torch.allclose(v_new, v, atol=1e-5)

    # Inverse
    x_rec, v_rec, _ = op(x_new, v_new, tok_emb, inverse=True)

    # Must recover exact inputs within FP64 precision
    diff_x = (x_rec - x).abs().max().item()
    diff_v = (v_rec - v).abs().max().item()
    assert diff_x < 1e-12, f"Inverse x reconstruction error: {diff_x:.2e}"
    assert diff_v < 1e-12, f"Inverse v reconstruction error: {diff_v:.2e}"


def test_write_operator_permutation_equivariance():
    """Verify that permuting particles commutes with the write operator."""
    dim = 4
    hidden_dim = 128
    n_particles = 64
    op = WriteOperator(dim=dim, hidden_dim=hidden_dim)

    # Non-zero weights
    for p in op.parameters():
        p.data.normal_(0.0, 0.1)

    x = torch.randn(n_particles, dim)
    v = torch.randn(n_particles, dim)
    tok_emb = torch.randn(hidden_dim)

    perm = torch.randperm(n_particles)
    x_perm = x[perm]
    v_perm = v[perm]

    # Forward on permuted
    x_out_perm, v_out_perm, _ = op(x_perm, v_perm, tok_emb)

    # Forward on original, then permute
    x_out, v_out, _ = op(x, v, tok_emb)
    x_out_then_perm = x_out[perm]
    v_out_then_perm = v_out[perm]

    assert torch.allclose(x_out_perm, x_out_then_perm, atol=1e-6)
    assert torch.allclose(v_out_perm, v_out_then_perm, atol=1e-6)


def test_write_operator_gradient_flow():
    """Verify that gradients flow cleanly to all write operator parameters."""
    dim = 4
    hidden_dim = 128
    n_particles = 64
    op = WriteOperator(dim=dim, hidden_dim=hidden_dim)

    x = torch.randn(n_particles, dim, requires_grad=True)
    v = torch.randn(n_particles, dim, requires_grad=True)
    tok_emb = torch.randn(hidden_dim, requires_grad=True)

    x_new, v_new, _ = op(x, v, tok_emb)
    # Scalar loss on transformed states
    loss = x_new.square().sum() + v_new.square().sum()
    loss.backward()

    # Verify input gradients
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert v.grad is not None and torch.isfinite(v.grad).all()
    assert tok_emb.grad is not None and torch.isfinite(tok_emb.grad).all()

    # Verify write operator parameters receive finite gradients
    for name, p in op.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Parameter {name} has None grad"
            assert torch.isfinite(p.grad).all(), f"Parameter {name} has non-finite grad"


def test_write_operator_invalid_inputs():
    """Verify validation on dimension and shapes."""
    op = WriteOperator(dim=4, hidden_dim=128)
    with pytest.raises(ValueError, match="2D tensors"):
        op(torch.randn(4), torch.randn(4), torch.randn(128))
    with pytest.raises(ValueError, match="shapes must match"):
        op(torch.randn(10, 4), torch.randn(12, 4), torch.randn(128))
    with pytest.raises(ValueError, match="token_embedding must have shape"):
        op(torch.randn(10, 4), torch.randn(10, 4), torch.randn(64))
