"""The displayed attention is the actual read map and does not change learning."""
import torch

from information_boltzmann.core.torus3d import CBIMTorus3D


def test_head_attention_monitoring_preserves_read_feature_and_gradients():
    torch.manual_seed(121)
    model = CBIMTorus3D(vocab_size=19, shape=(4, 4, 4), velocities=8,
                        content_dim=2, write_type="w4_predictive_agent",
                        readout_type="belief_agent")
    field = torch.randn(1, 4, 4, 4, 16, requires_grad=True)
    precision = torch.ones(1, 16)
    bare, _ = model.readout(field, precision, return_diag=False)
    monitored, diagnostics = model.readout(field, precision, return_diag=True)
    attention = diagnostics["read_attention_weights"]
    assert attention.shape == (1, 4, 4, 64)
    assert not attention.requires_grad
    torch.testing.assert_close(attention.sum(-1), torch.ones(1, 4, 4))
    torch.testing.assert_close(bare, monitored, rtol=0, atol=0)
    bare_grad = torch.autograd.grad(bare.square().sum(), field, retain_graph=True)[0]
    monitored_grad = torch.autograd.grad(monitored.square().sum(), field)[0]
    torch.testing.assert_close(bare_grad, monitored_grad, rtol=0, atol=0)
