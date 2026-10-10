"""Legacy algebra control and the production physical-junction integration."""
import pytest
import torch
from information_boltzmann.core.plastic_ports import PlasticMediumPorts3D
from information_boltzmann.core.hopf_recomposition import LocalHopfBranchPathway


def test_hopf_module_identity_and_energy_conservation():
    """Verify LocalHopfBranchPathway exact quadratic wave energy conservation and identity birth."""
    channels = 24
    hopf = LocalHopfBranchPathway(channels, coupled=True)
    hopf.theta.data.zero_()
    hopf.weight_left.data.zero_()
    hopf.weight_right.data.zero_()
    hopf.coupling.data.zero_()

    x = torch.randn(2, 7, channels)
    main, residual = hopf(x)

    # 1. Identity birth
    assert torch.allclose(main, x, atol=1e-6, rtol=1e-6), "Zero-init must be exact identity transformation"
    assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-7), "Residual branch must be zero at birth"

    # 2. Quadratic energy conservation under random parameter perturbation
    hopf.weight_left.data.normal_(0, 0.1)
    hopf.weight_right.data.normal_(0, 0.1)
    hopf.coupling.data.normal_(0, 0.1)
    hopf.theta.data.fill_(0.35)

    main_perturbed, residual_perturbed = hopf(x)
    in_energy = x.square().sum(dim=-1)
    out_energy = main_perturbed.square().sum(dim=-1) + residual_perturbed.square().sum(dim=-1)
    assert torch.allclose(in_energy, out_energy, atol=1e-5, rtol=1e-5), "Hopf pathway must preserve exact quadratic energy"


def test_plastic_medium_ports_hopf_forward_and_backward():
    """Verify PlasticMediumPorts3D forward execution and gradient propagation through Hopf pathway."""
    channels = 16
    vocab_size = 100
    model = PlasticMediumPorts3D(
        vocab_size=vocab_size,
        shape=(4, 4, 4),
        channels=channels,
        hopf_recomposition=True,
        bath_type='conductance',
        write_exchange='contact_mode',
        port_scope='compact',
        anisotropic_transport=True,
        structure_options=dict(resource_density=4., speed_reference=1., structure_time=3.,
                               prior_std=.4, initial_std=.4, maintenance_supply=1.4, initial_dual=.1)
    )
    assert model.hopf_recomposition is True
    assert model.hopf_pathway is not None
    assert '-physical-junction-gen3' in model.architecture

    belief = model.initial_belief(1)
    # A read measures persistent state; only evolution can route the flux.
    belief, _ = model.assimilate(belief, torch.tensor([3]), diagnostics=False)
    belief, _ = model.advance(belief, .07, substeps=3, diagnostics=False)
    feature, info = model.read(belief, decode=False, diagnostics=True)
    assert 'hopf_branch_fraction' not in info
    assert all(q.shape == belief.medium.field.shape for q in belief.medium.flux)

    # Loss backward to hopf parameters
    loss = feature.square().sum()
    loss.backward()

    assert model.hopf_pathway.gate.weight.grad is not None
    assert torch.isfinite(model.hopf_pathway.gate.weight.grad).all()
