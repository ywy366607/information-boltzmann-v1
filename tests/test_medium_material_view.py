"""CPU display export must reconstruct real physical coefficients and units."""
import numpy as np
import torch

from information_boltzmann.core.plastic_ports import PlasticMediumPorts3D
from scripts.ib.export_medium_material_view import export_maps


def test_export_matches_model_installed_factor_and_material_times(tmp_path):
    torch.manual_seed(43)
    constructor = dict(vocab_size=13, channels=4, shape=(2, 2, 2), material_reference_shape=None,
                       hidden=4, bath_type='conductance', anisotropic_transport=True,
                       short_term_plasticity=True, activity_adaptation=True,
                       plasticity_time_reference=3., response_time_reference=2.,
                       capacitance_reference=.7,
                       structure_options=dict(resource_density=4., speed_reference=2.,
                                              structure_time=10., prior_std=.05, initial_std=.05,
                                              maintenance_supply=3., initial_dual=0.))
    model = PlasticMediumPorts3D(**constructor)
    with torch.no_grad():
        model.medium.material.coefficients.normal_(std=.03)
        model.medium.structural_posterior.mean.normal_(std=.02)
        model.medium.transport_shear.weight.normal_(std=.04)
        prepared = model.medium.prepare_evolution()
    path = tmp_path / 'last.pt'
    torch.save({'model': model.state_dict(), 'config': {'constructor': constructor},
                'step': 3, 'cursor': 97}, path)
    view = export_maps(path)
    np.testing.assert_allclose(view['installed_factor'],
                               prepared.structural_factor.reshape(-1, 3, 3), rtol=1e-5, atol=1e-6)
    c = prepared.response
    membrane = (c.capacitance / c.leak).log().mean(-1).exp().reshape(-1)
    np.testing.assert_allclose(view['fields']['membrane_time'], membrane, rtol=1e-5)
    rate = model.medium.conduction_plasticity.coefficients(prepared.material)[1]
    np.testing.assert_allclose(view['fields']['conduction_time'],
                               (-rate.log().mean(-1)).exp().reshape(-1).detach().numpy(), rtol=1e-5)
    np.testing.assert_allclose(np.asarray(view['fields']['capacity']) + view['fields']['idle'], 4.)
    assert view['fresh_training_tokens'] == 96
