"""Display reconstruction must preserve the trained tensor's cross terms."""
import numpy as np

from scripts.ib.serve_medium_cosmos import material_tensor_view


def test_material_tensor_view_recovers_oblique_principal_axis():
    snapshot = {'speed': [[1., 2., 3.]], 'shear': [[.4, -.3, .7]]}
    report = material_tensor_view(snapshot)
    factor = np.diag([1., 2., 3.]) @ np.array([[1., 0., 0.], [.4, 1., 0.], [-.3, .7, 1.]])
    tensor = factor @ factor.T
    spectrum = np.asarray(report['material_tensor_eigenvalues'][0])
    direction = np.asarray(report['material_principal_axis'][0])
    assert spectrum.min() > 0
    np.testing.assert_allclose(tensor @ direction, spectrum[-1] * direction)
    np.testing.assert_allclose(spectrum.sum(), np.trace(tensor))
    assert np.count_nonzero(np.abs(direction) > .01) == 3
    assert 'material_principal_axis' not in snapshot


def test_actual_row_speed_uses_active_factor_instead_of_legacy_speed():
    report = material_tensor_view({'speed': [[100., 100., 100.]],
                                  'effective_transport_factor': [[[2., 0., 0.],
                                                                 [1., 2., 0.],
                                                                 [0., 0., 3.]]]})
    np.testing.assert_allclose(report['effective_row_speed'], [[2., np.sqrt(5.), 3.]])


def test_material_tensor_view_isotropic_and_empty():
    assert material_tensor_view(None) is None
    report = material_tensor_view({'speed': [[1., 1., 1.]], 'shear': None})
    np.testing.assert_allclose(report['material_tensor_eigenvalues'], [[1., 1., 1.]])


def test_legacy_display_apertures_follow_actual_learned_coordinates():
    snapshot = {'speed': [[1., 1., 1.]] * 2, 'shear': None,
                'coordinates': [[.1, .2, .3], [.2, .2, .3]],
                'read_coords': [[.1, .2, .3]]}
    config = {'read_port_radius': [.15, .15, .15]}
    first = material_tensor_view(snapshot, config)['read_footprint_by_probe']
    snapshot['read_coords'] = [[.2, .2, .3]]
    second = material_tensor_view(snapshot, config)['read_footprint_by_probe']
    np.testing.assert_allclose(first[0], second[0][::-1])
    assert first[0][0] > second[0][0]
