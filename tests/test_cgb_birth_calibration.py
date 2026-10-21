"""Birth-program numerical contracts, not a capability study."""
import json
import numpy as np
import pytest
from scipy.special import expit

from information_boltzmann.cgb.morphology import symmetric_log, symmetric_exp, SpatialIdentityProgram
from information_boltzmann.cgb.occurrence import OccurrenceProgram
from information_boltzmann.cgb.development import SpatialDevelopment, encounter


def test_log_covariance_roundtrip_preserves_spd():
    a = np.random.default_rng(8).normal(size=(7, 3, 3))
    covariance = a@a.transpose(0, 2, 1)+np.eye(3)*.1
    np.testing.assert_allclose(symmetric_exp(symmetric_log(covariance)), covariance, atol=1e-12)


def test_bilateral_spatial_program_reflects_position_and_covariance():
    code = np.zeros((1, 14)); code[0, :6] = [30, 50, 60, 5, 6, 7]
    covariance = np.array([[[4., 1., 0], [1., 3., .5], [0., .5, 2.]]])
    code[:, 7:13] = symmetric_log(covariance)
    p = SpatialIdentityProgram(dict(spatial_code=code, known=np.array([True]), midline=np.array(100)))
    v = p.describe([0, 0], ['L', 'R'])
    np.testing.assert_allclose(v['soma_mean'][:, 0], [130, 70])
    np.testing.assert_allclose(v['centroid_offset'][:, 0], [5, -5])
    np.testing.assert_allclose(v['covariance'][1, 0, 1], -1)
    with pytest.raises(ValueError): p.describe([0], ['unknown'])
    with pytest.raises(IndexError): p.describe([-1], ['L'])


def test_occurrence_is_directed_and_bounded_workspace_matches_formula():
    values = dict(code=np.eye(2), bias=np.array(-2.), geometry=np.array([.1, .3, .2, .4]),
                  out=np.array([1., 0.]), **{'in': np.array([0., 2.])},
                  affinity=np.array([[0., 3.], [-1., 0.]]), floors=np.array([.01, .02]),
                  feature_mean=np.zeros(4), feature_scale=np.ones(4))
    p = OccurrenceProgram(values)
    a, b = np.array([0, 1, 0]), np.array([1, 0, 1])
    e4, e2 = np.array([2., 2., 0.]), np.array([1., 1., 0.])
    raw = p.raw_features(e4, e2)
    expected = expit(-2+raw@values['geometry']+values['out'][a]+values['in'][b]+values['affinity'][a, b])
    np.testing.assert_allclose(p.probability(a, b, e4, e2, block_size=1), expected)
    assert expected[0] > expected[1]
    with pytest.raises(ValueError): p.raw_features([-1.], [1.])
    with pytest.raises(ValueError): p.probability(a, b, e4, e2, block_size=0)
    with pytest.raises(IndexError): p.probability([-1], [0], [1.], [1.])


def test_spatial_plan_resumes_through_json_and_pays_finite_length():
    n = 4
    plan = dict(soma_mean=np.zeros((n, 3)), centroid_offset=np.ones((n, 3))*4,
                covariance=np.broadcast_to(np.eye(3)*9, (n, 3, 3)), length=np.full(n, 20.),
                branch_density=np.full(n, .1))
    m = SpatialDevelopment(np.eye(n), np.arange(n), spatial_plan=plan)
    for _ in range(3): m.step()
    clone = SpatialDevelopment.from_state_dict(json.loads(json.dumps(m.state_dict(), default=lambda v: v.tolist())))
    for _ in range(12): m.step(); clone.step()
    for k, v in m.segments().items(): np.testing.assert_array_equal(v, clone.segments()[k])
    np.testing.assert_allclose(m.budget.reserved/m.params.material_per_um, plan['length'], atol=1e-10)
    np.testing.assert_allclose(m.budget.balance(np.zeros(n)), 0, atol=1e-12)
    overlap = encounter(m.segments(), n, 2, polarized=False).toarray()
    np.testing.assert_allclose(overlap, overlap.T)
    assert np.all(overlap >= 0)
