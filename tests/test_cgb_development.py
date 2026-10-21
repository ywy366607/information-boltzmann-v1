"""Spatial/material interfaces; no synthetic capability experiment."""
import json
import numpy as np

from information_boltzmann.cgb.development import DevelopmentParams, SpatialDevelopment, encounter, geometry_features
from information_boltzmann.cgb.structural_budget import MaterialParams, StructuralBudget


def test_arbor_growth_pays_material_and_stays_in_finite_cone_slots():
    code = np.random.default_rng(4).normal(size=(5, 4))
    m = SpatialDevelopment(code, np.arange(5), DevelopmentParams(initial_material=.03))
    for _ in range(20):
        m.step()
    s = m.segments()
    length = np.linalg.norm(s['end']-s['start'], axis=1).sum()
    np.testing.assert_allclose(length*m.params.material_per_um, m.budget.reserved.sum(), atol=1e-12)
    np.testing.assert_allclose(m.budget.balance(np.zeros(m.n)), 0, atol=1e-12)
    assert np.bincount(m.cone_cell).max() <= m.params.cone_slots_per_cell
    assert np.all(m.budget.free >= 0)


def test_spatial_resume_is_bitwise_including_branching_and_budget():
    code = np.random.default_rng(7).normal(size=(8, 4))
    m = SpatialDevelopment(code, np.arange(8))
    for _ in range(11):
        m.step()
    other = SpatialDevelopment.from_state_dict(m.state_dict())
    for _ in range(19):
        m.step()
        other.step()
    for k,v in m.segments().items():
        np.testing.assert_array_equal(v, other.segments()[k])
    for k in m.budget._fields:
        np.testing.assert_array_equal(getattr(m.budget,k), getattr(other.budget,k))
    assert m.rng.bit_generator.state == other.rng.bit_generator.state


def test_development_json_archive_replays_the_same_future_geometry():
    code = np.random.default_rng(7).normal(size=(8, 4))
    m = SpatialDevelopment(code,np.arange(8))
    for _ in range(10):
        m.step()
    disk = json.loads(json.dumps(m.state_dict(),default=lambda v:v.tolist()))
    restored = SpatialDevelopment.from_state_dict(disk)
    for _ in range(10):
        m.step()
        restored.step()
    for k,v in m.segments().items():
        np.testing.assert_array_equal(v,restored.segments()[k])
    assert m.rng.bit_generator.state == restored.rng.bit_generator.state


def test_encounter_requires_shared_voxel_and_respects_direction():
    segments = dict(start=np.array([[.1,.1,.1],[.1,.1,.1],[50.,50.,50.]]),
                    end=np.array([[.9,.1,.1],[.9,.1,.1],[50.9,50.,50.]]),
                    cell=np.array([0,1,2]), side=np.array([0,1,1]))
    overlap = encounter(segments,3,2).toarray()
    np.testing.assert_allclose(overlap[0,1], .64)
    assert overlap[1,0] == 0 and overlap[0,2] == 0
    features = geometry_features([overlap,overlap],np.arange(3),np.array([0]),np.array([1]),[.001,.001])
    np.testing.assert_allclose(features[0,:4], [np.log(.64),0,np.log(.001),1])


def test_reserved_arbor_material_survives_legacy_archive_and_pays_maintenance():
    b = StructuralBudget(1,[0],np.zeros(1),MaterialParams(maintenance=.1),1.)
    b.reserve(np.array([.5]))
    b.advance(np.zeros(1),1)
    np.testing.assert_allclose(b.spent,[.05])
    np.testing.assert_allclose(b.balance(np.zeros(1)),0,atol=1e-12)
    state=b.state_dict()
    del state['reserved']
    legacy=StructuralBudget.from_state_dict(state)
    np.testing.assert_array_equal(legacy.reserved,[0])
