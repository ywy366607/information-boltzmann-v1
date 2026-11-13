import numpy as np

from scripts.ib_local.address_spectrum_diagnostic import covariance_model, transform_keys


def test_lw_formula_matches_explicit_outer_product_variance():
    x=np.random.default_rng(11).normal(size=(17,8))
    model=covariance_model(x)
    centered=x-x.mean(0)
    c=centered.T@centered/len(x)
    mu=np.trace(c)/x.shape[1]
    beta=sum(np.square(np.outer(row,row)-c).sum() for row in centered)/len(x)**2
    delta=np.square(c-mu*np.eye(x.shape[1])).sum()
    np.testing.assert_allclose(model['strength'],min(beta/delta,1),rtol=1e-12)
    assert np.linalg.eigvalsh(model['cleaned']).min()>0


def test_affine_origin_removal_and_single_query_shapes():
    x=np.random.default_rng(7).normal(size=(12,6))
    for method in ('centered_cosine','lw'):
        k,q,_=transform_keys(x,x[:2],method)
        k2,q2,_=transform_keys(x+3,x[:2]+3,method)
        np.testing.assert_allclose(k,k2,atol=1e-12)
        np.testing.assert_allclose(q,q2,atol=1e-12)
        assert k.shape==(12,6) and q.shape==(2,6)


def test_constant_features_stay_finite_without_inventing_directions():
    x=np.ones((8,16))
    k,q,_=transform_keys(x,x[:1],'lw')
    assert np.isfinite(k).all() and not k.any() and not q.any()
