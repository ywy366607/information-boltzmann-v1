"""Numerical/serialization checks; not synthetic capability experiments."""
import io

import numpy as np
import torch

from information_boltzmann.cgb.matching import TypeMatching, linear_fit


def test_selected_pair_expression_and_gradients_equal_dense_reference():
    model = TypeMatching(7, 3, 2, seed=2)
    pre, post = torch.tensor([0, 2, 5, 1]), torch.tensor([3, 0, 4, 6])
    features = torch.tensor([[.2,.7], [.1,.3], [.4,-.2], [-.5,.8]])
    with torch.no_grad():
        model.geometry_weight.copy_(torch.tensor([.4,-.3]))
        model.affinity[0, 1] = .3
    actual = model(pre, post, features)
    dense = model.code @ model.affinity @ model.code.T
    reference = model.bias + model.out_bias[pre] + model.in_bias[post] + features @ model.geometry_weight + dense[pre, post]
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    grad_actual = torch.autograd.grad(actual.square().sum(), tuple(model.parameters()), retain_graph=True)
    grad_ref = torch.autograd.grad(reference.square().sum(), tuple(model.parameters()))
    for actual, expected in zip(grad_actual, grad_ref):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_symmetric_interaction_is_reciprocal_and_archive_roundtrip():
    model = TypeMatching(6, 3, 0, symmetric=True, seed=4)
    with torch.no_grad():
        model.affinity[0, 1] = 1.7
    pre, post = torch.tensor([0, 3, 5]), torch.tensor([2, 4, 1])
    empty = torch.empty(3, 0)
    a, b = model(pre, post, empty), model(post, pre, empty)
    torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-8)
    stream = io.BytesIO()
    torch.save({'specification': model.specification(), 'state': model.state_dict()}, stream)
    stream.seek(0)
    archive = torch.load(stream, weights_only=False)
    resumed = TypeMatching(**archive['specification'])
    resumed.load_state_dict(archive['state'])
    torch.testing.assert_close(resumed(pre, post, empty), a, rtol=0, atol=0)


def test_sparse_linear_baseline_and_standardization_preserve_function():
    pre, post = np.meshgrid(np.arange(5), np.arange(5), indexing='ij')
    pre, post = pre.ravel(), post.ravel()
    features = np.stack([np.sin(pre+2*post)], axis=1)
    target = 2 + pre*.3 - post*.7 + features[:, 0]*1.2
    train = np.arange(len(pre))
    _, prediction, info = linear_fit(5, pre, post, features, target, train)
    assert info['converged']
    np.testing.assert_allclose(prediction, target, rtol=1e-9, atol=1e-9)
    standardized = (features-features.mean(0))/features.std(0)
    _, normalized_prediction, _ = linear_fit(5, pre, post, standardized, target, train)
    np.testing.assert_allclose(normalized_prediction, prediction, rtol=1e-9, atol=1e-9)


def test_initialization_preserves_external_random_state():
    original = torch.get_rng_state().clone()
    TypeMatching(8, 2, 1, seed=19)
    assert torch.equal(original, torch.get_rng_state())


def test_cpu_generator_matches_selected_model_with_blocked_real_interface(tmp_path):
    from information_boltzmann.cgb.type_generator import TypeStrengthGenerator
    model = TypeMatching(7, 3, 2, symmetric=True, seed=7)
    with torch.no_grad():
        model.geometry_weight.copy_(torch.tensor([.4, -.7]))
        model.affinity[0, 2] = .9
    mean, scale = np.array([2., -1.]), np.array([.5, 2.])
    pre, post = np.array([0, 3, 5, 1]), np.array([2, 4, 6, 0])
    raw = np.array([[1., 2.], [.5, -2.], [.1, .8], [.2, -.3]])
    arrays = {k: p.detach().numpy() for k, p in model.state_dict().items()}
    path = tmp_path/'generator.npz'
    np.savez(path, **arrays, feature_mean=mean, feature_scale=scale,
             types=np.array([f'type{i}' for i in range(7)]), symmetric=np.array(True))
    generated = TypeStrengthGenerator.load(path).log_strength(pre, post, raw, block_size=2)
    with torch.no_grad():
        reference = model(torch.tensor(pre), torch.tensor(post),
                          torch.tensor((raw-mean)/scale, dtype=torch.float32)).numpy()
    np.testing.assert_allclose(generated, reference, rtol=1e-6, atol=1e-6)


def test_rank_regularization_scaling_preserves_identical_interaction_cost():
    from information_boltzmann.cgb.matching_analysis import balanced_rank_embedding, regularization_for_rank
    rng = np.random.default_rng(5)
    code, affinity = rng.normal(size=(11,3)), rng.normal(size=(3,3))
    c1,a1 = balanced_rank_embedding(code,affinity,3)
    c2,a2 = balanced_rank_embedding(code,affinity,12)
    np.testing.assert_allclose(c1@a1@c1.T,c2@a2@c2.T,rtol=1e-12,atol=1e-12)
    p1, p2 = np.mean(c1*c1)+np.mean(a1*a1), np.mean(c2*c2)+np.mean(a2*a2)
    np.testing.assert_allclose(np.mean(c2*c2),2*np.mean(a2*a2),rtol=1e-12)
    np.testing.assert_allclose(p1,regularization_for_rank(1,3,12)*p2,rtol=1e-12)
