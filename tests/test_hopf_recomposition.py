"""Numerical/interface acceptance for the third-generation recomposition layer."""

import pytest
import torch

from information_boltzmann.core.hopf_recomposition import (
    PairedBranchGenerator,
    RootedTree,
    WaveRecomposer,
    conjugate_generator,
    coproduct_counter,
    graft,
    iterated_coproduct_counters,
)


def test_ck_coproduct_is_coassociative_on_a_branched_tree():
    leaf_a = RootedTree("a")
    leaf_b = RootedTree("b")
    tree = graft((leaf_a, graft((leaf_b,), "branch")), "root")
    left, right = iterated_coproduct_counters(tree)
    assert left == right
    terms = coproduct_counter(tree)
    # Unit terms and at least one proper cut are all present.
    assert (((), (tree,)) in terms and ((tree,), ()) in terms
            and len(terms) > 2)


def test_split_merge_preserves_full_packed_wave_and_residual():
    torch.manual_seed(17)
    recomposer = WaveRecomposer()
    state = torch.randn(2, 3, 2, 4 * 5, dtype=torch.float64)
    split = recomposer.split(state, 0.37)
    merged = recomposer.merge(split.left, split.right, 0.37)
    torch.testing.assert_close(merged.main, state, atol=1e-13, rtol=1e-13)
    torch.testing.assert_close(merged.residual, torch.zeros_like(state),
                               atol=1e-13, rtol=1e-13)

    left = torch.randn_like(state)
    right = torch.randn_like(state)
    merged = recomposer.merge(left, right, .37)
    recovered = recomposer.unmerge(merged.main, merged.residual, .37)
    torch.testing.assert_close(recovered[0], left, atol=1e-13, rtol=1e-13)
    torch.testing.assert_close(recovered[1], right, atol=1e-13, rtol=1e-13)


def test_packed_wave_split_has_conserved_energy_and_explicit_capacity_sum():
    torch.manual_seed(18)
    recomposer = WaveRecomposer()
    state = torch.randn(1, 2, 2, 2, 4 * 3, dtype=torch.float64)
    result = recomposer.split(state, .41)
    before = state.square().sum()
    after = result.left.square().sum() + result.right.square().sum()
    torch.testing.assert_close(before, after, atol=1e-13, rtol=1e-13)
    torch.testing.assert_close(result.left_capacity + result.right_capacity,
                               state.new_tensor(1.0), atol=0, rtol=0)


def test_paired_branch_generator_is_skew_and_conserves_quadratic_norm():
    torch.manual_seed(19)
    a = torch.randn(6, 6, dtype=torch.float64)
    a = a - a.T
    b = torch.randn(6, 6, dtype=torch.float64)
    b = b - b.T
    c = torch.randn(6, 6, dtype=torch.float64) * .1
    generator = PairedBranchGenerator(a, b, c).matrix()
    torch.testing.assert_close(generator + generator.T,
                               torch.zeros_like(generator), atol=1e-13, rtol=1e-13)
    x = torch.randn(4, 12, dtype=torch.float64)
    derivative = (x @ generator.T * x).sum()
    assert abs(float(derivative)) < 1e-12


def test_pure_conjugacy_preserves_skew_spectrum_and_is_a_null_control():
    torch.manual_seed(20)
    raw = torch.randn(8, 8, dtype=torch.float64)
    a = raw - raw.T
    q, _ = torch.linalg.qr(torch.randn(8, 8, dtype=torch.float64))
    transformed = conjugate_generator(a, q)
    torch.testing.assert_close(transformed + transformed.T,
                               torch.zeros_like(transformed), atol=1e-13, rtol=1e-13)
    torch.testing.assert_close(torch.linalg.eigvalsh(1j * transformed),
                               torch.linalg.eigvalsh(1j * a), atol=1e-12, rtol=1e-12)


def test_fraction_keeps_branch_resource_away_from_singular_endpoints():
    recomposer = WaveRecomposer(epsilon=.02)
    theta = torch.tensor([-100., 0., 100.])
    p = recomposer.fraction(theta)
    assert torch.all(p >= .02) and torch.all(p <= .98)


def test_local_hopf_branch_pathway_conserves_energy_and_propagates_gradients():
    from information_boltzmann.core.hopf_recomposition import LocalHopfBranchPathway, coproduct
    torch.manual_seed(21)
    pathway = LocalHopfBranchPathway(channels=8).double()
    state = torch.randn(3, 8, dtype=torch.float64)
    main, residual = pathway(state, duration=0.5)

    before_energy = state.square().sum()
    after_energy = main.square().sum() + residual.square().sum()
    torch.testing.assert_close(before_energy, after_energy, atol=1e-12, rtol=1e-12)

    loss = main.sum() + 0.1 * residual.sum()
    loss.backward()
    assert pathway.theta.grad is not None and torch.isfinite(pathway.theta.grad)
    assert pathway.weight_left.grad is not None and torch.isfinite(pathway.weight_left.grad).all()
    assert pathway.weight_right.grad is not None and torch.isfinite(pathway.weight_right.grad).all()
    assert pathway.coupling.grad is not None and torch.isfinite(pathway.coupling.grad).all()

    # Tree representation
    tree = pathway.tree_representation("branch_layer")
    assert tree.label == "branch_layer"
    assert len(tree.children) == 2
    terms = coproduct(tree)
    assert len(terms) >= 3


def test_identical_branches_yield_zero_residual_under_equal_split():
    from information_boltzmann.core.hopf_recomposition import LocalHopfBranchPathway
    torch.manual_seed(22)
    pathway = LocalHopfBranchPathway(channels=8, coupled=False).double()
    # Force identical weights and zero theta (equal split p = 0.5)
    with torch.no_grad():
        pathway.theta.zero_()
        pathway.weight_right.copy_(pathway.weight_left)
    state = torch.randn(2, 8, dtype=torch.float64)
    main, residual = pathway(state, duration=0.8)
    # When branches are identical, residual difference must vanish
    torch.testing.assert_close(residual, torch.zeros_like(residual), atol=1e-13, rtol=1e-13)
    torch.testing.assert_close(main.square().sum(), state.square().sum(), atol=1e-13, rtol=1e-13)

