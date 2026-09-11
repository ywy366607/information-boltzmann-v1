"""Tests for T03: Exact score function gradient audit via branch enumeration."""
import math
import pytest
import torch

from fine_grain.information_boltzmann.collision import CollisionKernel
from scripts.ib_local.batched import collision_batched
from scripts.ib_local.reference import collision_serial
from scripts.ib_local.types import CandidateTable, FrozenContext


@pytest.fixture(autouse=True)
def set_deterministic():
    torch.set_num_threads(1)
    torch.manual_seed(11)


def test_score_gradient_1_event_branch_enumeration():
    """1-event branch enumeration: verify true expected gradient == surrogate gradient."""
    dim, hidden = 2, 8
    width, max_rate = 2.0, 5.0
    kernel = CollisionKernel(dim, hidden, width=width, max_rate=max_rate).double()

    x = torch.zeros(2, dim, dtype=torch.float64, requires_grad=True)
    v = torch.tensor([[1.0, 0.5], [-1.0, -0.5]], dtype=torch.float64, requires_grad=True)
    ctx_x = torch.randn(4, dim, dtype=torch.float64, requires_grad=True)
    ctx_v = torch.randn(4, dim, dtype=torch.float64, requires_grad=True)
    ctx = FrozenContext(ctx_x, ctx_v)

    normal = torch.tensor([[0.6, 0.8]], dtype=torch.float64)

    # 1 event has 2 branches:
    # Branch 0 (reject): uniform=0.999999 (prob < 1 -> reject)
    # Branch 1 (accept): uniform=0.0 (prob > 0 -> accept)
    t_rej = CandidateTable(torch.tensor([0]), torch.tensor([1]), normal, torch.tensor([0.999999], dtype=torch.float64))
    t_acc = CandidateTable(torch.tensor([0]), torch.tensor([1]), normal, torch.tensor([0.0], dtype=torch.float64))

    res_rej = collision_serial(x, v, ctx, t_rej, kernel, mode="strict_local_v1", width=width, max_rate=max_rate)
    res_acc = collision_serial(x, v, ctx, t_acc, kernel, mode="strict_local_v1", width=width, max_rate=max_rate)

    assert not res_rej.accepted[0]
    assert res_acc.accepted[0]

    # Exact branch probabilities: P(A=1) = exp(log_p), P(A=0) = 1 - P(A=1)
    p1 = res_acc.log_prob.exp()
    p0 = 1.0 - p1
    assert torch.allclose(p0 + p1, torch.tensor(1.0, dtype=torch.float64), atol=1e-12)

    # Differentiable endpoint loss on particle 0 (individual particle velocity changes upon collision)
    target_v = torch.tensor([2.0, -1.0], dtype=torch.float64)
    loss_rej = (res_rej.v[0] - target_v).square().sum()
    loss_acc = (res_acc.v[0] - target_v).square().sum()

    # Ground truth expected loss: L_bar = P(0)*L(0) + P(1)*L(1)
    L_bar = p0 * loss_rej + p1 * loss_acc

    # Surrogate expected loss: S_bar = sum_B P(B).detach() * [ L(B) + stopgrad(L(B) - b) * log_prob(B) ]
    b = 1.5
    S_acc = loss_acc + (loss_acc.detach() - b) * res_acc.log_prob
    S_rej = loss_rej + (loss_rej.detach() - b) * res_rej.log_prob
    S_bar = p0.detach() * S_rej + p1.detach() * S_acc

    params = list(kernel.parameters())
    targets = [x, v, ctx_x, ctx_v] + params

    grads_true = torch.autograd.grad(L_bar, targets, retain_graph=True)
    grads_surr = torch.autograd.grad(S_bar, targets, retain_graph=True)

    # 1. Verify exact equality between analytical expected gradient and surrogate gradient
    for name, gt, gs in zip(["x", "v", "ctx_x", "ctx_v"] + [f"p{i}" for i in range(len(params))], grads_true, grads_surr):
        diff = (gt - gs).abs().max().item()
        assert diff < 1e-10, f"1-event gradient mismatch for {name}: {diff:.2e}"

    # 2. Verify baseline invariance: changing constant b does not alter surrogate gradient
    for b_alt in [0.0, -10.0, 42.0]:
        S_acc_alt = loss_acc + (loss_acc.detach() - b_alt) * res_acc.log_prob
        S_rej_alt = loss_rej + (loss_rej.detach() - b_alt) * res_rej.log_prob
        S_bar_alt = p0.detach() * S_rej_alt + p1.detach() * S_acc_alt
        grads_alt = torch.autograd.grad(S_bar_alt, targets, retain_graph=True)
        for gt, ga in zip(grads_true, grads_alt):
            assert torch.allclose(gt, ga, atol=1e-10)

    # 3. Verify pure pathwise gradient fails on kernel parameters (exposes missing score term)
    S_pathwise = p0.detach() * loss_rej + p1.detach() * loss_acc
    grads_pathwise = torch.autograd.grad(S_pathwise, targets, allow_unused=True)

    # Kernel parameter gradient for pathwise is None / omitted (missing probability derivative)
    kernel_grad_norm_true = sum(g.abs().sum().item() for g in grads_true[4:])
    assert kernel_grad_norm_true > 1e-3, "True kernel gradient should be non-zero"
    for g_path in grads_pathwise[4:]:
        assert g_path is None or g_path.abs().sum().item() == 0.0, (
            "Pathwise kernel gradient must be None or zero because kinematics do not depend on rate parameters"
        )


def test_score_gradient_2_events_branch_enumeration_with_state_dependency():
    """2-event branch enumeration: verify state-dependent probability gradients and batched operator."""
    dim, hidden = 2, 8
    width, max_rate = 2.0, 5.0
    kernel = CollisionKernel(dim, hidden, width=width, max_rate=max_rate).double()

    # 3 particles with chain dependency: (0, 1) and (1, 2) share particle 1
    n = 3
    x = torch.zeros(n, dim, dtype=torch.float64, requires_grad=True)
    v = torch.tensor([[1.0, 0.5], [-0.8, -0.4], [0.2, 0.9]], dtype=torch.float64, requires_grad=True)
    ctx_x = torch.randn(4, dim, dtype=torch.float64, requires_grad=True)
    ctx_v = torch.randn(4, dim, dtype=torch.float64, requires_grad=True)
    ctx = FrozenContext(ctx_x, ctx_v)

    normal = torch.tensor([[1.0, 0.0], [0.6, 0.8]], dtype=torch.float64)
    ci = torch.tensor([0, 1], dtype=torch.int64)
    cj = torch.tensor([1, 2], dtype=torch.int64)

    # 4 branches: '00', '01', '10', '11'
    u_map = {
        '00': torch.tensor([0.999999, 0.999999], dtype=torch.float64),
        '01': torch.tensor([0.999999, 0.0], dtype=torch.float64),
        '10': torch.tensor([0.0, 0.999999], dtype=torch.float64),
        '11': torch.tensor([0.0, 0.0], dtype=torch.float64),
    }

    branch_results = {}
    for code, u in u_map.items():
        table = CandidateTable(ci, cj, normal, u)
        res = collision_batched(x, v, ctx, table, kernel, mode="strict_local_v1", width=width, max_rate=max_rate)
        acc_code = ''.join(['1' if a else '0' for a in res.accepted.tolist()])
        assert acc_code == code, f"Branch {code} got acceptance {acc_code}"
        branch_results[code] = res

    # Branch probabilities sum to 1
    probs = {code: res.log_prob.exp() for code, res in branch_results.items()}
    assert abs(sum(probs.values()).item() - 1.0) < 1e-12

    # Verify Event 1 acceptance probability differs between branch '01' and '11'
    # (proves state dependency on whether particle 1 collided in Event 0)
    p_event1_given_0_rej = branch_results['01'].log_prob - branch_results['00'].log_prob  # log p1^{(0)}
    p_event1_given_0_acc = branch_results['11'].log_prob - branch_results['10'].log_prob  # log p1^{(1)}
    assert not torch.allclose(p_event1_given_0_rej, p_event1_given_0_acc, atol=1e-5), (
        "Event 1 probability must depend on Event 0 state update"
    )

    # Differentiable endpoint loss on individual particles (not total kinetic energy invariant)
    target_v = torch.tensor([[1.5, -0.5], [-0.5, 1.2], [0.0, 0.0]], dtype=torch.float64)
    losses = {code: (res.v - target_v).square().sum() for code, res in branch_results.items()}
    L_bar = sum(probs[code] * losses[code] for code in branch_results)

    b = 2.5
    S_bar = sum(
        probs[code].detach() * (losses[code] + (losses[code].detach() - b) * branch_results[code].log_prob)
        for code in branch_results
    )

    params = list(kernel.parameters())
    targets = [x, v, ctx_x, ctx_v] + params

    grads_true = torch.autograd.grad(L_bar, targets, retain_graph=True)
    grads_surr = torch.autograd.grad(S_bar, targets, retain_graph=True)

    # Verify exact match
    for name, gt, gs in zip(["x", "v", "ctx_x", "ctx_v"] + [f"p{i}" for i in range(len(params))], grads_true, grads_surr):
        diff = (gt - gs).abs().max().item()
        assert diff < 1e-10, f"2-event gradient mismatch for {name}: {diff:.2e}"

    # Verify baseline invariance
    for b_alt in [0.0, -5.0, 10.0]:
        S_bar_alt = sum(
            probs[code].detach() * (losses[code] + (losses[code].detach() - b_alt) * branch_results[code].log_prob)
            for code in branch_results
        )
        grads_alt = torch.autograd.grad(S_bar_alt, targets, retain_graph=True)
        for gt, ga in zip(grads_true, grads_alt):
            assert torch.allclose(gt, ga, atol=1e-10)

    # Verify pathwise gradient omits the score term on kernel parameters
    S_pathwise = sum(probs[code].detach() * losses[code] for code in branch_results)
    grads_pathwise = torch.autograd.grad(S_pathwise, targets, allow_unused=True)

    kernel_grad_norm_true = sum(g.abs().sum().item() for g in grads_true[4:])
    assert kernel_grad_norm_true > 1e-3, "True kernel gradient should be non-zero"
    for g_path in grads_pathwise[4:]:
        assert g_path is None or g_path.abs().sum().item() == 0.0, (
            "Pathwise kernel gradient must be None or zero because kinematics do not depend on rate parameters"
        )


def test_score_gradient_finite_difference_cross_check():
    """Auxiliary verification: analytical expected gradient matches numerical finite difference."""
    dim, hidden = 2, 8
    width, max_rate = 2.0, 5.0
    kernel = CollisionKernel(dim, hidden, width=width, max_rate=max_rate).double()

    x = torch.zeros(2, dim, dtype=torch.float64)
    v = torch.tensor([[1.0, 0.5], [-1.0, -0.5]], dtype=torch.float64)
    ctx = FrozenContext(torch.randn(4, dim, dtype=torch.float64), torch.randn(4, dim, dtype=torch.float64))
    normal = torch.tensor([[0.6, 0.8]], dtype=torch.float64)

    t_rej = CandidateTable(torch.tensor([0]), torch.tensor([1]), normal, torch.tensor([0.999999], dtype=torch.float64))
    t_acc = CandidateTable(torch.tensor([0]), torch.tensor([1]), normal, torch.tensor([0.0], dtype=torch.float64))

    def evaluate_expected_loss(bias_val):
        with torch.no_grad():
            old_bias = kernel.output.bias.item()
            kernel.output.bias.fill_(bias_val)

        res_rej = collision_serial(x, v, ctx, t_rej, kernel, mode="strict_local_v1", width=width, max_rate=max_rate)
        res_acc = collision_serial(x, v, ctx, t_acc, kernel, mode="strict_local_v1", width=width, max_rate=max_rate)

        p1 = res_acc.log_prob.exp()
        p0 = 1.0 - p1
        loss = p0 * res_rej.v.square().sum() + p1 * res_acc.v.square().sum()

        with torch.no_grad():
            kernel.output.bias.fill_(old_bias)
        return loss

    # Autograd gradient
    res_rej0 = collision_serial(x, v, ctx, t_rej, kernel, mode="strict_local_v1", width=width, max_rate=max_rate)
    res_acc0 = collision_serial(x, v, ctx, t_acc, kernel, mode="strict_local_v1", width=width, max_rate=max_rate)
    p1 = res_acc0.log_prob.exp()
    p0 = 1.0 - p1
    L0 = p0 * res_rej0.v.square().sum() + p1 * res_acc0.v.square().sum()
    grad_autograd = torch.autograd.grad(L0, kernel.output.bias)[0].item()

    # Central finite difference
    eps = 1e-6
    b0 = kernel.output.bias.item()
    loss_pos = evaluate_expected_loss(b0 + eps).item()
    loss_neg = evaluate_expected_loss(b0 - eps).item()
    grad_fd = (loss_pos - loss_neg) / (2 * eps)

    # Must match within finite difference discretization error
    assert abs(grad_autograd - grad_fd) < 1e-6, f"Autograd {grad_autograd} vs FD {grad_fd}"


def test_score_gradient_legacy_eps_branch_enumeration():
    """Branch enumeration under legacy_eps mode: verify exact gradient equivalence."""
    dim, hidden = 2, 8
    width, max_rate = 2.0, 5.0
    kernel = CollisionKernel(dim, hidden, width=width, max_rate=max_rate).double()

    x = torch.zeros(2, dim, dtype=torch.float64, requires_grad=True)
    v = torch.tensor([[1.0, 0.5], [-1.0, -0.5]], dtype=torch.float64, requires_grad=True)
    ctx_x = torch.randn(4, dim, dtype=torch.float64, requires_grad=True)
    ctx_v = torch.randn(4, dim, dtype=torch.float64, requires_grad=True)
    ctx = FrozenContext(ctx_x, ctx_v)
    normal = torch.tensor([[0.6, 0.8]], dtype=torch.float64)

    t_rej = CandidateTable(torch.tensor([0]), torch.tensor([1]), normal, torch.tensor([0.999999], dtype=torch.float64))
    t_acc = CandidateTable(torch.tensor([0]), torch.tensor([1]), normal, torch.tensor([0.0], dtype=torch.float64))

    res_rej = collision_serial(x, v, ctx, t_rej, kernel, mode="legacy_eps", width=width, max_rate=max_rate)
    res_acc = collision_serial(x, v, ctx, t_acc, kernel, mode="legacy_eps", width=width, max_rate=max_rate)

    p1 = res_acc.log_prob.exp()
    p0 = 1.0 - p1

    target_v = torch.tensor([2.0, -1.0], dtype=torch.float64)
    loss_rej = (res_rej.v[0] - target_v).square().sum()
    loss_acc = (res_acc.v[0] - target_v).square().sum()

    L_bar = p0 * loss_rej + p1 * loss_acc

    b = 3.0
    S_acc = loss_acc + (loss_acc.detach() - b) * res_acc.log_prob
    S_rej = loss_rej + (loss_rej.detach() - b) * res_rej.log_prob
    S_bar = p0.detach() * S_rej + p1.detach() * S_acc

    params = list(kernel.parameters())
    targets = [x, v, ctx_x, ctx_v] + params

    grads_true = torch.autograd.grad(L_bar, targets, retain_graph=True)
    grads_surr = torch.autograd.grad(S_bar, targets)

    for gt, gs in zip(grads_true, grads_surr):
        diff = (gt - gs).abs().max().item()
        assert diff < 1e-10
