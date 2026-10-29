"""Deterministic CLO6 regression checks, without training/capability tasks."""
import importlib
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
V6 = importlib.import_module("run_optimizer_continual_v6")


@pytest.mark.parametrize("name", ["adam", "muon", "soap", "adam_decay"])
def test_baseline_dispatch_after_runner_installs_factory(monkeypatch, name):
    monkeypatch.setattr(V6.V3, "make", V6.make6)
    opt = V6.make6(name, 0.01, (2, 2), 10)
    assert isinstance(opt, (V6.V3.Plain, V6.V3.Decay))


def test_column_prior_follows_diagonal_covariance_transport():
    opt = V6.CLO6((2, 2))
    opt.tau2 = torch.tensor([[1.0, 9.0]])
    covariance = torch.diag(opt.tau2[0])
    old_q = opt.Q.clone()
    q = torch.tensor([[1.0, -1.0], [1.0, 1.0]]) / 2**0.5
    opt._rebase(q)
    transform = old_q.T @ opt.Q
    expected = torch.diagonal(transform.T @ covariance @ transform)[None]
    torch.testing.assert_close(opt.tau2, expected)
    torch.testing.assert_close(opt.tau2.sum(), torch.trace(covariance))


@pytest.mark.parametrize("prior", [None, torch.tensor([[-1.0, 9.0]])])
def test_identity_rebase_preserves_uninitialized_or_signed_prior(prior):
    opt = V6.CLO6((2, 2))
    opt.tau2 = None if prior is None else prior.clone()
    opt._rebase(torch.eye(2))
    if prior is None:
        assert opt.tau2 is None
    else:
        torch.testing.assert_close(opt.tau2, prior, rtol=0, atol=0)


def test_initial_covariance_bias_correction_recovers_constant_input_factor():
    opt = V6.CLO6((2, 2))
    opt.t = 1
    opt.Rx = 0.05 * torch.diag(torch.tensor([2.0, 8.0]))
    h = opt._curvature(torch.zeros(2, 2), torch.zeros(2, 2), False)
    torch.testing.assert_close(h, torch.tensor([[2.0, 8.0], [2.0, 8.0]]))


def test_age_risk_keeps_positive_process_variance_with_zero_drift():
    opt = V6.CLO6((1, 2))
    opt.J.fill_(2.0)
    opt.age = torch.tensor([[0.0, 1000.0]])
    q = opt._drift(torch.ones(1, 2), 1.0)
    assert 0 < q[0, 1] < q[0, 0]
    # Positive prediction variance can restore a nonzero Kalman gain even at P=0.
    gain = q / (q + 1)
    assert bool(((gain > 0) & (gain < 1)).all())
