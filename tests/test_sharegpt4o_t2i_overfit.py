from __future__ import annotations

import torch

from scripts.train_sharegpt4o_t2i_overfit import (
    finite_t2i_metrics,
    natural_gate,
    pairwise_mse,
)


def test_pairwise_t2i_metrics_reject_prompt_independent_average():
    target = torch.stack([
        torch.zeros(3, 4, 4),
        torch.ones(3, 4, 4),
    ])
    average = torch.full_like(target, 0.5)
    metrics = finite_t2i_metrics(average, target, average.flip(0))
    assert metrics["retrieval_top1"] < 2
    assert metrics["prompt_output_rms"] == 0.0
    assert not natural_gate(metrics)


def test_pairwise_t2i_metrics_accept_identified_high_fidelity_outputs():
    target = torch.stack([
        torch.zeros(3, 4, 4),
        torch.ones(3, 4, 4),
    ])
    pred = target * 0.99 + 0.005
    shuffled = pred.flip(0)
    energy = pairwise_mse(pred, target)
    assert energy.argmin(dim=1).tolist() == [0, 1]
    metrics = finite_t2i_metrics(pred, target, shuffled)
    assert metrics["retrieval_top1"] == 2
    assert metrics["shuffle_causal_gap"] > 0.9
    assert natural_gate(metrics)
