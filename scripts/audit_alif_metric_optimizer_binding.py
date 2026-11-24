"""CPU operator audit of Gram projection followed by the real BudgetedAdamW.

This is a numerical optimizer-contract check, with no capability experiment,
training checkpoint, world interaction, or running-experiment mutation.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from information_boltzmann.budgeted_adamw import BudgetedAdamW
from scripts.ib_local import encoder_subspace_metric as esm


def main():
    torch.set_num_threads(1)
    vectors = torch.zeros(32, 256)
    vectors[:, 0] = 3.0
    memory = SimpleNamespace(
        occupied=torch.ones(32, dtype=torch.bool),
        source=vectors, action=vectors.clone(), successor=vectors.clone(),
        long_successor=vectors.clone(), episodes=32,
    )
    engine = SimpleNamespace(model=SimpleNamespace(memory=memory))
    esm.configure(engine, damping=0.25)
    raw = torch.ones(128, 256)
    projected = esm.project_gradient(engine, raw)
    covariance, _ = esm.memory_gram_matrix(memory)

    def drift(direction):
        return float(((direction @ covariance) * direction).sum())

    def optimizer_step(gradient):
        parameter = torch.nn.Parameter(torch.zeros_like(gradient))
        optimizer = BudgetedAdamW(
            [parameter], lr=3e-4, weight_decay=0.0, allocation=True,
            cooling=True, wiener_gate=True, fused=False,
        )
        parameter.grad = gradient.clone()
        optimizer.step()
        return parameter.detach().clone(), optimizer.state[parameter]['budget_gate'].tolist()

    raw_delta, raw_gate = optimizer_step(raw)
    projected_delta, projected_gate = optimizer_step(projected)
    result = {
        'kind': 'CPU actual-operator contract check; zero world calls',
        'shape': [128, 256],
        'damping': 0.25,
        'old_covariance_eigenvalue': 9.0,
        'initial_optimizer_state': 'zero moments, real allocation/cooling/Wiener enabled',
        'protected_column_gradient_gain': float(projected[0, 0] / raw[0, 0]),
        'gradient_drift_ratio': drift(projected) / drift(raw),
        'actual_parameter_drift_ratio': drift(projected_delta) / drift(raw_delta),
        'protected_column_actual_step_gain': float(projected_delta[0, 0] / raw_delta[0, 0]),
        'raw_gate': raw_gate,
        'projected_gate': projected_gate,
        'interpretation': 'A bound on a projected gradient must be rebound to the final Adam/Wiener parameter displacement.',
    }
    output = ROOT / 'results/published/cgb_alif_metric_optimizer_binding_20261124.json'
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
