"""Compact type matching, fitted offline; this is the developmental outer loop.

The living medium's local learning rules are separate. Codes describe a shared
directed (or symmetric) interaction, not a table of individually fitted edges.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsmr
import torch
from torch import nn


class TypeMatching(nn.Module):
    def __init__(self, n_types, rank, n_features, *, symmetric=False, seed=0, device='cpu'):
        super().__init__()
        self.n_types, self.rank, self.n_features = n_types, rank, n_features
        self.symmetric, self.seed = symmetric, seed
        with torch.random.fork_rng(devices=[] if device == 'cpu' else [torch.device(device).index or 0]):
            torch.manual_seed(seed)
            self.code = nn.Parameter(.1 * torch.randn(n_types, rank, device=device))
        self.affinity = nn.Parameter(torch.eye(rank, device=device))
        self.out_bias = nn.Parameter(torch.zeros(n_types, device=device))
        self.in_bias = nn.Parameter(torch.zeros(n_types, device=device))
        self.bias = nn.Parameter(torch.zeros((), device=device))
        self.geometry_weight = nn.Parameter(torch.zeros(n_features, device=device))

    def forward(self, pre, post, features):
        result = self.bias + self.out_bias[pre] + self.in_bias[post] + features @ self.geometry_weight
        if self.rank:
            affinity = (self.affinity + self.affinity.T) * .5 if self.symmetric else self.affinity
            # GEMM is faster and smaller than materialising E x rank gathers.
            interaction = self.code @ affinity @ self.code.T
            result = result + interaction[pre, post]
        return result

    def penalty(self):
        return self.code.square().mean() + self.affinity.square().mean() if self.rank else self.bias * 0

    @torch.no_grad()
    def initialize_linear(self, coefficients):
        n = self.n_types
        for parameter, value in ((self.bias, coefficients[0]),
                (self.out_bias, coefficients[1:1+n]),
                (self.in_bias, coefficients[1+n:1+2*n]),
                (self.geometry_weight, coefficients[1+2*n:])):
            parameter.copy_(torch.as_tensor(value, dtype=parameter.dtype, device=parameter.device))

    def specification(self):
        return {'n_types': self.n_types, 'rank': self.rank, 'n_features': self.n_features,
                'symmetric': self.symmetric, 'seed': self.seed}


def linear_fit(n_types, pre, post, features, target, train):
    """Converged sparse least squares for degree terms plus optional geometry."""
    pre, post, features, target = (np.asarray(x) for x in (pre, post, features, target))
    a, b, f, y = pre[train], post[train], features[train], target[train]
    m = len(a)
    degree = sparse.coo_matrix((np.ones(3*m),
        (np.tile(np.arange(m), 3), np.concatenate((np.zeros(m, dtype=int), 1+a, 1+n_types+b)))),
        shape=(m, 1+2*n_types)).tocsr()
    design = sparse.hstack((degree, sparse.csr_matrix(f)), format='csr')
    solution = lsmr(design, y, atol=1e-11, btol=1e-11, maxiter=10000)
    coefficients = solution[0]
    prediction = (coefficients[0] + coefficients[1+pre] + coefficients[1+n_types+post]
                  + features @ coefficients[1+2*n_types:])
    return coefficients, prediction, {
        'stop_code': int(solution[1]), 'iterations': int(solution[2]),
        'residual_norm': float(solution[3]), 'normal_residual_norm': float(solution[4]),
        'condition_estimate': float(solution[6]), 'converged': int(solution[1]) in (1, 2, 4, 5)}


@dataclass
class PairBatch:
    pre: torch.Tensor
    post: torch.Tensor
    features: torch.Tensor
    target: torch.Tensor

    def selected(self, indices):
        return PairBatch(*(value[indices] for value in (self.pre, self.post, self.features, self.target)))


def objective(model, batch, regularization):
    mse = (model(batch.pre, batch.post, batch.features) - batch.target).square().mean()
    return mse + regularization * model.penalty(), mse


def optimize(model, train, val, regularization, *, steps=4000, polish=200, log_every=200):
    """Solve one training objective; validation only records model selection data."""
    curve = []
    optimizer = torch.optim.Adam(model.parameters(), lr=.02)
    for step in range(steps):
        lr = .0005 + .0195 * .5 * (1 + math.cos(math.pi * step / max(steps-1, 1)))
        optimizer.param_groups[0]['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        loss, mse = objective(model, train, regularization)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite type-matching training objective')
        loss.backward()
        optimizer.step()
        if step % log_every == 0 or step == steps-1:
            with torch.no_grad():
                current, train_mse = objective(model, train, regularization)
                val_mse = objective(model, val, 0)[1]
            curve.append({'stage': 'adam', 'update': step+1, 'lr': lr,
                'objective': float(current), 'train_mse': float(train_mse), 'val_mse': float(val_mse)})
    closures = 0
    before_polish = curve[-1]['objective'] if curve else float(objective(model, train, regularization)[0])
    if polish:
        lbfgs = torch.optim.LBFGS(model.parameters(), lr=1, max_iter=polish,
            max_eval=polish*2, tolerance_grad=1e-7, tolerance_change=1e-10,
            history_size=10, line_search_fn='strong_wolfe')

        def closure():
            nonlocal closures
            lbfgs.zero_grad(set_to_none=True)
            loss, mse = objective(model, train, regularization)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite matching refinement')
            loss.backward()
            closures += 1
            curve.append({'stage': 'lbfgs_closure', 'closure': closures,
                          'objective': float(loss.detach()), 'train_mse': float(mse.detach())})
            return loss

        lbfgs.step(closure)
        iterations = int(lbfgs.state[next(iter(model.parameters()))].get('n_iter', 0))
    else:
        iterations = 0
    model.zero_grad(set_to_none=True)
    final, train_mse = objective(model, train, regularization)
    final.backward()
    grad_norm = math.sqrt(sum(float(p.grad.square().sum()) for p in model.parameters() if p.grad is not None))
    with torch.no_grad():
        val_mse = objective(model, val, 0)[1]
    result = {'train_mse': float(train_mse.detach()), 'val_mse': float(val_mse),
        'objective': float(final.detach()), 'gradient_l2': grad_norm,
        'adam_updates': steps, 'lbfgs_iterations': iterations, 'lbfgs_closures': closures,
        'polish_objective_improvement': before_polish-float(final.detach()),
        'curve': curve}
    model.zero_grad(set_to_none=True)
    return result
