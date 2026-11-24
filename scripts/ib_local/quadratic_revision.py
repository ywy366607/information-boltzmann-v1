"""Two structured solvers for the same conditional quadratic revision law.

    E(z) = 1/2 ||z-prior||^2_P + 1/2 ||observation-H z||^2_R

The diagonal, identity-observation case gives a sigmoid gain when the neural
gate produces log(new_precision / old_precision). The relation mechanism uses
the full design geometry and a dual ridge solve. These share a variational
rule, not a scalar gain or a claim that imagined proposals are observations.
No parameters, persistent evidence, or additional optimization steps live here.
"""
from __future__ import annotations

import torch


def diagonal_gain(log_precision_ratio: torch.Tensor) -> torch.Tensor:
    """Stable new/(old+new) gain without exponentiating neural logits.

    The co-evolving workspace generates this relative precision. It is a
    learned weighting, not an independently calibrated likelihood precision.
    """
    return log_precision_ratio.sigmoid()


_EYE_CACHE: dict[tuple, torch.Tensor] = {}


def _eye(n: int, device, dtype) -> torch.Tensor:
    key = (n, device, dtype)
    mat = _EYE_CACHE.get(key)
    if mat is None:
        mat = torch.eye(n, device=device, dtype=dtype)
        if len(_EYE_CACHE) < 64:
            _EYE_CACHE[key] = mat
    return mat


def evidence_correction(design, residual, query, ridge, *, return_factor=False, return_gram=False):
    """Full relation correction in evidence coordinates, preserving old ops.

    Minimize ||design @ B - residual||² + ridge*||B||², then read query @ B.
    The kernel, coefficient and projected query also support the original
    predictive variance calculation. Evidence admission remains with U.
    """
    gram = design @ design.T
    kernel = gram + ridge * _eye(len(design), design.device, design.dtype)
    if return_factor:
        # One factor for the mean, query uncertainty AND subsequent operator
        # hypotheses. It belongs to this exact evidence graph, never a cache
        # across optimizer/evidence revisions.
        factor = torch.linalg.lu_factor(kernel)
        solved = torch.linalg.lu_solve(*factor, torch.cat((residual, design@query.T), -1))
        coefficient, projected = solved.split((residual.shape[-1], len(query)), -1)
        mean = (query@design.T)@coefficient
        if return_gram:
            return kernel, coefficient, mean, projected, factor, gram
        return kernel, coefficient, mean, projected, factor
    coefficient = torch.linalg.solve(kernel, residual)
    mean = (query @ design.T) @ coefficient
    projected = torch.linalg.solve(kernel, design @ query.T)
    if return_gram:
        return kernel, coefficient, mean, projected, gram
    return kernel, coefficient, mean, projected
