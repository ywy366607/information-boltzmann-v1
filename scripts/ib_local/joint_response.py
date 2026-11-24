"""One finite-evidence joint law for imagined and learned world responses.

Finite-U payment keeps the original marginal score. Only the predictive law
and its proper training score use the joint output geometry. The original
identity-coupled operator and observation draws remain common across actions.
"""
from functools import lru_cache

import torch


def enabled(model):
    return getattr(getattr(model, '_epistemic_value_engine', None),
                   'epistemic_value_schema', 0) >= 6


def distribution(d):
    """Same mean, df and coordinate marginals; restore residual directions."""
    from scripts.ib_local.successor_joint_reference import residual_scatter
    with torch.no_grad():
        scatter = residual_scatter(d['x'], d['y'], d['ridge']).to(d['mean'])
    off = scatter-torch.diag(scatter.diag())
    noise = torch.diag(d['noise'])+off/float(d['df'])
    scale = (1+d['leverage'][:, None, None])*noise
    return dict(d, scale_matrix=(scale+scale.transpose(-1, -2))*.5)


def predictive_score(model, d, target):
    from scripts.ib_local.causal_relation_program import log_score
    return log_score(distribution(d) if enabled(model) else d, target)


@lru_cache(maxsize=32)
def radial(df, count):
    """Paired deterministic chi-square strata, independent of live RNG."""
    import numpy as np
    from scipy.special import gammaincinv
    half = (count+1)//2
    p = (np.arange(half, dtype=np.float64)+.5)/half
    positive = torch.from_numpy(np.sqrt(2*gammaincinv(df/2, p)/df))
    return torch.cat((positive, positive))[:count]


def hypotheses(d, directions, observation_directions):
    from scripts.ib_local.causal_relation_program import mechanism_hypotheses
    # Remove the old per-coordinate noise transform before applying the
    # shared joint transform to BOTH mechanism and observation fluctuations.
    standardized = dict(d, mean=torch.zeros_like(d['mean']), noise=torch.ones_like(d['noise']))
    operator = mechanism_hypotheses(standardized, d['query'], directions, observation_directions)
    joint = distribution(d)
    noise = joint['scale_matrix'][0]/(1+d['leverage'][0])
    factor = torch.linalg.cholesky(noise)
    divisor = radial(float(d['df']), len(directions)).to(d['mean'])
    centers = d['mean'][:, None]+(operator@factor.T)/divisor[None, :, None]
    observation = (observation_directions[:, -1]@factor.T)/divisor[:, None]
    points = centers+observation[None]
    return centers, points, (factor, divisor)


def evidence(centers, geometry, observed):
    """Conditional hypothesis likelihood, with the same correlated noise."""
    factor, divisor = geometry
    delta = (observed[:, None]-centers)*divisor[None, :, None]
    white = torch.linalg.solve_triangular(factor, delta.transpose(-1, -2), upper=False)
    # The common determinant cancels; each hypothesis's radial scale remains.
    logits = -.5*white.square().sum(-2)+centers.shape[-1]*divisor.log()[None]
    return logits.log_softmax(-1)
