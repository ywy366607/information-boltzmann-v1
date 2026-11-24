"""Shadow universal-kernel regression for the planner's finite-use channel.

The existing source weights, feature map, ridge, labels and guarded prior
are reused. Only represented option intercepts are unpenalized. The small
Schur system separates those intercepts from the regularized content fit.
This module does not install a production predictor or retain observations.
"""
from __future__ import annotations

from contextlib import contextmanager

import torch


@torch.no_grad()
def fit(factor, scale, options, residual):
    """Solve A alpha + H beta = D residual, H.T alpha = 0.

    A is the original weighted Gram plus ridge, factor @ factor.T;
    D=diag(scale), H=D C, and C encodes the observed option identities.
    All solves use float64. No additional fitted hyperparameter is introduced.
    """
    factor = factor.double()
    scale = scale.to(factor)
    options = options.to(device=factor.device, dtype=torch.long)
    classes = options.unique(sorted=True)
    if len(classes) == 0:
        raise ValueError('fit needs at least one bank row')
    design = (options[:, None] == classes[None]).to(factor)
    h = scale[:, None]*design
    y = scale*residual.to(factor)
    alpha0 = torch.cholesky_solve(y[:, None], factor)[:, 0]
    w = torch.cholesky_solve(h, factor)
    schur = h.T@w
    beta = torch.linalg.solve(schur, h.T@alpha0)
    alpha = alpha0-w@beta
    a = factor@factor.T
    return dict(classes=classes, alpha=alpha, beta=beta,
                equation_error=(a@alpha+h@beta-y).abs().max(),
                orthogonality_error=(h.T@alpha).abs().max(),
                schur_condition=torch.linalg.cond(schur))


@torch.no_grad()
def evaluate(fitted, prior, options, cross):
    """Original prior + represented-option correction + content kernel."""
    alpha = fitted['alpha']
    design = (options.to(alpha.device)[:, None] == fitted['classes'][None]).to(alpha)
    return prior.to(alpha)+design@fitted['beta']+cross.to(alpha)@alpha


@torch.no_grad()
def predict_q(engine, packets):
    """Read-only candidate using the exact production fit's factor and rows."""
    from scripts.ib_local import latent_action_planner as planner
    from scripts.ib_local.native_amortized_decision import _kernel

    original, _ = planner.predict(engine, packets)
    cache = engine._latent_action_fit
    records = engine.latent_action_bank['records']
    if not records:
        return original[:, -1], dict(bank_rows=0, classes=[], beta=[]), None
    inputs = [r['packet'] for r in records]
    base = planner.prior(engine, inputs)[:, -1]
    target = torch.stack([r['target'] for r in records]).to(base)[:, -1]
    options = torch.tensor([p['option'] for p in inputs], device=base.device)
    fitted = fit(cache['factor'], cache['scale'], options, target.double()-base.double())
    query = planner.features(engine, packets)
    prior = planner.prior(engine, packets)[:, -1]
    query_options = torch.tensor([p['option'] for p in packets], device=base.device)
    cross = _kernel(query, cache['x'])*cache['scale'][None]
    got = evaluate(fitted, prior, query_options, cross).to(original)
    diagnostics = dict(bank_rows=len(records), classes=fitted['classes'].tolist(),
        beta=fitted['beta'].tolist(), equation_error=float(fitted['equation_error']),
        orthogonality_error=float(fitted['orthogonality_error']),
        schur_condition=float(fitted['schur_condition']))
    capsule = dict(factor=cache['factor'].cpu(), scale=cache['scale'].cpu(),
        options=options.cpu(), target=target.cpu(), bank_prior=base.cpu(),
        query_prior=prior.cpu(), query_options=query_options.cpu(), cross=cross.cpu(),
        original_q=original[:, -1].cpu())
    return got, diagnostics, capsule


@contextmanager
def calibrated_predictions(engine):
    """Explicit private-experiment scope; reuse one fit per native cache key.

    Only this engine's q column changes. All original teacher/settlement,
    state and endpoint prediction, novelty and actor code run normally. No
    runtime closure or fitted tensor is attached to a checkpointed object.
    """
    from scripts.ib_local import latent_action_planner as planner
    from scripts.ib_local.native_amortized_decision import _kernel

    original = planner.predict
    cached = dict(native=None, fitted=None, fits=0, queries=0)

    @torch.no_grad()
    def predict(e, packets):
        result, novelty = original(e, packets)
        if e is not engine or e._latent_action_fit['x'] is None:
            return result, novelty
        native = e._latent_action_fit
        if cached['native'] is not native:
            records = e.latent_action_bank['records']
            inputs = [r['packet'] for r in records]
            base = planner.prior(e, inputs)[:, -1]
            target = torch.stack([r['target'] for r in records]).to(base)[:, -1]
            options = torch.tensor([p['option'] for p in inputs], device=base.device)
            cached['fitted'] = fit(native['factor'], native['scale'], options,
                                   target.double()-base.double())
            cached['native'] = native
            cached['fits'] += 1
        query = planner.features(e, packets)
        cross = _kernel(query, native['x'])*native['scale'][None]
        prior = planner.prior(e, packets)[:, -1]
        options = torch.tensor([p['option'] for p in packets], device=prior.device)
        calibrated = evaluate(cached['fitted'], prior, options, cross)
        changed = result.clone()
        changed[:, -1] = calibrated.to(result)
        cached['queries'] += 1
        return changed, novelty

    planner.predict = predict
    try:
        yield cached
    finally:
        planner.predict = original
