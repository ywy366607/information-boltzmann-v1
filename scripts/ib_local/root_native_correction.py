"""Spend already computed native residuals at their actual root action ports.

The fitted predictor stays frozen. Only an evaluated successor contribution is
replaced, using its original quadrature mass and the existing gain conversion.
World labels and fitted-bank updates still wait for real settlement.
"""
from __future__ import annotations

import torch


def mode(engine):
    return getattr(engine, 'native_amortization', {}).get('root_native_correction')


def correct(engine, values, checks):
    """Return critic-unit root values; record the same patch in wood units."""
    selected_mode = mode(engine)
    if selected_mode is None:
        return values
    roots = engine._competing_action_roots
    if not roots or values is None:
        raise RuntimeError('native residual replacement requires actual root endpoints')
    delta = torch.zeros_like(values)
    contributions = []
    for row in checks:
        root = row['root_key']
        if root not in roots:
            raise RuntimeError('native residual belongs to a different decision scope')
        mass = row['mass']
        if not 0 <= mass <= 1.000001:
            raise ValueError('original successor probability mass required')
        correction = mass * row['error']
        for option in roots[root]:
            delta[option] += correction
        contributions.append(dict(key=row['key'], root=root,
            options=roots[root], mass=mass, predicted=row['predicted'],
            native=row['native'], future_correction=correction))
    # _relation_value returns 100*U; contextual_future is expressed as 10*U.
    critic_delta = 10 * engine.gamma * delta
    applied = critic_delta if selected_mode == 'apply' else torch.zeros_like(delta)
    result = values + applied
    receipt = dict(schema=1, mode=selected_mode,
        before_critic=values.tolist(), after_critic=result.tolist(),
        future_correction=delta.tolist(), applied_critic_delta=applied.tolist(),
        contributions=contributions, teacher_count=len(checks),
        bank_revision=engine.native_value_bank['revision'],
        semantics='replace evaluated original successor contributions before actual actor choice')
    engine._root_native_correction_last = receipt
    forecast = engine.epistemic_value_last
    forecast['root_native_correction'] = receipt
    if selected_mode == 'apply':
        forecast['uncorrected_contextual_future'] = forecast['contextual_future']
        forecast['uncorrected_contextual_total'] = forecast['contextual_total']
        forecast['contextual_future'] = (values.new_tensor(forecast['contextual_future']) + delta).tolist()
        forecast['contextual_total'] = (result / 10).tolist()
    return result
