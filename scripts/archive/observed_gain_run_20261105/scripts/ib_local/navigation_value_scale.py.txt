"""Optional PopArt-style normalization of the existing scalar value residual.

Rewards, replay labels, recalled skip values and confidence remain in their
original units. Only critic training uses normalized residual targets. Existing
Adam moments are retained; no optimizer-trajectory invariance is claimed.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def configure_value_scale(engine, kind):
    if kind not in ('none', 'popart'):
        raise ValueError(kind)
    previous = getattr(engine, 'value_normalization', 'none')
    if previous == kind:
        return
    if previous != 'none':
        raise ValueError('changing an existing normalization needs an explicit migration')
    # Identity coordinates preserve the source checkpoint. Decay reuses the
    # existing critic-error timescale, rather than adding a tuned time constant.
    engine.value_scale = dict(mean=0., variance=1., scale=1., count=0, decay=.98)
    engine.value_normalization = kind


def critic_output(engine, inputs):
    value = engine.model.kanli.critic(inputs)
    if getattr(engine, 'value_normalization', 'none') == 'popart':
        state = engine.value_scale
        return state['scale']*value + state['mean']
    return value


@torch.no_grad()
def update_value_scale(engine, residual_target):
    state = engine.value_scale
    old_mean, old_scale = state['mean'], state['scale']
    rate = 1-state['decay']
    delta = float(residual_target)-old_mean
    mean = old_mean + rate*delta
    variance = state['decay']*(state['variance'] + rate*delta*delta)
    scale = math.sqrt(max(variance, 1e-8))
    if not all(math.isfinite(x) for x in (mean, variance, scale)):
        raise FloatingPointError('nonfinite value normalization')
    layer = engine.model.kanli.critic[-1]
    if not isinstance(layer, torch.nn.Linear) or layer.out_features != 1 or layer.bias is None:
        raise TypeError('output-preserving scalar normalization requires the existing affine head')
    layer.weight.mul_(old_scale/scale)
    layer.bias.mul_(old_scale).add_(old_mean-mean).div_(scale)
    state.update(mean=mean, variance=variance, scale=scale, count=state['count']+1)


def normalized_critic_loss(engine, inputs, target):
    """Adapt coordinates before learning; preserve raw output including recall."""
    recalled = inputs[..., -2]
    residual_target = inputs.new_tensor(target)-recalled
    with torch.no_grad():
        before = critic_output(engine, inputs).squeeze(-1)
        update_value_scale(engine, float(residual_target))
        after = critic_output(engine, inputs).squeeze(-1)
        preservation_error = float((before-after).abs().max())
    state = engine.value_scale
    normalized_target = (residual_target-state['mean'])/state['scale']
    prediction = engine.model.kanli.critic(inputs).squeeze(-1)
    loss = F.mse_loss(prediction, normalized_target)
    return loss, dict(value_scale=state['scale'], value_mean=state['mean'],
        value_scale_count=state['count'], value_normalized_loss=float(loss.detach()),
        value_rebase_max_error=preservation_error,
        value_raw_residual_error=float((after-residual_target).square().mean()))
