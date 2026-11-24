"""Production actor in encoded coordinates, with an explicit decision view.

This reference uses the existing read, history projection, choice weights,
reuse provenance, interoception and gain units. It adds no head or policy.
The caller supplies the feeling/precision of the boundary being evaluated.
"""
from dataclasses import dataclass

import torch

from scripts.ib_local.navigation_gain_units import CRITIC_GAIN_SCALE,WOOD_GAIN_SCALE


@dataclass(frozen=True)
class ActorView:
    habit: torch.Tensor
    available: torch.Tensor
    reuse_weight: torch.Tensor
    slots: torch.Tensor
    read_weights: torch.Tensor


def encoded_view(model, latent, feeling, *, reward_valid):
    m=model.memory
    if latent.ndim!=2 or feeling.ndim!=2 or len(feeling) not in (1,len(latent)):
        raise ValueError('matched latent and interoceptive batches required')
    read=m.read_encoded(latent,model.encoder)
    history=(read.weights[...,None]*model.encoder(m.intervening)[read.slots]).sum(-2)
    decision=torch.cat((latent,read.value,history),-1)
    reusable=((m.word_count[read.slots]>0)&(m.progress[read.slots]>0)&
              reward_valid[read.slots]&m.occupied[read.slots])
    mass=(read.weights*reusable).sum(-1,keepdim=True)
    habit=model.kanli.choice(torch.cat((decision,read.value,mass,
                                      feeling.expand(len(latent),-1)),-1))
    available_reuse = mass > 0
    if getattr(model, '_expression_speech_only', False):
        available_reuse = torch.zeros_like(available_reuse)
    habit=torch.cat((habit[...,:2],habit[...,2:].masked_fill(~available_reuse,-1e4)),-1)
    return ActorView(habit,habit>-1e3,mass,read.slots,read.weights)


def evaluate(view, wood_values, *, precision, gain_units_schema=1):
    """Match CausalProgram._value -> GenerativeCycle._choice_offset.

    wood_values are 10*g, while the actual actor consumes critic units 100*g.
    They refer to the actual silence/generate/reuse endpoints at this boundary.
    """
    if wood_values.shape!=view.habit.shape or not bool(view.available.any(-1).all()):
        raise ValueError('one value for each actual available action required')
    # CausalProgram._value always multiplies utility by ten in this route.
    values=(CRITIC_GAIN_SCALE/WOOD_GAIN_SCALE)*wood_values
    if gain_units_schema!=1:
        raise ValueError('this contract requires the explicit repaired gain units')
    count=view.available.sum(-1,keepdim=True)
    center=values.masked_fill(~view.available,0).sum(-1,keepdim=True)/count
    offset=(precision*(values-center)).masked_fill(~view.available,0)
    logits=view.habit+offset
    probability=logits.softmax(-1)
    return dict(logits=logits,offset=offset,probability=probability,
                expected=(probability*wood_values).sum(-1))


def candidate_policy_decomposition(weights,gains,options,probability,actual_values):
    """Separate remembered physical endpoints from their three option labels."""
    if not (weights.ndim==gains.ndim==options.ndim==1 and
            len(weights)==len(gains)==len(options)):
        raise ValueError('matched finite remembered candidates required')
    mass=probability.new_zeros(3)
    # Three tiny deterministic reductions avoid unordered CUDA scatter sums.
    for option in range(3):
        mass[option]=weights[options==option].sum()
    candidate=(weights*(gains-actual_values[options])).sum()
    policy=((mass-probability)*actual_values).sum()
    recall=(weights*gains).sum()
    actual=(probability*actual_values).sum()
    variance=(probability*(actual_values-actual).square()).sum()
    return dict(option_mass=mass,candidate_term=candidate,policy_term=policy,
                recall_value=recall,actual_policy_value=actual,
                reconstruction_error=(recall-actual-candidate-policy).abs(),
                action_sampling_variance=variance)
