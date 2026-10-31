"""One normalized policy measure for sensing, waiting and vocabulary actions.

Values must come from one pre-observation event belief, in nats, optionally plus
an explicitly sourced log-preference. This module supplies normalization, not a
source of preferences, a perception likelihood, or an autonomous-life claim.
The caller's sensory measure may be quadrature weights over continuous gains;
it must not silently replace them with equal mass on an arbitrary number of bins.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class JointEventPolicy:
    log_wait: torch.Tensor       # [E], joint mass of sensing choice and waiting
    log_words: torch.Tensor      # [E,V], joint mass of sensing and this exact word
    log_normalizer: torch.Tensor
    timing_bias: torch.Tensor    # [E], exact word marginal minus waiting value
    word_value: torch.Tensor     # [E,V], ADD after the native decoder temperature

    @property
    def sense_probability(self):
        return torch.logaddexp(self.log_wait, torch.logsumexp(self.log_words, -1)).exp()

    @property
    def speech_probability(self):
        return self.log_words.exp().sum()

    def sample(self, generator=None):
        """Sample factorized joint mass; None is waiting, never a fake token."""
        sense = int(torch.multinomial(self.sense_probability.detach(), 1, generator=generator))
        log_spoken = torch.logsumexp(self.log_words[sense], -1)
        log_total = torch.logaddexp(self.log_wait[sense], log_spoken)
        branch = torch.stack((self.log_wait[sense], log_spoken)).sub(log_total).exp()
        spoken = bool(torch.multinomial(branch.detach(), 1, generator=generator))
        if not spoken:
            return sense, None
        word = int(torch.multinomial((self.log_words[sense] - log_spoken).detach().exp(),
                                    1, generator=generator))
        return sense, word


def joint_event_policy(sense_log_mass, motor_logits, word_logits, wait_value, word_value):
    """Tilt the existing native policy by values from a common event likelihood.

    sense_log_mass [E] includes quadrature/measure mass; motor_logits [E]
    are native speech-vs-wait logits; word_logits [E,V] are the current native
    decoder logits AFTER its temperature scaling. Values [E] / [E,V] are
    detached Bayesian inference results. No word-count bonus, speech target,
    independent epistemic reward or extra inverse-temperature is introduced.

    Constant values leave the prior invariant. Duplicating vocabulary entries
    with their prior mass split leaves the total speech mass invariant. These
    are essential: treating V words and one wait as V+1 equiprobable actions
    creates a spurious log(V) incentive to speak.
    """
    if not torch.is_tensor(word_logits) or word_logits.ndim != 2:
        raise ValueError('word_logits must have shape [senses,vocabulary]')
    e, v = word_logits.shape
    if e < 1 or v < 1:
        raise ValueError('The policy needs a nonempty sensory measure and vocabulary')
    like = word_logits
    sense_log_mass = torch.as_tensor(sense_log_mass, device=like.device, dtype=like.dtype)
    motor_logits = torch.as_tensor(motor_logits, device=like.device, dtype=like.dtype)
    wait_value = torch.as_tensor(wait_value, device=like.device, dtype=like.dtype).detach()
    word_value = torch.as_tensor(word_value, device=like.device, dtype=like.dtype).detach()
    if (sense_log_mass.shape != (e,) or motor_logits.shape != (e,)
            or wait_value.shape != (e,) or word_value.shape != (e, v)):
        raise ValueError('Policy branches must share sensory and vocabulary dimensions')
    for name, tensor in (('motor logits', motor_logits), ('wait value', wait_value),
                         ('word value', word_value)):
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f'Non-finite {name}')
    for name, logits in (('sensory measure', sense_log_mass), ('word logits', word_logits)):
        if bool((torch.isnan(logits) | torch.isposinf(logits)).any()):
            raise ValueError(f'Invalid {name}')
    if not bool(torch.isfinite(sense_log_mass).any()) or not bool(torch.isfinite(word_logits).any(-1).all()):
        raise ValueError('Every native branch requires nonzero prior mass')
    log_sense = F.log_softmax(sense_log_mass, -1)
    log_word = F.log_softmax(word_logits, -1)
    word_partition = torch.logsumexp(log_word + word_value, -1)
    wait = log_sense + F.logsigmoid(-motor_logits) + wait_value
    words = log_sense[:, None] + F.logsigmoid(motor_logits)[:, None] + log_word + word_value
    normalizer = torch.logsumexp(torch.cat((wait, words.flatten())), -1)
    return JointEventPolicy(wait - normalizer, words - normalizer, normalizer,
                            word_partition - wait_value, word_value)
