"""Probability/gradient contracts, not evidence of spontaneous exploration."""
import torch
import pytest

from information_boltzmann.core.fly_generative_policy import joint_event_policy


def make(values=False):
    torch.manual_seed(4)
    s = torch.randn(3, dtype=torch.float64)
    h = torch.randn(3, dtype=torch.float64)
    words = torch.randn(3, 7, dtype=torch.float64)
    wait = torch.randn(3, dtype=torch.float64) if values else torch.zeros(3, dtype=torch.float64)
    u = torch.randn(3, 7, dtype=torch.float64) if values else torch.zeros_like(words)
    return s, h, words, wait, u


def test_one_partition_normalizes_sense_timing_and_word_marginals():
    s, h, words, wait, u = make(True)
    p = joint_event_policy(s, h, words, wait, u)
    reference = torch.cat((s.softmax(0) * (-h).sigmoid() * wait.exp(),
                           (s.softmax(0)[:, None] * h.sigmoid()[:, None]
                            * words.softmax(-1) * u.exp()).flatten()))
    reference /= reference.sum()
    torch.testing.assert_close(torch.cat((p.log_wait.exp(), p.log_words.exp().flatten())), reference)
    torch.testing.assert_close(p.sense_probability.sum(), s.new_tensor(1.))
    conditional_speech = p.log_words.exp().sum(-1) / p.sense_probability
    torch.testing.assert_close(conditional_speech, (h + p.timing_bias).sigmoid())
    torch.testing.assert_close(p.log_words.softmax(-1), (words + u).softmax(-1))


def test_no_information_or_constant_information_preserves_native_joint_prior():
    s, h, words, wait, u = make()
    p = joint_event_policy(s, h, words, wait + 21., u + 21.)
    torch.testing.assert_close(p.sense_probability, s.softmax(0))
    torch.testing.assert_close(p.log_words.exp().sum(-1), s.softmax(0) * h.sigmoid())
    torch.testing.assert_close(p.timing_bias, torch.zeros_like(h), atol=1e-14, rtol=0)


def test_vocabulary_refinement_does_not_invent_speech_preference():
    s, h, words, wait, u = make(True)
    original = joint_event_policy(s, h, words, wait, u)
    # Each original word is represented twice, sharing its native prior mass.
    refined = joint_event_policy(s, h, words.repeat_interleave(2, -1) - torch.log(s.new_tensor(2.)),
                                 wait, u.repeat_interleave(2, -1))
    torch.testing.assert_close(refined.speech_probability, original.speech_probability)
    torch.testing.assert_close(refined.sense_probability, original.sense_probability)


def test_posterior_values_are_fixed_while_native_policy_is_differentiable():
    s, h, words, wait, u = [t.requires_grad_() for t in make(True)]
    p = joint_event_policy(s, h, words, wait, u)
    (-p.log_words[1, 2]).backward()
    for t in (s, h, words):
        assert t.grad is not None and torch.isfinite(t.grad).all() and t.grad.abs().sum() > 0
    assert wait.grad is None and u.grad is None


def test_masked_native_word_mass_remains_zero():
    s, h, words, wait, u = make(True)
    words[:, 2] = -torch.inf
    p = joint_event_policy(s, h, words, wait, u)
    assert p.log_words[:, 2].isneginf().all()
    words[0] = -torch.inf
    with pytest.raises(ValueError, match='nonzero prior'):
        joint_event_policy(s, h, words, wait, u)
