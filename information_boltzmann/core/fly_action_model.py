"""Opt-in ordered action consequences and posterior-guided live policies.

This adapter uses the existing mushroom-body context, vocabulary and motor acts.
It does not assign reward to self-hearing or turn phi into a novelty counter.
"""
import math

import torch
import torch.nn.functional as F

from .fly_consequence_belief import GaussianConsequences, NativeConsequenceCredit
from .fly_active_inference import motor_logits


SCHEMA = 'ordered-native-actions-v2'


def external_prediction_loss(losses, sources):
    """Conditioning on a known motor command removes self-samples as external labels.

    Return a vector whose ordinary mean is the external-target mean. The graph
    remains present for all-self windows; sensory input and raw NLL are untouched.
    """
    if len(sources) != losses.numel():
        raise ValueError('Prediction source attribution must match every target')
    external = losses.new_tensor([s != 'self_hearing' for s in sources])
    return losses * external * (losses.numel() / external.sum().clamp_min(1))


def cosine_projection(size, width, like):
    """Fixed orthonormal rows; width follows the existing MBON destination count."""
    if width > size:
        raise ValueError('Action projection exceeds the available embedding dimensions')
    i = torch.arange(width, device=like.device, dtype=like.dtype)[:, None]
    j = torch.arange(size, device=like.device, dtype=like.dtype)[None, :]
    p = torch.cos(math.pi * i * (j + .5) / size) * math.sqrt(2. / size)
    p[0] /= math.sqrt(2.)
    return p


class OrderedConsequenceCredit(NativeConsequenceCredit):
    """The v1 pending response drains before an explicit new feature schema starts."""

    def __init__(self, learner):
        super().__init__(learner)
        self.native_events = []
        self.timing_biases = []
        self.timing_bias = 0.
        self.word_bias = None
        self.decision_bounds = (0, 0)
        self.width = len(self.posts)
        self.retention = float(torch.sigmoid(self.model.comp_logit_trace.detach()).mean())
        self.word_table = None
        if self.state.get('action_schema') not in (None, SCHEMA):
            raise ValueError('Unknown ordered action schema; migrate explicitly')
        if self.active:
            self._word_table()
        elif self.state.get('response') is None and not self.state.get('in_child_turn', False):
            self._activate()

    @property
    def active(self):
        return self.state.get('action_schema') == SCHEMA

    def _word_table(self):
        # Frozen input coordinates avoid the old moving-decoder action dictionary.
        embedding = self.model.embedding.weight
        if embedding.requires_grad:
            raise ValueError('Ordered action coordinates require a frozen input embedding')
        key = (SCHEMA, self.width, embedding.data_ptr(), embedding._version)
        cache = getattr(self.model, '_consequence_action_coordinates', None)
        if cache is None or cache[0] != key:
            with torch.no_grad():
                p = cosine_projection(embedding.shape[1], self.width, embedding)
                table = F.normalize(F.normalize(embedding.detach(), dim=1) @ p.T, dim=1)
            cache = (key, table)
            self.model._consequence_action_coordinates = cache
        self.word_table = cache[1]

    def _activate(self):
        if self.state.get('response') is not None:
            raise RuntimeError('Never replace a posterior with an unfinished legacy response')
        self.state['legacy_v1'] = {k: self.state.get(k) for k in
                                   ('belief', 'reward_sum', 'reward_n', 'action', 'utterance', 'utterance_words')}
        self.state.update(action_schema=SCHEMA, belief={}, reward_sum=0., reward_n=0,
                          action=None, in_child_turn=False)
        self.belief = GaussianConsequences(self.state['belief'])
        self._word_table()
        self.begin_child_turn()
        self.state['in_child_turn'] = False

    def begin_child_turn(self):
        if not self.active:
            return super().begin_child_turn()
        z = self.learner.state.h.new_zeros(self.width)
        self.state['ordered_action'] = {'n': 0, 'sum': z.clone(), 'weighted': z.clone(), 'last': z.clone()}
        self.state['in_child_turn'] = True

    def action_features(self, context, said, mother_turn=False, candidate=None):
        """Affine in a candidate word; AB/BA and repetition counts are distinguishable."""
        q = self.state['ordered_action']
        n = q['n'] + int(candidate is not None)
        s, t, last = q['sum'], q['weighted'], q['last']
        if candidate is not None:
            s = s + candidate
            t = t + n * candidate
            last = candidate
        mean = s / max(1, n)
        order = (2 * t - (n + 1) * s) / (n * (n - 1)) if n > 1 else torch.zeros_like(s)
        shape = mean.shape[:-1]
        c = context.expand(*shape, context.numel())
        count = mean.new_full((*shape, 1), math.log1p(n))
        summary = torch.cat((mean, order, last.expand_as(mean), count), -1)
        # Three order-aware blocks share each native destination's context coordinate.
        joint_word = (mean + order + last) / math.sqrt(3.)
        joint = (c.reshape(*shape, 5, self.width) * joint_word.unsqueeze(-2)).flatten(-2)
        return torch.cat((mean.new_ones((*shape, 1)), c, c * float(said),
                          mean.new_full((*shape, 1), float(said)),
                          mean.new_full((*shape, 1), float(mother_turn)), summary, joint), -1)

    @torch.no_grad()
    def policy_values(self, context, mother_turn=False):
        """All vocabulary values via one small quadratic form, without vocabulary x P."""
        zero = context.new_zeros(self.width)
        b = self.action_features(context, True, mother_turn, zero).double()
        basis = torch.eye(self.width, device=context.device, dtype=context.dtype)
        a = (self.action_features(context, True, mother_turn, basis).double() - b).T
        p = self.state['belief'].get('P')
        if p is None:
            pa, pb, noise = a, b, 1.
        else:
            noise = self.state['belief']['noise_sum'] / self.state['belief']['noise_count']
            pa, pb = p @ a, p @ b
        quadratic, linear, constant = a.T @ pa, a.T @ pb, b @ pb
        words = self.word_table.double()
        variance = ((words @ quadratic) * words).sum(-1) + 2 * (words @ linear) + constant
        values = .5 * torch.log1p(variance.clamp_min(0) / max(noise, torch.finfo(words.dtype).eps))
        wait = .5 * math.log1p(self.belief.ratio(self.action_features(context, False, mother_turn)))
        return values.to(context.dtype), wait

    @torch.no_grad()
    def prepare(self, context, physical, enabled=True, mother_turn=False, temperature=1.):
        self.timing_bias, self.word_bias = 0., None
        if not self.active or not enabled:
            return
        from .fly_bptt_learning import read_fly_motor
        values, wait = self.policy_values(context, mother_turn)
        log_prior = F.log_softmax(
            self.model.decoder(self.model.read_norm(read_fly_motor(self.model, physical))).float()[0]
            / max(float(temperature), 1e-6), -1)
        # Integrate the word branch of the same posterior-tilted policy.
        # E[V] would be only a Jensen lower bound, not its log partition.
        self.timing_bias = float(torch.logsumexp(log_prior + values, dim=0) - wait)
        self.word_bias = values.detach()
        self.state['pre_action_wait_value'] = wait

    def advance_ticks(self, ticks):
        self.native_events.append({'kind': 'clock', 'ticks': int(ticks)})
        self.timing_biases.extend([self.timing_bias] * ticks)
        if self.active:
            for e in self.state['eligibility'].values():
                e.mul_(self.retention ** ticks)

    def _advance_trace(self):
        if not self.active:
            return super()._advance_trace()
        self.state['clock_words'] = self.state.get('clock_words', 0) + 1

    def choose(self, context, frames, first_commit, mother_turn=False):
        start, length = self.decision_bounds
        tail = length - first_commit - 1 if first_commit is not None else 0
        self.native_events.append({'kind': 'timing', 'start': start, 'length': length,
                                   'first': first_commit, 'tail_ticks': tail})
        self.content_tick = start + (first_commit if first_commit is not None else length - 1)
        self.content_tail = tail
        if not self.active:
            return super().choose(context, frames, first_commit, mother_turn)
        with torch.enable_grad():
            logits = motor_logits(self.model, frames.detach())[:, 1] + self.timing_bias
            score = (F.logsigmoid(-logits).sum() if first_commit is None else
                     F.logsigmoid(-logits[:first_commit]).sum() + F.logsigmoid(logits[first_commit]))
            self._add_score(score * self.retention ** tail, [k for k in self.params if k in ('ai_w', 'ai_b')])
        self.decisions.append(first_commit is not None)
        self.state['action_context'] = context.detach()
        self.state['action_mother_turn'] = mother_turn
        self.state['action'] = self.action_features(context, first_commit is not None, mother_turn).detach()

    def say(self, read, word, temperature):
        event = {'kind': 'content', 'tick': self.content_tick, 'word': int(word),
                 'temperature': float(temperature), 'tail_ticks': self.content_tail}
        if self.word_bias is not None:
            event['logit_bias'] = self.word_bias
        self.native_events.append(event)
        if not self.active:
            return super().say(read, word, temperature)
        if self.state.get('response') is not None:
            raise RuntimeError('Settle the external response before starting speech')
        with torch.enable_grad():
            logits = self.model.decoder(self.model.read_norm(read.detach())).float() / max(float(temperature), 1e-6)
            if self.word_bias is not None:
                logits = logits + self.word_bias
            self._add_score(F.log_softmax(logits, -1)[0, int(word)] * self.retention ** self.content_tail,
                            [k for k in self.params if k.startswith(('decoder.', 'read_norm.'))])
        q = self.state['ordered_action']
        with torch.no_grad():
            self.state.setdefault('pending_self_echo', []).append(int(word))
            e = self.word_table[int(word)]
            q['n'] += 1
            q['sum'].add_(e); q['weighted'].add_(e, alpha=q['n']); q['last'].copy_(e)
            self.state['action'] = self.action_features(self.state['action_context'], True,
                                                        self.state['action_mother_turn']).detach()

    def hear_token(self, token, source, exposure):
        """Verify the known action->echo identity without treating it as discovery."""
        if source != 'self_hearing' or not self.state.get('pending_self_echo'):
            return
        expected = self.state['pending_self_echo'].pop(0)
        counts = self.state.setdefault('self_echo', {'matched': 0, 'mismatched': 0, 'exposure_sum': 0.})
        counts['matched' if expected == int(token) else 'mismatched'] += 1
        counts['exposure_sum'] += float(exposure)

    def observe(self, physical, source, exposure, end=True):
        if source == 'mother_reply' and exposure > 0 and self.state.get('action') is not None \
                and self.state.get('response') is None:
            self.native_events.append({'kind': 'response_begin'})
        return super().observe(physical, source, exposure, end)

    def end_response(self):
        response = self.state.get('response')
        n = len(self.gains)
        baseline = self.state['reward_sum'] / max(1, self.state['reward_n'])
        super().end_response()
        if len(self.gains) != n:
            self.native_events.append({'kind': 'response_end', 'advantage': self.gains[-1] - baseline})
            if not self.active:
                self._activate()
        return response

    def finish(self):
        super().finish()
        self.learner.last_epistemic.update(action_schema=self.state.get('action_schema', 'legacy_draining'),
                                           native_credit_enabled=True, timing_information_bias=self.timing_bias)
        self.learner.last_epistemic['self_echo'] = dict(self.state.get('self_echo') or {})
