"""V4 stage 2: wood and metal assist water-fire (Claude, 2026-11-04).

User, 11-04: tuning water-fire already tunes metal-wood; the rise from water
to fire IS wood. When the topic is exhausted (the partner only repeats), the
body is sitting: no new information enters, and water should rise by itself
into a motive and an act. Stage 1 failed exactly there (K4: silence was as
likely in an exhausted topic as in a fresh one).

This is the other half of active inference, added to the stage-1 body:
* Interoception: the body's own last surprise (reply NLL), last learning
  (pre-post on the probe) and last latent prediction error are given to it as
  part of its own state, standardized by its own running statistics. These
  quantities already existed in the loop; they were never given to the body.
* Imagination (wood): for silence, generate and reuse, the outcome model F
  predicts the landing; a critic G values each imagined landing (with the
  interoception) as expected usable progress.
* Recall (water rising into wood, amendment after gate_stage2_1 failed I3):
  for each option, the progress that the same kind of act brought from
  similar states, read from the experience table, is part of the critic input.
* Calibration (metal): G learns from the realized usable progress of the
  option actually taken. Its precision is 1/RMS of its own recent errors.
* Choice: softmax(habit logits + precision * (G - mean G)). The habit (the
  stage-1 REINFORCE head) is kept; imagination adds to it in proportion to
  how well it has been predicting. Nothing schedules or sets a speech rate.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from scripts.ib_local.language_action import peek
from scripts.ib_local.navigation_settlement import complete_action_labels, settlement_code
from scripts.ib_local.navigation_value_scale import critic_output
from scripts.ib_local.language_navigation_kanli_v4 import GENERATE, REUSE, SILENCE, KanLiEngine

INTERO = 3
SCALE = 100.  # progress in centi-nats for the critic


class ImaginationHeads(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.choice = nn.Linear(4*latent_dim+1+INTERO, 3)
        self.outcome = nn.Sequential(nn.Linear(2*latent_dim, latent_dim), nn.SiLU(),
                                     nn.Linear(latent_dim, latent_dim))
        self.generate_intent = nn.Parameter(torch.zeros(latent_dim))
        self.critic = nn.Sequential(nn.Linear(2*latent_dim+INTERO+3+2, latent_dim), nn.SiLU(),
                                    nn.Linear(latent_dim, 1))


def attach_imagination(model, seed=29):
    from scripts.ib_local.language_navigation_kanli_v4 import attach
    attach(model, seed=seed)
    latent = model.encoder.out_features
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed+1)
        model.kanli = ImaginationHeads(latent).to(model.memory.source.device)
    # Which option each experience was (silence / generate / reuse); -1 unknown.
    model.memory.register_buffer('option', torch.full((model.memory.capacity,), -1, dtype=torch.long,
                                                      device=model.memory.source.device))
    # What the body felt when it decided (standardized interoception).
    model.memory.register_buffer('feeling', torch.zeros(model.memory.capacity, INTERO,
                                                        device=model.memory.source.device))
    return model


class Running:
    def __init__(self, n, decay=.99):
        self.mean, self.var, self.count, self.decay = np.zeros(n), np.ones(n), 0, decay

    def update(self, x):
        x = np.asarray(x, float)
        if self.count == 0:
            self.mean = x.copy()
        else:
            d = x-self.mean
            self.mean = self.mean + (1-self.decay)*d
            self.var = self.decay*self.var + (1-self.decay)*d*d
        self.count += 1

    def standardize(self, x):
        return (np.asarray(x, float)-self.mean)/np.sqrt(self.var+1e-8)

    def state_dict(self):
        return dict(mean=self.mean.tolist(), var=self.var.tolist(), count=self.count)


class ImaginingEngine(KanLiEngine):
    def __init__(self, *args, warmup=32, felt_recall=True, **kw):
        super().__init__(*args, **kw)
        self.felt_recall = felt_recall
        self.intero_stats = Running(INTERO)
        self.intero_raw = np.zeros(INTERO)
        self.critic_error = None
        self.critic_count = 0
        self.warmup = warmup

    def reward_label_valid(self, slots=None, *, wood=False):
        """Objective provenance only; never discards the underlying experience."""
        name = 'wood_settlement_versions' if wood else 'settlement_versions'
        versions = getattr(self, name, None)
        if versions is None:
            mask = torch.ones_like(self.model.memory.occupied)
        else:
            mask = versions == settlement_code(getattr(self, 'settlement_kind', 'witness'))
        if not wood and getattr(self, 'reward_memory_kind', 'td') == 'observed':
            memory = self.model.memory
            # The same mask gates both conditional gain recall and reusable
            # words. Mixed-action content remains available to memory.read.
            mask = mask & memory.progress_valid & complete_action_labels(memory)
        return mask if slots is None else mask[slots]

    def intero(self):
        q = self.intero_stats.standardize(self.intero_raw) if self.intero_stats.count > 1 else np.zeros(INTERO)
        return torch.tensor(np.clip(q, -5, 5), dtype=torch.float32, device=self.device)[None]

    def precision(self):
        if self.critic_count < self.warmup or self.critic_error is None:
            return 0.
        return 1/np.sqrt(self.critic_error+1e-4)

    def plan(self, h):
        plan = self.model.plan_from_state(h)
        memory = self.model.memory
        reusable = ((memory.word_count[plan['slots']] > 0) & (memory.progress[plan['slots']] > 0) &
                    self.reward_label_valid(plan['slots']) &
                    memory.occupied[plan['slots']])
        reuse_weight = (plan['weights']*reusable).sum(-1, keepdim=True)
        q = self.intero().expand(len(h), -1)
        logits = self.model.kanli.choice(torch.cat((plan['decision_feature'], plan['action'], reuse_weight, q), -1))
        logits = torch.cat((logits[..., :2], logits[..., 2:].masked_fill(reuse_weight <= 0, -1e4)), -1)
        return logits, plan, reusable

    @torch.no_grad()
    def recall_outcomes(self, z, k=8, temperature=.2):
        """Water rising: what each kind of act brought from similar states. [3, 2] = value, support."""
        memory, enc = self.model.memory, self.model.encoder
        keys = F.normalize(enc(memory.source), dim=-1)
        similarity = keys @ F.normalize(z, dim=-1)
        if self.felt_recall:
            # State-dependent recall (user 11-04: when we remember, we also
            # recall how it felt): content similarity minus felt distance,
            # both on the same unit scale (declared choice, not tuned).
            felt = self.intero()[0]
            similarity = similarity - (memory.feeling-felt).norm(dim=-1)/INTERO**.5
        rows = []
        for option in range(3):
            valid = memory.occupied & (memory.option == option) & self.reward_label_valid()
            count = int(valid.sum())
            if not count:
                rows.append(torch.zeros(2, device=z.device)); continue
            score, slot = similarity.masked_fill(~valid, -1e4).topk(min(k, count))
            w = (score/temperature).softmax(-1)
            rows.append(torch.stack(((w*memory.progress[slot]).sum()*SCALE, torch.tensor(min(k, count)/k, device=z.device))))
        return torch.stack(rows)

    @torch.no_grad()
    def imagine(self, h, plan, reusable):
        """Imagined landing and value of each option; returns values [3] and critic inputs [3, n]."""
        heads, enc, memory = self.model.kanli, self.model.encoder, self.model.memory
        z = enc(h)[0]
        weights = plan['weights'][0]*reusable[0]
        if float(weights.sum()) > 0:
            slot = int(plan['slots'][0, int(weights.argmax())])
            reuse_act = enc(memory.action[slot])
        else:
            reuse_act = z
        actions = torch.stack((z, z+heads.generate_intent, reuse_act))
        landing = heads.outcome(torch.cat((z.expand(3, -1), actions), -1))
        onehot = torch.eye(3, device=z.device)
        recalled = self.recall_outcomes(z)
        inputs = torch.cat((z.expand(3, -1), landing, self.intero().expand(3, -1), onehot, recalled), -1)
        # The recalled outcome is the estimate itself; the critic learns only the
        # correction (amendment 2 after gate_stage2_2: a critic trained one sample
        # per cycle did not separate the options within 256 cycles).
        return critic_output(self, inputs).squeeze(-1)+recalled[:, 0], inputs

    @torch.no_grad()
    def _utter(self):
        self.decision.start(self.recurrent)
        p = peek(self.table, self.pending)
        _, h = self.decision.step(self.pending, p)
        habit, plan, reusable = self.plan(h[None])
        values, inputs = self.imagine(h[None], plan, reusable)
        available = habit[0] > -1e3
        centered = values-values[available].mean()
        offset = (self.precision()*centered).masked_fill(~available, 0.)
        logits = habit[0]+offset
        record = dict(feeling=self.intero()[0].clone(), choice_logits=logits.tolist(), habit_logits=habit[0].tolist(), offset=offset.tolist(),
                      values=values.tolist(), critic_inputs=inputs, stop_logits=[], byte_logp=[], probs=[],
                      truncated=False, choice=SILENCE, reused_slot=None)
        self._pending_record = record
        if self.mode == 'silence':
            return b'', record
        choice = int(torch.multinomial(logits.softmax(-1), 1))
        record['choice'] = choice
        if choice == SILENCE:
            return b'', record
        self.decision.start(self.recurrent)
        if choice == GENERATE:
            return self._speak_bytes(None, record, True), record
        k = int(torch.multinomial(plan['weights'][0]*reusable[0]/(plan['weights'][0]*reusable[0]).sum(), 1))
        slot = int(plan['slots'][0, k])
        record['reused_slot'] = slot
        memory = self.model.memory
        words = memory.words[slot, :int(memory.word_count[slot])].cpu().numpy()
        return self._speak_bytes(words, record, False), record

    def _policy_logp(self, h, logits, targets, own, record):
        habit, _, _ = self.plan(h[:1])
        offset = torch.tensor(record['offset'], device=habit.device)
        combined = habit[0]+offset
        logp_choice = combined.log_softmax(-1)
        terms = [logp_choice[record['choice']]]
        replay = dict(choice=combined.detach().tolist(), stops=[])
        if record['choice'] == GENERATE and own:
            decisions = self.decision_logits(h[:own+1])
            n = len(record['stop_logits'])
            stopped = n == own
            for k in range(1, n+1):
                logit = decisions[k, 1]
                terms.append(F.logsigmoid(logit) if stopped and k == n else F.logsigmoid(-logit))
            replay['stops'] = [float(decisions[k, 1].detach()) for k in range(1, n+1)]
            byte_logp = logits[0, :own].log_softmax(-1).gather(1, targets[0, :own, None]).flatten()
            terms.extend(byte_logp)
            replay['byte_logp'] = byte_logp.detach()
        return torch.stack([t.reshape(()) for t in terms]).mean(), replay

    def step(self, directions, age_ablate=False):
        """Stage-1 cycle, then critic calibration and the interoception update."""
        return self._step_with_critic(directions, age_ablate)

    def _step_with_critic(self, directions, age_ablate):
        # The critic is trained in its own small step after the stage-1 cycle,
        # on the realized outcome of the option actually taken.
        result = KanLiEngine.step(self, directions, age_ablate)
        record = self._pending_record
        self.model.memory.option[result['episode_slot']] = record['choice']
        self.model.memory.feeling[result['episode_slot']] = record['feeling']
        target = SCALE*result['usable_signal']
        if self.mode != 'silence':
            critic = self.model.kanli.critic
            inputs = record['critic_inputs'][record['choice']].detach()
            prediction = critic(inputs[None]).squeeze()+inputs[-2]
            loss = F.mse_loss(prediction, torch.tensor(target, device=self.device))
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.)
            self.optimizer.step()
            error = (float(record['values'][record['choice']])-target)**2
            self.critic_error = error if self.critic_error is None else .98*self.critic_error+.02*error
            self.critic_count += 1
        # Interoception for the next decision: what this body just felt.
        # Last reply surprise, last learning on the probe, last latent prediction error.
        self.intero_raw = np.array([result['response_nll'], SCALE*result['progress'], result['short_loss']])
        self.intero_stats.update(self.intero_raw)
        probs = torch.tensor(record['choice_logits']).softmax(-1).tolist()
        result.update(choice_probability=probs, habit_logits=record['habit_logits'], imagination_offset=record['offset'],
                      imagined_value=record['values'], target_value=target, precision=self.precision(),
                      critic_rms=None if self.critic_error is None else float(np.sqrt(self.critic_error)),
                      interoception=self.intero_raw.tolist())
        return result
