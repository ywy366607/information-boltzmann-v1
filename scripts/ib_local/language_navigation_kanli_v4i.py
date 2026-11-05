"""V4 stage 7: wood is the body's own film (Claude, 2026-11-04).

Hetu reading (user, 11-04): heaven = the world, earth = the organism, 5 = the
contact between them. 天三生木，地八成之: wood is born of the world's law
(I do X, then Y follows), and the body completes it. The symmetric
action-effect probe found where that law sits in the reply world (接话).
* It is in the body's own byte prediction. After its own words, the first 4
  reply bytes are predicted about 1.9 nats/byte better than after other words.
* It is not in the 64-d latent of a single state. F was looking in the wrong
  space.
The user agreed (11-04): wood's imagination is done by the body itself.

Each cycle, before choosing:
* Inner speech: the body drafts its utterance (32 bytes, sampled exactly as
  speaking would), and samples the stored words it would reuse.
* Films: for silence, the draft and the reused words, the body plays on by
  itself what the world would say next, max_utterance (32) bytes, and records
  its mean predictive entropy (expected surprise, nats/byte). Speaking and
  hearing in imagination run on a COPY of the fast table, so imagination
  never writes the real table. The body's recurrent state is untouched.
* Valuation: the critic sees the option's film entropy in place of F's
  latent landing, plus the same content, feeling, option and recall inputs.
  Metal calibrates it on the realized TD return exactly as in stage 4.
* Acting: if generate is chosen, the draft is what is said (its byte
  probabilities are checked by the causal policy replay). If reuse is
  chosen, the sampled stored words are said.
F is still trained (unchanged), but it no longer enters the choice.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from scripts.ib_local.language_action import peek
from scripts.ib_local.language_navigation_kanli_v4 import GENERATE, REUSE, SILENCE
from scripts.ib_local.language_navigation_kanli_v4b import INTERO
from scripts.ib_local.language_navigation_kanli_v4e import EqualEngine, torch_multinomial


class FilmEngine(EqualEngine):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        latent = self.model.encoder.out_features
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(31)
            critic = nn.Sequential(nn.Linear(latent+1+INTERO+3+2, latent), nn.SiLU(), nn.Linear(latent, 1))
        self.model.kanli.critic = critic.to(self.device)
        self.optimizer.add_param_group(dict(params=list(critic.parameters()), lr=self.optimizer.param_groups[0]['lr']))

    @torch.no_grad()
    def _film(self, words, table):
        """Say `words` (or draft them if None) then imagine the reply, all on `table`.

        Returns (words, byte_logp, probs, mean entropy of the imagined reply).
        """
        byte = self.pending
        self.decision.start(self.recurrent)
        p = peek(table, byte)
        logits, _ = self.decision.step(byte, p)
        said, logps, probs = [], [], []
        n = self.max_utterance if words is None else len(words)
        for i in range(n):
            if words is None:
                logp = logits.log_softmax(-1)
                chosen = int(torch_multinomial(logp))
                logps.append(float(logp[chosen]))
            else:
                chosen = int(words[i])
            probs.append(p)
            said.append(chosen)
            type(table).run_credit(table, np.array([byte, chosen], np.uint8), max_window=1)
            byte = chosen
            p = peek(table, byte)
            logits, _ = self.decision.step(byte, p)
        entropy = []
        for _ in range(self.max_utterance):
            logp = logits.log_softmax(-1)
            entropy.append(float(-(logp.exp()*logp).sum()))
            nxt = int(torch_multinomial(logp))
            type(table).run_credit(table, np.array([byte, nxt], np.uint8), max_window=1)
            byte = nxt
            p = peek(table, byte)
            logits, _ = self.decision.step(byte, p)
        return bytes(said), logps, probs, float(np.mean(entropy))

    @torch.no_grad()
    def _value(self, h, films):
        heads, enc = self.model.kanli, self.model.encoder
        z = enc(h)[0]
        recalled = self.recall_outcomes(z)
        film = torch.tensor(films, dtype=torch.float32, device=z.device)[:, None]
        inputs = torch.cat((z.expand(3, -1), film, self.intero().expand(3, -1), torch.eye(3, device=z.device), recalled), -1)
        return heads.critic(inputs).squeeze(-1)+recalled[:, 0], inputs

    @torch.no_grad()
    def _utter(self):
        self.decision.start(self.recurrent)
        p = peek(self.table, self.pending)
        _, h = self.decision.step(self.pending, p)
        habit, plan, reusable = self.plan(h[None])
        memory = self.model.memory
        w = plan['weights'][0]*reusable[0]
        slot, reuse_words = None, None
        if float(w.sum()) > 0:
            k = int(torch.multinomial(w/w.sum(), 1))
            slot = int(plan['slots'][0, k])
            reuse_words = memory.words[slot, :int(memory.word_count[slot])].cpu().numpy()
        films, draft = [0., 0., 0.], None
        if self.mode != 'silence':
            _, _, _, films[SILENCE] = self._film(b'', copy.deepcopy(self.table))
            draft, draft_logp, _, films[GENERATE] = self._film(None, copy.deepcopy(self.table))
            if slot is not None:
                films[REUSE] = self._film(reuse_words, copy.deepcopy(self.table))[3]
        values, inputs = self._value(h[None], films)
        available = habit[0] > -1e3
        centered = values-values[available].mean()
        offset = (self.precision()*centered).masked_fill(~available, 0.)
        logits = habit[0]+offset
        record = dict(feeling=self.intero()[0].clone(), choice_logits=logits.tolist(), habit_logits=habit[0].tolist(),
                      offset=offset.tolist(), values=values.tolist(), critic_inputs=inputs, films=list(films),
                      stop_logits=[], byte_logp=[], probs=[], truncated=False, choice=SILENCE, reused_slot=None)
        self._pending_record = record
        if self.mode == 'silence':
            return b'', record
        choice = int(torch.multinomial(logits.softmax(-1), 1))
        record['choice'] = choice
        if choice == SILENCE:
            return b'', record
        self.decision.start(self.recurrent)
        if choice == GENERATE:
            record['byte_logp'] = list(draft_logp)
            return self._speak_bytes(np.frombuffer(draft, np.uint8), record, False), record
        record['reused_slot'] = slot
        return self._speak_bytes(reuse_words, record, False), record

    def step(self, directions, age_ablate=False):
        result = super().step(directions, age_ablate)
        result['films'] = self._pending_record['films']
        return result
