"""Action-conditioned latent imagination on the existing language body.

An imagined utterance is replayed through the same recurrent decision state
and a temporary fast table.  The post-utterance state is the action input to
the existing outcome model, matching the state used by its training loss.
No extra encoder, teacher, or persistent memory is introduced.
"""
from __future__ import annotations

import numpy as np
import torch

from scripts.ib_local.language_action import peek
from scripts.ib_local.language_navigation_kanli_v4e import EqualEngine, torch_multinomial
from scripts.ib_local.language_navigation_kanli_v4i import FilmEngine


class LatentDraftEngine(FilmEngine):
    def __init__(self, *args, **kwargs):
        # Keep the original latent-sized critic and existing optimizer groups.
        EqualEngine.__init__(self, *args, **kwargs)
        self._draft_actions = []
        self.ablate_action_outcome = False

    @torch.no_grad()
    def _film(self, words, table):
        # Only replay the proposed action; no unused imagined-reply sampling.
        byte = self.pending
        self.decision.start(self.recurrent)
        p = peek(table, byte)
        logits, state = self.decision.step(byte, p)
        said, logps, probs = [], [], []
        count = self.max_utterance if words is None else len(words)
        for i in range(count):
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
            logits, state = self.decision.step(byte, p)
        self._draft_actions.append(state.detach().clone().reshape(-1))
        return bytes(said), logps, probs, 0.

    @torch.no_grad()
    def _value(self, h, films):
        del films  # the scalar reply-entropy feature failed its prior gate
        enc, heads = self.model.encoder, self.model.kanli
        z = enc(h)[0]
        actions = [enc(x) for x in self._draft_actions]
        while len(actions) < 3:
            actions.append(z)
        landing = heads.outcome(torch.cat((z.expand(3, -1), torch.stack(actions)), -1))
        self.draft_diagnostics = dict(
            action_generate_silent_l2=float((actions[1] - actions[0]).norm()),
            outcome_generate_silent_l2=float((landing[1] - landing[0]).norm()),
            action_reuse_generate_l2=(float((actions[2] - actions[1]).norm())
                                      if len(self._draft_actions) == 3 else None),
            outcome_reuse_generate_l2=(float((landing[2] - landing[1]).norm())
                                       if len(self._draft_actions) == 3 else None))
        recalled = self.recall_outcomes(z)
        chosen_landing = landing[0].expand_as(landing) if self.ablate_action_outcome else landing
        inputs = torch.cat((z.expand(3, -1), chosen_landing, self.intero().expand(3, -1),
                            torch.eye(3, device=z.device), recalled), -1)
        values = heads.critic(inputs).squeeze(-1) + recalled[:, 0]
        # Counterfactual read only: retain every critic input except the
        # F outcome, which is replaced by the silence outcome for all options.
        blind = inputs.clone()
        width = z.numel()
        blind[:, width:2*width] = landing[0]
        blind_values = heads.critic(blind).squeeze(-1) + recalled[:, 0]
        self.draft_diagnostics['critic_F_effect_max'] = float((values-blind_values).abs().max())
        available = slice(None) if len(self._draft_actions) == 3 else slice(0, 2)
        delta = (values-blind_values)[available]
        self.draft_diagnostics['policy_logit_F_effect_max'] = float(
            self.precision() * (delta-delta.mean()).abs().max())
        return values, inputs

    def _utter(self):
        self._draft_actions = []
        return super()._utter()

    def _outcome_loss(self, source, action, long):
        loss, diagnostics = super()._outcome_loss(source, action, long)
        choice = self._pending_record['choice']
        if choice < len(self._draft_actions):
            diagnostics['draft_actual_action_l2'] = float(
                (self._draft_actions[choice] - action.detach()).norm())
        return loss, diagnostics
