"""V4 stage 5: fire's own metal (Claude, 2026-11-04).

Nested five-phase analysis of stage 4: fire's inner fire was empty and its
inner metal was dead.
* Generated speech has median length 1 byte.
* The world understands speech iff it is >= 4 bytes long (95-100% vs 0%).
  Only 10-17% of utterances reach 4 bytes, the geometric tail of a newborn
  stop probability of about 0.5 per byte.
* The stop head never learned. Its habit gradient is the TD advantage in nats
  (median |A| = 0.013) times an actor gradient. In the same update the
  language gradient is about 1.4, so fire's metal received about 1% of the
  signal.

The change stays inside fire. Its metal calibrates itself against its own
noise: for the decision heads (choice and stop, all non-body parameters), the
delayed actor is scaled by A / RMS(A), with RMS(A) its own running RMS. This is
the habit's own precision, the same principle as the critic's 1/RMS. The body
(language model) parameters keep the raw advantage, so earth is not washed by
policy gradients. Nothing sets a speech length or rate.

Amendment 1 (gate_stage5_1, H3 failed: first stop probability 0.454 -> 0.461).
Adam rescales the head gradients per parameter, so scale was not the cause.
Fire's metal was judging with the big metal's verdict: one TD return per cycle
(noise about +-3 centi-nat), in which "stopped too early, not taken up" drowns.
Under 五行互藏, fire needs its own metal judging its own act. Fire's purpose is
contact, so its metal asks whether my speech changed what I hear
(contingency), and it reads the answer from the body's own interoception:
  c = reply NLL - the body's own running expectation of reply NLL.
When speech is taken up, the partner continues somewhere new and surprise
jumps (1.64 vs 0.91). The stop decisions are credited with (c - b_c)/RMS, at
once and on the decision heads only. The choice keeps the big metal's TD
return, so each act is judged by the metal of its own phase.

Amendment 2 (gate_stage5_2 passed formally, but stop probability ROSE 0.460 ->
0.485). Measured on the stage-4 records, reply NLL minus its long-run mean is
+0.44 for speech that was not taken up in a new topic (the topic itself
surprises), so short utterances in new topics were rewarded. Against the
previous turn's reply NLL instead, taken-up speech in an exhausted topic gives
+0.69 (median) and speech that was not taken up gives -0.02 / -0.04 in either
state. The expectation is therefore the last felt reply surprise.

Amendment 3 (gate_stage5_3, H3 failed: first stop probability 0.485 -> 0.480).
The signal had the right sign (n >= 4: +0.70, n = 1: -0.07, n = 2-3: -0.28),
but one advantage for the whole utterance also punished the correct "go on"
after byte 1 whenever the utterance later failed at byte 2 or 3. The pushes
on the first stop cancelled. Fire's metal lacked fire's own water: a memory of
what each length led to. V(k) = running mean contingency of utterances that
reached k bytes (decay 0.98 per update, declared). Each stop decision gets its
own credit: go on at k -> V(k+1) - V(k); stop at n -> c - V(n). The sum is
normalized by its own running RMS.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from scripts.ib_local.language_navigation_kanli_v4 import GENERATE
from scripts.ib_local.language_navigation_kanli_v4c import WitnessEngine


class FireEngine(WitnessEngine):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        names = [n for n, p in self.model.named_parameters() if p.requires_grad]
        if len(names) != len(self.params):
            raise RuntimeError('parameter list mismatch')
        self.head_mask = [not n.startswith('body.') for n in names]
        self.adv_ms = None
        self.c_mean = self.c_ms = None
        self.v_len = {}
        self.fire_grad = None
        self.fire_last = None

    def _settle_previous(self, y):
        out = super()._settle_previous(y)
        actor, raw = out['actor'], out['advantage']
        out['advantage_raw'] = raw
        if actor is None:
            return out
        self.adv_ms = raw*raw if self.adv_ms is None else self.baseline_decay*self.adv_ms+(1-self.baseline_decay)*raw*raw
        scaled = raw/np.sqrt(self.adv_ms+1e-12)
        out['advantage_head'] = scaled
        # Pre-scale per parameter; _advance then multiplies by advantage = 1.
        out['actor'] = tuple(None if g is None else g*(scaled if head else raw)
                             for g, head in zip(actor, self.head_mask))
        out['advantage'] = 1.
        return out

    def _policy_logp(self, h, logits, targets, own, record):
        """Choice term -> delayed big-metal actor; stop terms -> fire's own metal, now."""
        habit, _, _ = self.plan(h[:1])
        combined = habit[0]+torch.tensor(record['offset'], device=habit.device)
        choice_term = combined.log_softmax(-1)[record['choice']]
        replay = dict(choice=combined.detach().tolist(), stops=[])
        self.fire_grad, self.fire_last = None, None
        if record['choice'] == GENERATE and own:
            decisions = self.decision_logits(h[:own+1])
            n = len(record['stop_logits'])
            stopped = n == own
            terms = [F.logsigmoid(decisions[k, 1]) if stopped and k == n else F.logsigmoid(-decisions[k, 1])
                     for k in range(1, n+1)]
            replay['stops'] = [float(decisions[k, 1].detach()) for k in range(1, n+1)]
            byte_logp = logits[0, :own].log_softmax(-1).gather(1, targets[0, :own, None]).flatten()
            replay['byte_logp'] = byte_logp.detach()
            if terms:
                reply = float(F.cross_entropy(logits[0, own:].detach(), targets[0, own:]))
                # Amendment 2: compare with what was heard just before (the last felt
                # reply surprise), not with the long-run mean.
                expect = float(self.intero_raw[0]) if self.intero_stats.count > 0 else reply
                c = reply-expect
                V = lambda k: self.v_len.get(k, self.v_len.get(max(self.v_len), 0.) if self.v_len else 0.)
                adv = [(c-V(k)) if stopped and k == n else (V(k+1)-V(k)) for k in range(1, n+1)]
                for k in range(1, n+1):
                    self.v_len[k] = c if k not in self.v_len else .98*self.v_len[k]+.02*c
                ms = float(np.mean(np.square(adv)))
                self.c_ms = ms if self.c_ms is None else self.baseline_decay*self.c_ms+(1-self.baseline_decay)*ms
                scale = 1/np.sqrt(self.c_ms+1e-12) if self.c_ms > 0 else 0.
                weighted = sum(float(a)*scale*t for a, t in zip(adv, terms))
                grads = torch.autograd.grad(weighted, self.params, retain_graph=True, allow_unused=True)
                self.fire_grad = tuple(None if g is None or not head else -g.detach()
                                       for g, head in zip(grads, self.head_mask))
                self.fire_last = dict(contingency=c, advantages=[float(a)*scale for a in adv], reply_nll=reply,
                                      expected=expect, stops=n, v_len={k: round(v, 4) for k, v in sorted(self.v_len.items())[:8]})
        return choice_term, replay

    def _advance(self, loss, features, diagnostics, next_state, actor=None, advantage=0.):
        if self.fire_grad is None:
            return super()._advance(loss, features, diagnostics, next_state, actor, advantage)
        # Fire's own metal acts in the first update of the cycle it judged.
        # The parent adds -1 * combined, so combined = advantage*actor - fire.
        fire, self.fire_grad = self.fire_grad, None
        actor = actor if actor is not None else (None,)*len(fire)
        combined = []
        for a, f in zip(actor, fire):
            x = None if a is None else advantage*a
            if f is not None:
                x = -f if x is None else x-f
            combined.append(x)
        return super()._advance(loss, features, diagnostics, next_state, tuple(combined), 1.)

    def step(self, directions, age_ablate=False):
        result = super().step(directions, age_ablate)
        result['stop_logits'] = list(self._pending_record['stop_logits'])
        result['fire_metal'] = self.fire_last
        return result
