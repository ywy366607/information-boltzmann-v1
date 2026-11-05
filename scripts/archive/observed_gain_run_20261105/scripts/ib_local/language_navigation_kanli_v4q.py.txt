"""V4q: one common level and centered action contrasts, learned from the loop.

No oracle location/freshness enters the body. Historical residuals use the
baseline available at their own decision, so replay does not turn old absolute
progress into a new target for today's changed baseline.
"""
from __future__ import annotations

import torch

from scripts.ib_local.language_navigation_kanli_v4p import ActionEvidence, ResidualWoodEngine


class SharedActionEvidence(ActionEvidence):
    """Normalized innovation on [1, one_hot(action)-1/3].

`blind` is the common level and `mean` contains centered contrasts. The
feature norm squared is 5/3: the selected forecast receives exactly gain*error,
other forecasts receive 2/5 of that update. Common motion need not wait until
each option is selected. This is predictive pooling, not identified causal
effects from observational samples.
"""
    def __init__(self, decay=.95):
        super().__init__(decay)
        self.register_buffer('unconditional', torch.zeros(4))

    def bases(self):
        return self.blind[None] + self.mean

    @torch.no_grad()
    def observe(self, choice, target, base, raw):
        n = int(self.seen[choice])
        if n:
            rate = 1. if n == 1 else 1-self.decay
            self.base_error[choice].lerp_((base-target).square(), rate)
            self.raw_error[choice].lerp_((raw-target).square(), rate)
        any_seen = bool(self.seen.any())
        self.unconditional.lerp_(target, 1-self.decay if any_seen else 1.)
        if not any_seen:
            self.blind.copy_(target)
        elif not n:
            # Initialize only the newly observed option, preserving all others.
            self.mean[choice].copy_(target-self.blind)
            shift = self.mean.mean(0)
            self.mean.sub_(shift)
            self.blind.add_(shift)
        else:
            innovation = (1-self.decay)*(target-base)/(5/3)
            self.blind.add_(innovation)
            self.mean.sub_(innovation/3)
            self.mean[choice].add_(innovation)
        self.seen[choice] += 1


class SharedContextWoodEngine(ResidualWoodEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model.wood_evidence = SharedActionEvidence(self.wood_decay).to(self.device)
        m = self.model.memory
        m.register_buffer('wood_baseline', torch.zeros_like(m.wood_target))
        # v4p replaced F's first layer after DialogueEngine cached these lists.
        # Refresh references so clipping and actor snapshots use actual params.
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.groups = (*self.groups[:2], [p for n,p in self.model.named_parameters()
                                         if not n.startswith('body.')])
        # Read-only comparison on the SAME chosen transitions; never passed to
        # the policy or learning losses. Included in engine continuation state.
        self.audit_unpooled = ActionEvidence(self.wood_decay).to(self.device)

    @torch.no_grad()
    def _value(self, h, films):
        values, inputs = super()._value(h, films)
        self._forecast['common'] = self.model.wood_evidence.blind.clone()
        self._forecast['blind'] = self.model.wood_evidence.unconditional.clone()
        self._forecast['contrasts'] = self.model.wood_evidence.mean.clone()
        return values, inputs

    def _wood_replay_batch(self):
        m = self.model.memory
        valid = torch.nonzero(m.occupied & (m.wood_target_count > 0) & (m.wood_option >= 0)).flatten()
        if not len(valid):
            return None
        pick = valid[torch.as_tensor(self.replay_rng.integers(0, len(valid), self.replay_batch),
                                     device=valid.device)]
        self._reward_replay_valid = self.reward_label_valid(pick, wood=True).clone()
        return tuple(x[pick].detach().clone() for x in
                     (m.source, m.action, m.wood_target, m.wood_option, m.wood_baseline))

    def _training_batch(self, target):
        choice = self._pending_record['choice']
        src, act = self._wood_source[None], self._wood_action[None]
        options = torch.tensor([choice], device=self.device)
        labels = target[None]
        self._batch_baseline = self._forecast['base'][choice:choice+1]
        if self._wood_replay is not None:
            rs, ra, rt, ri, rb = self._wood_replay
            src, act = torch.cat((src, rs)), torch.cat((act, ra))
            labels, options = torch.cat((labels, rt)), torch.cat((options, ri))
            self._batch_baseline = torch.cat((self._batch_baseline, rb))
        return src, act, options, labels

    def _training_base(self, options):
        assert len(options) == len(self._batch_baseline)
        return self._batch_baseline.detach()

    @torch.no_grad()
    def _store_label(self, result, choice, target):
        m, slot = self.model.memory, result['episode_slot']
        old_count = int(m.wood_target_count[slot])
        super()._store_label(result, choice, target)
        if not result['episode_merged']:
            m.wood_baseline[slot].copy_(self._forecast['base'][choice])
        elif int(m.wood_target_count[slot]) > 0:
            m.wood_baseline[slot].lerp_(self._forecast['base'][choice], 1/(old_count+1))

    def step(self, directions, age_ablate=False):
        before = dict(anchor=int(self.world.anchor), offset=int(self.world.offset),
                      moves=int(self.world.moves), witness_count=len(self.w_ids),
                      feeling=self.intero_raw.tolist())
        result = super().step(directions, age_ablate)
        choice = self._pending_record['choice']
        target = self.model.memory.wood_target.new_tensor(result['wood_target'])
        old_base = self.audit_unpooled.bases()[choice].clone()
        result['wood_unpooled_error'] = (old_base-target).square().tolist()
        result['wood_unpooled_base'] = old_base.tolist()
        self.audit_unpooled.observe(choice, target, old_base, old_base)
        result.update(world_before=before, wood_common=self._forecast['common'].tolist(),
            wood_action_contrasts=self._forecast['contrasts'].tolist(),
            wood_all_predictions=self._forecast['used'].tolist(),
            wood_all_bases=self._forecast['base'].tolist(),
            wood_all_confidence=self._forecast['confidence'].tolist(),
            wood_centering_error=float(self.model.wood_evidence.mean.sum(0).abs().max()))
        return result
