"""V4p: observed action means plus a calibrated, learned contextual residual.

Forecast before observing; calibrate after settlement. The four coordinates
are NLL4/3, NLL32/3, match, 10*g, exactly as v4m. Replay uses fixed stored
features re-encoded by today's encoder, without an old autograd graph.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from scripts.ib_local.language_navigation_kanli_v4l import LiveTargetEngine
from scripts.ib_local.language_navigation_kanli_v4m import WoodCalibratedEngine
from scripts.ib_local.navigation_settlement import settlement_code
from scripts.ib_local.navigation_value_scale import critic_output


class ActionEvidence(nn.Module):
    """Bounded online statistics; replay never counts as fresh evidence."""
    def __init__(self, decay=.95):
        super().__init__()
        self.decay = decay
        self.register_buffer('mean', torch.zeros(3, 4))
        self.register_buffer('blind', torch.zeros(4))
        self.register_buffer('seen', torch.zeros(3, dtype=torch.long))
        self.register_buffer('base_error', torch.zeros(3, 4))
        self.register_buffer('raw_error', torch.zeros(3, 4))

    def bases(self):
        return torch.where(self.seen[:, None] > 0, self.mean, self.blind)

    def confidence(self):
        value = (1 - self.raw_error / self.base_error.clamp_min(1e-8)).clamp(0, 1)
        return torch.where(self.seen[:, None] > 1, value, 0.)

    @torch.no_grad()
    def observe(self, choice, target, base, raw):
        # base/raw are the actual decision-time forecasts, not recomputations.
        n = int(self.seen[choice])
        b, r = (base-target).square(), (raw-target).square()
        if n:
            rate = 1. if n == 1 else 1-self.decay
            self.base_error[choice].lerp_(b, rate)
            self.raw_error[choice].lerp_(r, rate)
        self.mean[choice].lerp_(target, 1-self.decay if n else 1.)
        self.blind.lerp_(target, 1-self.decay if bool(self.seen.any()) else 1.)
        self.seen[choice] += 1


class ResidualWoodEngine(WoodCalibratedEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        heads = self.model.kanli
        # Retain F's architecture and optimizer cadence, exposing the already
        # chosen option as three additional inputs. Start at exactly the mean.
        old = heads.outcome[0]
        with torch.random.fork_rng(devices=[old.weight.device] if old.weight.is_cuda else []):
            new = nn.Linear(old.in_features+3, old.out_features, device=old.weight.device)
        with torch.no_grad():
            new.weight.zero_()
            new.weight[:, :old.in_features].copy_(old.weight)
            new.bias.copy_(old.bias)
            heads.outcome[-1].weight[:4].zero_()
            heads.outcome[-1].bias[:4].zero_()
        old_ids = {id(p) for p in old.parameters()}
        for group in self.optimizer.param_groups:
            group['params'] = [p for p in group['params'] if id(p) not in old_ids]
        heads.outcome[0] = new
        self.optimizer.add_param_group(dict(params=list(new.parameters()),
                                             lr=self.optimizer.param_groups[0]['lr']))
        self.model.wood_evidence = ActionEvidence(self.wood_decay).to(self.device)
        self.model.memory.register_buffer('wood_option', torch.full(
            (self.model.memory.capacity,), -1, dtype=torch.long, device=self.device))
        self._forecast = None

    def wood_state_dict(self):
        return dict(decay=self.model.wood_evidence.decay,
                    replay_rng=self.replay_rng.bit_generator.state)

    def load_wood_state_dict(self, state):
        if not 0 <= float(state['decay']) < 1:
            raise ValueError('invalid evidence decay')
        self.model.wood_evidence.decay = float(state['decay'])
        self.replay_rng.bit_generator.state = state['replay_rng']

    def _residual(self, sources, actions, options):
        z, a = self.model.encoder(sources), self.model.encoder(actions)
        identity = F.one_hot(options, 3).to(z.dtype)
        return self.model.kanli.outcome(torch.cat((z, a-z, identity), -1))[:, :4]

    def _compose_prediction(self, base, residual):
        return base + residual

    def _compose_used_prediction(self, base, residual, confidence):
        return base + confidence * residual

    @torch.no_grad()
    def _value(self, h, films):
        del films
        z = self.model.encoder(h)[0]
        actions = list(self._draft_actions)
        while len(actions) < 3:
            actions.append(h.reshape(-1))
        residual = self._residual(h.expand(3, -1), torch.stack(actions),
                                  torch.arange(3, device=self.device))
        evidence = self.model.wood_evidence
        base, confidence = evidence.bases().clone(), evidence.confidence().clone()
        raw = self._compose_prediction(base, residual)
        used = self._compose_used_prediction(base, residual, confidence)
        self._forecast = dict(base=base, raw=raw, used=used, confidence=confidence,
                              blind=evidence.blind.clone())
        recalled = self.recall_outcomes(z)
        outcome = z.new_zeros(3, z.numel()); outcome[:, :4] = used
        inputs = torch.cat((z.expand(3, -1), outcome, self.intero().expand(3, -1),
                            torch.eye(3, device=self.device), recalled), -1)
        values = critic_output(self, inputs).squeeze(-1)+recalled[:, 0]
        base_inputs = inputs.clone(); base_inputs[:, z.numel():z.numel()+4] = base
        base_values = critic_output(self, base_inputs).squeeze(-1)+recalled[:, 0]
        blind = inputs.clone(); blind[:, z.numel():2*z.numel()] = 0
        blind_values = critic_output(self, blind).squeeze(-1)+recalled[:, 0]
        available = len(self._draft_actions) if self._draft_actions else 1
        def offset(delta):
            out = torch.zeros_like(delta)
            out[:available] = self.precision()*(delta[:available]-delta[:available].mean())
            return out
        self._forecast['total_offset'] = offset(values-blind_values)
        self._forecast['residual_offset'] = offset(values-base_values)
        self.draft_diagnostics = dict(
            policy_logit_F_effect_max=float(self._forecast['total_offset'].abs().max()),
            policy_logit_residual_effect_max=float(self._forecast['residual_offset'].abs().max()),
            confidence=confidence.tolist())
        return values, inputs

    def _wood_replay_batch(self):
        m = self.model.memory
        valid = torch.nonzero(m.occupied & (m.wood_target_count > 0) & (m.wood_option >= 0)).flatten()
        if not len(valid):
            return None
        pick = valid[torch.as_tensor(self.replay_rng.integers(0, len(valid), self.replay_batch),
                                     device=valid.device)]
        return tuple(x[pick].detach().clone() for x in
                     (m.source, m.action, m.wood_target, m.wood_option))

    def _training_batch(self, target):
        choice = self._pending_record['choice']
        src, act = self._wood_source[None], self._wood_action[None]
        options = torch.tensor([choice], device=self.device)
        labels = target[None]
        if self._wood_replay is not None:
            rs, ra, rt, ri = self._wood_replay
            src, act = torch.cat((src, rs)), torch.cat((act, ra))
            labels, options = torch.cat((labels, rt)), torch.cat((options, ri))
        return src, act, options, labels

    def _outcome_loss(self, source, action, long):
        del long
        self._wood_source, self._wood_action = source.detach().clone(), action.detach().clone()
        self._wood_replay = self._wood_replay_batch()
        target = torch.cat((self.model._wood_nll/3., source.new_tensor([float(self._wood_match), 0.])))
        src, act, options, labels = self._training_batch(target)
        base = self._training_base(options)
        prediction = self._compose_prediction(base, self._residual(src, act, options))
        loss = F.mse_loss(prediction[:, :3], labels[:, :3])
        return loss, dict(outcome_error=float(loss.detach()),
            wood_replay_samples=0 if self._wood_replay is None else len(self._wood_replay[0]),
            draft_actual_action_l2=float((self._draft_actions[self._pending_record['choice']]-action).norm())
                if self._draft_actions else 0.)

    def _training_base(self, options):
        return self.model.wood_evidence.bases()[options].detach()

    @torch.no_grad()
    def _store_label(self, result, choice, target):
        m, slot = self.model.memory, result['episode_slot']
        if not result['episode_merged']:
            m.wood_target[slot].copy_(target)
            m.wood_target_count[slot] = 1
            m.wood_option[slot] = choice
        elif (int(m.wood_option[slot]) == choice and
              int(m.wood_target_count[slot]) == int(m.count[slot])-1):
            count = int(m.wood_target_count[slot])
            m.wood_target[slot].lerp_(target, 1/(count+1))
            m.wood_target_count[slot] = count+1
        else:
            # A merged source/action cannot be relabelled as the latest option.
            m.wood_target_count[slot] = 0
            m.wood_option[slot] = -1

    def step(self, directions, age_ablate=False):
        original_take = self.world.take
        def observed_take(*args, **kwargs):
            response, heldout, info = original_take(*args, **kwargs)
            self._wood_match = info['match'] > 0
            return response, heldout, info
        self.world.take = observed_take
        try:
            # Reuse the same complete water/fire/body/critic transition, without
            # the superseded v4m scalar calibration and absolute-output update.
            result = LiveTargetEngine.step(self, directions, age_ablate)
        finally:
            self.world.take = original_take
        choice = self._pending_record['choice']
        target = torch.cat((self.model._wood_nll/3., self.model._wood_nll.new_tensor(
            [float(self._wood_match), 10*result['metal_g']])))
        forecast = self._forecast
        base, raw, used = (forecast[k][choice] for k in ('base', 'raw', 'used'))
        # g only exists after settlement. Statistics are still from BEFORE
        # this turn, including during the delayed residual training below.
        self.optimizer.zero_grad(set_to_none=True)
        src, act, options, labels = self._training_batch(target)
        prediction = self._compose_prediction(self._training_base(options), self._residual(src, act, options))
        squared = (prediction[:, 3]-labels[:, 3]).square()
        if hasattr(self, 'wood_settlement_versions'):
            # Old labels still train the first three channels. Only the
            # incompatible reward coordinate is excluded from replay.
            valid = torch.cat((torch.ones(1, device=self.device),
                self._reward_replay_valid.to(squared.dtype))) if self._wood_replay is not None else torch.ones_like(squared)
            loss = self.outcome_weight*(squared*valid).sum()/valid.sum().clamp_min(1)
        else:
            loss = self.outcome_weight*F.mse_loss(prediction[:, 3], labels[:, 3])
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(list(self.model.encoder.parameters())+
                                              list(self.model.kanli.outcome.parameters()), 1.)
        self.optimizer.step()
        self.model.wood_evidence.observe(choice, target, base, raw)
        if hasattr(self, 'wood_settlement_versions'):
            slot = result['episode_slot']
            old_version = int(self.wood_settlement_versions[slot])
        self._store_label(result, choice, target)
        if hasattr(self, 'wood_settlement_versions'):
            version = settlement_code(self.settlement_kind)
            self.wood_settlement_versions[slot] = version if not result['episode_merged'] or old_version == version else 2
        logits = target.new_tensor(self._pending_record['choice_logits'])
        def tv(delta):
            return float((logits.softmax(-1)-(logits-delta).softmax(-1)).abs().sum()/2)
        result.update(wood_target=target.tolist(), wood_base=base.tolist(), wood_raw=raw.tolist(),
            wood_prediction=used.tolist(), wood_blind=forecast['blind'].tolist(),
            wood_confidence_used=forecast['confidence'][choice].tolist(),
            wood_confidence_next=self.model.wood_evidence.confidence()[choice].tolist(),
            wood_base_error=(base-target).square().tolist(), wood_raw_error=(raw-target).square().tolist(),
            wood_used_error=(used-target).square().tolist(),
            wood_blind_error=(forecast['blind']-target).square().tolist(),
            wood_policy_tv=tv(forecast['total_offset']),
            residual_policy_tv=tv(forecast['residual_offset']), wood_to_fire=self.draft_diagnostics,
            wood_g_update_loss=float(loss.detach()), wood_g_gradient_norm=float(grad),
            wood_valid_replay_slots=int((self.model.memory.wood_target_count > 0).sum()),
            wood_reward_replay_samples=(int(self._reward_replay_valid.sum())
                if hasattr(self, 'wood_settlement_versions') and self._wood_replay is not None
                else (0 if self._wood_replay is None else len(self._wood_replay[0]))),
            updates=result['updates']+1)
        return result
