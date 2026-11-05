"""Low-dimensional action outcome prediction with prequential wood calibration.

Candidate on the existing v4l individual. Wood predicts normalized reply NLL4,
NLL32, match and witness progress. After the observed turn, its own prediction
is compared to a running constant baseline; only past errors affect the next
turn. The world match flag is a simulator observation, not a universal sensor.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from scripts.ib_local.language_navigation_kanli_v4l import LiveSigregBody, LiveTargetEngine


class WoodObservationBody(LiveSigregBody):
    def objective(self, ids, targets, state, table_logp, directions,
                  prediction_weight=.1, regularization_weight=.001, score_mask=None):
        loss, state, h, diag = super().objective(ids, targets, state, table_logp, directions,
                                                 prediction_weight, regularization_weight, score_mask)
        token = F.cross_entropy(diag['logits'].flatten(0, 1), targets.flatten(), reduction='none')
        if score_mask is None:
            language = token.mean()
        else:
            mask = score_mask.reshape(-1).to(token.dtype)
            language = (token*mask).sum()/mask.sum().clamp_min(1)
        loss = loss + 9.*language
        if score_mask is not None:
            own = int((score_mask[0] == 0).sum())
            with torch.no_grad():
                ce = F.cross_entropy(diag['logits'][0, own:], targets[0, own:], reduction='none')
                self._wood_nll = torch.stack((ce[:4].mean(), ce[:32].mean())).detach()
        return loss, state, h, diag


class WoodCalibratedEngine(LiveTargetEngine):
    """The existing F's first four outputs predict one observed turn's outcome."""
    def __init__(self, *args, wood_decay=.95, **kwargs):
        super().__init__(*args, **kwargs)
        self.wood_decay = wood_decay
        self.wood_count = 0
        self.wood_mean = np.zeros(4, dtype=np.float64)
        self.wood_error = 0.
        self.wood_baseline_error = 0.
        self.wood_confidence = 0.
        self._wood_source = self._wood_action = None
        self._wood_predictions = None
        memory = self.model.memory
        memory.register_buffer('wood_target', torch.zeros(memory.capacity, 4, device=memory.source.device))
        memory.register_buffer('wood_target_count', torch.zeros(memory.capacity, dtype=torch.long,
                                                                 device=memory.source.device))
        self._wood_replay = None

    def _wood_replay_batch(self):
        memory = self.model.memory
        valid = torch.nonzero(memory.occupied & (memory.wood_target_count > 0)).flatten()
        if not len(valid):
            return None
        sampled = valid[torch.as_tensor(self.replay_rng.integers(0, len(valid), self.replay_batch),
                                        device=valid.device)]
        # Snapshot before this turn can merge or evict a slot.
        return (memory.source[sampled].detach().clone(), memory.action[sampled].detach().clone(),
                memory.wood_target[sampled].detach().clone())

    def wood_state_dict(self):
        """Persist the prequential calibration along with a full body snapshot."""
        return dict(decay=self.wood_decay, count=self.wood_count,
                    mean=self.wood_mean.tolist(), error=self.wood_error,
                    baseline_error=self.wood_baseline_error, confidence=self.wood_confidence,
                    replay_rng=self.replay_rng.bit_generator.state)

    def load_wood_state_dict(self, state):
        mean = np.asarray(state['mean'], dtype=np.float64)
        decay, confidence, count = float(state['decay']), float(state['confidence']), int(state['count'])
        error, baseline = float(state['error']), float(state['baseline_error'])
        if (mean.shape != (4,) or not np.isfinite(mean).all() or not 0 <= decay < 1
                or not 0 <= confidence <= 1 or count < 0 or not np.isfinite(error)
                or not np.isfinite(baseline) or min(error, baseline) < 0):
            raise ValueError('invalid wood calibration state')
        self.wood_decay, self.wood_count, self.wood_mean = decay, count, mean.copy()
        self.wood_error, self.wood_baseline_error, self.wood_confidence = error, baseline, confidence
        self.replay_rng.bit_generator.state = state['replay_rng']

    @torch.no_grad()
    def _value(self, h, films):
        del films
        enc, heads = self.model.encoder, self.model.kanli
        z = enc(h)[0]
        actions = [enc(a) for a in self._draft_actions]
        while len(actions) < 3:
            actions.append(z)
        raw = heads.outcome(torch.cat((z.expand(3, -1), torch.stack(actions)), -1))
        self._wood_predictions = raw[:, :4].detach().cpu().numpy().copy()
        # At zero reliability the outcome channels are action-independent;
        # the existing habit, recalled experience and critic remain available.
        outcome = torch.zeros_like(raw)
        outcome[:, :4] = self.wood_confidence * raw[:, :4]
        recalled = self.recall_outcomes(z)
        inputs = torch.cat((z.expand(3, -1), outcome, self.intero().expand(3, -1),
                            torch.eye(3, device=z.device), recalled), -1)
        values = heads.critic(inputs).squeeze(-1) + recalled[:, 0]
        blind = inputs.clone(); width = z.numel(); blind[:, width:2*width] = 0
        blind_values = heads.critic(blind).squeeze(-1) + recalled[:, 0]
        available = slice(None) if len(self._draft_actions) == 3 else slice(0, 2)
        delta = (values-blind_values)[available]
        self.draft_diagnostics = dict(
            action_generate_silent_l2=float((actions[1]-actions[0]).norm()),
            outcome_generate_silent_l2=float((raw[1,:4]-raw[0,:4]).norm()),
            critic_F_effect_max=float((values-blind_values).abs().max()),
            policy_logit_F_effect_max=float(self.precision()*(delta-delta.mean()).abs().max()),
            wood_confidence=self.wood_confidence)
        return values, inputs

    def _outcome_loss(self, source, action, long):
        del long
        # 3 observable targets exist by the second body update. The witness
        # result g is measured only after this update and is settled below.
        self._wood_source, self._wood_action = source.detach().clone(), action.detach().clone()
        self._wood_replay = self._wood_replay_batch()
        target = torch.cat((self.model._wood_nll / 3.,
                            torch.tensor([float(self._wood_match)], device=source.device)))
        sources, actions, targets = source[None], action[None], target[None]
        if self._wood_replay is not None:
            rs, ra, rt = self._wood_replay
            sources = torch.cat((sources, rs)); actions = torch.cat((actions, ra))
            targets = torch.cat((targets, rt[:, :3]))
        z, a = self.model.encoder(sources), self.model.encoder(actions)
        prediction = self.model.kanli.outcome(torch.cat((z, a), -1))[:, :3]
        loss = F.mse_loss(prediction, targets)
        with torch.no_grad():
            source_substitute = self.model.kanli.outcome(torch.cat((z, z), -1))[:, :3]
            substitute_loss = float(F.mse_loss(source_substitute, targets))
            shuffled_loss = None
            if len(actions) > 1:
                shuffled = self.model.kanli.outcome(torch.cat((z, a.roll(1, 0)), -1))[:, :3]
                shuffled_loss = float(F.mse_loss(shuffled, targets))
        return loss, dict(outcome_error=float(loss.detach()),
                          outcome_error_current=float(F.mse_loss(prediction[0], targets[0]).detach()),
                          outcome_error_shuffled=shuffled_loss,
                          outcome_error_source_substitute=substitute_loss,
                          wood_replay_samples=0 if self._wood_replay is None else len(self._wood_replay[0]))

    def step(self, directions, age_ablate=False):
        original_take = self.world.take
        def observed_take(*args, **kwargs):
            response, heldout, info = original_take(*args, **kwargs)
            self._wood_match = info['match'] > 0
            return response, heldout, info
        self.world.take = observed_take
        confidence_used = self.wood_confidence
        try:
            result = super().step(directions, age_ablate)
        finally:
            self.world.take = original_take
        choice = self._pending_record['choice']
        prediction = self._wood_predictions[choice].astype(np.float64)
        target = np.array([float(self.model._wood_nll[0])/3., float(self.model._wood_nll[1])/3.,
                           float(self._wood_match), 10.*result['metal_g']], dtype=np.float64)
        # Compare the decision-time forecast against the previous running mean;
        # a warm-start observation initializes the baseline, not its own score.
        if self.wood_count:
            model_err = float(np.mean((prediction-target)**2))
            baseline_err = float(np.mean((self.wood_mean-target)**2))
            if self.wood_count == 1:
                self.wood_error, self.wood_baseline_error = model_err, baseline_err
            else:
                decay = self.wood_decay
                self.wood_error = decay*self.wood_error+(1-decay)*model_err
                self.wood_baseline_error = decay*self.wood_baseline_error+(1-decay)*baseline_err
            self.wood_confidence = float(np.clip(1-self.wood_error/(self.wood_baseline_error+1e-8), 0, 1))
        else:
            model_err = baseline_err = None
        self.wood_mean = (self.wood_mean*self.wood_decay + (1-self.wood_decay)*target
                          if self.wood_count else target.copy())
        self.wood_count += 1
        # The fourth label exists only after witness re-evaluation; it gets one
        # explicit delayed optimizer update on F and its shared encoder.
        self.optimizer.zero_grad(set_to_none=True)
        sources, actions = self._wood_source[None], self._wood_action[None]
        labels = torch.tensor([target[3]], dtype=sources.dtype, device=sources.device)
        if self._wood_replay is not None:
            rs, ra, rt = self._wood_replay
            sources = torch.cat((sources, rs)); actions = torch.cat((actions, ra))
            labels = torch.cat((labels, rt[:, 3]))
        z = self.model.encoder(sources); a = self.model.encoder(actions)
        p = self.model.kanli.outcome(torch.cat((z,a), -1))[:, 3]
        g_loss = F.mse_loss(p, labels) * self.outcome_weight
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.model.encoder.parameters())+
                                      list(self.model.kanli.outcome.parameters()), 1.)
        self.optimizer.step()
        # Attach the same observed label to the same bounded episode slot.
        # Merge only when every contributing episode already had a label;
        # otherwise the averaged source/action and target would be misaligned.
        memory = self.model.memory
        slot = result['episode_slot']
        with torch.no_grad():
            if not result['episode_merged']:
                memory.wood_target[slot].copy_(torch.as_tensor(target, dtype=memory.wood_target.dtype,
                                                               device=memory.wood_target.device))
                memory.wood_target_count[slot] = 1
            elif int(memory.wood_target_count[slot]) == int(memory.count[slot])-1:
                count = int(memory.wood_target_count[slot])
                value = torch.as_tensor(target, dtype=memory.wood_target.dtype, device=memory.wood_target.device)
                memory.wood_target[slot].copy_((memory.wood_target[slot]*count+value)/(count+1))
                memory.wood_target_count[slot] = count+1
            else:
                memory.wood_target_count[slot] = 0
        result.update(wood_target=target.tolist(), wood_prediction=prediction.tolist(),
                      wood_confidence_used=confidence_used, wood_confidence_next=self.wood_confidence,
                      wood_model_error=model_err, wood_baseline_error=baseline_err,
                      wood_g_update_loss=float(g_loss.detach()), updates=result['updates']+1)
        return result
