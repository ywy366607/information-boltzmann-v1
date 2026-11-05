"""V4 stage 4: metal gets its own water (Claude, 2026-11-04).

User, 11-04: every phase must itself be alive, so each unfolds into its own
five phases. Stage 3 showed metal's own water was dry. Metal judged progress
on the probe it had just learned. With no memory of what it had already
learned, it counted memorizing repeats as progress (+0.47 centi-nat median on
exhausted turns). The world's `fresh` flag then clipped that to 0: the world
remembered on metal's behalf, so the body never felt the cost of silence. Yet
that cost was large: the silent arm drifted from 1.51 to 6.20 nats on
validation.

Changes, all inside metal (water and wood unchanged from stage 3):
* Metal's water, the witness reservoir. The body keeps, by reservoir
  sampling, 128-byte windows of what it heard. A window is admitted only when
  it surprised the body at hearing time: its NLL then exceeded the body's
  running mean reply NLL, which is its own interoception. The world's `fresh`
  flag is no longer used by the body.
* Metal's judgment, generalization. Progress g_t is the witness NLL before the
  cycle minus after it, on the fixed admitted set. Witnesses are scored as one
  cold-start stream with the byte-table probabilities stored at hearing, so g
  measures the network's change only.
* Metal returns to the cause, multi-turn. The label of cycle t is the TD
  return y_t = g_t + GAMMA * V_{t+1}. V_{t+1} is the policy-weighted imagined
  value at the next decision. Labels, critic and habit all use y_t, so the
  habit's actor gradient is applied one cycle late. GAMMA = 0.8, a horizon of
  about 5 turns, is a declared choice and is not tuned.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from scripts.ib_local.language_navigation_kanli_v4 import GENERATE, KanLiEngine
from scripts.ib_local.language_navigation_kanli_v4b import SCALE, ImaginingEngine
from scripts.ib_local.language_navigation_v2 import FLOOR
from scripts.ib_local.navigation_settlement import SETTLEMENT_CODES, settlement_code
from scripts.ib_local.navigation_live_witness import frozen_table_logp
from scripts.ib_local.navigation_value_scale import normalized_critic_loss

GAMMA = .8
WITNESS = 32
WINDOW = 128


def choice_replay_error(replayed, sampled, habit):
    """Compare available choices even when their finite logits are large."""
    a, b, h = (np.asarray(x, dtype=float) for x in (replayed, sampled, habit))
    if a.shape != b.shape or h.shape != b.shape:
        raise ValueError('choice replay shapes differ')
    available = h > -1e3  # The habit mask declares unavailable choices.
    if not available.any():
        raise RuntimeError('policy has no available choices')
    if not np.isfinite(a[available]).all() or not np.isfinite(b[available]).all():
        raise FloatingPointError('nonfinite available choice logits')
    return float(np.abs(a[available]-b[available]).max())


class WitnessEngine(ImaginingEngine):
    def __init__(self, *args, gamma=GAMMA, witnesses=WITNESS, **kw):
        super().__init__(*args, **kw)
        self.gamma, self.capacity = gamma, witnesses
        self.w_ids, self.w_tgt, self.w_logp, self.w_hear = [], [], [], []
        self.w_admitted = 0
        self.w_rng = np.random.default_rng(131)
        self.prev = None          # what cycle t-1 left for its TD label
        self.delayed_applied = 0

    def settlement_value(self, witness_change, continuation_progress, joint_progress=None):
        """Select the actual label before memory, TD and interoception consume it."""
        kind = getattr(self, 'settlement_kind', 'witness')
        if kind not in SETTLEMENT_CODES:
            raise ValueError(f'unknown settlement: {kind}')
        value = witness_change if kind == 'witness' else continuation_progress
        if kind == 'continuation_joint':
            if joint_progress is None:
                raise ValueError('joint settlement requires both frozen body scores')
            value = joint_progress
        if not np.isfinite(value):
            raise FloatingPointError('nonfinite settlement')
        return float(value)

    # ---- metal's water -------------------------------------------------
    @torch.no_grad()
    def witness_nll(self):
        """Witnesses scored as one cold-start stream; each window keeps its own targets."""
        if not self.w_ids:
            return 0.
        ids, tgt, logp = (torch.cat(x, 1) for x in (self.w_ids, self.w_tgt, self.w_logp))
        state = self.model.initial_state(1, device=self.device)
        total = 0.
        for i in range(0, ids.shape[1], 512):
            logits, state, _ = self.model.predict(ids[:, i:i+512], state, logp[:, i:i+512])
            total += float(F.cross_entropy(logits[0], tgt[0, i:i+512], reduction='sum'))
        return total/ids.shape[1]

    @torch.no_grad()
    def _admit(self, response, response_probs, token_ce, own):
        """Offer one random window of the reply; admitted only if it surprised the body."""
        if len(response) < WINDOW+1:
            return False
        s = int(self.w_rng.integers(0, len(response)-WINDOW))
        heard = float(token_ce[own+s+1:own+s+WINDOW+1].mean())
        felt = self.intero_stats.mean[0] if self.intero_stats.count > 1 else -np.inf
        if heard <= felt:
            return False
        ids = torch.as_tensor(response[s:s+WINDOW].astype(np.int64), device=self.device)[None]
        tgt = torch.as_tensor(response[s+1:s+WINDOW+1].astype(np.int64), device=self.device)[None]
        logp = torch.as_tensor(np.log(np.maximum(response_probs[s+1:s+WINDOW+1], FLOOR)),
                               device=self.device)[None]
        self.w_admitted += 1
        if len(self.w_ids) < self.capacity:
            self.w_ids.append(ids); self.w_tgt.append(tgt); self.w_logp.append(logp); self.w_hear.append(heard)
        else:
            j = int(self.w_rng.integers(0, self.w_admitted))
            if j < self.capacity:
                self.w_ids[j], self.w_tgt[j], self.w_logp[j], self.w_hear[j] = ids, tgt, logp, heard
        return True

    # ---- one cycle -----------------------------------------------------
    def step(self, directions, age_ablate=False):
        memory = self.model.memory
        w_pre = self.witness_nll()
        start_state = self.recurrent.clone()
        start_pending = self.pending
        source_time = int(memory.clock)
        utterance, record = self._utter()
        # Value of the state the body now stands in, under its own policy.
        probs3 = torch.tensor(record['choice_logits']).softmax(-1)
        v_now = float((probs3*torch.tensor(record['values'])).sum())
        own = len(utterance)
        response, heldout, info = self.world.take(utterance, self.response_bytes, self.holdout_bytes)
        last_action = utterance[-1] if own else start_pending
        joint = (getattr(self, 'settlement_kind', 'witness') == 'continuation_joint'
                 or getattr(self, 'audit_joint_score', False))
        joint_pre = joint_post = None
        if joint:
            # Both reads use the same actual prefix. Teacher forcing exposes
            # prior bytes as context, but never trains on the scored targets.
            score_prefix = np.concatenate((np.array([last_action], np.uint8), response, heldout[:-1]))
            joint_old_logp = torch.as_tensor(
                frozen_table_logp(self.table, score_prefix)[len(response):], device=self.device)[None]
        _, response_probs, _ = self.table.run_credit(
            np.concatenate((np.array([last_action], np.uint8), response)), max_window=self.response_bytes)
        if joint:
            joint_new_logp = torch.as_tensor(
                frozen_table_logp(self.table, score_prefix)[len(response):], device=self.device)[None]
        raw = np.concatenate((np.array([start_pending], np.uint8), np.frombuffer(utterance, np.uint8), response))
        probs = np.concatenate((np.asarray(record['probs'], np.float32).reshape(-1, 256), response_probs), 0)
        ids, targets, logp = self._inputs(raw, probs)
        mask = torch.ones_like(targets, dtype=torch.float32)
        mask[:, :own] = 0
        component_ablation = self._age_component_ablation(ids, targets, start_state, logp, mask) if age_ablate else None
        loss, state_after_response, features, diag = self.model.objective(
            ids, targets, start_state, logp, directions, score_mask=mask)
        logits = diag['logits']
        token_ce = F.cross_entropy(logits[0].detach(), targets[0], reduction='none').cpu().numpy()
        actor, policy_error, actor_norm = None, 0., 0.
        if self.mode != 'silence':
            policy, replay = self._policy_logp(features, logits, targets, own, record)
            errors = [choice_replay_error(replay['choice'], record['choice_logits'], record['habit_logits'])]
            errors += [abs(a-b) for a, b in zip(replay['stops'], record['stop_logits'])]
            if record['choice'] == GENERATE and own:
                errors.append(float((replay['byte_logp'] -
                                     torch.tensor(record['byte_logp'], device=self.device)).abs().max()))
            policy_error = max(errors)
            if policy_error > 1e-3:
                raise RuntimeError(f'sampled decisions and causal replay differ: {policy_error}')
            actor = torch.autograd.grad(policy, self.params, retain_graph=True, allow_unused=True)
            actor = tuple(None if g is None else g.detach().clone() for g in actor)
            actor_norm = float(torch.stack([g.norm() for g in actor if g is not None]).norm())
        # Settle cycle t-1 only after this cycle's policy replay: its label
        # changes which recalled words count as reusable.
        td = None
        if self.prev is not None:
            # A pending old-objective credit remains payable at migration,
            # but must not bootstrap into the value of a different objective.
            same_kind = self.prev.get('settlement_kind', 'witness') == getattr(self, 'settlement_kind', 'witness')
            td = self._settle_previous(self.prev['g'] + (self.gamma*v_now/SCALE if same_kind else 0.))
            if not same_kind:
                td['objective_boundary'] = True
        probe_raw = np.concatenate((response[-1:], heldout))
        _, probe_probs, _ = self.table.run_credit(probe_raw, max_window=self.holdout_bytes)
        probe_ids, probe_targets, probe_logp = self._inputs(probe_raw, probe_probs)
        with torch.no_grad():
            pre_logits, _, _ = self.model.predict(probe_ids, state_after_response.detach(), probe_logp)
            pre = float(F.cross_entropy(pre_logits.flatten(0, 1), probe_targets.flatten()))
            if joint:
                joint_logits, _, _ = self.model.predict(probe_ids, state_after_response.detach(), joint_old_logp)
                joint_pre = float(F.cross_entropy(joint_logits.flatten(0, 1), probe_targets.flatten()))
        h_cycle = [features.detach()]
        response_norm = self._advance(loss, features, diag, state_after_response)
        if joint:
            with torch.no_grad():
                joint_logits, _, _ = self.model.predict(probe_ids, self.recurrent, joint_new_logp)
                joint_post = float(F.cross_entropy(joint_logits.flatten(0, 1), probe_targets.flatten()))
        after_loss, after_state, after_features, after_diag = self.model.objective(
            probe_ids, probe_targets, self.recurrent, probe_logp, directions)
        timeline = torch.cat(h_cycle+[after_features.detach()], 0)
        short_at = min(own+memory.short, len(timeline)-1)
        long_at = min(memory.long, len(timeline)-1)
        source, action, long = timeline[0], timeline[own], timeline[long_at]
        outcome_loss, outcome_diag = self._outcome_loss(source, action, long)  # same method for stages 4-5b; overridable from stage 6
        post = float(after_diag['ce'])
        # The habit's actor of the PREVIOUS cycle is applied now, with its TD advantage.
        delayed, advantage = (None, 0.) if td is None else (td.pop('actor'), td['advantage'])
        if delayed is not None:
            self.delayed_applied += 1
        heldout_norm = self._advance(after_loss+self.outcome_weight*outcome_loss, after_features, after_diag,
                                     after_state, actor=delayed, advantage=advantage)
        self.pending = int(heldout[-1])
        w_post = self.witness_nll()
        witness_change = w_pre-w_post
        g = self.settlement_value(witness_change, pre-post,
                                  joint_pre-joint_post if joint else None)
        # Observed mode keeps this gain (or its merged mean). Legacy mode
        # replaces it with a TD return when the next state is known.
        slot, merged = memory.commit_experience(source, action, timeline[short_at], long, self.model.encoder,
                                                utterance, g, source_time)
        if hasattr(self, 'settlement_versions'):
            version = settlement_code(self.settlement_kind)
            old = int(self.settlement_versions[slot])
            self.settlement_versions[slot] = version if not merged or old == version else 2
        memory.option[slot] = record['choice']
        memory.feeling[slot] = record['feeling']
        admitted = self._admit(response, response_probs, token_ce, own)
        self.prev = dict(g=g, slot=slot, actor=actor, inputs=record['critic_inputs'][record['choice']].detach(),
                         value=float(record['values'][record['choice']]), choice=record['choice'],
                         settlement_kind=getattr(self, 'settlement_kind', 'witness'))
        self.intero_raw = np.array([float(diag['ce']), SCALE*g, float(diag['latent'])])
        self.intero_stats.update(self.intero_raw)
        try:
            utterance.decode('utf-8'); utf8_valid = True
        except UnicodeDecodeError:
            utf8_valid = False
        return dict(mode=self.mode, choice=('silence', 'generate', 'reuse')[record['choice']],
            choice_probability=probs3.tolist(), reused_slot=record['reused_slot'],
            utterance_hex=utterance.hex(), utf8_valid=utf8_valid, spoke=own > 0,
            truncated=record['truncated'], external_bytes=len(response)+len(heldout), self_bytes=own,
            world=info, response_nll=float(diag['ce']), heldout_pre_nll=pre, heldout_post_nll=post,
            progress=pre-post, fresh=bool(info['holdout_position_fresh']),
            joint_pre_nll=joint_pre, joint_post_nll=joint_post,
            joint_progress=joint_pre-joint_post if joint else None,
            witness_pre=w_pre, witness_post=w_post, witness_g=witness_change,
            settlement_kind=getattr(self, 'settlement_kind', 'witness'),
            settlement=g, metal_g=g, usable_signal=g,
            witnesses=len(self.w_ids), witness_admitted=self.w_admitted, admitted=admitted,
            td_previous=td, value_now=v_now, delayed_actor_applied=self.delayed_applied,
            actor_norm=actor_norm, policy_replay_max_error=policy_error,
            short_loss=float(diag['latent']), long_loss=float(diag['latent_long']), **outcome_diag,
            episode_slot=slot, episode_merged=merged, age_component_ablation=component_ablation,
            gradient_norms=[response_norm, heldout_norm], updates=2,
            habit_logits=record['habit_logits'], imagination_offset=record['offset'],
            imagined_value=record['values'], precision=self.precision(),
            critic_rms=None if self.critic_error is None else float(np.sqrt(self.critic_error)),
            interoception=self.intero_raw.tolist())

    def _write_return_label(self, y):
        """Only the legacy return-memory contract writes TD into the table."""
        if getattr(self, 'reward_memory_kind', 'td') == 'observed':
            return
        prev, memory = self.prev, self.model.memory
        memory.progress[prev['slot']] = y
        if hasattr(self, 'settlement_versions'):
            # TD overwrites this scalar; it is no longer a mixed average.
            self.settlement_versions[prev['slot']] = settlement_code(prev.get('settlement_kind', 'witness'))

    def _settle_previous(self, y):
        """Learn a TD target while preserving measured gains in observed mode."""
        prev = self.prev
        self._write_return_label(y)
        target = SCALE*y
        out = dict(g=prev['g'], y=y, imagined=prev['value'], choice=prev['choice'], actor=None, advantage=0.)
        if getattr(self, 'reward_memory_kind', 'td') == 'observed':
            memory = self.model.memory
            out.update(memory_label_kind='observed_gain',
                       memory_gain=float(memory.progress[prev['slot']]),
                       memory_gain_valid=bool(memory.progress_valid[prev['slot']]))
        if self.mode != 'silence':
            critic = self.model.kanli.critic
            inputs = prev['inputs']
            if getattr(self, 'value_normalization', 'none') == 'popart':
                loss, scale_diag = normalized_critic_loss(self, inputs[None], target)
                out.update(scale_diag)
            else:
                prediction = critic(inputs[None]).squeeze()+inputs[-2]
                loss = F.mse_loss(prediction, torch.tensor(target, device=self.device))
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.)
            if getattr(self, 'value_normalization', 'none') == 'popart':
                out['value_gradient_norm'] = float(grad_norm)
            self.optimizer.step()
            error = (prev['value']-target)**2
            self.critic_error = error if self.critic_error is None else .98*self.critic_error+.02*error
            self.critic_count += 1
            out['actor'] = prev['actor']
            out['advantage'] = y-self.baseline
        self.baseline = self.baseline_decay*self.baseline + (1-self.baseline_decay)*y
        return out
