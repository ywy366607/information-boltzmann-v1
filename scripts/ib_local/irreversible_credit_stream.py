"""One committed token history, with bounded reexecution for window learning.

Actual inputs advance the live native cache once. At a 128-input boundary,
reexecution computes local gradients from a held incoming cache; its output
cache is discarded. The already-lived cache is never replaced or rolled back.
Generated tokens are inputs too, but are not independent teaching targets.
"""
from __future__ import annotations

import hashlib
import math

import torch
from torch.nn import functional as F


class IrreversibleCreditStream:
    def __init__(self, backend, *, window=128, on_window=None):
        if window != 128:
            raise ValueError('the existing capsule contract requires 128 inputs')
        self.backend, self.window, self.on_window = backend, window, on_window
        self.inputs, self.targets, self.learn_mask, self.target_nll = [], [], [], []
        self.boundary = None
        self.next_logits = None
        self.events = self.windows = self.update_windows = 0
        self.chain = hashlib.sha256(b'CGB irreversible token stream v1').hexdigest()
        self.phase_counts = {}

    def _target(self, token, learn):
        score = float(F.cross_entropy(self.next_logits.float().reshape(1, -1),
                                     torch.tensor([token], device=self.next_logits.device)))
        if not math.isfinite(score):
            raise FloatingPointError('nonfinite pre-update prediction')
        self.targets.append(token)
        self.learn_mask.append(bool(learn))
        self.target_nll.append(score)
        if len(self.targets) == self.window:
            self._commit()
        return score

    def _commit(self):
        if len(self.inputs) != self.window or len(self.targets) != self.window:
            raise RuntimeError('one next-token observation per committed input required')
        expected_ce = (sum(v for v, m in zip(self.target_nll, self.learn_mask) if m)
                       / sum(self.learn_mask) if any(self.learn_mask) else None)
        result = self.backend.commit(self.inputs, self.targets, self.learn_mask, self.boundary)
        self.windows += 1
        self.update_windows += int(any(self.learn_mask))
        result.update(stream_events=self.events, window=self.windows,
                      external_teaching_tokens=sum(self.learn_mask),
                      lived_preupdate_ce=expected_ce,
                      reexecution_ce_difference=(None if expected_ce is None else
                                                result['ce'] - expected_ce),
                      history_chain=self.chain)
        if self.on_window is not None:
            self.on_window(result)
        self.inputs, self.targets, self.learn_mask, self.target_nll = [], [], [], []
        self.boundary = None

    def receive(self, tokens, *, phase, teach=False):
        """Consume externally supplied or self-produced tokens; never pad/reset.

        Batched known inputs remain causal. Learning waits for the actual next
        token, including across phase boundaries. Only external targets teach.
        """
        tokens = [int(t) for t in tokens]
        if phase == 'self' and teach:
            raise ValueError('self-produced text cannot provide independent teaching')
        scores, position = [], 0
        while position < len(tokens):
            if self.inputs:
                scores.append(self._target(tokens[position], teach))
            if not self.inputs:
                self.boundary = self.backend.boundary()
            count = min(len(tokens)-position, self.window-len(self.inputs))
            piece = tokens[position:position+count]
            output = self.backend.advance(piece, clock_offset=len(self.inputs))
            if output.shape[0] != count:
                raise ValueError('one causal prediction per consumed token required')
            self.inputs.extend(piece)
            for i, token in enumerate(piece):
                self.events += 1
                self.chain = hashlib.sha256(
                    bytes.fromhex(self.chain) + f'{phase}:{token};'.encode()).hexdigest()
                self.phase_counts[phase] = self.phase_counts.get(phase, 0) + 1
                if i + 1 < count:
                    self.next_logits = output[i]
                    scores.append(self._target(piece[i+1], teach))
            self.next_logits = output[-1].detach().clone()
            position += count
        return scores

    def generate(self, *, eos_token_id, max_tokens=16, pending=None):
        if self.next_logits is None:
            raise RuntimeError('generation needs an actually observed prefix')
        generated = []
        for _ in range(max_tokens):
            token = int(self.next_logits.argmax())
            if pending is not None:
                pending.add(self.backend.last_prediction_participation, token)
                self.backend.mark_prediction_used()
            # EOS is a real emitted event too; it is consumed before stopping.
            self.receive([token], phase='self', teach=False)
            generated.append(token)
            if token == eos_token_id:
                break
        if pending is not None:
            pending.finish_emission(self.backend.learner.cache._seen_tokens)
        return generated

    def state_dict(self):
        from scripts.ib_local.pretrained_gdn_experience import copy_tree
        return dict(schema=1, window=self.window, inputs=list(self.inputs),
                    targets=list(self.targets), learn_mask=list(self.learn_mask),
                    target_nll=list(self.target_nll), boundary=copy_tree(self.boundary),
                    next_logits=copy_tree(self.next_logits), events=self.events,
                    windows=self.windows, update_windows=self.update_windows,
                    chain=self.chain, phase_counts=dict(self.phase_counts))

    def restore_latest(self, state):
        """Fault recovery only; caller also restores the matching backend state."""
        if self.events or self.inputs:
            raise RuntimeError('cannot rewind an already running life')
        if state['schema'] != 1 or state['window'] != self.window:
            raise ValueError('incompatible stream state')
        for name in ('inputs', 'targets', 'learn_mask', 'target_nll'):
            setattr(self, name, list(state[name]))
        if len(self.targets) != max(0, len(self.inputs)-1) or len(self.inputs) > self.window:
            raise ValueError('invalid pending history')
        from scripts.ib_local.pretrained_gdn_experience import copy_tree
        self.boundary = copy_tree(state['boundary'])
        self.next_logits = (None if state['next_logits'] is None else
                            state['next_logits'].to(self.backend.device))
        for name in ('events', 'windows', 'update_windows', 'chain'):
            setattr(self, name, state[name])
        self.phase_counts = dict(state['phase_counts'])


class PretrainedStreamBackend:
    """Reuse native learner and frozen replay cache; no alternate learning rule."""
    def __init__(self, learner):
        self.learner = learner
        self.device = next(learner.model.parameters()).device
        self._decode_context = None
        self.record_predictions = False
        self.last_prediction_participation = None
        self._capture_boundary = True
        self._captured_views = None
        self._captured_view_key = None

    def set_prediction_capture(self, enabled):
        self.record_predictions = bool(enabled)
        self.last_prediction_participation = None
        self._capture_boundary = True
        self._captured_views = self._captured_view_key = None

    def mark_prediction_used(self):
        # Later answer positions are causally reconstructed from this first
        # boundary and the actual emitted prefix; do not copy all24 S again.
        self._capture_boundary = False

    def _close_decode(self):
        if self._decode_context is not None:
            self._decode_context.__exit__(None, None, None)
            self._decode_context = None

    def boundary(self):
        from scripts.ib_local.pretrained_gdn_experience import copy_tree
        return dict(states=copy_tree(self.learner.cache.states),
                    seen_tokens=self.learner.cache._seen_tokens)

    @torch.no_grad()
    def advance(self, tokens, *, clock_offset):
        from scripts.ib_local.evaluate_pretrained_gdn_ruler import frozen_decode_replays
        if self._decode_context is None:
            self._decode_context = frozen_decode_replays(self.learner)
            self._decode_context.__enter__()
        record = None
        if self.record_predictions:
            from scripts.ib_local.delayed_prediction_credit import PredictionParticipation, capture_views
            key = (self.learner.updates, tuple(int(w.memory.clock) for w in self.learner.memories.values()))
            if key != self._captured_view_key:
                self._captured_views = capture_views(self.learner)
                self._captured_view_key = key
            record = PredictionParticipation(torch.tensor([tokens]),
                self.boundary() if self._capture_boundary else None,
                self._captured_views, clock_offset, self.learner.updates,
                self.learner.cache._seen_tokens+len(tokens))
        x = torch.tensor([tokens], device=self.device)
        options = {}
        if record is not None:
            from scripts.ib_local.delayed_prediction_credit import capture_reads
            options['on_prediction'] = lambda: setattr(record, 'reads', capture_reads(self.learner))
        logits, cache = self.learner.predict(x, clock_offset=clock_offset, **options)
        self.learner.cache = cache
        if record is not None:
            record.logits = logits[0, -1].detach().float().cpu().clone()
        self.last_prediction_participation = record
        return logits[0]

    def commit(self, inputs, targets, mask, boundary):
        from scripts.ib_local.pretrained_gdn import clone_cache
        from scripts.ib_local.pretrained_gdn_experience import copy_tree
        self._close_decode()
        live = self.learner.cache
        seen = live._seen_tokens
        # Restored window boundaries are serialized without any autograd graph.
        if isinstance(boundary, dict):
            from scripts.ib_local.pretrained_gdn import new_recurrent_cache
            cache = new_recurrent_cache(boundary['seen_tokens'])
            cache.states = copy_tree(boundary['states'], self.device)
        else:
            cache = clone_cache(boundary)
        x = torch.tensor([inputs], device=self.device)
        parameter_version=self.learner.updates
        if any(mask):
            result = self.learner.step(x, torch.tensor([targets], device=self.device),
                loss_mask=torch.tensor([mask], device=self.device, dtype=torch.bool),
                cache_override=cache, preserve_live_cache=True)
            slots=result.get('committed_slots')
        else:
            slots=self.learner.observe(x, cache_override=cache, preserve_live_cache=True)
            result = dict(updates=self.learner.updates, tokens=self.learner.tokens,
                          observed_only=True)
        if getattr(self.learner,'joint_capture_enabled',False):
            if slots is None:
                raise RuntimeError('joint archive needs the actual committed local identities')
            from scripts.ib_local.pretrained_gdn_experience import tree_nbytes
            joint=self.learner.archive_joint_participation(x,cache,slots,parameter_version)
            result.update(joint_boundary_bytes=tree_nbytes(joint.boundary),
                          joint_input_bytes=tree_nbytes(joint.incoming_features),
                          joint_entries_share_one_archive=len(slots),
                          experience_cpu_bytes=self.learner.experience_storage_nbytes)
        if self.learner.cache is not live or live._seen_tokens != seen:
            raise AssertionError('local reexecution replaced the actually lived cache')
        result['live_cache_preserved'] = True
        return result

    def close(self):
        self._close_decode()
