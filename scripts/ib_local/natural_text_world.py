"""API-free natural-text contact, retaining the continuous caregiver history."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np

from scripts.ib_local import inquiry_navigation as inquiry


def extended_source(prefix, manifest):
    """Validate the immutable append on both migration and checkpoint restore."""
    if manifest.get('schema') != 1:
        raise ValueError('unknown natural-text source schema')
    path = Path(manifest['train_path'])
    raw = path.read_bytes()
    if len(raw) != manifest['train_bytes'] or hashlib.sha256(raw).hexdigest() != manifest['train_sha256']:
        raise ValueError('natural-text source changed')
    if len(prefix) != manifest['prefix_bytes']:
        raise ValueError('natural-text append would change existing addresses')
    return np.concatenate((prefix, np.frombuffer(raw, dtype=np.uint8)))


def append_source(base, manifest):
    """Append bytes without changing old addresses, seen flags, or world RNG."""
    data = extended_source(base.data, manifest)
    state = base.state_dict()
    expanded = type(base)(data, window=base.window, seed=base.seed,
                          buckets=base.positions.shape[0], per_bucket=base.positions.shape[1])
    state['seen'] = np.pad(state['seen'], (0, len(data)-len(base.data)))
    expanded.load_state_dict(state)
    return expanded


class NaturalTextWorld:
    """The existing suffix-navigation world, with no caregiver/provider calls.

    Mother history remains checkpoint data, not synthetic observations. The
    actual text response boundary feeds the existing world-relation port.
    """
    def __init__(self, base, state, source_manifest=None, navigation_policy=None):
        self.base = base
        self.source_manifest = source_manifest
        self.retained = copy.deepcopy({k: v for k, v in state.items() if k.startswith('mother_')})
        self.navigation_policy = navigation_policy or state.get('text_navigation_policy', 'legacy')
        if self.navigation_policy not in ('legacy', 'stable_context_v1', *inquiry.POLICIES):
            raise ValueError('unknown text navigation policy')

    def __getattr__(self, name):
        return getattr(self.__dict__['base'], name)

    def take(self, utterance, response_bytes, holdout_bytes):
        # Reading proceeds when no suffix is taken up. Silence must not turn
        # this language-growth phase into endless rereading of one 4KB topic.
        cursor = self.base.cursor
        if cursor + self.base.window > len(self.base.data):
            cursor = 0
        self.base.anchor, self.base.offset = cursor, 0
        explicit = self.navigation_policy.startswith('inquire_')
        term = b''
        if explicit:
            response_bytes -= inquiry.HEADER
        if self.navigation_policy == 'legacy':
            response, continuation, info = self.base.take(utterance, response_bytes, holdout_bytes)
        else:
            from scripts.ib_local.text_navigation import stable_suffix
            required = response_bytes + holdout_bytes
            if response_bytes <= 0 or holdout_bytes <= 0 or required > self.base.window:
                raise ValueError('positive observation pieces must fit the world window')
            # Keep the inherited world RNG clock, although tie-breaking no
            # longer uses randomness. No body RNG or hidden feature is read.
            if utterance:
                self.base.rng.integers(0, 2**32, dtype=np.uint64)
            if self.navigation_policy == 'stable_context_v1':
                start, score, ties, distance = stable_suffix(
                    np.frombuffer(utterance, np.uint8), self.base.data, self.base.window, cursor)
            elif self.navigation_policy == 'inquire_terms_v1' and utterance:
                if '_search_bytes' not in vars(self):
                    self._search_bytes = self.base.data.tobytes().lower()
                start, score, ties, distance, term = inquiry.address(
                    utterance, self._search_bytes, self.base.window, cursor)
            else:
                start, score, ties, distance = -1, 0, 0, -1
            if score:
                self.base.anchor, self.base.offset = int(start), 0
                self.base.moves += 1
            else:
                start = cursor
            end = int(start) + required
            prior_response = int(self.base.seen[start:start+response_bytes].sum())
            prior_holdout = int(self.base.seen[start+response_bytes:end].sum())
            self.base.seen[start:end] = True
            self.base.offset = required % self.base.window
            self.base.cursor = end
            response = self.base.data[start:start+response_bytes].copy()
            continuation = self.base.data[start+response_bytes:end].copy()
            info = dict(start=int(start), anchor=int(self.base.anchor), match=int(score),
                response_previously_seen_bytes=prior_response,
                holdout_previously_seen_bytes=prior_holdout,
                holdout_position_fresh=prior_holdout == 0,
                source='matched' if score else 'topic',
                match_candidates=int(ties), match_distance=int(distance),
                address_policy=self.navigation_policy)
        accounting = {}
        if self.source_manifest is not None:
            # take() starts at offset zero here, so this actual delivered turn
            # is contiguous, including turns whose suffix selected an old address.
            start, split = int(info['start']), int(self.source_manifest['prefix_bytes'])
            def counts(begin, size):
                inherited = max(0, min(begin + size, split) - begin)
                return dict(inherited_prefix=inherited, natural_append=size-inherited)
            accounting = dict(text_source_bytes=counts(start, len(response)+len(continuation)),
                text_response_source_bytes=counts(start, len(response)),
                text_holdout_source_bytes=counts(start+len(response), len(continuation)))
        if explicit:
            control = inquiry.header(bool(utterance), term)
            response = np.concatenate((np.frombuffer(control, np.uint8), response))
            accounting.update(control_bytes=len(control), inquiry_term=term.decode('ascii'))
        return response, continuation, dict(info, **accounting, source='text',
            navigation_source=info.get('source'), navigation_response_bytes=len(response),
            text_navigation=(self.navigation_policy if self.navigation_policy in inquiry.POLICIES else
                'longest actual suffix; nearest equally specific continuation; EOF wrap only'
                if self.navigation_policy == 'stable_context_v1' else
                'suffix match or sequential cursor; EOF wrap only'),
            mother_bytes=0, provider_calls=0,
            child_text=utterance.decode('utf-8', 'replace'), text_reply=bytes(response).decode('utf-8', 'replace'))

    def state_dict(self, *, copy_arrays=True):
        state = self.base.state_dict() if copy_arrays else self.base.checkpoint_state_dict()
        return dict(state, **copy.deepcopy(self.retained), text_navigation_policy=self.navigation_policy)

    def checkpoint_state_dict(self):
        return self.state_dict(copy_arrays=False)

    def load_state_dict(self, state):
        self.base.load_state_dict(state)
        self.retained = copy.deepcopy({k: v for k, v in state.items() if k.startswith('mother_')})
        policy = state.get('text_navigation_policy', 'legacy')
        if policy not in ('legacy', 'stable_context_v1', *inquiry.POLICIES):
            raise ValueError('unknown text navigation policy')
        self.navigation_policy = policy
