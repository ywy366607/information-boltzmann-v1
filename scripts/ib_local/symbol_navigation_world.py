"""User-authorized, semantic-free environment for the existing navigation body.

Only emitted bytes enter language learning. Room IDs, entropy rates and the
routing rule are researcher diagnostics. No room reward enters the learner.
"""
import copy

import numpy as np


class SymbolNavigationWorld:
    """An utterance ending in a/b selects periodic/random binary input.

    Silence or any other ending stays in the current room. The periodic
    source is aabb; the random source is an independent fair a/b process.
    Both use the same alphabet, so frequency alone does not separate them.
    """
    def __init__(self, seed=11):
        self.rng = np.random.default_rng(seed)
        self.anchor = 1
        self.offset = 0
        self.moves = 0
        self.emitted = 0

    def take(self, utterance, response_bytes, holdout_bytes):
        if response_bytes < 1 or holdout_bytes < 1:
            raise ValueError('both observation intervals must be positive')
        previous = self.anchor
        accepted = bool(utterance and utterance[-1] in (ord('a'), ord('b')))
        if accepted:
            self.anchor = int(utterance[-1] == ord('b'))
            self.moves += int(self.anchor != previous)
        n = response_bytes + holdout_bytes
        if self.anchor == 0:
            motif = np.frombuffer(b'aabb', dtype=np.uint8)
            data = motif[(self.offset + np.arange(n)) % len(motif)].copy()
        else:
            data = self.rng.integers(ord('a'), ord('b') + 1, n, dtype=np.uint8)
        start = self.emitted
        self.offset = (self.offset + n) % 4
        self.emitted += n
        return data[:response_bytes], data[response_bytes:], dict(
            start=start, anchor=self.anchor, match=int(accepted),
            room=('periodic', 'random')[self.anchor], room_before=previous,
            moved=self.anchor != previous,
            # Each event is newly emitted, even if its byte content repeats.
            # This field is only inherited reporting metadata, not a reward.
            holdout_position_fresh=True,
            source='matched' if accepted else 'stay')

    def state_dict(self):
        return dict(anchor=self.anchor, offset=self.offset, moves=self.moves,
                    emitted=self.emitted, rng=copy.deepcopy(self.rng.bit_generator.state))

    def load_state_dict(self, state):
        self.anchor = int(state['anchor'])
        self.offset = int(state['offset'])
        self.moves = int(state['moves'])
        self.emitted = int(state['emitted'])
        self.rng.bit_generator.state = copy.deepcopy(state['rng'])


def validation_stream(seed=701, room_bytes=2048):
    """Fixed independent samples, used only in the existing validation sidecar."""
    periodic = np.resize(np.frombuffer(b'aabb', np.uint8), room_bytes + 1)
    random = np.random.default_rng(seed).integers(97, 99, room_bytes + 1, dtype=np.uint8)
    return np.concatenate((periodic, random))
