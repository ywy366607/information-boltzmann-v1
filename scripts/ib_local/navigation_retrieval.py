"""Exact words belong to retrieval, separately from continuous experience state.

The existing slot alignment is retained: no additional history, index, or byte
copy is needed. This is an exact payload store, not an n-gram probability table.
"""
from __future__ import annotations

import torch
from torch import nn


class ExactUtteranceTable(nn.Module):
    def __init__(self, capacity, max_utterance=32):
        super().__init__()
        self.register_buffer('words', torch.zeros(capacity, max_utterance, dtype=torch.uint8))
        self.register_buffer('word_count', torch.zeros(capacity, dtype=torch.long))

    @torch.no_grad()
    def write(self, slot, utterance, *, merged, keep):
        if not merged:
            self.words[slot].zero_()
            self.word_count[slot] = 0
        if utterance and keep:
            payload = torch.tensor(list(utterance[:self.words.shape[1]]),
                                   dtype=torch.uint8, device=self.words.device)
            self.words[slot].zero_()
            self.words[slot, :len(payload)] = payload
            self.word_count[slot] = len(payload)


def split_retrieval_state(module, state, prefix, local_metadata, strict,
                          missing_keys, unexpected_keys, error_msgs):
    """Losslessly admit old whole-body checkpoints without changing weights."""
    for field in ('words', 'word_count'):
        target = prefix + 'retrieval.' + field
        for old in (prefix + 'memory.' + field, prefix + 'memory.retrieval.' + field):
            if old in state:
                if target in state and not torch.equal(state[target], state[old]):
                    raise ValueError('conflicting exact retrieval payloads in checkpoint')
                state[target] = state.pop(old)
