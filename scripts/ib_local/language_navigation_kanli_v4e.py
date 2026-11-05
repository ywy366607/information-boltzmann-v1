"""V4 stage 5b: equal-length speech window, no stop head (Claude, 2026-11-04).

User, 11-04: "Does there have to be a stop? OWT itself has no stop;
pretraining has no stop token. What if the window is equal length?"
Checked: the world's source had document separators, but building the world
filtered them out (raw >= offset). The body's whole life is one stream with
no stop, and the stop head was something we added. At birth it stopped about
half the time after each byte, with no source for that. It was the dead,
borrowed link in fire's inner loop, and stage 5's three amendments were
teaching it something the world does not have.

Here, when the body generates, it says a full window of `max_utterance`
(32) bytes, the same fixed length as the old upper limit. This is symmetric
with the world's own fixed 480-byte turn and with equal pretraining windows.
Whether to speak (silence stays a choice) and what to say are unchanged.
Metal is stage 4's witness reservoir and TD return; fire's own metal from
stage 5 is not used (it judged the stop).
"""
from __future__ import annotations

import numpy as np

from scripts.ib_local.language_action import peek
from scripts.ib_local.language_navigation_kanli_v4c import WitnessEngine


class EqualEngine(WitnessEngine):
    def _speak_bytes(self, utterance_iter, record, sample):
        byte = self.pending
        p = peek(self.table, byte)
        logits, h = self.decision.step(byte, p)
        words = []
        length = self.max_utterance if sample else len(utterance_iter)
        for i in range(length):
            if sample:
                logp = logits.log_softmax(-1)
                chosen = int(torch_multinomial(logp))
                record['byte_logp'].append(float(logp[chosen]))
            else:
                chosen = int(utterance_iter[i])
            record['probs'].append(p)
            words.append(chosen)
            _, observed, _ = self.table.run_credit(np.array([byte, chosen], np.uint8), max_window=1)
            if not np.array_equal(p, observed[0]):
                raise RuntimeError('self-hearing peek differs from prequential table read')
            byte = chosen
            p = peek(self.table, byte)
            logits, h = self.decision.step(byte, p)
        record['truncated'] = False
        return bytes(words)


def torch_multinomial(logp):
    import torch
    return torch.multinomial(logp.exp(), 1)
