"""ForageWorld: intake depends on the organism's own actions (generation, layer B').

Two real text sources, each with a hidden rich/poor state that switches with
hazard `hazard` per tick. The organism hears one byte per tick from the source
it is at and receives one unit of intake with probability p_rich or p_poor:
single-tick intake is noisy evidence of the hidden state. Actions: 0 stay,
1 switch (the next tick is spent travelling: no byte, no intake). Every tick
costs `cost` from a bounded reserve; an empty reserve is absorbing (dead), and
the world keeps advancing. Supply is a world property: no prediction score is
converted into resources. Hidden states, intake draws and the ghost use separate
random streams, so every agent with the same seed lives through the same
hidden-state history whatever it does.

Stage 2 (`doc_rich` given): a source's hidden state is no longer a coin with a
hazard; it is a fixed world rule on the current document (`doc_rich[k][i]`, with
document end offsets `doc_ends[k]`), so it changes at document boundaries and
can be inferred from the text itself.

Stage 3 (`rich_mask` given): the hidden state of source k at its position i is
`rich_mask[k][i]`, a world rule on the local text (for example "a digit occurred
in the last W bytes"), so it flips on the text's own time scale.

`cut=True` gives the loop-cut world of `SelfMaintains`: the world moves on the
actions of a ghost that switches at the organism's own empirical switch rate,
independently of the world state; the organism still senses what it receives.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Tick:
    byte: int | None
    intake: int
    reserve: float
    alive: bool
    travelled: bool


class ForageWorld:
    def __init__(self, sources, *, hazard=0.01, p_rich=0.7, p_poor=0.3, cost=0.55,
                 reserve=20.0, capacity=40.0, cut=False, doc_ends=None, doc_rich=None, rich_mask=None, seed=0):
        self.src = [np.asarray(s, dtype=np.uint8) for s in sources]
        self.pos = [0] * len(self.src)
        self.seed = seed
        self.rng = np.random.default_rng(seed)  # hidden states only
        self.irng = np.random.default_rng([seed, 1])  # intake draws
        self.grng = np.random.default_rng([seed, 2])  # loop-cut ghost
        self.rich = [bool(self.rng.random() < 0.5) for _ in self.src]
        self.at, self.travelling = 0, False
        self.hazard, self.p_rich, self.p_poor = hazard, p_rich, p_poor
        self.cost, self.reserve, self.capacity = cost, reserve, capacity
        self.alive, self.cut = True, cut
        self.requested = self.ticks = 0
        self.doc_ends, self.doc_rich = doc_ends, doc_rich
        self.rich_mask = rich_mask
        if doc_rich is not None:
            self.rich = [self._doc_rich(k) for k in range(len(self.src))]
        if rich_mask is not None:
            self.rich = [bool(m[0]) for m in rich_mask]

    def _doc_rich(self, k):
        import bisect
        i = bisect.bisect_right(self.doc_ends[k], self.pos[k] % len(self.src[k]))
        return bool(self.doc_rich[k][min(i, len(self.doc_rich[k]) - 1)])

    def step(self, action: int) -> Tick:
        if self.rich_mask is not None:
            self.rich = [bool(m[p % len(m)]) for m, p in zip(self.rich_mask, self.pos)]
        elif self.doc_rich is None:
            for k in range(len(self.rich)):
                if self.rng.random() < self.hazard:
                    self.rich[k] = not self.rich[k]
        else:
            self.rich = [self._doc_rich(k) for k in range(len(self.src))]
        self.ticks += 1
        self.requested += int(action == 1)
        if self.cut:  # ghost with the organism's own switch rate, blind to the world
            action = int(self.grng.random() < self.requested / self.ticks)
        if action == 1 and not self.travelling:
            self.at = (self.at + 1) % len(self.src)
            self.travelling = True
        travelled = self.travelling
        byte, intake = None, 0
        if self.travelling:
            self.travelling = False
        else:
            k = self.at
            byte = int(self.src[k][self.pos[k] % len(self.src[k])])
            if self.alive:
                p = self.p_rich if self.rich[k] else self.p_poor
                intake = int(self.irng.random() < p)
        if self.alive:
            self.reserve = min(self.capacity, self.reserve + intake) - self.cost
            if self.reserve <= 0:
                self.reserve, self.alive = 0.0, False
        self.pos = [p + 1 for p in self.pos]  # every source runs on; the world does not wait
        return Tick(byte, intake, self.reserve, self.alive, travelled)
