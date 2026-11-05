"""Carriers compared by the language gate through one interface.

Every candidate here is an existing idea reduced to the same byte protocol, so
the gate can compare carriers instead of checking one chosen design.  Nothing
in this file uses temporal backpropagation.  `reference_only` candidates are
yardsticks and cannot be admitted.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ib_local.gated_deltanet_2 import GatedDeltaNet2Layer, GatedDeltaNet2LM


def _copy_learning_state(fields: dict) -> dict:
    """Deepcopy aliased model/optimizer state, including unstepped gradients."""
    copied = copy.deepcopy(fields)
    for name, value in fields.items():
        if isinstance(value, torch.optim.Optimizer):
            for old_group, new_group in zip(value.param_groups, copied[name].param_groups):
                for old, new in zip(old_group['params'], new_group['params']):
                    new.grad = None if old.grad is None else old.grad.detach().clone()
    return copied


class Candidate:
    """One continuous learner.  `predict` must not depend on the coming byte."""

    name = "candidate"
    reference_only = False
    fixed: tuple[str, ...] = ()

    def predict(self) -> torch.Tensor:
        """Return log-probabilities over the 256 bytes."""
        raise NotImplementedError

    def log_prob(self, byte: int) -> float:
        return float(self.predict()[byte])

    def observe(self, byte: int) -> None:
        raise NotImplementedError

    def advance(self, elapsed: float) -> None:
        """Apply real elapsed time.  The default body ignores time."""

    def state_vector(self) -> torch.Tensor:
        """A bounded reading of the fast state, used for held-out decoding."""
        raise NotImplementedError

    def state_bytes(self) -> int:
        """All persistent state and learned numbers, in bytes."""
        raise NotImplementedError

    def capacity_bytes(self) -> int:
        """The most `state_bytes` can ever reach.  Fixed arrays are their own bound."""
        return self.state_bytes()

    def load_shares(self) -> torch.Tensor | None:
        """Share of writes per region, or None for a body without regions."""
        return None

    def clone(self) -> dict:
        # One deepcopy call keeps shared references intact, such as an
        # optimizer's reference to its network's parameters.
        return _copy_learning_state({key: value for key, value in self.__dict__.items()
                                     if key not in self.fixed})

    def restore(self, saved: dict) -> None:
        self.__dict__.update(_copy_learning_state(saved))


class OnlineHead:
    """G14 affine byte head: the arrived-minus-predicted output times the feature."""

    def __init__(self, width: int, step: float) -> None:
        self.weight = torch.zeros(width, 256)
        self.bias = torch.zeros(256)
        self.step = step

    def log_probs(self, feature: torch.Tensor) -> torch.Tensor:
        return torch.log_softmax(feature @ self.weight + self.bias, dim=0)

    def update(self, feature: torch.Tensor, byte: int) -> None:
        innovation = -torch.softmax(feature @ self.weight + self.bias, dim=0)
        innovation[byte] += 1.0
        self.weight += self.step * torch.outer(feature, innovation)
        self.bias += self.step * innovation


class FadingContext:
    """An untrained contraction on arrivals with one fixed leak."""

    def __init__(self, width: int, leak: float, generator: torch.Generator, gain: float = 0.95) -> None:
        self.recurrent = torch.randn(width, width, generator=generator) / width ** 0.5 * gain
        self.encoder = torch.randn(256, width, generator=generator) * 0.6
        self.leak = leak
        self.state = torch.zeros(width)

    def step(self, byte: int) -> None:
        drive = torch.tanh(self.recurrent @ self.state + self.encoder[byte])
        self.state = (1.0 - self.leak) * self.state + self.leak * drive


class GDNLifeBytes(Candidate):
    """`GDNLife`: the arrival is written at its own address and read back next tick."""

    name = "gdn_life"
    fixed = ("address", "value", "weights")

    def __init__(self, budget_kib: int, seed: int = 3) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.depth = 64
        slots = max(64, budget_kib * 1024 // (4 * (self.depth + 1)))
        self.slots = slots
        self.address = torch.randint(0, slots, (256, 8), generator=generator)
        self.weights = torch.full((8,), 1.0 / 8)
        self.value = torch.randn(256, self.depth, generator=generator) / self.depth ** 0.5 * 2
        self.natural = torch.zeros(slots, self.depth)
        self.precision = torch.full((slots, 1), 0.127)
        self.head = OnlineHead(self.depth, 0.05)
        self.last = 0
        self.writes = torch.zeros(64)

    def feature(self) -> torch.Tensor:
        read = self.address[self.last]
        # Unscaled: a larger feature makes the online head's fixed step diverge.
        return (self.natural[read] / self.precision[read]).mean(0)

    def predict(self) -> torch.Tensor:
        return self.head.log_probs(self.feature())

    def observe(self, byte: int) -> None:
        self.head.update(self.feature(), byte)
        write = self.address[byte]
        self.natural[write] += self.value[byte]
        self.precision[write] += 1.0
        self.writes[write % 64] += 1.0
        self.last = byte

    def advance(self, elapsed: float) -> None:
        released = self.precision / (1.0 + elapsed * 0.05 * self.precision)
        self.natural = self.natural * released / self.precision
        self.precision = released

    def state_vector(self) -> torch.Tensor:
        return (self.natural[:256] / self.precision[:256]).flatten()

    def state_bytes(self) -> int:
        return 4 * (self.natural.numel() + self.precision.numel() + self.head.weight.numel() + 256)

    def load_shares(self) -> torch.Tensor:
        return self.writes / self.writes.sum().clamp_min(1.0)


class ColumnMatrix(Candidate):
    """The 10-28 design: M columns of r x r, tied read and delta write on one key."""

    name = "column_matrix"
    fixed = ("value", "projection", "key_projection", "keys")

    def __init__(self, budget_kib: int, seed: int = 3, *, active: int = 4, step: float = 0.5) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.columns = 64
        self.width = max(8, int((budget_kib * 1024 / (4 * self.columns)) ** 0.5))
        self.context = FadingContext(1024, 0.5, generator)
        self.projection = torch.randn(1024, 64, generator=generator)
        self.key_projection = torch.randn(1024, self.width, generator=generator)
        keys = torch.randn(self.columns, 64, generator=generator)
        self.keys = keys / keys.norm(dim=1, keepdim=True)
        self.value = torch.randn(256, self.width, generator=generator) / self.width ** 0.5
        self.memory = torch.zeros(self.columns, self.width, self.width)
        self.head = OnlineHead(self.width, 0.05)
        self.active, self.step = active, step
        self.query_mean = torch.zeros(64)
        self.key_mean = torch.zeros(self.width)
        self.seen = 0
        self.writes = torch.zeros(self.columns)

    def route(self) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.context.state @ self.projection - self.query_mean
        key = self.context.state @ self.key_projection - self.key_mean
        return (query @ self.keys.T).topk(self.active).indices, key / key.norm().clamp_min(1e-6)

    def predict(self) -> torch.Tensor:
        selected, key = self.route()
        return self.head.log_probs((self.memory[selected] @ key).mean(0) * 2)

    def observe(self, byte: int) -> None:
        selected, key = self.route()
        prediction = self.memory[selected] @ key
        self.head.update(prediction.mean(0) * 2, byte)
        self.memory[selected] += self.step * torch.einsum(
            "ci,j->cij", self.value[byte] - prediction, key)
        self.writes[selected] += 1.0
        self.seen += 1
        rate = 1.0 / min(self.seen, 5000)
        self.query_mean += rate * (self.context.state @ self.projection - self.query_mean)
        self.key_mean += rate * (self.context.state @ self.key_projection - self.key_mean)
        self.context.step(byte)

    def advance(self, elapsed: float) -> None:
        self.memory *= math.exp(-0.01 * elapsed)

    def state_vector(self) -> torch.Tensor:
        return self.context.state.clone()

    def state_bytes(self) -> int:
        return 4 * (self.memory.numel() + self.head.weight.numel() + 256 + 1024 + 64 + self.width)

    def load_shares(self) -> torch.Tensor:
        return self.writes / self.writes.sum().clamp_min(1.0)


class FadingCells(Candidate):
    """Fading contexts at several depths address conjugate successor counts.

    A cell's count vector is the Dirichlet natural parameter of what followed
    its context.  Reads run from the shallowest context to the deepest, so a
    deep cell with evidence dominates and an empty one falls back.  There is no
    gradient anywhere in this body.
    """

    name = "fading_cells"
    fixed = ("planes",)

    # Each scale is (leak, recurrent gain).
    scales = ((0.95, 0.95), (0.8, 0.95), (0.6, 0.95), (0.4, 0.95))

    def __init__(self, budget_kib: int, seed: int = 3, *, symbols: int = 32) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.tables_per_scale = 2
        count = len(self.scales) * self.tables_per_scale
        # Sized for the observed mean cell, not the rare full one; O03 measures the result.
        # Scales without recurrence stay low order and occupy few cells.
        deep = sum(gain > 0 for _, gain in self.scales) * self.tables_per_scale
        self.bits = max(6, int(math.log2(budget_kib * 1024 / ((deep or count) * 16))))
        self.symbols = symbols
        self.contexts = [FadingContext(512, leak, generator, gain) for leak, gain in self.scales]
        self.planes = [torch.randn(512, self.bits, generator=generator) for _ in range(count)]
        self.means = [torch.zeros(512) for _ in self.scales]
        self.cells: list[dict[int, dict[int, float]]] = [{} for _ in range(count)]
        self.order0 = np.ones(256)
        self.seen = 0
        self.writes = torch.zeros(64)
        self.powers = 1 << np.arange(self.bits)

    def addresses(self) -> list[tuple[int, int]]:
        out = []
        for scale, context in enumerate(self.contexts):
            centred = context.state - self.means[scale]
            for table in range(self.tables_per_scale):
                index = scale * self.tables_per_scale + table
                signs = (centred @ self.planes[index] > 0).numpy()
                out.append((index, int(signs @ self.powers)))
        return out

    def log_prob(self, byte: int) -> float:
        probability = self.order0[byte] / self.order0.sum()
        for table, cell in self.addresses():
            counts = self.cells[table].get(cell)
            if counts:
                total, kinds = sum(counts.values()), len(counts)
                probability = (counts.get(byte, 0.0) + kinds * probability) / (total + kinds)
        return math.log(probability)

    def predict(self) -> torch.Tensor:
        probability = self.order0 / self.order0.sum()
        for table, cell in self.addresses():
            counts = self.cells[table].get(cell)
            if counts:
                vector = np.zeros(256)
                for key, value in counts.items():
                    vector[key] = value
                kinds = len(counts)
                probability = (vector + kinds * probability) / (vector.sum() + kinds)
        return torch.as_tensor(np.log(probability), dtype=torch.float32)

    def observe(self, byte: int) -> None:
        for table, cell in self.addresses():
            counts = self.cells[table].setdefault(cell, {})
            if byte not in counts and len(counts) >= self.symbols:
                del counts[min(counts, key=counts.get)]
            counts[byte] = counts.get(byte, 0.0) + 1.0
            self.writes[cell % 64] += 1.0
        self.order0[byte] += 1.0
        self.seen += 1
        rate = 1.0 / min(self.seen, 5000)
        for scale, context in enumerate(self.contexts):
            self.means[scale] = self.means[scale] + rate * (context.state - self.means[scale])
            context.step(byte)

    def advance(self, elapsed: float) -> None:
        factor = math.exp(-0.01 * elapsed)
        for table in self.cells:
            for counts in table.values():
                for key in counts:
                    counts[key] *= factor

    def state_vector(self) -> torch.Tensor:
        return torch.cat([context.state for context in self.contexts])

    def state_bytes(self) -> int:
        entries = sum(len(counts) for table in self.cells for counts in table.values())
        cells = sum(len(table) for table in self.cells)
        return 2 * entries + 2 * cells + 512 + 4 * 512 * len(self.scales) * 2

    def capacity_bytes(self) -> int:
        return (len(self.cells) * (1 << self.bits) * (2 + 2 * self.symbols)
                + 512 + 4 * 512 * len(self.scales) * 2)

    def load_shares(self) -> torch.Tensor:
        return self.writes / self.writes.sum().clamp_min(1.0)


class FadingGraded(FadingCells):
    """`FadingCells` whose scales have the depths their leaks claim.

    Recurrence stretches every scale to about the same depth, so the base body
    has no low-order cell to fall back on.  Without it the state is a geometric
    sum of arrivals and `1 - leak` alone sets how fast older bytes fade.
    """

    name = "fading_graded"
    scales = ((0.98, 0.0), (0.8, 0.0), (0.55, 0.0), (0.3, 0.0))


class PooledRead:
    """A log-opinion pool over a body's cells, replacing the cascade.

    Each occupied cell states evidence against order-0.  How far a cell is
    trusted depends on its table and how much it has seen, and that trust follows
    only the gradient of the current byte's log loss.  With `gated`, trust also
    depends on how many cells spoke, so shallow cells can take over when an
    unfamiliar context leaves the deep ones empty.
    """

    gated = False
    grades: tuple[int, ...] = ()

    def __init__(self, budget_kib: int, seed: int = 3, *, rate: float = 0.005) -> None:
        super().__init__(budget_kib, seed)
        self.levels = 8
        gates = len(self.cells) + 1 if self.gated else 1
        self.trust = np.full((gates, len(self.cells), self.levels) + self.grades, 0.25)
        self.rate = rate
        self.opinion: tuple | None = None

    def entry(self, table: int, cell: int, total: float) -> tuple[int, ...]:
        return table, min(int(math.log2(total + 1.0)), self.levels - 1)

    def speakers(self, entries: list[tuple[int, ...]]) -> int:
        return len(entries)

    def pooled(self) -> tuple:
        """Log probabilities, the evidence rows and the trust entries they used."""
        if self.opinion is None:
            base = self.order0 / self.order0.sum()
            rows, entries = [], []
            for table, cell in self.addresses():
                counts = self.cells[table].get(cell)
                if counts:
                    vector = np.zeros(256)
                    for key, value in counts.items():
                        vector[key] = value
                    total, kinds = vector.sum(), len(counts)
                    row = np.log((vector + kinds * base) / (total + kinds) / base)
                    rows.append(row)
                    entries.append(self.entry(table, cell, total))
            gate = self.speakers(entries) if self.gated else 0
            entries = [(gate, *entry) for entry in entries]
            logits = np.log(base)
            for row, entry in zip(rows, entries):
                logits = logits + self.trust[entry] * row
            logits = logits - logits.max()
            total = np.exp(logits).sum()
            self.opinion = (logits - math.log(total), rows, entries)
        return self.opinion

    def log_prob(self, byte: int) -> float:
        return float(self.pooled()[0][byte])

    def predict(self) -> torch.Tensor:
        return torch.as_tensor(self.pooled()[0], dtype=torch.float32)

    def observe(self, byte: int) -> None:
        logits, rows, entries = self.pooled()
        probability = np.exp(logits)
        for row, entry in zip(rows, entries):
            self.trust[entry] += self.rate * (row[byte] - float(probability @ row))
        super().observe(byte)
        self.opinion = None

    def advance(self, elapsed: float) -> None:
        super().advance(elapsed)
        self.opinion = None

    def state_bytes(self) -> int:
        return super().state_bytes() + 4 * self.trust.size

    def capacity_bytes(self) -> int:
        return super().capacity_bytes() + 4 * self.trust.size


class FadingPooled(PooledRead, FadingGraded):
    name = "fading_pooled"


class FadingCellsPooled(PooledRead, FadingCells):
    name = "fading_cells_pooled"


class FadingMixed(PooledRead, FadingCells):
    """Shallow scales without recurrence for low orders, recurrent ones for reach."""

    name = "fading_mixed"
    scales = ((0.98, 0.0), (0.8, 0.0), (0.55, 0.0), (0.3, 0.0), (0.6, 0.95), (0.4, 0.95))


class FadingGated(FadingPooled):
    name = "fading_gated"
    gated = True


class FadingCellsGated(FadingCellsPooled):
    name = "fading_cells_gated"
    gated = True


class FadingCellsLow(FadingCellsPooled):
    """The kept body with two low-order scales to fall back on."""

    name = "fading_cells_low"
    scales = FadingCells.scales + ((0.98, 0.0), (0.8, 0.0))


class FadingCellsLowGated(FadingCellsLow):
    name = "fading_cells_low_gated"
    gated = True


class FadingCellsMatched(FadingCellsLowGated):
    """A cell also keeps the signature of the context that last wrote it.

    A hashed cell always answers, even for a context it has never seen.  The
    distance between the signature and the query says how far to believe it:
    an exact match acts as a lookup, a near match lends what a similar context
    saw, and a far one is someone else's evidence.
    """

    name = "fading_cells_matched"
    fixed = ("planes", "tag_planes")
    grades = (4,)

    def __init__(self, budget_kib: int, seed: int = 3, *, rate: float = 0.005) -> None:
        super().__init__(budget_kib, seed, rate=rate)
        generator = torch.Generator().manual_seed(seed + 1)
        self.tag_planes = [torch.randn(512, 16, generator=generator) for _ in self.cells]
        self.tags: list[dict[int, int]] = [{} for _ in self.cells]
        self.query: list[int] | None = None

    def queries(self) -> list[int]:
        if self.query is None:
            powers, self.query = 1 << np.arange(16), []
            for scale, context in enumerate(self.contexts):
                centred = context.state - self.means[scale]
                for table in range(self.tables_per_scale):
                    index = scale * self.tables_per_scale + table
                    self.query.append(int((centred @ self.tag_planes[index] > 0).numpy() @ powers))
        return self.query

    def entry(self, table: int, cell: int, total: float) -> tuple[int, ...]:
        distance = bin(self.tags[table].get(cell, 0) ^ self.queries()[table]).count("1")
        grade = 0 if distance == 0 else 1 if distance <= 2 else 2 if distance <= 5 else 3
        return (*super().entry(table, cell, total), grade)

    def speakers(self, entries: list[tuple[int, ...]]) -> int:
        return sum(entry[-1] <= 1 for entry in entries)

    def observe(self, byte: int) -> None:
        query, cells = self.queries(), self.addresses()
        super().observe(byte)
        for table, cell in cells:
            self.tags[table][cell] = query[table]
        self.query = None

    def state_bytes(self) -> int:
        return super().state_bytes() + 2 * sum(len(table) for table in self.tags)

    def capacity_bytes(self) -> int:
        return super().capacity_bytes() + 2 * len(self.cells) * (1 << self.bits)


class WindowNetwork(Candidate):
    """A shift register read by a small network with a one-tick gradient.

    The register has no parameters, so there is nothing to backpropagate
    through time.  This stands in for attention-style carriers.
    """

    name = "window_network"
    fixed = ()

    def __init__(self, budget_kib: int, seed: int = 3, *, window: int = 12, embedding: int = 16) -> None:
        torch.manual_seed(seed)
        # Adam keeps two more numbers per parameter, all counted against the budget.
        budget = budget_kib * 1024 // 12

        def size(width: int) -> int:
            return (257 * embedding + (window * embedding + 1) * width
                    + (width + 1) * width + (width + 1) * 256)

        hidden = 32
        while size(hidden * 2) <= budget:
            hidden *= 2
        self.network = nn.ModuleDict(dict(
            embed=nn.Embedding(257, embedding), first=nn.Linear(window * embedding, hidden),
            second=nn.Linear(hidden, hidden), last=nn.Linear(hidden, 256),
        ))
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.002)
        self.register = torch.full((1, window), 256, dtype=torch.long)

    def logits(self) -> torch.Tensor:
        net = self.network
        hidden = F.gelu(net["first"](net["embed"](self.register).flatten(1)))
        return net["last"](hidden + F.gelu(net["second"](hidden)))[0]

    def predict(self) -> torch.Tensor:
        with torch.no_grad():
            return torch.log_softmax(self.logits(), dim=0)

    def observe(self, byte: int) -> None:
        loss = F.cross_entropy(self.logits().unsqueeze(0), torch.tensor([byte]))
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.register = torch.cat((self.register[:, 1:], torch.tensor([[byte]])), dim=1)

    def state_vector(self) -> torch.Tensor:
        with torch.no_grad():
            return self.network["embed"](self.register).flatten()

    def state_bytes(self) -> int:
        return 4 * sum(value.numel() for value in self.network.parameters()) * 3 + 8 * self.register.numel()


class GDN2(Candidate):
    """The official Gated DeltaNet-2 layers on bytes, learning from the current byte.

    The state enters each tick as a constant, so credit reaches the current
    write, read and gates but never an earlier tick.
    """

    name = "gdn2"
    layers, heads, step = 2, 4, 0.002
    expansion = 0

    def __init__(self, budget_kib: int, seed: int = 3) -> None:
        torch.manual_seed(seed)
        layers, heads = self.layers, self.heads

        def size(width: int) -> int:
            # Adam keeps two more numbers per parameter, all counted against the budget.
            numbers = 257 * width + layers * (8 * width * width + 4 * width)
            numbers += layers * self.expansion * (2 * width * width + 2 * width)
            return 12 * numbers + 4 * layers * width * width // heads

        width = heads * 2
        while size(width + heads) <= budget_kib * 1024:
            width += heads
        self.model = GatedDeltaNet2LM(256, width, layers, heads)
        if self.expansion:
            self.model.feedforward = nn.ModuleList(
                nn.Sequential(nn.RMSNorm(width), nn.Linear(width, self.expansion * width), nn.GELU(),
                              nn.Linear(self.expansion * width, width)) for _ in range(layers))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.step)
        self.state = self.model.initial_state(1)
        self.last = 32
        self.read()

    def read(self) -> None:
        """Prepare the coming prediction from the last byte and the held state."""
        model = self.model
        hidden = model.source.embedding(torch.tensor([[self.last]]))
        written = []
        for index, layer in enumerate(model.layers):
            hidden, state, _ = layer(hidden, self.state[index])
            if self.expansion:
                hidden = hidden + model.feedforward[index](hidden)
            written.append(state)
        self.pending = (model.decoder(model.final_norm(hidden))[0, 0], torch.stack(written))

    def predict(self) -> torch.Tensor:
        return torch.log_softmax(self.pending[0].detach(), dim=0)

    def learn(self, loss: torch.Tensor) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

    def observe(self, byte: int) -> None:
        logits, state = self.pending
        self.learn(F.cross_entropy(logits.unsqueeze(0), torch.tensor([byte])))
        self.state = state.detach()
        self.last = byte
        self.read()

    def state_vector(self) -> torch.Tensor:
        return self.pending[1].detach().flatten()

    def state_bytes(self) -> int:
        return 12 * sum(value.numel() for value in self.model.parameters()) + 4 * self.state.numel() + 8

    def clone(self) -> dict:
        pending = self.__dict__.pop("pending")
        try:
            saved = super().clone()
        finally:
            self.pending = pending
        return saved

    def restore(self, saved: dict) -> None:
        super().restore(saved)
        self.read()


class GDN2Truncated(GDN2):
    """The same body with truncated temporal backpropagation.  A yardstick, never admissible."""

    name = "gdn2_tbptt"
    reference_only = True
    span = 64

    def __init__(self, budget_kib: int, seed: int = 3) -> None:
        self.loss, self.held = 0.0, 0
        super().__init__(budget_kib, seed)

    def observe(self, byte: int) -> None:
        logits, state = self.pending
        self.loss = self.loss + F.cross_entropy(logits.unsqueeze(0), torch.tensor([byte]))
        self.held += 1
        self.state = state
        if self.held == self.span:
            self.learn(self.loss / self.span)
            self.loss, self.held = 0.0, 0
            self.state = state.detach()
        self.last = byte
        self.read()

    def clone(self) -> dict:
        loss, state = self.loss, self.state
        self.loss, self.held, self.state = 0.0, 0, state.detach()
        saved = super().clone()
        self.loss, self.state = loss, state
        return saved


class GDN2Batched(GDN2):
    """One-tick credit, averaged over the same span the truncated yardstick uses."""

    name = "gdn2_batched"
    span = 64

    def __init__(self, budget_kib: int, seed: int = 3) -> None:
        self.held = 0
        super().__init__(budget_kib, seed)

    def observe(self, byte: int) -> None:
        logits, state = self.pending
        (F.cross_entropy(logits.unsqueeze(0), torch.tensor([byte])) / self.span).backward()
        self.held += 1
        if self.held == self.span:
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.held = 0
        self.state = state.detach()
        self.last = byte
        self.read()


class GDN2MlpBatched(GDN2Batched):
    """Each layer followed by the feed-forward block GDN layers ship with in language models."""

    name = "gdn2_mlp_batched"
    expansion = 2


class GDN2WriteCredit(GDN2Batched):
    """One-tick credit plus credit for past writes, taken from the read.

    A read of past writes is `r = W h`, and the exact gradient `e hᵀ` names a
    history nobody holds.  `e rᵀ W` needs only this tick's numbers and never
    opposes it, so no history, trace or temporal gradient is kept.
    """

    name = "gdn2_write"
    values, keys = True, False

    def __init__(self, budget_kib: int, seed: int = 3) -> None:
        super().__init__(budget_kib, seed)
        self.credit = [(torch.zeros_like(layer.v_proj.weight), torch.zeros_like(layer.k_proj.weight))
                       for layer in self.model.layers]

    def read(self) -> None:
        """The layer's own equations, keeping what the read saw of past writes."""
        model = self.model
        hidden = model.source.embedding(torch.tensor([[self.last]]))
        written, records = [], []
        for index, layer in enumerate(model.layers):
            norm = layer.norm(hidden)[:, 0]
            shape = (1, layer.heads, layer.dk)
            query = F.normalize(layer.q_proj(norm).view(shape), dim=-1)
            key = F.normalize(layer.k_proj(norm).view(shape), dim=-1)
            value = layer.v_proj(norm).view(shape)
            decay = torch.sigmoid(layer.decay_proj(norm)).view(shape)
            erase = torch.sigmoid(layer.erase_proj(norm)).view(shape)
            write = torch.sigmoid(layer.write_proj(norm)).view(shape)
            held = decay.unsqueeze(-1) * self.state[index]
            retrieved = torch.einsum("bhk,bhkd->bhd", erase * key, held)
            prior = held - torch.einsum("bhk,bhd->bhkd", key, retrieved)
            state = prior + torch.einsum("bhk,bhd->bhkd", key, write * value)
            read = torch.einsum("bhkd,bhk->bhd", state, query)
            read.retain_grad()
            past = torch.einsum("bhkd,bhk->bhd", prior, query)
            gated = read.reshape(1, layer.d) * F.silu(layer.g_proj(norm))
            hidden = hidden + layer.o_proj(gated).unsqueeze(1)
            if self.expansion:
                hidden = hidden + model.feedforward[index](hidden)
            written.append(state)
            carried = self.settle(index, key, decay, erase, write, norm)
            records.append((read, past.detach()[0], prior.detach()[0], query.detach()[0], carried))
        logits = model.decoder(model.final_norm(hidden))[0, 0]
        self.pending = (logits, torch.stack(written), records)

    def settle(self, index: int, key: torch.Tensor, decay: torch.Tensor, erase: torch.Tensor,
               write: torch.Tensor, norm: torch.Tensor) -> tuple | None:
        return None

    def value_credit(self, layer: nn.Module, slope: torch.Tensor, past: torch.Tensor, query: torch.Tensor,
                     carried: tuple | None) -> torch.Tensor:
        weight = layer.v_proj.weight.view(layer.heads, layer.dk, -1)
        return torch.einsum("hd,he,hei->hdi", slope, past, weight)

    def observe(self, byte: int) -> None:
        logits, state, records = self.pending
        (F.cross_entropy(logits.unsqueeze(0), torch.tensor([byte])) / self.span).backward()
        with torch.no_grad():
            for index, (layer, record, (value, key)) in enumerate(zip(self.model.layers, records, self.credit)):
                read, past, prior, query, carried = record
                slope = read.grad[0]
                if carried is not None:
                    self.trace[index] = carried[1]
                if self.values:
                    value += self.value_credit(layer, slope, past, query, carried).flatten(0, 1)
                if self.keys:
                    weight = layer.k_proj.weight.view(layer.heads, layer.dk, -1)
                    held = torch.einsum("hkd,hd->hk", prior, slope)
                    key += torch.einsum("hk,hl,hli->hki", query, held, weight).flatten(0, 1)
        self.held += 1
        if self.held == self.span:
            for layer, (value, key) in zip(self.model.layers, self.credit):
                layer.v_proj.weight.grad += value
                layer.k_proj.weight.grad += key
                value.zero_()
                key.zero_()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.held = 0
        self.state = state.detach()
        self.last = byte
        self.read()


class GDN2MlpWriteCredit(GDN2WriteCredit):
    """Write credit with the feed-forward block after each read."""

    name = "gdn2_mlp_write"
    expansion = 2


class GDN2MlpWriteKeyCredit(GDN2MlpWriteCredit):
    """The same with the key rule, which aligned far less than the value rule."""

    name = "gdn2_mlp_write_keys"
    keys = True


class GDN2TraceCredit(GDN2WriteCredit):
    """Value credit from a forward eligibility trace, the memory with inputs in place of values.

    The trace obeys the memory's own decay and erase, so `q` reads which past
    inputs the current read holds and `e (q T)` credits them.  It keeps one number
    per key and input, shared by the output weights of that input, and no history.
    """

    name = "gdn2_trace"

    def __init__(self, budget_kib: int, seed: int = 3) -> None:
        self.trace: dict[int, torch.Tensor] = {}
        super().__init__(budget_kib, seed)

    def settle(self, index: int, key: torch.Tensor, decay: torch.Tensor, erase: torch.Tensor,
               write: torch.Tensor, norm: torch.Tensor) -> tuple:
        key, decay, erase, write, norm = (value.detach()[0] for value in (key, decay, erase, write, norm))
        trace = self.trace.get(index)
        if trace is None:
            trace = torch.zeros(*key.shape, norm.numel())
        held = decay.unsqueeze(-1) * trace
        retrieved = torch.einsum("hk,hki->hi", erase * key, held)
        prior = held - torch.einsum("hk,hi->hki", key, retrieved)
        return prior, prior + torch.einsum("hk,i->hki", key * write.mean(-1, keepdim=True), norm)

    def value_credit(self, layer: nn.Module, slope: torch.Tensor, past: torch.Tensor, query: torch.Tensor,
                     carried: tuple) -> torch.Tensor:
        return torch.einsum("hd,hk,hki->hdi", slope, query, carried[0])

    def state_bytes(self) -> int:
        return super().state_bytes() + 4 * sum(trace.numel() for trace in self.trace.values())


class GDN2TraceKeyCredit(GDN2TraceCredit):
    """Trace credit for the values and the read rule for the keys."""

    name = "gdn2_trace_keys"
    keys = True


class GDN2MlpTruncated(GDN2Truncated):
    """The feed-forward layers with truncated temporal backpropagation.  A yardstick, never admissible."""

    name = "gdn2_mlp_tbptt"
    expansion = 2


class ColumnLayer(GatedDeltaNet2Layer):
    """One GDN-2 layer whose S gains a column axis: r x r becomes M x r x r.

    The input's read key selects a few columns by learned column keys.  Only
    those are decayed, erased, written and read, so the rest keep what they hold.
    """

    def __init__(self, width: int, heads: int, columns: int, active: int) -> None:
        super().__init__(width, heads)
        self.column_keys = nn.Parameter(torch.randn(heads, columns, self.dk))
        self.active = active

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        count, heads, size = x.shape[0], self.heads, self.dk
        norm = self.norm(x)[:, 0]
        q = F.normalize(self.q_proj(norm).view(count, heads, size), dim=-1)
        k = F.normalize(self.k_proj(norm).view(count, heads, size), dim=-1)
        v = self.v_proj(norm).view(count, heads, size)
        decay = torch.sigmoid(self.decay_proj(norm)).view(count, heads, size)
        erase = torch.sigmoid(self.erase_proj(norm)).view(count, heads, size)
        write = torch.sigmoid(self.write_proj(norm)).view(count, heads, size)
        scores = torch.einsum("bhk,hmk->bhm", q, F.normalize(self.column_keys, dim=-1))
        top, chosen = scores.topk(self.active, dim=-1)
        shares = torch.softmax(top * 8.0, dim=-1)
        index = chosen[..., None, None].expand(-1, -1, -1, size, size)
        held = decay[:, :, None, :, None] * state.gather(2, index)
        retrieved = torch.einsum("bhk,bhakd->bhad", erase * k, held)
        written = (held - torch.einsum("bhk,bhad->bhakd", k, retrieved)
                   + torch.einsum("bha,bhk,bhd->bhakd", shares, k, write * v))
        read = torch.einsum("bha,bhakd,bhk->bhd", shares, written, q)
        output = self.o_proj(read.reshape(count, 1, -1) * F.silu(self.g_proj(norm)[:, None]))
        return x + output, state.scatter(2, index, written)


class GDN2Column(GDN2):
    """The real column medium: GDN-2 with S upgraded to M x r x r."""

    name = "gdn2_column"
    columns, active = 64, 4

    def __init__(self, budget_kib: int, seed: int = 3) -> None:
        torch.manual_seed(seed)
        layers, heads, columns = self.layers, self.heads, self.columns

        def size(width: int) -> int:
            numbers = 257 * width + layers * (8 * width * width + 4 * width + columns * width)
            return 12 * numbers + 4 * layers * columns * width * width // heads

        width = heads * 2
        while size(width + heads) <= budget_kib * 1024:
            width += heads
        self.model = GatedDeltaNet2LM(256, width, 0, heads)
        self.model.layers = nn.ModuleList(ColumnLayer(width, heads, columns, self.active) for _ in range(layers))
        self.model.num_layers = layers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.step)
        self.state = torch.zeros(layers, 1, heads, columns, width // heads, width // heads)
        self.last = 32
        self.read()

    def read(self) -> None:
        model = self.model
        hidden = model.source.embedding(torch.tensor([[self.last]]))
        written = []
        for index, layer in enumerate(model.layers):
            hidden, state = layer(hidden, self.state[index])
            written.append(state)
        self.pending = (model.decoder(model.final_norm(hidden))[0, 0], torch.stack(written))


class CausalAttention(Candidate):
    """A real Transformer over a register of recent bytes, with a one-tick gradient.

    The register has no parameters, so nothing is backpropagated through time.
    """

    name = "transformer"
    window, layers, heads, step = 32, 2, 4, 0.002

    def __init__(self, budget_kib: int, seed: int = 3) -> None:
        torch.manual_seed(seed)

        def size(width: int) -> int:
            numbers = (256 + self.window + 1) * width + self.layers * (10 * width * width + 8 * width)
            return 12 * numbers + 8 * self.window

        width = self.heads * 2
        while size(width + self.heads) <= budget_kib * 1024:
            width += self.heads
        self.embed = nn.Embedding(257, width)
        self.position = nn.Parameter(torch.zeros(self.window, width))
        layer = nn.TransformerEncoderLayer(width, self.heads, width * 2, dropout=0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, self.layers, enable_nested_tensor=False)
        self.last = nn.Linear(width, 256)
        self.parts = nn.ModuleList((self.embed, self.encoder, self.last))
        self.optimizer = torch.optim.Adam([self.position, *self.parts.parameters()], lr=self.step)
        self.register = torch.full((1, self.window), 256, dtype=torch.long)
        self.mask = nn.Transformer.generate_square_subsequent_mask(self.window)
        self.prepared = None

    def logits(self) -> torch.Tensor:
        hidden = self.encoder(self.embed(self.register) + self.position, mask=self.mask, is_causal=True)
        return self.last(hidden[0, -1])

    def predict(self) -> torch.Tensor:
        if self.prepared is None:
            with torch.no_grad():
                self.prepared = torch.log_softmax(self.logits(), dim=0)
        return self.prepared

    def observe(self, byte: int) -> None:
        loss = F.cross_entropy(self.logits().unsqueeze(0), torch.tensor([byte]))
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.parts.parameters(), 1.0)
        self.optimizer.step()
        self.register = torch.cat((self.register[:, 1:], torch.tensor([[byte]])), dim=1)
        self.prepared = None

    def state_vector(self) -> torch.Tensor:
        with torch.no_grad():
            return self.embed(self.register).flatten()

    def state_bytes(self) -> int:
        numbers = self.position.numel() + sum(value.numel() for value in self.parts.parameters())
        return 12 * numbers + 8 * self.register.numel()


class CausalAttentionBatched(CausalAttention):
    """The same Transformer, with gradients averaged over a span."""

    name = "transformer_batched"
    span = 64

    def __init__(self, budget_kib: int, seed: int = 3) -> None:
        super().__init__(budget_kib, seed)
        self.held = 0

    def observe(self, byte: int) -> None:
        (F.cross_entropy(self.logits().unsqueeze(0), torch.tensor([byte])) / self.span).backward()
        self.held += 1
        if self.held == self.span:
            nn.utils.clip_grad_norm_(self.parts.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.held = 0
        self.register = torch.cat((self.register[:, 1:], torch.tensor([[byte]])), dim=1)
        self.prepared = None


class ContextTable(Candidate):
    """Finite PPM-style context memory.  A yardstick, never admissible."""

    name = "context_table"
    reference_only = True
    fixed = ("embed",)

    def __init__(self, budget_kib: int, seed: int = 3, *, order: int = 5) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.bits = max(4, int(math.log2(budget_kib * 1024 / 2.6)))
        self.slots: list = [None] * (1 << self.bits)
        self.order = order
        self.history: list[int] = []
        self.embed = torch.randn(256, 16, generator=generator)

    def depths(self) -> range:
        return range(0, min(self.order, len(self.history)) + 1)

    def digest(self, depth: int) -> tuple[int, int]:
        """The slot and check bits of the suffix of this length."""
        value = depth + 1
        for byte in self.history[len(self.history) - depth:]:
            value = (value * 0x100000001B3 + byte + 1) & 0xFFFFFFFFFFFFFFFF
        value ^= value >> 29
        value = (value * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        return (value >> 40) & ((1 << self.bits) - 1), value & 0xFFFF

    def lookups(self) -> list[tuple[int, int, list | None]]:
        found = []
        for depth in self.depths():
            index, check = self.digest(depth)
            first, second = self.slots[index], self.slots[index ^ 1]
            if first is not None and first[0] == check:
                found.append((index, check, first))
            elif second is not None and second[0] == check:
                found.append((index ^ 1, check, second))
            else:
                first_mass = sum(first[1].values()) if first is not None else -1
                second_mass = sum(second[1].values()) if second is not None else -1
                found.append((index if first_mass <= second_mass else index ^ 1, check, None))
        return found

    def log_prob(self, byte: int) -> float:
        probability = 1.0 / 256.0
        for _, _, entry in self.lookups():
            if entry is not None:
                counts = entry[1]
                total, kinds = sum(counts.values()), len(counts)
                probability = (counts.get(byte, 0) + kinds * probability) / (total + kinds)
        return math.log(probability)

    def predict(self) -> torch.Tensor:
        probability = np.full(256, 1.0 / 256.0)
        for _, _, entry in self.lookups():
            if entry is not None:
                vector = np.zeros(256)
                for key, value in entry[1].items():
                    vector[key] = value
                kinds = len(entry[1])
                probability = (vector + kinds * probability) / (vector.sum() + kinds)
        return torch.as_tensor(np.log(probability), dtype=torch.float32)

    def observe(self, byte: int) -> None:
        for index, check, entry in self.lookups():
            if entry is None:
                self.slots[index] = [check, {byte: 1}]
            else:
                entry[1][byte] = entry[1].get(byte, 0) + 1
        self.history = (self.history + [byte])[-self.order:]

    def state_vector(self) -> torch.Tensor:
        rows = [self.embed[byte] for byte in self.history]
        rows = [torch.zeros(16)] * (self.order - len(rows)) + rows
        return torch.cat(rows)

    def state_bytes(self) -> int:
        live = [slot for slot in self.slots if slot is not None]
        return 2 * len(live) + sum(2 * len(slot[1]) for slot in live)

    def capacity_bytes(self) -> int:
        return len(self.slots) * (2 + 2 * 256)


class ContextCells(ContextTable):
    """Cells addressed by exact recent suffixes, read through a log-opinion pool.

    The diagnosis of O21 showed the read was sound and the address was not: the
    same pool over exact suffixes beats the table on contexts never seen before.
    Trust depends on the suffix length, the cell's evidence and the longest
    suffix found, and follows only the gradient of the current byte's log loss.
    Time is settled lazily: a cell pays for elapsed time when it is next touched,
    so an event never visits cells outside its address.

    Intake is finite.  The pool keeps one number, the count of held symbols.  At
    its limit a new symbol must displace the weakest symbol of its own cell, and
    a new cell must displace a resident, so the budget is a hard bound and not an
    average.
    """

    name = "context_cells"
    reference_only = False
    orders: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    span = 12
    cell_share = 8.0
    count_limit = 60000.0
    ways = 2

    def __init__(self, budget_kib: int, seed: int = 3, *, rate: float = 0.005) -> None:
        super().__init__(budget_kib, seed, order=max(self.orders))
        self.bits = max(4, int(math.log2(budget_kib * 1024 / self.cell_share)))
        self.slots = [None] * (1 << self.bits)
        self.embed = torch.randn(256, 64, generator=torch.Generator().manual_seed(seed))
        self.order0 = np.ones(256)
        self.levels = 8
        self.trust = np.full((len(self.orders) + 1, len(self.orders), self.levels), 0.25)
        self.rate = rate
        self.held = 0
        self.limit = max(16, (budget_kib * 1024 - self.fixed_bytes() - 3 * len(self.slots)) // 2)
        self.clock = 0.0
        self.writes = torch.zeros(64)
        self.opinion: tuple | None = None

    def depths(self) -> list[int]:
        return [depth for depth in self.orders if depth <= len(self.history)]

    def lookups(self) -> list[tuple[int, int, list | None]]:
        found = []
        for depth in self.depths():
            index, check = self.digest(depth)
            places = [index ^ way for way in range(self.ways)]
            for place in places:
                slot = self.slots[place]
                if slot is not None and slot[0] == check:
                    found.append((place, check, slot))
                    break
            else:
                found.append((min(places, key=self.mass), check, None))
        return found

    def mass(self, place: int) -> float:
        slot = self.slots[place]
        return -1.0 if slot is None else sum(slot[1].values())

    def released(self, entry: list) -> dict:
        """The counts of a cell after the time since it was last touched."""
        if len(entry) < 3 or entry[2] == self.clock:
            return entry[1]
        factor = math.exp(-0.01 * (self.clock - entry[2]))
        return {key: value * factor for key, value in entry[1].items()}

    def pooled(self) -> tuple:
        """Log probabilities, evidence rows, trust entries and the address found."""
        if self.opinion is None:
            base = self.order0 / self.order0.sum()
            found, rows, entries = self.lookups(), [], []
            for depth, (_, _, slot) in zip(self.depths(), found):
                if slot is not None:
                    counts = self.released(slot)
                    vector = np.zeros(256)
                    for key, value in counts.items():
                        vector[key] = value
                    total, kinds = vector.sum(), len(counts)
                    rows.append(np.log((vector + kinds * base) / (total + kinds) / base))
                    entries.append((self.orders.index(depth), min(int(math.log2(total + 1.0)), self.levels - 1)))
            gate = entries[-1][0] + 1 if entries else 0
            entries = [(gate, *entry) for entry in entries]
            logits = np.log(base)
            for row, entry in zip(rows, entries):
                logits = logits + self.trust[entry] * row
            logits = logits - logits.max()
            self.opinion = (logits - math.log(np.exp(logits).sum()), rows, entries, found)
        return self.opinion

    def log_prob(self, byte: int) -> float:
        return float(self.pooled()[0][byte])

    def predict(self) -> torch.Tensor:
        return torch.as_tensor(self.pooled()[0], dtype=torch.float32)

    def observe(self, byte: int) -> None:
        logits, rows, entries, found = self.pooled()
        probability = np.exp(logits)
        for row, entry in zip(rows, entries):
            self.trust[entry] += self.rate * (row[byte] - float(probability @ row))
        for index, check, slot in found:
            if slot is None:
                resident = self.slots[index]
                self.held -= len(resident[1]) if resident is not None else 0
                self.slots[index] = None
                counts: dict[int, float] = {}
            else:
                counts = self.released(slot)
            if byte not in counts:
                if self.held >= self.limit:
                    if not counts:
                        continue
                    del counts[min(counts, key=counts.get)]
                    self.held -= 1
                counts[byte] = 0.0
                self.held += 1
            counts[byte] += 1.0
            if counts[byte] >= self.count_limit:
                # Each held count is two bytes, and a low limit keeps a cell able to change.
                counts = {key: value / 2.0 for key, value in counts.items()}
            self.slots[index] = [check, counts, self.clock]
            self.writes[index % 64] += 1.0
        self.order0[byte] += 1.0
        if self.order0.sum() >= 1e9:
            self.order0 /= 2.0
        self.history = (self.history + [byte])[-self.span:]
        self.opinion = None

    def advance(self, elapsed: float) -> None:
        self.clock += elapsed
        self.opinion = None

    def state_vector(self) -> torch.Tensor:
        rows = [self.embed[byte] for byte in self.history]
        return torch.cat([torch.zeros(64)] * (self.span - len(rows)) + rows)

    def clone(self) -> dict:
        # A flat copy of the cells; deepcopy's memo costs more than the cells do.
        return {"slots": [slot and [slot[0], dict(slot[1]), slot[2]] for slot in self.slots],
                "trust": self.trust.copy(), "order0": self.order0.copy(),
                "history": list(self.history), "held": self.held, "clock": self.clock,
                "writes": self.writes.clone()}

    def restore(self, saved: dict) -> None:
        self.slots = [slot and [slot[0], dict(slot[1]), slot[2]] for slot in saved["slots"]]
        self.trust, self.order0 = saved["trust"].copy(), saved["order0"].copy()
        self.history, self.held = list(saved["history"]), saved["held"]
        self.clock, self.writes = saved["clock"], saved["writes"].clone()
        self.opinion = None

    def fixed_bytes(self) -> int:
        return 4 * self.trust.size + 4 * 256 + self.span + 4

    def state_bytes(self) -> int:
        live = [slot for slot in self.slots if slot is not None]
        return 3 * len(live) + sum(2 * len(slot[1]) for slot in live) + self.fixed_bytes()

    def capacity_bytes(self) -> int:
        return 3 * len(self.slots) + 2 * self.limit + self.fixed_bytes()

    def load_shares(self) -> torch.Tensor:
        return self.writes / self.writes.sum().clamp_min(1.0)


class ContextCellsLong(ContextCells):
    name = "context_cells_long"
    orders = (1, 2, 3, 4, 5, 6, 8)


class ContextCellsFourWay(ContextCells):
    """More places per address, so a new context can outlast its neighbours."""

    name = "context_cells_fourway"
    ways = 4


class ContextCellsChanging(ContextCells):
    """Cells that stay able to change: old evidence is halved at a small count."""

    name = "context_cells_changing"
    count_limit = 255.0


CANDIDATES = {cls.name: cls for cls in (GDNLifeBytes, ColumnMatrix, FadingCells, FadingGraded,
                                        FadingPooled, FadingCellsPooled, FadingMixed, FadingGated,
                                        FadingCellsGated, FadingCellsLow, FadingCellsLowGated,
                                        FadingCellsMatched, ContextCells, ContextCellsLong,
                                        ContextCellsChanging, ContextCellsFourWay,
                                        WindowNetwork, GDN2, GDN2Truncated, GDN2Column,
                                        GDN2Batched, GDN2MlpBatched, GDN2MlpTruncated, GDN2WriteCredit,
                                        GDN2MlpWriteCredit, GDN2MlpWriteKeyCredit, GDN2TraceCredit, GDN2TraceKeyCredit,
                                        CausalAttention,
                                        CausalAttentionBatched, ContextTable)}
