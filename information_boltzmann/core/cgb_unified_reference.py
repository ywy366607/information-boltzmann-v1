"""Executable reference of the existing SensoryMaintenance body/world process.

The default specialization is the original finite mathematical witness.
FiniteFamily changes declared carrier/world parameters, not the learning rule.
The exact ensemble, rational arithmetic and observer history are verification
workspace, not free organism memory. The byte port below checks a conditional
distribution representation; it does not supply a generative language law.
"""

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import struct
from typing import Callable


CELLS = ("vacuum", "search", "aSpent", "bSpent", "aReady", "bReady", "aHeld", "bHeld")
SIGNALS = ("calm", "answerA", "answerB", "feed", "scarce", "miss", "probe")
WORLDS = ("feedA", "feedB", "missA", "missB", "probeA", "probeB", "answerA", "answerB", "trap")
BASE_CONTENT = (0, 2, 1, 1, 4, 4, 3, 3)
CREDIT = (0, 1, 0, 0, 2, 2, 2, 2)
SENSE = (3, 3, 5, 5, 6, 6, 1, 2, 4)
Protocol = Callable[[int, int], int]  # pure external tick/world -> signal; -1 is loss


@dataclass(frozen=True)
class FiniteFamily:
    situations: int = 2
    capacity: int = 2

    def __post_init__(self):
        if type(self.situations) is not int or self.situations < 2:
            raise ValueError("ambiguous-search family needs at least two situations")
        if type(self.capacity) is not int or self.capacity < 0:
            raise ValueError("capacity must be nonnegative")

    @property
    def cells(self):
        return 2 + 3 * self.situations

    def base(self, cell):
        if cell < 2:
            return 2 * cell
        return (1, 4, 3)[(cell - 2) // self.situations]

    def credit(self, cell):
        return 0 if cell == 0 or 2 <= cell < 2 + self.situations else 1 if cell == 1 else 2

    def signal(self, world):
        n = self.situations
        if type(world) is not int or not 0 <= world <= 4 * n:
            raise ValueError("invalid world")
        phase, label = divmod(world, n)
        return (n + 1, n + 3, n + 4, 1 + label, n + 2)[phase]


DEFAULT = FiniteFamily()


@dataclass(frozen=True)
class Body:
    cell: int = 1
    reserve: int = 2
    family: FiniteFamily = DEFAULT

    def __post_init__(self):
        if not isinstance(self.family, FiniteFamily):
            raise ValueError("invalid family")
        if type(self.cell) is not int or not 0 <= self.cell < self.family.cells:
            raise ValueError("invalid cell")
        if type(self.reserve) is not int or not 0 <= self.reserve <= self.family.capacity:
            raise ValueError("invalid reserve")

    @property
    def content(self):
        return self.family.base(self.cell) + self.reserve

    @property
    def action(self):
        n = self.family.situations
        return (self.cell - 2) % n if self.cell >= 2 else n


EMPTY = Body(0, 0)


@dataclass(frozen=True)
class State:
    body: Body = Body()
    world: int = 4

    def __post_init__(self):
        if (not isinstance(self.body, Body) or type(self.world) is not int
                or not 0 <= self.world <= 4 * self.body.family.situations):
            raise ValueError("invalid joint state")


@dataclass(frozen=True)
class Update:
    body: Body
    incoming: int = 0
    outgoing: int = 0
    writes: int = 0
    topups: int = 0
    lost_credit: int = 0


def local_step(x: Body, signal: int) -> Update:
    """SensoryMaintenance.next and its five ports, including inactive rows."""
    f = x.family
    n, capacity = f.situations, f.capacity
    if type(signal) is not int or not -1 <= signal < n + 5:
        raise ValueError("invalid signal")
    q, r = x.cell, x.reserve
    empty = Body(0, 0, f)
    phase = (q - 2) // n if q >= 2 else -1
    if signal == 0:
        return Update(x)
    if q == 0:
        return Update(empty, outgoing=r)
    if signal == -1:
        return Update(empty if r == 0 else Body(q, r - 1, f),
                      outgoing=f.base(q) if r == 0 else 1,
                      lost_credit=f.credit(q) if r == 0 else 0)
    nxt = q
    if signal == n + 2:
        nxt = 0
    elif 1 <= signal <= n and q == 1:
        nxt = signal + 1
    elif signal == n + 1 and phase in (0, 2):
        nxt = 2 + n + (q - 2) % n
    elif signal == n + 3:
        nxt = 1 if phase == 1 else 0
    incoming = 0 if signal in (n + 2, n + 3) else (
        4 if signal == n + 1 and phase == 0 else
        2 if signal == n + 1 and phase == 2 else 1)
    outgoing = (f.base(q) if signal == n + 2 else
                (2 if phase == 1 else f.base(q)) if signal == n + 3 else
                (2 if q == 1 else 1) if 1 <= signal <= n else 1)
    writes = int((1 <= signal <= n and q == 1) or (signal == n + 3 and phase == 1))
    topups = 2 if signal == n + 1 and phase == 0 else 0
    lost = f.credit(q) if signal == n + 2 or (signal == n + 3 and phase != 1) else 0
    if nxt == 0:
        return Update(empty, incoming, outgoing + r, writes, topups, lost)
    return Update(Body(nxt, capacity if incoming else r, f),
                  incoming + (capacity - r if incoming else 0),
                  outgoing, writes, topups, lost)


def world_row(cell: int, world: int, family: FiniteFamily = DEFAULT) -> dict[int, Fraction]:
    """GraceRecovery.moves; both this and local_step read the OLD body/world."""
    n = family.situations
    family.signal(world)  # validate before branching
    action = Body(cell, 0, family).action
    phase, label = divmod(world, n)
    if cell == 0 or world == 4 * n:
        target = 4 * n
    elif phase == 0:
        if action == n:
            target = 3 * n + label
        elif action == label:
            return {v: Fraction(1, n) for v in range(n)}
        else:
            target = n + label
    elif phase == 1:
        target = 2 * n + label
    elif phase == 2:
        target = (3 * n if action == n else n) + label
    else:
        target = label
    return {target: Fraction(1)}


def natural(_tick: int, world: int) -> int:
    return SENSE[world]


def interrupted(start: int, duration: int, family: FiniteFamily = DEFAULT) -> Protocol:
    if any(type(v) is not int or v < 0 for v in (start, duration)):
        raise ValueError("invalid interruption")
    return lambda tick, world: -1 if start <= tick < start + duration else family.signal(world)


@dataclass(frozen=True)
class Carrier:
    """Two old-stock buffers; 0=body, 1=source, 2=returned.

    Choosing the lowest old identities realizes the transport contract. It is
    not asserted identical to Lean's noncomputable Classical.choose selection.
    The default has two 12-token buffers; a larger reserve needs a wider one.
    """
    owners: tuple[tuple[int, ...], tuple[int, ...]]

    @staticmethod
    def seed(body: Body = Body()):
        widths = (4, max(4, body.family.capacity))
        return Carrier(tuple(tuple([0] * b + [1] * c + [2] * (2 * c - b))
                             for b, c in zip((body.family.base(body.cell), body.reserve), widths)))

    def advance(self, before: Body, signal: int) -> "Carrier":
        change = local_step(before, signal)
        structure_in = change.incoming
        if change.incoming and change.body.cell != 0:
            structure_in -= before.family.capacity - before.reserve
        structure_out = change.outgoing
        if change.body.cell == 0 and change.body.reserve == 0 and signal != -1:
            structure_out -= before.reserve
        elif signal == -1 and before.reserve:
            structure_out = 0
        old_content = (before.family.base(before.cell), before.reserve)
        new_content = (before.family.base(change.body.cell), change.body.reserve)
        inputs = (structure_in, change.incoming - structure_in)
        outputs = (structure_out, change.outgoing - structure_out)
        result = []
        for j, owners in enumerate(self.owners):
            width = 4 if j == 0 else max(4, before.family.capacity)
            assert len(owners) == 3 * width and owners.count(0) == old_content[j]
            assert owners.count(1) == width and set(owners) <= {0, 1, 2}
            selected = [tuple(k for k, place in enumerate(owners) if place == p)[:n]
                        for p, n in ((1, inputs[j]), (0, outputs[j]), (2, inputs[j]))]
            assert tuple(map(len, selected)) == (inputs[j], outputs[j], inputs[j])
            after = list(owners)
            for identities, place in zip(selected, (0, 2, 1)):
                for identity in identities:
                    after[identity] = place
            assert after.count(0) == new_content[j] and after.count(1) == width
            result.append(tuple(after))
        return Carrier(tuple(result))


class ReferenceRun:
    """Exact law and first-hit observer of ONE fixed body/world kernel.

    The cut branch constructs a fresh ghost from its OWN current body marginal.
    The evolving ensemble is required by that counterfactual's definition.
    """

    def __init__(self, protocol: Protocol | None = None, *, cut: bool = False,
                 family: FiniteFamily = DEFAULT):
        if type(cut) is not bool:
            raise ValueError("cut must be boolean")
        self.family = family
        self.protocol = protocol if protocol is not None else lambda _t, w: family.signal(w)
        self.cut = cut
        self.laws = [{State(Body(1, family.capacity, family), 2 * family.situations): Fraction(1)}]
        self.world_kernels = []
        self.ledger = []

    def extend(self, steps: int):
        if type(steps) is not int or steps < 0:
            raise ValueError("invalid observer horizon")
        while len(self.world_kernels) < steps:
            tick = len(self.world_kernels)
            law = self.laws[-1]
            marginal = {}
            for z, p in law.items():
                assert z.body.family == self.family
                marginal[z.body.cell] = marginal.get(z.body.cell, Fraction(0)) + p
            kernels = {}
            if self.cut:
                for w in range(4 * self.family.situations + 1):
                    row = {}
                    for q, p in marginal.items():
                        for v, r in world_row(q, w, self.family).items():
                            row[v] = row.get(v, Fraction(0)) + p * r
                    kernels[w] = row
            future, ports = {}, [Fraction(0)] * 5
            for z, p in law.items():
                change = local_step(z.body, self.protocol(tick, z.world))
                for j, field in enumerate((change.incoming, change.outgoing, change.writes,
                                           change.topups, change.lost_credit)):
                    ports[j] += p * field
                row = kernels[z.world] if self.cut else world_row(z.body.cell, z.world, self.family)
                for w, r in row.items():
                    target = State(change.body, w)
                    future[target] = future.get(target, Fraction(0)) + p * r
            assert sum(future.values()) == 1
            assert all(p > 0 for p in future.values())
            before = sum(p * z.body.content for z, p in law.items())
            after = sum(p * z.body.content for z, p in future.items())
            cb = sum(p * self.family.credit(z.body.cell) for z, p in law.items())
            ca = sum(p * self.family.credit(z.body.cell) for z, p in future.items())
            assert after + ports[1] == before + ports[0]
            assert ca + ports[2] + ports[4] == cb + ports[3]
            self.world_kernels.append(kernels)
            self.laws.append(future)
            self.ledger.append(tuple(ports))
        return self

    def recovery(self, tick: int, state: State, horizon: int = 3) -> Fraction:
        if any(type(v) is not int or v < 0 for v in (tick, horizon)):
            raise ValueError("invalid first-hit query")
        self.extend(tick + horizon)

        @lru_cache(maxsize=None)
        def hit(t, z, k):
            if not k:
                return Fraction(0)
            change = local_step(z.body, self.protocol(t, z.world))
            if change.incoming:
                return Fraction(1)
            row = self.world_kernels[t][z.world] if self.cut else world_row(z.body.cell, z.world, self.family)
            return sum((p * hit(t + 1, State(change.body, w), k - 1)
                        for w, p in row.items()), Fraction(0))

        return hit(tick, state, horizon)

    def survival(self, tick: int) -> Fraction:
        self.extend(tick + 3)
        return sum((p for z, p in self.laws[tick].items()
                    if z.body.content > 0 and self.recovery(tick, z) >= Fraction(1, 2)), Fraction(0))

    def balance(self, steps: int) -> dict[str, Fraction]:
        self.extend(steps)
        totals = [sum((row[j] for row in self.ledger[:steps]), Fraction(0)) for j in range(5)]
        final = sum(p * z.body.content for z, p in self.laws[steps].items())
        credit = sum(p * self.family.credit(z.body.cell) for z, p in self.laws[steps].items())
        initial_content = 2 + self.family.capacity
        source = 4 + max(4, self.family.capacity)
        assert final + totals[1] == initial_content + totals[0]
        assert credit + totals[2] + totals[4] == 1 + totals[3]
        return dict(final_content=final, incoming=totals[0], outgoing=totals[1],
                    final_credit=credit, writes=totals[2], topups=totals[3], lost_credit=totals[4],
                    source_stock=Fraction(source), final_body_and_source=final + source,
                    initial_body_and_source=Fraction(initial_content + source), recharge_work=totals[0])


@dataclass(frozen=True)
class TicketMemory:
    """Finite statistical reference: H byte counters with total M in 1..255.

    Tickets encode probability, not conserved matter. The world holds the only
    resource reserve; this state must not introduce a second source of credit.
    Global normalization is an observer algorithm, not a P1b-local medium.
    """
    payload: bytes

    def __post_init__(self):
        if (type(self.payload) is not bytes or not 1 <= len(self.payload) <= 256
                or not 1 <= sum(self.payload) <= 255):
            raise ValueError("invalid finite ticket memory")

    @property
    def counts(self):
        return self.payload


@dataclass(frozen=True)
class ByteFrame:
    """Frozen prior and registered H-by-256 uint16 likelihood program data.

    All rows are positive, have the same total <=32767, and precede the target.
    The caller still owes the provenance, computation and locality contract of
    these rows. This interface does not invent or certify a language model.
    """
    prior: TicketMemory
    table: bytes

    def __post_init__(self):
        if not isinstance(self.prior, TicketMemory) or type(self.table) is not bytes:
            raise ValueError("invalid byte frame")
        if len(self.table) != len(self.prior.counts) * 512:
            raise ValueError("likelihood shape must be H by 256")
        totals = []
        for h in range(len(self.prior.counts)):
            row = struct.unpack_from("<256H", self.table, h * 512)
            if min(row) < 1:
                raise ValueError("this byte contract requires full support")
            totals.append(sum(row))
        if max(totals) > 32767 or len(set(totals)) != 1:
            raise ValueError("candidate likelihood rows need one common bounded denominator")

    def masses(self, byte: int):
        if type(byte) is not int or not 0 <= byte < 256:
            raise ValueError("invalid native byte")
        return tuple(c * struct.unpack_from("<H", self.table, 512 * h + 2 * byte)[0]
                     for h, c in enumerate(self.prior.counts))

    @property
    def denominator(self):
        return sum(self.prior.counts) * sum(struct.unpack_from("<256H", self.table))

    def probability(self, byte: int):
        return Fraction(sum(self.masses(byte)), self.denominator)

    def project(self, byte: int):
        """Propose a finite conditional state, before any resource payment.

        Hamilton rounding bounds only this projection, not accumulated history
        error. A zero count is absorbing without a separately funded reopening.
        """
        masses = self.masses(byte)
        total, tickets = sum(masses), sum(self.prior.counts)
        divisions = [divmod(tickets * m, total) for m in masses]
        counts = [q for q, _r in divisions]
        remainder = tickets - sum(counts)
        order = sorted(range(len(counts)), key=lambda h: (-divisions[h][1], h))
        for h in order[:remainder]:
            counts[h] += 1
        projected = bytes(counts)
        flips = sum((a ^ b).bit_count() for a, b in zip(self.prior.counts, projected))
        return TicketMemory(projected), flips


def byte_step(memory: TicketMemory, table: bytes, world):
    """Paid diagnostic port: one public byte and one world tick.

    Likelihood rows are registered model data, not learned or generated here.
    Tariff = changed encoding bits; computation/accounting costs still require
    physical calibration. Receipts and action selection are not modeled here.
    This is not a refinement of the finite body's survival certificates or a
    locally realized CGB learning law; those are distinct admission obligations.
    """
    frame = ByteFrame(memory, table)  # validate/freeze before touching the world
    boundary = world.begin()
    observation = world.publish()
    if not boundary.active:
        # Evaluation fallback, not computation attributed to an inactive body.
        return memory, dict(tick=observation.tick, byte=observation.payload,
                            probability=Fraction(1, 256), active=False,
                            required_write_units=0, spent=0, changed=False, funded=False)
    probability = frame.probability(observation.payload)
    proposed, fee = frame.project(observation.payload)
    funded = world.pay_write(fee)
    successor = proposed if funded else memory
    return successor, dict(tick=observation.tick, byte=observation.payload,
                           probability=probability, active=boundary.active,
                           required_write_units=fee, spent=fee if funded else 0,
                           changed=successor != memory, funded=funded)


@dataclass(frozen=True)
class BoundaryKernel:
    """One-particle conditional sampler on a declared boundary clique.

    Frozen positive uint16 weights define the target law. One logical proposal
    reads only its source and destination weights; no normalizer is consulted.
    The caller owes uniform proposal/ticket entropy, clique realization, and
    the history-to-weight law. A position is a sample, not a probability vector.
    """
    weights: bytes
    attempt_fee: int = 1

    def __post_init__(self):
        if (type(self.weights) is not bytes or len(self.weights) % 2
                or not 4 <= len(self.weights) <= 512
                or type(self.attempt_fee) is not int or self.attempt_fee < 1):
            raise ValueError("invalid boundary weights or attempt tariff")
        if any(w == 0 for (w,) in struct.iter_unpack("<H", self.weights)):
            raise ValueError("boundary weights must be positive")

    @property
    def alphabet(self):
        return len(self.weights) // 2

    def weight(self, label: int):
        if type(label) is not int or not 0 <= label < self.alphabet:
            raise ValueError("invalid boundary label")
        return struct.unpack_from("<H", self.weights, 2 * label)[0]

    def hop(self, position: int, proposal: int, ticket: int):
        """proposal ~ Uniform(B), ticket ~ Uniform(weight(position)), independent.

        Rejection and self-proposal are transitions too. Ticket production is
        an explicit stochastic primitive, not an unbounded hidden inner loop.
        """
        old, new = self.weight(position), self.weight(proposal)
        if type(ticket) is not int or not 0 <= ticket < old:
            raise ValueError("invalid boundary acceptance ticket")
        return proposal if ticket < min(old, new) else position

    def tick(self, position: int, proposal: int, ticket: int, world):
        """Publish once, pay one attempt, commit NEXT tick's sample.

        All valid attempts, including rejection and self-proposal, cost the same
        declared amount. Refusal keeps the old position. Only a fully funded
        attempt has the registered mixing kernel; these units are not joules.
        """
        candidate = self.hop(position, proposal, ticket)
        boundary = world.begin()
        observation = world.publish()
        funded = world.pay_write(self.attempt_fee)
        return candidate if funded else position, dict(
            tick=observation.tick, byte=observation.payload,
            predicted_label=position if boundary.active else None,
            funded=funded, spent=self.attempt_fee if funded else 0)
