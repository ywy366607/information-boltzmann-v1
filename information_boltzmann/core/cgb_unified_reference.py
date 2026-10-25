"""Executable reference of the existing SensoryMaintenance body/world process.

This consolidates a finite mathematical witness, not a language architecture.
The exact ensemble, rational arithmetic and observer history are verification
workspace, not free organism memory. No new carrier or learning law is chosen.
"""

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from typing import Callable


CELLS = ("vacuum", "search", "aSpent", "bSpent", "aReady", "bReady", "aHeld", "bHeld")
SIGNALS = ("calm", "answerA", "answerB", "feed", "scarce", "miss", "probe")
WORLDS = ("feedA", "feedB", "missA", "missB", "probeA", "probeB", "answerA", "answerB", "trap")
BASE_CONTENT = (0, 2, 1, 1, 4, 4, 3, 3)
CREDIT = (0, 1, 0, 0, 2, 2, 2, 2)
SENSE = (3, 3, 5, 5, 6, 6, 1, 2, 4)
Protocol = Callable[[int, int], int]  # pure external tick/world -> signal; -1 is loss


@dataclass(frozen=True)
class Body:
    cell: int = 1
    reserve: int = 2

    def __post_init__(self):
        if type(self.cell) is not int or not 0 <= self.cell < 8:
            raise ValueError("invalid cell")
        if type(self.reserve) is not int or not 0 <= self.reserve < 3:
            raise ValueError("invalid reserve")

    @property
    def content(self):
        return BASE_CONTENT[self.cell] + self.reserve

    @property
    def action(self):
        return 0 if self.cell in (2, 4, 6) else 1 if self.cell in (3, 5, 7) else 2


EMPTY = Body(0, 0)


@dataclass(frozen=True)
class State:
    body: Body = Body()
    world: int = 4

    def __post_init__(self):
        if not isinstance(self.body, Body) or type(self.world) is not int or not 0 <= self.world < 9:
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
    if type(signal) is not int or not -1 <= signal < 7:
        raise ValueError("invalid signal")
    q, r = x.cell, x.reserve
    if signal == 0:
        return Update(x)
    if q == 0:
        return Update(EMPTY, outgoing=r)
    if signal == -1:
        return Update(EMPTY if r == 0 else Body(q, r - 1),
                      outgoing=BASE_CONTENT[q] if r == 0 else 1,
                      lost_credit=CREDIT[q] if r == 0 else 0)
    nxt = q
    if signal == 4:
        nxt = 0
    elif signal in (1, 2) and q == 1:
        nxt = signal + 1
    elif signal == 3 and q in (2, 3, 6, 7):
        nxt = 4 if q in (2, 6) else 5
    elif signal == 5:
        nxt = 1 if q in (4, 5) else 0
    incoming = 0 if signal in (4, 5) else (
        4 if signal == 3 and q in (2, 3) else
        2 if signal == 3 and q in (6, 7) else 1)
    outgoing = (BASE_CONTENT[q] if signal == 4 else
                (2 if q in (4, 5) else BASE_CONTENT[q]) if signal == 5 else
                (2 if q == 1 else 1) if signal in (1, 2) else 1)
    writes = int((signal in (1, 2) and q == 1) or (signal == 5 and q in (4, 5)))
    topups = 2 if signal == 3 and q in (2, 3) else 0
    lost = CREDIT[q] if signal == 4 or (signal == 5 and q not in (4, 5)) else 0
    if nxt == 0:
        return Update(EMPTY, incoming, outgoing + r, writes, topups, lost)
    return Update(Body(nxt, 2 if incoming else r), incoming + (2 - r if incoming else 0),
                  outgoing, writes, topups, lost)


def world_row(cell: int, world: int) -> dict[int, Fraction]:
    """GraceRecovery.moves; both this and local_step read the OLD body/world."""
    action = Body(cell, 0).action
    if cell == 0 or world == 8:
        target = 8
    elif world in (0, 1):
        if action == 2:
            target = 6 + world
        elif action == world:
            return {0: Fraction(1, 2), 1: Fraction(1, 2)}
        else:
            target = 2 + world
    elif world in (2, 3):
        target = world + 2
    elif world in (4, 5):
        target = world + 2 if action == 2 else world - 2
    else:
        target = world - 6
    return {target: Fraction(1)}


def natural(_tick: int, world: int) -> int:
    return SENSE[world]


def interrupted(start: int, duration: int) -> Protocol:
    if any(type(v) is not int or v < 0 for v in (start, duration)):
        raise ValueError("invalid interruption")
    return lambda tick, world: -1 if start <= tick < start + duration else SENSE[world]


@dataclass(frozen=True)
class Carrier:
    """Two existing 12-token buffers; 0=body, 1=source, 2=returned.

    Choosing the lowest old identities realizes the transport contract. It is
    not asserted identical to Lean's noncomputable Classical.choose selection.
    """
    owners: tuple[tuple[int, ...], tuple[int, ...]]

    @staticmethod
    def seed(body: Body = Body()):
        return Carrier(tuple(tuple([0] * b + [1] * 4 + [2] * (8 - b))
                             for b in (BASE_CONTENT[body.cell], body.reserve)))

    def advance(self, before: Body, signal: int) -> "Carrier":
        change = local_step(before, signal)
        structure_in = change.incoming
        if change.incoming and change.body != EMPTY:
            structure_in -= 2 - before.reserve
        structure_out = change.outgoing
        if change.body == EMPTY and signal != -1:
            structure_out -= before.reserve
        elif signal == -1 and before.reserve:
            structure_out = 0
        old_content = (BASE_CONTENT[before.cell], before.reserve)
        new_content = (BASE_CONTENT[change.body.cell], change.body.reserve)
        inputs = (structure_in, change.incoming - structure_in)
        outputs = (structure_out, change.outgoing - structure_out)
        result = []
        for j, owners in enumerate(self.owners):
            assert len(owners) == 12 and owners.count(0) == old_content[j]
            assert owners.count(1) == 4 and set(owners) <= {0, 1, 2}
            selected = [tuple(k for k, place in enumerate(owners) if place == p)[:n]
                        for p, n in ((1, inputs[j]), (0, outputs[j]), (2, inputs[j]))]
            assert tuple(map(len, selected)) == (inputs[j], outputs[j], inputs[j])
            after = list(owners)
            for identities, place in zip(selected, (0, 2, 1)):
                for identity in identities:
                    after[identity] = place
            assert after.count(0) == new_content[j] and after.count(1) == 4
            result.append(tuple(after))
        return Carrier(tuple(result))


class ReferenceRun:
    """Exact law and first-hit observer of ONE fixed body/world kernel.

    The cut branch constructs a fresh ghost from its OWN current body marginal.
    The evolving ensemble is required by that counterfactual's definition.
    """

    def __init__(self, protocol: Protocol = natural, *, cut: bool = False):
        if type(cut) is not bool:
            raise ValueError("cut must be boolean")
        self.protocol, self.cut = protocol, cut
        self.laws = [{State(): Fraction(1)}]
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
                marginal[z.body.cell] = marginal.get(z.body.cell, Fraction(0)) + p
            kernels = {}
            if self.cut:
                for w in range(9):
                    row = {}
                    for q, p in marginal.items():
                        for v, r in world_row(q, w).items():
                            row[v] = row.get(v, Fraction(0)) + p * r
                    kernels[w] = row
            future, ports = {}, [Fraction(0)] * 5
            for z, p in law.items():
                change = local_step(z.body, self.protocol(tick, z.world))
                for j, field in enumerate((change.incoming, change.outgoing, change.writes,
                                           change.topups, change.lost_credit)):
                    ports[j] += p * field
                row = kernels[z.world] if self.cut else world_row(z.body.cell, z.world)
                for w, r in row.items():
                    target = State(change.body, w)
                    future[target] = future.get(target, Fraction(0)) + p * r
            assert sum(future.values()) == 1
            assert all(p > 0 for p in future.values())
            before = sum(p * z.body.content for z, p in law.items())
            after = sum(p * z.body.content for z, p in future.items())
            cb = sum(p * CREDIT[z.body.cell] for z, p in law.items())
            ca = sum(p * CREDIT[z.body.cell] for z, p in future.items())
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
            row = self.world_kernels[t][z.world] if self.cut else world_row(z.body.cell, z.world)
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
        credit = sum(p * CREDIT[z.body.cell] for z, p in self.laws[steps].items())
        assert final + totals[1] == 4 + totals[0]
        assert credit + totals[2] + totals[4] == 1 + totals[3]
        return dict(final_content=final, incoming=totals[0], outgoing=totals[1],
                    final_credit=credit, writes=totals[2], topups=totals[3], lost_credit=totals[4],
                    source_stock=Fraction(8), final_body_and_source=final + 8,
                    initial_body_and_source=Fraction(12), recharge_work=totals[0])
