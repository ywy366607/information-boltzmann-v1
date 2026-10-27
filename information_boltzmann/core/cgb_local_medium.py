"""Local reactions and finite boundary exchange of the registered LG-1 family.

One shared state holds end charges (p,q) and plastic slots w. The rook contact
graph is implicit. Rates realize pi proportional to 2**(-2*E), without reading
the global energy or normalizer. This is a reaction reference, not an admitted
language organism or an equivalence to the LG1 mean-field approximation.
"""

from dataclasses import asdict, dataclass
from functools import cached_property


def _read_packed(data, index, bits):
    start, shift = divmod(index * bits, 8)
    width = (shift + bits + 7) // 8
    return (int.from_bytes(data[start:start + width], "little") >> shift) & ((1 << bits) - 1)


def _write_packed(data, index, bits, value):
    start, shift = divmod(index * bits, 8)
    width = (shift + bits + 7) // 8
    old = int.from_bytes(data[start:start + width], "little")
    mask = ((1 << bits) - 1) << shift
    data[start:start + width] = ((old & ~mask) | (value << shift)).to_bytes(width, "little")


@dataclass(frozen=True)
class LocalLaw:
    rows: int
    labels: int
    charge_cap: int
    slot_cap: int
    potential: tuple[int, ...]  # same U(N) for every compartment
    epsilon: int = 1
    coupling: int = 1
    attempt_fee: int = 1

    def __post_init__(self):
        values = (self.rows, self.labels, self.charge_cap, self.slot_cap,
                  self.epsilon, self.coupling, self.attempt_fee)
        if (any(type(v) is not int for v in values) or self.rows < 1
                or not 1 <= self.labels <= 256 or not 1 <= self.charge_cap <= 15
                or not 1 <= self.slot_cap <= 15 or self.attempt_fee < 1
                or type(self.potential) is not tuple
                or len(self.potential) != max(self.rows, self.labels) * self.charge_cap + 1
                or any(type(v) is not int for v in self.potential)):
            raise ValueError("invalid finite local law")

    @property
    def edges(self):
        return self.rows * self.labels

    @property
    def events(self):
        return self.edges * (self.rows + self.labels + 4)

    @property
    def states(self):
        return (self.charge_cap + 1) ** 2 * (self.slot_cap + 1)

    @property
    def bits(self):
        return (self.states - 1).bit_length()

    @property
    def state_bytes(self):
        return (self.edges * self.bits + 7) // 8

    @cached_property
    def bounds(self):
        q, w, j = self.charge_cap, self.slot_cap, abs(self.coupling)
        delta_u = max(abs(b - a) for a, b in zip(self.potential, self.potential[1:]))
        return (j * w * (q - 1) + 2 * delta_u,
                abs(self.epsilon) + j * q * max(q - 1, w - 1) + delta_u,
                j * w * q, j * w * q)


class LocalMedium:
    """Three <=4-bit fields per edge, with one minimal fixed-width mixed-radix code.

    Uniform event indices range over a FIXED set, including disabled events.
    Uniform acceptance tickets are explicit stochastic primitives. Optional
    packed clique totals are derived execution storage, not extra history.
    No index array, eligibility array or autograd history is stored.
    """

    def __init__(self, law: LocalLaw, payload: bytes, *, cache_totals=True):
        if (not isinstance(law, LocalLaw) or type(payload) is not bytes
                or type(cache_totals) is not bool):
            raise ValueError("invalid local medium")
        bits = law.bits * law.edges
        if len(payload) != law.state_bytes or (bits % 8 and payload[-1] >> (bits % 8)):
            raise ValueError("invalid packed shape or padding")
        self.law, self._payload = law, bytearray(payload)
        self._total_bits = ((law.labels * law.charge_cap).bit_length(),
                            (law.rows * law.charge_cap).bit_length())
        self._totals = (tuple(bytearray((n * b + 7) // 8)
                              for n, b in zip((law.rows, law.labels), self._total_bits))
                        if cache_totals else None)
        for edge in range(law.edges):
            p, q, w = self.get(edge)
            if max(p, q) > law.charge_cap or w > law.slot_cap:
                raise ValueError("packed state exceeds capacity")
            if self._totals is not None:
                row, col = divmod(edge, law.labels)
                for channel, index, value in ((0, row, p), (1, col, q)):
                    data, width = self._totals[channel], self._total_bits[channel]
                    _write_packed(data, index, width, _read_packed(data, index, width) + value)
        _ = law.bounds  # compile static rate bounds once, not during local events

    @property
    def payload(self):
        """Read-only view; writes must maintain the derived total invariant."""
        return memoryview(self._payload).toreadonly()

    @property
    def cache_bytes(self):
        return sum(map(len, self._totals)) if self._totals is not None else 0

    def total(self, channel, index):
        """Exact sum over ONE allowed clique; the uncached path is the oracle."""
        f = self.law
        if (type(channel) is not int or channel not in (0, 1)
                or type(index) is not int or not 0 <= index < (f.rows, f.labels)[channel]):
            raise ValueError("invalid compartment")
        if self._totals is not None:
            return _read_packed(self._totals[channel], index, self._total_bits[channel])
        ids = (range(index * f.labels, (index + 1) * f.labels) if channel == 0
               else range(index, f.edges, f.labels))
        return sum(self.get(e)[channel] for e in ids)

    def get(self, edge):
        if type(edge) is not int or not 0 <= edge < self.law.edges:
            raise ValueError("invalid implicit edge")
        code = _read_packed(self._payload, edge, self.law.bits)
        if code >= self.law.states:
            raise ValueError("unused local code")
        pq, w = divmod(code, self.law.slot_cap + 1)
        p, q = divmod(pq, self.law.charge_cap + 1)
        return p, q, w

    def _put(self, edge, state):
        p, q, w = state
        if self._totals is not None:
            old = self.get(edge)
            row, col = divmod(edge, self.law.labels)
            for channel, index in ((0, row), (1, col)):
                if state[channel] != old[channel]:
                    data, bits = self._totals[channel], self._total_bits[channel]
                    value = _read_packed(data, index, bits) - old[channel] + state[channel]
                    _write_packed(data, index, bits, value)
        code = (p * (self.law.charge_cap + 1) + q) * (self.law.slot_cap + 1) + w
        _write_packed(self._payload, edge, self.law.bits, code)

    def decode(self, index):
        f = self.law
        if type(index) is not int or not 0 <= index < f.events:
            raise ValueError("invalid fixed proposal index")
        if index < 4 * f.edges:
            edge, choice = divmod(index, 4)
            return 0, edge, None, 1 if choice < 2 else -1, choice % 2
        index -= 4 * f.edges
        if index < 2 * f.edges:
            edge, choice = divmod(index, 2)
            return 1, edge, None, 1 if choice == 0 else -1, 1
        index -= 2 * f.edges
        if index < f.edges * (f.labels - 1):
            edge, choice = divmod(index, f.labels - 1)
            row, col = divmod(edge, f.labels)
            other = row * f.labels + choice + (choice >= col)
            return 2, edge, other, 1, 0
        index -= f.edges * (f.labels - 1)
        edge, choice = divmod(index, f.rows - 1)
        row, col = divmod(edge, f.labels)
        other = (choice + (choice >= row)) * f.labels + col
        return 3, edge, other, 1, 1

    def reverse(self, index):
        kind, edge, other, _sign, _channel = self.decode(index)
        f = self.law
        if kind == 0:
            return index ^ 3  # reverse direction AND exchange the two cliques
        if kind == 1:
            return index ^ 1
        if kind == 2:
            col, target = edge % f.labels, other % f.labels
            return 6 * f.edges + other * (f.labels - 1) + col - (col > target)
        row, target = edge // f.labels, other // f.labels
        return (6 * f.edges + f.edges * (f.labels - 1)
                + other * (f.rows - 1) + row - (row > target))

    def unit_energy(self, state):
        p, q, w = state
        return self.law.epsilon * w - self.law.coupling * w * p * q

    def prepare(self, index):
        """Return local edits, log2 rate (None if blocked), and its fixed bound."""
        kind, edge, other, sign, channel = self.decode(index)
        f = self.law
        old = self.get(edge)
        new = list(old)
        if kind == 0:
            new[0] -= sign
            new[1] += sign
        elif kind == 1:
            new[1] -= sign
            new[2] += sign
        else:
            target = list(self.get(other))
            end = kind - 2
            new[end] -= 1
            target[end] += 1
        edits = [(edge, tuple(new))]
        if other is not None:
            edits.append((other, tuple(target)))
        if any(not (0 <= p <= f.charge_cap and 0 <= q <= f.charge_cap
                    and 0 <= w <= f.slot_cap) for _e, (p, q, w) in edits):
            return (), None, f.bounds[kind]
        delta = sum(self.unit_energy(state) - self.unit_energy(self.get(e)) for e, state in edits)
        if kind < 2:
            row, col = divmod(edge, f.labels)
            total = self.total(channel, row if channel == 0 else col)
            changed = new[channel] - old[channel]
            delta_u = f.potential[total + changed] - f.potential[total]
            delta += (2 if kind == 0 else 1) * delta_u
        exponent = -delta
        assert exponent <= f.bounds[kind]
        return tuple(edits), exponent, f.bounds[kind]

    def tick(self, index, ticket, world):
        """One publication and one paid local attempt, including blocked moves.

        Select index uniformly from law.events, then ticket uniformly from
        [0,2**random_bits). The attempt tariff is declared, not joule-calibrated;
        producing that random ticket can itself be costly. No extra free steps.
        """
        edits, exponent, bound = self.prepare(index)
        bits = 0 if exponent is None else bound - exponent
        if type(ticket) is not int or ticket < 0 or ticket.bit_length() > bits:
            raise ValueError("invalid exact dyadic ticket")
        boundary = world.begin()
        observed = world.publish()
        funded = world.pay_write(self.law.attempt_fee)
        accepted = funded and exponent is not None and ticket == 0
        if accepted:
            for edge, state in edits:
                self._put(edge, state)
        return dict(tick=observed.tick, byte=observed.payload, active=boundary.active,
                    funded=funded, accepted=accepted, random_bits=bits,
                    spent=self.law.attempt_fee if funded else 0,
                    input_conditions_reaction=False)

    def state_dict(self):
        config = asdict(self.law)
        config["potential"] = list(config["potential"])
        return dict(version=2, encoding="mixed-radix-le", law=config, payload=self.payload.hex())

    def load_state_dict(self, snapshot):
        expected = self.state_dict()["law"]
        if (type(snapshot) is not dict or set(snapshot) != {"version", "encoding", "law", "payload"}
                or type(snapshot["version"]) is not int or snapshot["version"] != 2
                or type(snapshot["encoding"]) is not str or snapshot["encoding"] != "mixed-radix-le"
                or type(snapshot["law"]) is not dict or set(snapshot["law"]) != set(expected)
                or any(type(snapshot["law"][k]) is not type(v) or snapshot["law"][k] != v
                       for k, v in expected.items())
                or any(type(v) is not int for v in snapshot["law"]["potential"])
                or type(snapshot["payload"]) is not str):
            raise ValueError("local medium checkpoint mismatch")
        restored = LocalMedium(self.law, bytes.fromhex(snapshot["payload"]),
                               cache_totals=self._totals is not None)
        self._payload, self._totals = restored._payload, restored._totals


@dataclass(frozen=True)
class ReservoirLaw:
    source_initial: int
    content_cost: int = 0
    chemical_base: int = 0
    stimulus: int = 1

    def __post_init__(self):
        if (any(type(v) is not int for v in asdict(self).values())
                or self.source_initial < 0 or self.stimulus < 0):
            raise ValueError("invalid finite reservoir law")


class OpenMedium:
    """The same body with finite, reversible boundary material exchange.

    A fixed proposal set includes all closed moves and two exchange directions
    per edge. The byte selects one Q compartment's external chemical potential.
    It does not select a proposal or write weights. Mass-based predictions are
    an explicit OBSERVER readout; the body never reads their global normalizer.
    Resource credits in the world are distinct from this reservoir's material.
    """

    def __init__(self, body: LocalMedium, reservoir: ReservoirLaw, *, opportunities_per_tick=1):
        if (not isinstance(body, LocalMedium) or not isinstance(reservoir, ReservoirLaw)
                or type(opportunities_per_tick) is not int or opportunities_per_tick < 0):
            raise ValueError("invalid open medium")
        self.body, self.reservoir = body, reservoir
        self._opportunities_per_tick = opportunities_per_tick
        self.stock = reservoir.source_initial
        self.initial_content = sum(sum(body.get(e)) for e in range(body.law.edges))
        self.intake = self.released = 0
        f = body.law
        delta_u = max(abs(b - a) for a, b in zip(f.potential, f.potential[1:]))
        self.exchange_bound = (abs(f.coupling) * f.slot_cap * f.charge_cap + delta_u
                               + abs(reservoir.content_cost)
                               + max(abs(reservoir.chemical_base),
                                     abs(reservoir.chemical_base + reservoir.stimulus)))

    @property
    def events(self):
        return self.body.law.events + 2 * self.body.law.edges

    @property
    def opportunities_per_tick(self):
        """Registered finite opportunity cap, not an adaptive convergence loop."""
        return self._opportunities_per_tick

    @property
    def proposal_families(self):
        """State-independent spans; disabled moves remain in each family."""
        f = self.body.law
        sizes = (4 * f.edges, 2 * f.edges, f.edges * (f.labels - 1),
                 f.edges * (f.rows - 1), 2 * f.edges)
        start, spans = 0, []
        for size in sizes:
            if size:
                spans.append((start, size))
            start += size
        return tuple(spans)

    def sample_proposal(self, rng):
        """Uniform family, then uniform event: fixed reciprocal conductances.

        This avoids combinatorial dilution of boundary attempts; it changes
        relative family clocks versus a uniform draw over the entire index set.
        Both reverse partners still have the same proposal probability.
        """
        families = self.proposal_families
        start, size = families[rng.randrange(len(families))]
        return start + rng.randrange(size)

    def proposal_bound(self, index):
        if type(index) is not int or not 0 <= index < self.events:
            raise ValueError("invalid open proposal")
        if index < self.body.law.events:
            return self.body.law.bounds[self.body.decode(index)[0]]
        return self.exchange_bound

    def reverse(self, index):
        self.proposal_bound(index)
        if index < self.body.law.events:
            return self.body.reverse(index)
        return self.body.law.events + ((index - self.body.law.events) ^ 1)

    def prepare(self, index, label):
        bound = self.proposal_bound(index)
        f, r = self.body.law, self.reservoir
        if type(label) is not int or not 0 <= label < f.labels:
            raise ValueError("invalid boundary stimulus")
        if index < f.events:
            edits, exponent, _ = self.body.prepare(index)
            return edits, exponent, bound, 0
        edge, direction = divmod(index - f.events, 2)
        sign = 1 if direction == 0 else -1
        p, q, w = old = self.body.get(edge)
        if not 0 <= q + sign <= f.charge_cap or (sign == 1 and self.stock == 0):
            return (), None, bound, 0
        new = (p, q + sign, w)
        col = edge % f.labels
        total = self.body.total(1, col)
        delta = (self.body.unit_energy(new) - self.body.unit_energy(old)
                 + f.potential[total + sign] - f.potential[total])
        mu = r.chemical_base + (r.stimulus if col == label else 0)
        exponent = -delta - (r.content_cost - mu) * sign
        assert -bound <= exponent <= bound
        return ((edge, new),), exponent, bound, sign

    def masses(self):
        """Observer workspace: current Q masses, without a Gibbs approximation."""
        f = self.body.law
        values = tuple(sum(self.body.get(e)[1] for e in range(b, f.edges, f.labels))
                       for b in range(f.labels))
        total = sum(values)
        return (values, total) if total else ((1,) * f.labels, f.labels)

    def _commit(self, edits, sign):
        for edge, state in edits:
            self.body._put(edge, state)
        self.stock -= sign
        self.intake += int(sign == 1)
        self.released += int(sign == -1)

    def tick(self, index, ticket, world):
        """Compatibility wrapper for one opportunity and the original receipt."""
        result = self.tick_window(((index, ticket),), world)
        return dict(tick=result["tick"], byte=result["byte"], active=result["active"],
                    funded=bool(result["funded_attempts"]), accepted=bool(result["accepted_reactions"]),
                    material_delta=result["material_delta"], target_mass=result["target_mass"],
                    total_mass=result["total_mass"], random_bits=result["planned_random_bits"],
                    spent=result["spent"])

    def tick_window(self, plan, world):
        """One publication followed by a bounded sequence of the SAME paid kernel.

        The caller supplies an exact tuple of (proposal, uniform dyadic ticket)
        pairs sampled independently of the new byte, using sample_proposal.
        All ranges are checked before publication, including unused late pairs.
        A fixed opportunity cap changes the per-publication dynamics: it is a
        registered exposure/work budget, not an equivalence to the old clock.
        No future input is inspected, and no target-dependent extra work occurs.

        Every evaluated reaction uses the state left by its predecessor. Blocked
        and randomly rejected reactions are paid; the first refusal ends work.
        The whole plan's ticket bits count as generated even when funding stops.
        These declared operation charges do not certify joules or real-time CPU
        deadlines. Snapshots are supported between complete windows, not inside.
        """
        if self.body.law.labels != 256:
            raise ValueError("tick requires the native 256-byte boundary")
        if type(plan) is not tuple or len(plan) > self.opportunities_per_tick:
            raise ValueError("reaction plan exceeds the registered finite window")
        planned_bits = 0
        for pair in plan:
            if type(pair) is not tuple or len(pair) != 2:
                raise ValueError("invalid reaction plan entry")
            index, ticket = pair
            bound = self.proposal_bound(index)
            if type(ticket) is not int or ticket < 0 or ticket.bit_length() > 2 * bound:
                raise ValueError("invalid fixed-width dyadic ticket")
            planned_bits += 2 * bound
        masses, total = self.masses()  # observer measurement BEFORE the target
        boundary = world.begin()
        observed = world.publish()
        evaluated = funded = accepted = material_delta = evaluated_bits = 0
        for index, ticket in plan:
            if not world.pay_write(self.body.law.attempt_fee):
                break
            funded += 1
            edits, exponent, bound, sign = self.prepare(index, observed.payload)
            evaluated += 1
            evaluated_bits += 2 * bound
            if exponent is not None and ticket >> (bound + exponent) == 0:
                self._commit(edits, sign)
                accepted += 1
                material_delta += sign
        return dict(tick=observed.tick, byte=observed.payload, active=boundary.active,
                    target_mass=masses[observed.payload], total_mass=total,
                    scheduled_attempts=len(plan), evaluated_attempts=evaluated,
                    funded_attempts=funded, accepted_reactions=accepted,
                    material_delta=material_delta, planned_random_bits=planned_bits,
                    evaluated_random_bits=evaluated_bits,
                    spent=funded * self.body.law.attempt_fee,
                    opportunities_per_tick=self.opportunities_per_tick)

    def state_dict(self):
        return dict(version=2, opportunities_per_tick=self.opportunities_per_tick,
                    reservoir=asdict(self.reservoir), body=self.body.state_dict(),
                    initial_content=self.initial_content, stock=self.stock,
                    intake=self.intake, released=self.released)

    def load_state_dict(self, snapshot):
        expected = self.state_dict()
        # The old interface meant exactly one opportunity per publication.
        if type(snapshot) is dict and type(snapshot.get("version")) is int and snapshot["version"] == 1:
            if self.opportunities_per_tick != 1 or "opportunities_per_tick" in snapshot:
                raise ValueError("legacy checkpoint has a different reaction clock")
            snapshot = dict(snapshot, version=2, opportunities_per_tick=1)
        if (type(snapshot) is not dict or set(snapshot) != set(expected)
                or type(snapshot["version"]) is not int or snapshot["version"] != 2
                or type(snapshot["opportunities_per_tick"]) is not int
                or snapshot["opportunities_per_tick"] != self.opportunities_per_tick
                or type(snapshot["reservoir"]) is not dict
                or set(snapshot["reservoir"]) != set(expected["reservoir"])
                or any(type(v) is not int or v != expected["reservoir"][k]
                       for k, v in snapshot["reservoir"].items())
                or any(type(snapshot[k]) is not int or snapshot[k] < 0
                       for k in ("stock", "intake", "released", "initial_content"))
                or snapshot["initial_content"] != self.initial_content):
            raise ValueError("open medium checkpoint mismatch")
        restored = LocalMedium(self.body.law, bytes(self.body.payload),
                               cache_totals=self.body._totals is not None)
        restored.load_state_dict(snapshot["body"])
        content = sum(sum(restored.get(e)) for e in range(restored.law.edges))
        if (content + snapshot["stock"] != self.initial_content + self.reservoir.source_initial
                or snapshot["stock"] != self.reservoir.source_initial
                - snapshot["intake"] + snapshot["released"]):
            raise ValueError("open material balance mismatch")
        self.body._payload, self.body._totals = restored._payload, restored._totals
        self.stock, self.intake, self.released = (snapshot[k] for k in ("stock", "intake", "released"))
