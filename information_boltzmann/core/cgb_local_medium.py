"""Closed local reactions of the registered LG-1 integer energy family.

One shared state holds end charges (p,q) and plastic slots w. The rook contact
graph is implicit. Rates realize pi proportional to 2**(-2*E), without reading
the global energy or normalizer. This is a reaction reference, not an admitted
language organism or an equivalence to the LG1 mean-field approximation.
"""

from dataclasses import asdict, dataclass
from functools import cached_property


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
    Uniform acceptance tickets are explicit stochastic primitives. No index
    array, aggregate cache, eligibility array or autograd history is stored.
    """

    def __init__(self, law: LocalLaw, payload: bytes):
        if not isinstance(law, LocalLaw) or type(payload) is not bytes:
            raise ValueError("invalid local medium")
        bits = law.bits * law.edges
        if len(payload) != law.state_bytes or (bits % 8 and payload[-1] >> (bits % 8)):
            raise ValueError("invalid packed shape or padding")
        self.law, self.payload = law, bytearray(payload)
        for edge in range(law.edges):
            p, q, w = self.get(edge)
            if max(p, q) > law.charge_cap or w > law.slot_cap:
                raise ValueError("packed state exceeds capacity")
        _ = law.bounds  # compile static rate bounds once, not during local events

    def get(self, edge):
        if type(edge) is not int or not 0 <= edge < self.law.edges:
            raise ValueError("invalid implicit edge")
        start, shift = divmod(edge * self.law.bits, 8)
        width = (shift + self.law.bits + 7) // 8
        word = int.from_bytes(self.payload[start:start + width], "little")
        code = (word >> shift) & ((1 << self.law.bits) - 1)
        if code >= self.law.states:
            raise ValueError("unused local code")
        pq, w = divmod(code, self.law.slot_cap + 1)
        p, q = divmod(pq, self.law.charge_cap + 1)
        return p, q, w

    def _put(self, edge, state):
        p, q, w = state
        code = (p * (self.law.charge_cap + 1) + q) * (self.law.slot_cap + 1) + w
        start, shift = divmod(edge * self.law.bits, 8)
        width = (shift + self.law.bits + 7) // 8
        old = int.from_bytes(self.payload[start:start + width], "little")
        mask = ((1 << self.law.bits) - 1) << shift
        word = (old & ~mask) | (code << shift)
        self.payload[start:start + width] = word.to_bytes(width, "little")

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
            ids = (range(row * f.labels, (row + 1) * f.labels) if channel == 0
                   else range(col, f.edges, f.labels))
            total = sum(self.get(e)[channel] for e in ids)
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
        restored = LocalMedium(self.law, bytes.fromhex(snapshot["payload"]))
        self.payload = restored.payload
