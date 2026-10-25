"""Finite real-byte world with paid, delayed replay of its public prefix.

The immutable archive belongs to the environment, not the learner. Tariffs
are declared units, not joules; this interface proves no learning capability
or infinite supply. One publish is one world tick, including after failure.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib


class ContractViolation(RuntimeError):
    """A source, timing, resource, or restoration contract was violated."""


@dataclass(frozen=True)
class Observation:
    tick: int
    payload: object
    arrived: bool = True


@dataclass(frozen=True)
class BoundaryFrame:
    tick: int
    active: bool
    receipt: bytes | None


class PrefixReplayWorld:
    """Single-slot prefix replay; callbacks never receive the private source.

    begin supplies resources, pays maintenance, then holding and due reading.
    Holding is charged once per elapsed interval, including the delivery tick.
    Repeated begin is idempotent until publish. Requests follow publication;
    origin excludes earlier archive bytes from the public prefix. A receipt is
    available in its open frame and is consumed when that frame is published.
    """

    COSTS = ("maintenance_spent", "request_spent", "holding_spent", "read_spent")
    COUNTS = COSTS + ("offered", "spill", "accepted", "delivered", "cancelled",
                      "replayed_bytes", "fresh")

    def __init__(self, source: bytes, *, source_id: str, origin=0, reserve=0,
                 capacity=32, maintenance=1, request_fee=1, holding_fee=1,
                 read_per_byte=1, delay=1, max_block=64, supply_per_tick=0):
        if type(source) is not bytes or type(source_id) is not str or not source_id:
            raise ContractViolation("source must be immutable bytes with an audit identity")
        values = dict(origin=origin, initial_reserve=reserve, capacity=capacity,
                      maintenance=maintenance, request_fee=request_fee,
                      holding_fee=holding_fee, read_per_byte=read_per_byte,
                      delay=delay, max_block=max_block, supply_per_tick=supply_per_tick)
        if (any(type(v) is not int or v < 0 for v in values.values())
                or origin > len(source) or reserve > capacity or maintenance < 1 or delay < 1 or max_block < 1):
            raise ContractViolation("invalid world range, tariff, capacity, or delay")
        self._source = source
        self._config = dict(values, source_id=source_id, source_bytes=len(source),
                            source_sha256=hashlib.sha256(source).hexdigest())
        self._state = dict(tick=0, cursor=origin, phase="ready", reserve=reserve,
                           active=True, pending=None, receipt=None,
                           counts=dict.fromkeys(self.COUNTS, 0))

    def _pay(self, category, amount):
        state = self._state
        if state["reserve"] < amount:
            return False
        state["reserve"] -= amount
        state["counts"][category] += amount
        return True

    def _cancel(self):
        if self._state["pending"] is not None:
            self._state["pending"] = None
            self._state["counts"]["cancelled"] += 1

    def _frame(self):
        state = self._state
        receipt = state["receipt"]
        return BoundaryFrame(state["tick"], state["active"],
                             None if receipt is None else receipt["payload"])

    def begin(self) -> BoundaryFrame:
        state, config = self._state, self._config
        if state["phase"] == "open":
            return self._frame()
        if state["cursor"] == len(self._source):
            raise StopIteration("finite public byte source exhausted")
        state["phase"], state["receipt"] = "open", None
        amount = config["supply_per_tick"]
        admitted = min(amount, config["capacity"] - state["reserve"])
        state["reserve"] += admitted
        state["counts"]["offered"] += amount
        state["counts"]["spill"] += amount - admitted
        if state["active"]:
            required = config["maintenance"]
            paid = min(required, state["reserve"])
            self._pay("maintenance_spent", paid)
            if paid < required:
                state["active"] = False
                self._cancel()
        pending = state["pending"]
        if pending is not None:
            if not self._pay("holding_spent", config["holding_fee"]):
                self._cancel()
            elif pending["due"] <= state["tick"]:
                if self._pay("read_spent", len(pending["payload"]) * config["read_per_byte"]):
                    state["receipt"], state["pending"] = pending, None
                    state["counts"]["delivered"] += 1
                    state["counts"]["replayed_bytes"] += len(pending["payload"])
                else:
                    self._cancel()
        return self._frame()

    def publish(self) -> Observation:
        state = self._state
        if state["phase"] != "open":
            raise ContractViolation("publication requires an open boundary frame")
        observation = Observation(state["tick"], self._source[state["cursor"]])
        state["cursor"] += 1
        state["tick"] += 1
        state["counts"]["fresh"] += 1
        state["phase"], state["receipt"] = "published", None
        return observation

    def request(self, lag: int, length: int) -> bool:
        state, config = self._state, self._config
        if state["phase"] != "published":
            raise ContractViolation("requests must follow this tick's publication")
        if (type(lag) is not int or lag < 0 or type(length) is not int
                or not 1 <= length <= config["max_block"]):
            raise ContractViolation("invalid replay lag or bounded length")
        end, start = state["cursor"] - lag, state["cursor"] - lag - length
        if not config["origin"] <= start < end <= state["cursor"]:
            raise ContractViolation("request is outside the actually published prefix")
        if (not state["active"] or state["pending"] is not None
                or state["reserve"] < config["request_fee"]):
            return False
        payload = self._source[start:end]
        self._pay("request_spent", config["request_fee"])
        requested_at = state["tick"] - 1
        state["pending"] = dict(start=start, end=end, requested_at=requested_at,
                                due=requested_at + config["delay"], payload=payload)
        state["counts"]["accepted"] += 1
        return True

    def state_dict(self) -> dict:
        """Complete JSON checkpoint; source identity and costs are audit-only."""
        state = dict(self._state, counts=dict(self._state["counts"]))
        for key in ("pending", "receipt"):
            record = state[key]
            state[key] = None if record is None else dict(record, payload=list(record["payload"]))
        return dict(version=1, config=dict(self._config), state=state)

    def load_state_dict(self, snapshot: dict) -> None:
        """Validate a complete checkpoint before replacing any live state.

        Integrity checks certify source/range/accounting invariants, not the
        authenticity of an untrusted checkpoint's claimed exposure history.
        """
        def require(condition, message):
            if not condition:
                raise ContractViolation(message)

        require(type(snapshot) is dict and set(snapshot) == {"version", "config", "state"},
                "invalid world checkpoint fields")
        require(type(snapshot["version"]) is int and snapshot["version"] == 1,
                "unsupported world checkpoint version")
        config = snapshot["config"]
        require(type(config) is dict and set(config) == set(self._config)
                and all(type(config[k]) is type(v) and config[k] == v
                        for k, v in self._config.items()), "world/source configuration mismatch")
        raw = snapshot["state"]
        require(type(raw) is dict and set(raw) == set(self._state), "invalid world state fields")
        state = dict(raw)
        counts = state["counts"]
        require(type(counts) is dict and set(counts) == set(self.COUNTS)
                and all(type(v) is int and v >= 0 for v in counts.values()), "invalid world counters")
        state["counts"] = dict(counts)
        require(all(type(state[k]) is int and state[k] >= 0 for k in ("tick", "cursor", "reserve"))
                and type(state["active"]) is bool and type(state["phase"]) is str,
                "invalid world control types")
        tick, phase = state["tick"], state["phase"]
        require(phase in ("ready", "open", "published")
                and state["cursor"] == config["origin"] + tick
                and state["cursor"] <= len(self._source)
                and state["reserve"] <= config["capacity"] and counts["fresh"] == tick,
                "world clock, cursor, or capacity mismatch")
        begins = tick + int(phase == "open")
        require((phase != "open" or state["cursor"] < len(self._source))
                and (phase != "published" or tick > 0), "impossible world phase")
        require(counts["offered"] == begins * config["supply_per_tick"]
                and counts["spill"] <= counts["offered"]
                and state["reserve"] + sum(counts[k] for k in self.COSTS) + counts["spill"]
                == config["initial_reserve"] + counts["offered"], "world resource balance mismatch")
        require(counts["request_spent"] == counts["accepted"] * config["request_fee"]
                and counts["read_spent"] == counts["replayed_bytes"] * config["read_per_byte"]
                and counts["accepted"] <= tick
                and counts["delivered"] <= counts["replayed_bytes"]
                <= counts["delivered"] * config["max_block"], "world replay accounting mismatch")
        maintenance, holding = config["maintenance"], config["holding_fee"]
        require(counts["holding_spent"] <= max(0, begins - 1) * holding
                and (counts["holding_spent"] % holding == 0 if holding else counts["holding_spent"] == 0),
                "world holding accounting mismatch")
        require(counts["delivered"] * config["delay"] <= max(0, begins - 1),
                "world deliveries exceed elapsed request intervals")
        require((counts["maintenance_spent"] == begins * maintenance if state["active"] else
                 maintenance > 0 and counts["maintenance_spent"] < begins * maintenance),
                "world failure/maintenance mismatch")
        for key in ("pending", "receipt"):
            record = state[key]
            if record is None:
                continue
            require(type(record) is dict and set(record) == {"start", "end", "requested_at", "due", "payload"},
                    "invalid replay record fields")
            require(all(type(record[k]) is int and record[k] >= 0
                        for k in ("start", "end", "requested_at", "due")), "invalid replay record types")
            payload = record["payload"]
            require(type(payload) is list and 1 <= len(payload) <= config["max_block"]
                    and all(type(v) is int and 0 <= v <= 255 for v in payload), "invalid replay payload")
            start, end, requested_at, due = (record[k] for k in ("start", "end", "requested_at", "due"))
            require(config["origin"] <= start < end <= config["origin"] + requested_at + 1
                    and requested_at < tick and end - start == len(payload)
                    and due == requested_at + config["delay"]
                    and bytes(payload) == self._source[start:end], "replay source/timing mismatch")
            require(state["active"] and phase != "ready", "inactive world retains replay")
            require((due > tick if phase == "open" else due >= tick) if key == "pending" else
                    phase == "open" and due == tick and counts["delivered"] > 0
                    and len(payload) <= counts["replayed_bytes"],
                    "replay phase mismatch")
            state[key] = dict(record, payload=bytes(payload))
        require(counts["accepted"] - counts["delivered"] - counts["cancelled"]
                == int(state["pending"] is not None), "single-slot request accounting mismatch")
        require(state["receipt"] is None or state["pending"] is None,
                "receipt and pending request cannot coexist")
        pending = state["pending"]
        held = 0 if pending is None else begins - pending["requested_at"] - 1
        require(counts["holding_spent"] >= holding * (held + counts["delivered"] * config["delay"]),
                "world checkpoint omits paid replay intervals")
        if phase == "ready":
            require(tick == 0 and state["active"] and state["reserve"] == config["initial_reserve"]
                    and not any(counts.values()) and state["pending"] is None and state["receipt"] is None,
                    "invalid initial world checkpoint")
        self._state = state
