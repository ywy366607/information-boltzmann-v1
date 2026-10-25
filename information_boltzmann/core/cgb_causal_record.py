"""Support records for the existing live-body world actuator port.

An exact, known-model inference reference; not a learned causal model or a
paid physical learner. World transitions use the PRE-update body's action.
Only the evaluator supplies a true world state to live_world_weights(). The
record updater receives a candidate mask, executed action and next sensor.
Vacuum, missed sensors and the marginal-ghost cut require separate contracts.
Delayed receipts below refer to the source of ONE observed world move; they
do not identify the current world or a long-delayed multistep reward's cause.
"""

from __future__ import annotations

WORLD_NAMES = ("feedA", "feedB", "missA", "missB", "probeA", "probeB",
               "answerA", "answerB", "trap")
SIGNALS = ("calm", "answerA", "answerB", "feed", "scarce", "miss", "probe")
WORLD_SENSORS = ("feed", "feed", "miss", "miss", "probe", "probe",
                 "answerA", "answerB", "scarce")
ALL_WORLDS = (1 << len(WORLD_NAMES)) - 1


def _integer(value: int, upper: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < upper:
        raise ValueError(f"expected integer in 0..{upper - 1}")


def live_world_weights(world: str, executed_action: int) -> tuple[int, ...]:
    """Original GraceRecovery world probabilities multiplied by two.

    The live-body world law depends on its policy output alone. Evaluating
    another action characterizes that port, without resetting a body or
    claiming its current policy can choose that action.
    """
    if world not in WORLD_NAMES:
        raise ValueError("unknown world state")
    _integer(executed_action, 3)
    a = executed_action
    if world in ("feedA", "feedB"):
        branch = world[-1]
        if a == 2:
            target = "answer" + branch
        elif a == (0 if branch == "A" else 1):
            return (1, 1, 0, 0, 0, 0, 0, 0, 0)
        else:
            target = "miss" + branch
    elif world in ("probeA", "probeB"):
        target = ("answer" if a == 2 else "miss") + world[-1]
    elif world in ("missA", "missB"):
        target = "probe" + world[-1]
    elif world in ("answerA", "answerB"):
        target = "feed" + world[-1]
    else:
        target = "trap"
    return tuple(2 if name == target else 0 for name in WORLD_NAMES)


def observe_support(prior: int, sensor: str) -> int:
    """Intersect existing candidates with an actual current sensor."""
    _integer(prior, 512)
    if sensor not in SIGNALS:
        raise ValueError("unknown sensor")
    return sum(1 << i for i, value in enumerate(WORLD_SENSORS)
               if prior & (1 << i) and value == sensor)


def update_live_support(prior: int, executed_action: int, next_sensor: str) -> int:
    """Smallest support containing every compatible possible successor.

    This stores possibilities, not their probabilities. The caller must know
    the body was live when this action entered the world law and must supply
    the corresponding next observation. Unknown/delayed action matching is
    not resolved by this interface. Empty output reports inconsistent evidence.
    """
    _integer(prior, 512)
    _integer(executed_action, 3)
    if next_sensor not in SIGNALS:
        raise ValueError("unknown sensor")
    predicted = 0
    for i, world in enumerate(WORLD_NAMES):
        if prior & (1 << i):
            for j, mass in enumerate(live_world_weights(world, executed_action)):
                if mass:
                    predicted |= 1 << j
    return observe_support(predicted, next_sensor)


def source_screen(prior: int, executed_action: int, next_sensor: str) -> int:
    """Filter hypotheses about the frozen source, rather than its successors."""
    _integer(prior, 512)
    _integer(executed_action, 3)
    if next_sensor not in SIGNALS:
        raise ValueError("unknown sensor")
    return sum(1 << w for w, world in enumerate(WORLD_NAMES)
               if prior & (1 << w) and any(
                   mass and WORLD_SENSORS[v] == next_sensor
                   for v, mass in enumerate(live_world_weights(world, executed_action))))


def _natural(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("expected nonnegative integer")


def paid_source_update(prior: int, reserve: int, executed_action: int,
                       next_sensor: str, overhead: int) -> tuple[int, int, int, bool]:
    """Prepay a read, then use the existing one-credit effective-write rule.

    Return source candidates, reserve, total fee and read admission. These
    are declared write-credit units, not measured physical energy. A funded
    read with no remaining write credit retains the original source candidates.
    """
    screened = source_screen(prior, executed_action, next_sensor)
    _natural(reserve)
    _natural(overhead)
    if overhead > reserve:
        return prior, reserve, 0, False
    after_read = reserve - overhead
    write = int(after_read > 0 and screened != prior)
    return (screened if write else prior, after_read - write, overhead + write, True)


class DelayedSourceCarrier:
    """One self-contained receipt with a separately endowed credit reserve.

    capture() accepts an already paired receipt and trusts its provenance.
    For an actual boundary sequence, begin()/complete() separately pay for
    storing context and appending its immediately following sensor. The
    SensoryRecordObserver adapter calls tick() once per world interval.
    Each admitted capture/holding tick/read is charged its declared tariff.
    Unfunded holding makes the receipt unavailable, without claiming a free
    physical erasure. Busy captures are rejected; world input need not wait.
    """

    def __init__(self, reserve: int, *, capture_fee: int, holding_fee: int, read_fee: int):
        for value in (reserve, capture_fee, holding_fee, read_fee):
            _natural(value)
        self._reserve = reserve
        self._spent = 0
        self._capture_fee = capture_fee
        self._holding_fee = holding_fee
        self._read_fee = read_fee
        self._receipt: tuple[int, int, str] | None = None
        self._pending: tuple[int, int] | None = None
        self._status = "idle"

    @property
    def reserve(self) -> int:
        return self._reserve

    @property
    def spent(self) -> int:
        return self._spent

    @property
    def status(self) -> str:
        return self._status

    def _pay(self, tariff: int) -> bool:
        if tariff > self._reserve:
            return False
        self._reserve -= tariff
        self._spent += tariff
        return True

    def capture(self, prior: int, executed_action: int, actual_next_sensor: str) -> bool:
        source_screen(prior, executed_action, actual_next_sensor)  # Validate only.
        if self._status in ("waiting", "awaiting") or not self._pay(self._capture_fee):
            return False
        self._receipt = (prior, executed_action, actual_next_sensor)
        self._pending = None
        self._status = "waiting"
        return True

    def tick(self) -> bool:
        if self._status not in ("waiting", "awaiting"):
            return False
        if not self._pay(self._holding_fee):
            self._receipt = None
            self._pending = None
            self._status = "expired"
            return False
        return True

    def begin(self, prior: int, executed_action: int) -> bool:
        """Pay to save the context BEFORE its next-world sensor exists."""
        _integer(prior, 512)
        _integer(executed_action, 3)
        if self._status in ("waiting", "awaiting") or not self._pay(self._capture_fee):
            return False
        self._pending = (prior, executed_action)
        self._receipt = None
        self._status = "awaiting"
        return True

    def complete(self, actual_next_sensor: str | None) -> bool:
        """Append the immediately following sensor, paying another capture fee.

        None denotes sensory loss. Discard the unmatched context then; later
        observations cannot be substituted for this missing one-step outcome.
        """
        if actual_next_sensor is not None and actual_next_sensor not in SIGNALS:
            raise ValueError("unknown sensor")
        if self._status != "awaiting":
            return False
        assert self._pending is not None
        if actual_next_sensor is None or not self._pay(self._capture_fee):
            self._pending = None
            self._status = "unobserved" if actual_next_sensor is None else "expired"
            return False
        prior, action = self._pending
        self._receipt = (prior, action, actual_next_sensor)
        self._pending = None
        self._status = "waiting"
        return True

    def read(self) -> tuple[int, int, str] | None:
        """Consume once; return source candidates and the original action/sensor."""
        if self._status != "waiting":
            return None
        assert self._receipt is not None
        prior, action, sensor = self._receipt
        candidates, reserve, fee, admitted = paid_source_update(
            prior, self._reserve, action, sensor, self._read_fee)
        self._reserve = reserve
        self._spent += fee
        self._receipt = None
        self._status = "consumed" if admitted else "expired"
        return (candidates, action, sensor) if admitted else None
