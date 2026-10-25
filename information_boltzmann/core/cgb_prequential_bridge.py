"""Causal prequential scoring and finite joint-state prediction/update ports.
Learners own state; delivered auxiliary input is transient conditional context.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import isfinite, log
from numbers import Integral
from random import Random
from typing import Any, Callable, Protocol

try:
    from .cgb_information_ledger import InformationLedgerRow, PredictiveInformationLedger
    from .cgb_low_resource_life import Observation
except ImportError:
    from cgb_information_ledger import InformationLedgerRow, PredictiveInformationLedger
    from cgb_low_resource_life import Observation


class PrequentialBridgeViolation(ValueError):
    """Raised when an observation cannot be admitted to the learner bridge."""


class PrequentialLearner(Protocol):
    def predict(self) -> Any: ...
    def observe(self, token: int, *, external: bool = True) -> float: ...


@dataclass(frozen=True)
class PrequentialResult:
    tick: int
    target: int
    prediction: Any
    model_nll_nats: float
    ledger_row: InformationLedgerRow


class CGBPrequentialBridge:
    """Admit contiguous observations to an existing online learner."""

    def __init__(self, learner: PrequentialLearner, ledger: PredictiveInformationLedger,
        reference_nll_nats: Callable[[int], float],
    ):
        self.learner = learner
        self.ledger = ledger
        self.reference_nll_nats = reference_nll_nats
        self.last_tick = ledger.next_tick - 1

    def step(
        self, observation: Observation, *,
        write_units: int, dissipation_units: int,
        calibrated_energy: float | None = None,
        auxiliary: tuple[int, ...] = (),
    ) -> PrequentialResult:
        auxiliary = _context(auxiliary)
        if observation is None or not observation.arrived:
            raise PrequentialBridgeViolation("continuous input contract violated")
        if observation.tick != self.last_tick + 1:
            raise PrequentialBridgeViolation(
                f"observation tick {observation.tick} is out of order; expected {self.last_tick + 1}"
            )
        target = observation.payload
        if isinstance(target, bool) or not isinstance(target, Integral):
            raise PrequentialBridgeViolation("token observation payload must be an integer")
        for name, value in (("write_units", write_units), ("dissipation_units", dissipation_units)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise PrequentialBridgeViolation(f"{name} must be a nonnegative integer")
        if calibrated_energy is not None and (not isfinite(calibrated_energy) or calibrated_energy < 0):
            raise PrequentialBridgeViolation("calibrated energy must be finite and nonnegative")
        kwargs = {"auxiliary": auxiliary} if auxiliary else {}
        validate = getattr(self.learner, "validate_input", None)
        if validate is not None:
            validate(int(target), **kwargs)
        reference_nll = float(self.reference_nll_nats(int(target)))
        if not isfinite(reference_nll) or reference_nll < 0:
            raise PrequentialBridgeViolation("reference NLL must be finite and nonnegative")
        prediction = self.learner.predict(**kwargs)
        model_nll_nats = float(self.learner.observe(int(target), external=True, **kwargs))
        row = self.ledger.record(tick=observation.tick, model_nll_nats=model_nll_nats,
            reference_nll_nats=reference_nll, write_units=write_units,
            dissipation_units=dissipation_units, calibrated_energy=calibrated_energy)
        self.last_tick = observation.tick
        return PrequentialResult(observation.tick, int(target), prediction, model_nll_nats, row)


def _context(value, bits=None):
    if (not isinstance(value, tuple) or len(value) > 62
            or any(type(x) is not int or not -(2**31) <= x < 2**31 for x in value)):
        raise PrequentialBridgeViolation("context must contain at most 62 int32 values")
    if bits is not None and (len(value) != 1 or not 0 <= value[0] < 2**bits):
        raise PrequentialBridgeViolation("joint context must encode the registered hidden bits")
    return value


@dataclass(frozen=True)
class PackedTreeFrame:
    """Immutable old-model conditional head; pure callbacks own allocation safety."""

    vocab_size: int
    codes: tuple[int, ...]
    context: tuple[int, ...]
    tick: int
    rule: Callable

    def _law(self, depth, prefix, observed=None):
        bits = tuple(b for b in (0, 1)
                     if ((2 * prefix + b) << (len(self.codes) - depth - 1)) < self.vocab_size)
        if observed is not None:
            if observed not in bits:
                raise PrequentialBridgeViolation("illegal token branch")
            bits = (observed,)
        def rule(code, config, ctx):
            hidden, selected = divmod(config, len(bits))
            bit = bits[selected]
            return self.rule(code, hidden, bit, ctx[:-2], ctx[-2], ctx[-1]), bit
        return rule, 2 * len(bits), self.context + (depth, prefix)

    def _path(self, token):
        if type(token) is not int or not 0 <= token < self.vocab_size:
            raise PrequentialBridgeViolation("token outside vocabulary")
        prefix = 0
        for depth in range(len(self.codes)):
            bit = (token >> (len(self.codes) - depth - 1)) & 1
            yield depth, prefix, bit
            prefix = 2 * prefix + bit

    def probability(self, token):
        probability = Fraction(1)
        for depth, prefix, bit in self._path(token):
            rule, count, context = self._law(depth, prefix)
            total, event, _, _ = kernel.local_masses(
                self.codes[depth], self.codes[depth], context, count, bit, rule)
            probability *= Fraction(event, total)
        return probability

    def nll(self, token):
        p = self.probability(token)
        return log(p.denominator) - log(p.numerator)

    def sample(self, rng):
        prefix = 0
        for depth, code in enumerate(self.codes):
            rule, count, context = self._law(depth, prefix)
            total = kernel.local_masses(code, code, context, count, 0, rule)[0]
            _, bit = kernel.local_sample(code, context, count, rule, rng.randrange(total))
            prefix = 2 * prefix + bit
        return prefix


class PackedTreePort:
    """Joint hidden-state transition; explicit advance retains legacy semantics.
    One selected depth changes per observation; no optimizer or temporal graph."""

    def __init__(self, vocab_size, rule, advance=None, context=(), seed=0, learning=True, *, rule_id=None):
        global torch, kernel
        import torch
        try:
            from . import cgb_language_kernel as kernel
        except ImportError:
            import cgb_language_kernel as kernel
        self.rule_id = rule_id if rule_id is not None else getattr(rule, "rule_id", None)
        if (type(vocab_size) is not int or not 1 <= vocab_size < 2**31
                or not callable(rule) or (advance is not None and not callable(advance)) or type(seed) is not int
                or type(learning) is not bool or not isinstance(self.rule_id, str) or not self.rule_id):
            raise PrequentialBridgeViolation("invalid vocabulary, callbacks, seed or registered rule_id")
        self.vocab_size, self.rule, self.advance = vocab_size, rule, advance
        self.depths = (vocab_size - 1).bit_length()
        context = (0,) if advance is None and context == () else context
        self.context = _context(context, self.depths if advance is None else None)
        self.seed, self.learning = seed, learning
        self.rng, self.tick, self.updates = Random(seed), 0, 0
        self.generation_rng = Random(seed ^ 0x5DEECE66D)
        codes = [self.rng.randrange(16) for _ in range(self.depths)]
        self.packed = torch.tensor([codes[i] | ((codes[i + 1] if i + 1 < len(codes) else 8) << 4)
                                    for i in range(0, len(codes), 2)], dtype=torch.uint8)

    def validate_input(self, token, auxiliary=()):
        """Pure preflight before consuming a reference value or updating state."""
        if type(token) is not int or not 0 <= token < self.vocab_size:
            raise PrequentialBridgeViolation("token outside vocabulary")
        _context(self.context + _context(auxiliary))

    def predict(self, auxiliary=()):
        codes = tuple(((int(self.packed[i // 2]) >> (4 * (i % 2))) & 15) - 8
                      for i in range(self.depths))
        return PackedTreeFrame(self.vocab_size, codes,
                               _context(self.context + _context(auxiliary)), self.tick, self.rule)

    def observe(self, token, *, external=True, auxiliary=()):
        if external is not True:
            raise PrequentialBridgeViolation("generated tokens are not external observations")
        posterior_rng = Random()
        posterior_rng.setstate(self.rng.getstate())
        frame = self.predict(auxiliary=auxiliary)
        nll = frame.nll(token)
        hidden = []
        for depth, prefix, bit in frame._path(token):
            rule, count, context = frame._law(depth, prefix, bit)
            code = frame.codes[depth]
            total = kernel.local_masses(code, code, context, count, bit, rule)[0]
            selected, _ = kernel.local_sample(code, context, count, rule, posterior_rng.randrange(total))
            hidden.append(selected)
        context_next = ((sum(h << d for d, h in enumerate(hidden)),) if self.advance is None
                        else _context(self.advance(frame.context, token, tuple(hidden))))
        if self.learning and self.depths:
            selected = self.tick % self.depths
            delta = 1 if (self.tick // self.depths) % 2 == 0 else -1
            depth, prefix, bit = tuple(frame._path(token))[selected]
            if -8 <= frame.codes[depth] + delta <= 7:
                rule, count, context = frame._law(depth, prefix)
                accepted, _ = kernel.local_edit(self.packed, depth, delta, context, count, bit, rule)
                self.updates += int(accepted)
        self.context, self.tick, self.rng = context_next, self.tick + 1, posterior_rng
        return nll

    def sample(self):
        rng = Random()
        rng.setstate(self.generation_rng.getstate())
        token = self.predict().sample(rng)
        self.generation_rng = rng
        return token

    def state_dict(self):
        return dict(version=2 if self.advance is None else 1, vocab_size=self.vocab_size, rule_id=self.rule_id, seed=self.seed,
                    transition="joint" if self.advance is None else "legacy",
                    learning=self.learning, packed=tuple(self.packed.tolist()), context=self.context,
                    tick=self.tick, updates=self.updates, rng_state=self.rng.getstate(),
                    generation_rng_state=self.generation_rng.getstate())

    def load_state_dict(self, state):
        def tuples(value):
            return tuple(tuples(x) for x in value) if isinstance(value, (list, tuple)) else value
        if (not isinstance(state, dict) or type(state.get("version")) is not int
                or state["version"] != (2 if self.advance is None else 1) or type(state.get("vocab_size")) is not int
                or state["vocab_size"] != self.vocab_size
                or state.get("transition", "legacy") != ("joint" if self.advance is None else "legacy")
                or state.get("rule_id") != self.rule_id or type(state.get("seed")) is not int
                or any(not isinstance(state.get(k), (list, tuple)) or len(state[k]) != 3 for k in ("rng_state", "generation_rng_state"))
                or type(state.get("learning")) is not bool
                or any(type(state.get(k)) is not int or state[k] < 0 for k in ("tick", "updates"))
                or state["updates"] > state["tick"]):
            raise PrequentialBridgeViolation("checkpoint identity or counters do not match")
        context = _context(tuples(state["context"]), self.depths if self.advance is None else None)
        packed = tuples(state["packed"])
        if (not isinstance(packed, tuple) or len(packed) != (self.depths + 1) // 2
                or any(type(x) is not int or not 0 <= x <= 255 for x in packed)
                or (self.depths % 2 and packed[-1] >> 4 != 8)):
            raise PrequentialBridgeViolation("invalid packed checkpoint")
        rng = Random()
        rng.setstate(tuples(state["rng_state"]))
        generation_rng = Random()
        generation_rng.setstate(tuples(state["generation_rng_state"]))
        self.packed = torch.tensor(packed, dtype=torch.uint8)
        self.context, self.rng = context, rng
        self.generation_rng = generation_rng
        self.seed, self.learning, self.tick, self.updates = (
            state[k] for k in ("seed", "learning", "tick", "updates"))
