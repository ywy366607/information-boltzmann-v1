"""Causal prequential scoring and finite joint-state prediction/update ports.
Learners own state; delivered auxiliary input is transient conditional context.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, fields, is_dataclass
from fractions import Fraction
from math import isfinite, log
from numbers import Integral, Real
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


def _clone_numeric_state(value: Any) -> Any:
    """Copy a state tree as values, never as an autograd continuation."""
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().clone()
    if is_dataclass(value) and not isinstance(value, type):
        return type(value)(**{field.name: _clone_numeric_state(getattr(value, field.name))
                              for field in fields(value)})
    if isinstance(value, dict):
        return {key: _clone_numeric_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_numeric_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_numeric_state(item) for item in value)
    return copy.deepcopy(value)


@dataclass(frozen=True)
class GDNBytePortSnapshot:
    """A read-only branch point for a declared external-continuation audit.

    This is evaluation scaffolding, not organism state.  The GDN state and,
    when it can learn locally, the affine byte head are copied so a probe can
    replay a possible external continuation without changing the individual.
    The memory kernel, query and observation encoder are deliberately shared
    and must remain fixed during one audit.
    """

    state: Any
    head_state: dict[str, Any] | None
    diagnostics: Any
    port_state: Any = None


class GDNByteReadoutPort:
    """Interface-only causal byte readout over one existing GDN state.

    The caller supplies the state-dependent query, byte head and observation
    encoder. An optional local two-phase update trains an affine byte head;
    it does not train the query or observation encoder. Fast state is
    updated only after the byte arrives, without keeping an autograd history.
    The world may explicitly advance elapsed time through the existing GDN
    process prior before prediction. Action and full
    checkpointing are not provided by this interface.
    """

    def __init__(self, memory: Any, state: Any, query: Callable, head: Callable,
                 encode_observation: Callable, *, head_step_size: Real = 0):
        import torch
        if (isinstance(head_step_size, bool) or not isinstance(head_step_size, Real)
                or not isfinite(head_step_size) or head_step_size < 0):
            raise PrequentialBridgeViolation("head step size must be a finite nonnegative real")
        if head_step_size and (not isinstance(head, torch.nn.Linear) or head.out_features != 256):
            raise PrequentialBridgeViolation("local head learning requires an affine 256-byte head")
        self.memory = memory
        self.state = state
        self.query = query
        self.head = head
        self.encode_observation = encode_observation
        self.head_step_size = float(head_step_size)
        self._pending = None
        self._pending_features = None

    def validate_input(self, token: int) -> None:
        if isinstance(token, bool) or not isinstance(token, Integral) or not 0 <= token < 256:
            raise PrequentialBridgeViolation("GDN byte must be an integer from 0 to 255")

    def advance_time(self, elapsed: Real) -> None:
        """Apply the existing process prior before preparing the next prediction.

        ``elapsed`` is supplied by the world, not inferred from a byte count.
        A prepared prediction must be consumed first; zero time is an exact
        no-op. No per-tick state history or autograd graph is retained.
        """
        import torch
        if self._pending is not None:
            raise PrequentialBridgeViolation("observe the pending GDN prediction before advancing time")
        if isinstance(elapsed, bool) or not isinstance(elapsed, Real) or not isfinite(elapsed) or elapsed < 0:
            raise PrequentialBridgeViolation("elapsed time must be a finite nonnegative real")
        if elapsed:
            with torch.no_grad():
                self.state = self.memory.dynamic_prior(self.state, elapsed)

    def predict(self) -> Any:
        import torch
        if self._pending is not None:
            raise PrequentialBridgeViolation("observe the pending GDN prediction first")
        with torch.no_grad():
            keys = self.query(self.state)
            expected = (1, 1, self.memory.n_slots)
            if not isinstance(keys, torch.Tensor) or tuple(keys.shape) != expected:
                raise PrequentialBridgeViolation("GDN byte query must have shape [1, 1, slots]")
            features = self.memory.association_readout(self.state, keys)[0, 0]
            logits = self.head(features)
            if not isinstance(logits, torch.Tensor) or tuple(logits.shape) != (256,) or not logits.is_floating_point():
                raise PrequentialBridgeViolation("GDN byte head must return 256 floating logits")
            if not bool(torch.isfinite(logits).all()):
                raise PrequentialBridgeViolation("GDN byte logits must be finite")
            self._pending = logits.detach().clone()
            if self.head_step_size:
                self._pending_features = features.detach().clone()
        return self._pending.clone()

    def observe(self, token: int, *, external: bool = True) -> float:
        import torch
        self.validate_input(token)
        if self._pending is None:
            raise PrequentialBridgeViolation("predict before observing a GDN byte")
        with torch.no_grad():
            nll = float(-torch.log_softmax(self._pending, dim=0)[int(token)].item())
        self.observe_pre_scored(token, external=external)
        return nll

    def observe_pre_scored(self, token: int, *, external: bool = True) -> None:
        """Consume a byte whose prepared prediction was already scored."""
        import torch
        self.validate_input(token)
        if self._pending is None:
            raise PrequentialBridgeViolation("predict before observing a GDN byte")
        with torch.no_grad():
            encoded = self.encode_observation(int(token))
            if not isinstance(encoded, tuple) or len(encoded) not in (3, 5):
                raise PrequentialBridgeViolation(
                    "GDN observation encoder must return value/center/precision "
                    "or value/address/likelihood/center/precision"
                )
            if len(encoded) == 3:
                tokens, centers, precision = encoded
                address_tokens = likelihood_tokens = None
            else:
                tokens, address_tokens, likelihood_tokens, centers, precision = encoded
            next_state = self.memory.assimilate(
                self.state, tokens, centers, precision, update_motion=False,
                address_tokens=address_tokens, likelihood_tokens=likelihood_tokens)
            if self.head_step_size:
                # G14: the observed-minus-predicted output times the
                # prepared feature. This is a current-byte head update;
                # it carries no gradient through earlier states.
                innovation = -torch.softmax(self._pending, dim=0)
                innovation[int(token)] += 1
                self.head.weight.addmm_(innovation.unsqueeze(1),
                                        self._pending_features.unsqueeze(0),
                                        alpha=self.head_step_size)
                if self.head.bias is not None:
                    self.head.bias.add_(innovation, alpha=self.head_step_size)
            self.state = next_state
            self._pending = None
            self._pending_features = None

    def snapshot(self) -> GDNBytePortSnapshot:
        """Capture the complete mutable predictor state at a causal boundary.

        A pending prediction has already selected a temporal boundary and is
        therefore not a legal branch point.  The optional affine head belongs
        to the predictor whenever local learning is enabled, so it is included
        in the snapshot.  No extra history is retained after the audit.
        """
        if self._pending is not None:
            raise PrequentialBridgeViolation("cannot snapshot a pending GDN prediction")
        head_state = None
        if self.head_step_size:
            head_state = {name: value.detach().clone()
                          for name, value in self.head.state_dict().items()}
        return GDNBytePortSnapshot(
            state=_clone_numeric_state(self.state),
            head_state=head_state,
            diagnostics=_clone_numeric_state(getattr(self.memory, "last_diagnostics", None)),
        )

    def restore(self, snapshot: GDNBytePortSnapshot) -> None:
        """Restore a snapshot without retaining an audit branch in the port."""
        if not isinstance(snapshot, GDNBytePortSnapshot):
            raise PrequentialBridgeViolation("invalid GDN snapshot")
        if self._pending is not None:
            raise PrequentialBridgeViolation("cannot restore over a pending GDN prediction")
        self.state = _clone_numeric_state(snapshot.state)
        if snapshot.head_state is not None:
            self.head.load_state_dict({name: value.detach().clone()
                                       for name, value in snapshot.head_state.items()})
        if hasattr(self.memory, "last_diagnostics"):
            self.memory.last_diagnostics = _clone_numeric_state(snapshot.diagnostics)

    def external_future_signature(self, continuation: Any) -> Any:
        """Read the post-continuation byte prediction without changing this port.

        ``continuation`` yields ``(elapsed, StreamEvent)`` pairs from an
        external world.  Its events must carry a nonempty source other than
        ``"self"`` and valid byte payloads.  The returned 256-logit vector is
        the runtime counterpart of a registered future-language signature:
        it records what this *same* predictor would say after that external
        continuation.  It is not an extra reserve counter, and equality or
        tolerance between signatures remains a declared evaluation contract.
        """
        snapshot = self.snapshot()
        try:
            for elapsed, event in continuation:
                source = getattr(event, "source", None)
                token = getattr(event, "payload", None)
                if not isinstance(source, str) or not source or source == "self":
                    raise PrequentialBridgeViolation(
                        "future GDN continuation must use a sourced external event"
                    )
                self.validate_input(token)
                self.advance_time(elapsed)
                self.predict()
                self.observe_pre_scored(int(token), external=True)
            logits = self.predict()
            # Reading this final signature must not leave a fabricated pending
            # event in the live port.  `restore` below then reinstates the
            # original causal boundary exactly.
            self._pending = None
            self._pending_features = None
            return logits.detach().clone()
        finally:
            self._pending = None
            self._pending_features = None
            self.restore(snapshot)


class LastByteGDNReadoutPort(GDNByteReadoutPort):
    """The concrete causal byte port used by the fixed GDN Lean model.

    Its query is not a caller-supplied callback: at birth it is a declared
    seed distribution, and after an arrival it is the address of the stored
    last byte encoding.  This is the runtime form of
    ``GDNLife.query``.  The encoder, address kernel and seed logits are model
    parameters, but this port deliberately supplies no unproved update rule
    for them; the existing local two-phase rule only updates the affine head.
    """

    def __init__(self, memory: Any, state: Any, byte_encoder: Callable,
                 head: Callable, seed_logits: Any, anchor: Any,
                 *, observation_precision: Real = 1.0, head_step_size: Real = 0,
                 address_encoder: Callable | None = None,
                 likelihood_encoder: Callable | None = None,
                 value_step_size: Real = 0):
        import torch
        if not callable(byte_encoder):
            raise PrequentialBridgeViolation("GDN byte value encoder must be callable")
        address_encoder = byte_encoder if address_encoder is None else address_encoder
        likelihood_encoder = byte_encoder if likelihood_encoder is None else likelihood_encoder
        if not callable(address_encoder) or not callable(likelihood_encoder):
            raise PrequentialBridgeViolation("GDN address and likelihood encoders must be callable")
        if (isinstance(value_step_size, bool) or not isinstance(value_step_size, Real)
                or not isfinite(value_step_size) or value_step_size < 0):
            raise PrequentialBridgeViolation("value step size must be a finite nonnegative real")
        if value_step_size:
            if not isinstance(byte_encoder, torch.nn.Embedding):
                raise PrequentialBridgeViolation("value learning requires an embedding value encoder")
            if address_encoder is byte_encoder or likelihood_encoder is byte_encoder:
                raise PrequentialBridgeViolation(
                    "value learning requires distinct fixed address and likelihood encoders"
                )
            if not isinstance(head, torch.nn.Linear) or head.out_features != 256:
                raise PrequentialBridgeViolation("value learning requires an affine 256-byte head")
        if (isinstance(observation_precision, bool)
                or not isinstance(observation_precision, Real)
                or not isfinite(observation_precision) or observation_precision < 0):
            raise PrequentialBridgeViolation("observation precision must be a finite nonnegative real")
        if not isinstance(seed_logits, torch.Tensor) or tuple(seed_logits.shape) != (memory.n_slots,):
            raise PrequentialBridgeViolation("GDN seed logits must have shape [slots]")
        if not bool(torch.isfinite(seed_logits).all()):
            raise PrequentialBridgeViolation("GDN seed logits must be finite")
        if not isinstance(anchor, torch.Tensor) or tuple(anchor.shape) != (2,):
            raise PrequentialBridgeViolation("GDN byte anchor must have shape [2]")
        if not bool(torch.isfinite(anchor).all()):
            raise PrequentialBridgeViolation("GDN byte anchor must be finite")
        self.byte_encoder = byte_encoder
        self.address_encoder = address_encoder
        self.likelihood_encoder = likelihood_encoder
        self.value_step_size = float(value_step_size)
        self._last_byte_token: int | None = None
        # The scalar is exactly the one-event associative read eligibility for
        # the previous value.  It is independent of elapsed prior time under
        # the Gaussian prior/read pair, so it is an O(1) causal carrier.
        self._previous_value_eligibility: float | None = None
        self.seed_logits = seed_logits
        # A fixed location contributes the same positional offset to every
        # byte address; byte-specific addressing remains in the encoder and
        # existing content-address map, exactly as in the fixed Lean model.
        self.anchor = anchor.detach().clone()
        self.observation_precision = float(observation_precision)
        super().__init__(memory, state, self._last_byte_query, head,
                         self._encode_byte, head_step_size=head_step_size)

    def _last_byte_query(self, state: Any) -> Any:
        import torch
        if state.last_tokens is None:
            logits = self.seed_logits.to(device=state.natural.device, dtype=state.natural.dtype)
            return torch.softmax(logits, dim=0).reshape(1, 1, self.memory.n_slots)
        if state.last_centers is None or tuple(state.last_tokens.shape[:2]) != (1, 1):
            raise PrequentialBridgeViolation("last-byte GDN state lacks one causal byte observation")
        return self.memory.address(state.last_tokens, state.last_centers)

    def _encode_byte(self, token: int) -> tuple[Any, Any, Any, Any, float]:
        import torch
        byte = torch.tensor([[int(token)]], device=self.state.natural.device)
        tokens = self.byte_encoder(byte)
        address_tokens = self.address_encoder(byte)
        likelihood_tokens = self.likelihood_encoder(byte)
        expected = (1, 1, self.memory.d_model)
        for name, value in (("value", tokens), ("address", address_tokens),
                            ("likelihood", likelihood_tokens)):
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected:
                raise PrequentialBridgeViolation(
                    f"GDN byte {name} encoder must return [1, 1, d_model] features"
                )
            if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
                raise PrequentialBridgeViolation(
                    f"GDN byte {name} encoder must return finite floating features"
                )
        anchor = self.anchor.to(device=tokens.device, dtype=tokens.dtype).reshape(1, 1, 2)
        return tokens, address_tokens, likelihood_tokens, anchor, self.observation_precision

    def _record_value_eligibility(self, token: int) -> None:
        """Carry the preceding value's exact one-arrival read sensitivity.

        With address and likelihood fixed, the write contribution is
        ``address * observation_precision * local_precision``.  Dividing by
        the posterior slot precision gives its contribution to any later
        read.  The dynamic Gaussian prior scales numerator and denominator by
        the same factor, so elapsed time does not enlarge this carrier.
        """
        import torch
        if not self.value_step_size:
            return
        _, address_tokens, likelihood_tokens, centers, precision = self._encode_byte(token)
        address = self.memory.address(address_tokens, centers)[0, 0]
        local_precision = torch.nn.functional.softplus(
            self.memory.log_likelihood_precision(likelihood_tokens)
        )[0, 0]
        query = self._last_byte_query(self.state)[0, 0]
        posterior_precision = self.state.precision[0, :, 0].clamp_min(1e-6)
        coefficient = (query * address * (float(precision) * local_precision)
                       / posterior_precision).sum()
        self._previous_value_eligibility = float(coefficient.detach().item())

    def _previous_value_credit(self, target: int) -> tuple[int, Any] | None:
        """Read Lean's value-eligibility times the prepared language error."""
        import torch
        if not self.value_step_size or self._last_byte_token is None:
            return None
        eligibility = self._previous_value_eligibility
        if eligibility is None:
            return None
        probabilities = torch.softmax(self._pending, dim=0)
        language_error = self.head.weight[int(target)] - probabilities @ self.head.weight
        return self._last_byte_token, (
            self.value_step_size * eligibility * language_error.detach()
        )

    def observe_pre_scored(self, token: int, *, external: bool = True) -> None:
        """Assimilate this byte, then credit the prior byte through its carrier."""
        import torch
        self.validate_input(token)
        if self._pending is None:
            raise PrequentialBridgeViolation("predict before observing a GDN byte")
        # The prepared frame fixes score and third factor before this arrival.
        # We apply the resulting parameter change after the current write, so
        # even repeated bytes cannot alter their own just-arrived observation.
        credit = self._previous_value_credit(int(token))
        super().observe_pre_scored(token, external=external)
        if credit is not None:
            previous_token, update = credit
            with torch.no_grad():
                self.byte_encoder.weight[previous_token].add_(update)
        self._last_byte_token = int(token)
        self._record_value_eligibility(int(token))

    def snapshot(self) -> GDNBytePortSnapshot:
        snapshot = super().snapshot()
        return GDNBytePortSnapshot(
            state=snapshot.state,
            head_state=snapshot.head_state,
            diagnostics=snapshot.diagnostics,
            port_state={
                "last_byte_token": self._last_byte_token,
                "previous_value_eligibility": self._previous_value_eligibility,
            },
        )

    def restore(self, snapshot: GDNBytePortSnapshot) -> None:
        super().restore(snapshot)
        port_state = snapshot.port_state
        if port_state is None:
            self._last_byte_token = None
            self._previous_value_eligibility = None
            return
        if not isinstance(port_state, dict):
            raise PrequentialBridgeViolation("invalid last-byte GDN port snapshot")
        token = port_state.get("last_byte_token")
        if token is not None:
            self.validate_input(token)
        eligibility = port_state.get("previous_value_eligibility")
        if eligibility is not None and (not isinstance(eligibility, Real) or not isfinite(eligibility)):
            raise PrequentialBridgeViolation("invalid last-byte GDN eligibility snapshot")
        self._last_byte_token = token
        self._previous_value_eligibility = eligibility


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
        *, strict_scoring: bool = False,
    ):
        if type(strict_scoring) is not bool:
            raise PrequentialBridgeViolation("strict_scoring must be a bool")
        self.learner = learner
        self.ledger = ledger
        self.reference_nll_nats = reference_nll_nats
        self.strict_scoring = strict_scoring
        self.last_tick = ledger.next_tick - 1
        self._prepared = None

    def prepare(self, *, auxiliary: tuple[int, ...] = ()) -> Any:
        """Capture this tick's prediction before the world publishes its target."""
        auxiliary = _context(auxiliary)
        if self._prepared is not None:
            raise PrequentialBridgeViolation("a prediction is already prepared")
        kwargs = {"auxiliary": auxiliary} if auxiliary else {}
        prediction = self.learner.predict(**kwargs)
        self._prepared = (self.last_tick + 1, auxiliary, _snapshot_prediction(prediction))
        return prediction

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
        if self.strict_scoring and self._prepared is None:
            raise PrequentialBridgeViolation("strict scoring requires a prepared prediction")
        if self._prepared is not None:
            prepared_tick, prepared_auxiliary, _ = self._prepared
            if prepared_tick != observation.tick or prepared_auxiliary != auxiliary:
                raise PrequentialBridgeViolation("prepared prediction context does not match observation")
        validate = getattr(self.learner, "validate_input", None)
        if validate is not None:
            validate(int(target), **kwargs)
        reference_nll = float(self.reference_nll_nats(int(target)))
        if not isfinite(reference_nll) or reference_nll < 0:
            raise PrequentialBridgeViolation("reference NLL must be finite and nonnegative")
        prediction = (self._prepared[2] if self._prepared is not None
                      else self.learner.predict(**kwargs))
        # Score the prepared frame before observe can change the state.
        # Legacy learners with no supported frame retain their old behavior.
        model_nll_nats = _snapshot_nll(prediction, int(target))
        if self.strict_scoring and model_nll_nats is None:
            raise PrequentialBridgeViolation("strict scoring requires a scoreable prediction frame")
        if model_nll_nats is not None and (not isfinite(model_nll_nats) or model_nll_nats < 0):
            raise PrequentialBridgeViolation("prediction NLL must be finite and nonnegative")
        pre_scored = getattr(self.learner, "observe_pre_scored", None)
        if model_nll_nats is not None and callable(pre_scored):
            pre_scored(int(target), external=True, **kwargs)
        else:
            observed_nll = float(self.learner.observe(int(target), external=True, **kwargs))
            if model_nll_nats is None:
                model_nll_nats = observed_nll
        row = self.ledger.record(tick=observation.tick, model_nll_nats=model_nll_nats,
            reference_nll_nats=reference_nll, write_units=write_units,
            dissipation_units=dissipation_units, calibrated_energy=calibrated_energy)
        self.last_tick = observation.tick
        self._prepared = None
        return PrequentialResult(observation.tick, int(target), prediction, model_nll_nats, row)


def _snapshot_prediction(prediction: Any) -> Any:
    """Copy tensor logits so reused learner buffers cannot alter an earlier prediction.

    Non-tensor frames retain their existing contract: their `nll` method and
    all data it reads must remain stable through scoring.
    """
    try:
        import torch
    except ImportError:
        return prediction
    if isinstance(prediction, torch.Tensor):
        return prediction.detach().clone()
    return prediction


def _snapshot_nll(prediction: Any, target: int) -> float | None:
    """Score immutable probability frames or one-dimensional torch logits."""
    frame_nll = getattr(prediction, "nll", None)
    if callable(frame_nll):
        return float(frame_nll(target))
    # Torch is optional for non-tensor bridge users.
    try:
        import torch
    except ImportError:
        return None
    if not isinstance(prediction, torch.Tensor):
        return None
    if prediction.ndim != 1 or not prediction.is_floating_point():
        raise PrequentialBridgeViolation("prediction logits must be a 1D floating tensor")
    if not 0 <= target < prediction.numel():
        raise PrequentialBridgeViolation("target outside prediction vocabulary")
    return float(-torch.log_softmax(prediction.detach(), dim=0)[target].item())


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
