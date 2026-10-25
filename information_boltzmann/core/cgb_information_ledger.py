"""Bounded-memory information/dissipation ledger for real token predictions.

The ledger converts a model negative log likelihood and a declared
context-free reference into a signed predictive gain.  Positive gain is
recorded as useful bits; negative gain is kept separately as predictive loss.
This is an operational predictive measure, not a universal mutual-information
or joule-per-bit theorem.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


class InformationLedgerViolation(ValueError):
    """Raised when a prediction row cannot be accounted for."""


@dataclass(frozen=True)
class InformationLedgerRow:
    tick: int
    model_nll_bits: float
    reference_nll_bits: float
    useful_bits: float
    predictive_loss_bits: float
    write_units: int
    dissipation_units: int
    calibrated_energy: float | None = None

    @property
    def energy_per_useful_bit(self) -> float | None:
        if self.calibrated_energy is None or self.useful_bits <= 0.0:
            return None
        return self.calibrated_energy / self.useful_bits

    @property
    def dissipation_units_per_useful_bit(self) -> float | None:
        if self.useful_bits <= 0.0:
            return None
        return self.dissipation_units / self.useful_bits


class PredictiveInformationLedger:
    """Accumulate contiguous, independently labelled prediction rows."""

    def __init__(self, *, start_tick: int = 0, reference_name: str = "fixed_reference"):
        if isinstance(start_tick, bool) or not isinstance(start_tick, int) or start_tick < 0:
            raise InformationLedgerViolation("start_tick must be a nonnegative integer")
        if not isinstance(reference_name, str) or not reference_name:
            raise InformationLedgerViolation("reference_name must be non-empty")
        self.next_tick = start_tick
        self.reference_name = reference_name
        self.rows = 0
        self.useful_bits = 0.0
        self.predictive_loss_bits = 0.0
        self.useful_rows = 0
        self.loss_rows = 0
        self.write_units = 0
        self.dissipation_units = 0
        self.calibrated_energy = 0.0
        self.energy_rows = 0

    def record(
        self,
        *,
        tick: int,
        model_nll_nats: float,
        reference_nll_nats: float,
        write_units: int,
        dissipation_units: int,
        calibrated_energy: float | None = None,
    ) -> InformationLedgerRow:
        if tick != self.next_tick:
            raise InformationLedgerViolation(
                f"prediction tick {tick} is out of order; expected {self.next_tick}"
            )
        values = (model_nll_nats, reference_nll_nats)
        if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in values):
            raise InformationLedgerViolation("NLL values must be finite and nonnegative")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
               for value in (write_units, dissipation_units)):
            raise InformationLedgerViolation("discrete ledger values must be nonnegative integers")
        if calibrated_energy is not None and (
            not math.isfinite(float(calibrated_energy)) or calibrated_energy < 0.0
        ):
            raise InformationLedgerViolation("calibrated energy must be finite and nonnegative")
        model_bits = float(model_nll_nats) / math.log(2.0)
        reference_bits = float(reference_nll_nats) / math.log(2.0)
        gain_bits = reference_bits - model_bits
        useful_bits = max(0.0, gain_bits)
        loss_bits = max(0.0, -gain_bits)
        row = InformationLedgerRow(
            tick=tick,
            model_nll_bits=model_bits,
            reference_nll_bits=reference_bits,
            useful_bits=useful_bits,
            predictive_loss_bits=loss_bits,
            write_units=write_units,
            dissipation_units=dissipation_units,
            calibrated_energy=calibrated_energy,
        )
        self.rows += 1
        self.next_tick += 1
        self.useful_bits += useful_bits
        self.predictive_loss_bits += loss_bits
        self.useful_rows += int(useful_bits > 0.0)
        self.loss_rows += int(loss_bits > 0.0)
        self.write_units += write_units
        self.dissipation_units += dissipation_units
        if calibrated_energy is not None:
            self.calibrated_energy += float(calibrated_energy)
            self.energy_rows += 1
        return row

    def summary(self) -> dict[str, float | int | str | None]:
        return {
            "reference_name": self.reference_name,
            "rows": self.rows,
            "useful_bits": self.useful_bits,
            "predictive_loss_bits": self.predictive_loss_bits,
            "net_predictive_gain_bits": self.useful_bits - self.predictive_loss_bits,
            "useful_rows": self.useful_rows,
            "loss_rows": self.loss_rows,
            "write_units": self.write_units,
            "dissipation_units": self.dissipation_units,
            "dissipation_units_per_useful_bit": (
                self.dissipation_units / self.useful_bits
                if self.useful_bits > 0.0 else None
            ),
            "write_units_per_useful_bit": (
                self.write_units / self.useful_bits
                if self.useful_bits > 0.0 else None
            ),
            "calibrated_energy": self.calibrated_energy if self.energy_rows else None,
            "energy_per_useful_bit": (
                self.calibrated_energy / self.useful_bits
                if self.energy_rows and self.useful_bits > 0.0 else None
            ),
            "energy_coverage_rows": self.energy_rows,
            "measure_kind": "predictive_gain_relative_to_declared_reference",
        }
