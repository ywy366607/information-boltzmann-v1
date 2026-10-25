"""Validate the real-prediction information ledger without running a model."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys


def load_module():
    path = Path(__file__).resolve().parents[1] / "information_boltzmann" / "core" / "cgb_information_ledger.py"
    spec = importlib.util.spec_from_file_location("cgb_information_ledger", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def expect_rejection(module, fn, fragment: str) -> None:
    try:
        fn()
    except module.InformationLedgerViolation as exc:
        assert fragment in str(exc), str(exc)
    else:
        raise AssertionError("invalid ledger row was accepted")


def main() -> None:
    m = load_module()
    ledger = m.PredictiveInformationLedger(reference_name="train_unigram")
    row = ledger.record(
        tick=0,
        model_nll_nats=0.0,
        reference_nll_nats=math.log(2.0),
        write_units=1,
        dissipation_units=2,
        calibrated_energy=4.0,
    )
    assert abs(row.useful_bits - 1.0) < 1e-12
    assert row.predictive_loss_bits == 0.0
    assert row.energy_per_useful_bit == 4.0
    assert row.dissipation_units_per_useful_bit == 2.0
    loss = ledger.record(
        tick=1,
        model_nll_nats=math.log(4.0),
        reference_nll_nats=math.log(2.0),
        write_units=0,
        dissipation_units=1,
    )
    assert loss.useful_bits == 0.0 and abs(loss.predictive_loss_bits - 1.0) < 1e-12
    assert ledger.summary()["useful_bits"] == 1.0
    assert ledger.summary()["net_predictive_gain_bits"] == 0.0
    assert ledger.summary()["useful_rows"] == 1 and ledger.summary()["loss_rows"] == 1
    expect_rejection(
        m,
        lambda: ledger.record(
            tick=3,
            model_nll_nats=0.0,
            reference_nll_nats=0.0,
            write_units=0,
            dissipation_units=0,
        ),
        "expected 2",
    )
    print("cgb_information_ledger_ok")


if __name__ == "__main__":
    main()
