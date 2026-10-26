"""The admission gate rejects unlicensed architectures and its checks run on real code."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import cgb_gate  # noqa: E402


def test_charge_filter_is_rejected():
    r = cgb_gate.gate("charge_filter")
    assert not r["admitted"]
    assert any("global_posterior_normalization" in f for f in r["failures"])


def test_g15_checks_pass_on_the_implementation():
    for name in ("softmax_readout", "g15_policy_gradient", "g15_outcome_blind_zero",
                 "g14_delta_rule", "miss_counter_distinguishes", "interior_symmetric"):
        ok, info = cgb_gate.CHECKS[name]()
        assert ok, (name, info)


def test_cited_theorems_exist():
    decls, _, _ = cgb_gate.load_inventory()
    for p in cgb_gate.ARCH.glob("*.json"):
        for mech in json.loads(p.read_text(encoding="utf-8"))["mechanisms"]:
            for t in mech.get("theorems", []):
                assert t in decls, (p.name, t)


def test_unlicensed_structure_blocks_admission():
    r = cgb_gate.gate("g15_organism")
    assert not r["admitted"]
    assert "no Lean Medium instance with an Axioms proof" in r["failures"]


def test_grown_organism_checks_pass_and_only_w_in_and_medium_block():
    r = cgb_gate.gate("g15_grown_organism")
    assert all(e["passed"] for e in r["execution"])
    assert sorted(r["failures"]) == sorted([
        "hidden_write_weights (structure): no valid Lean theorem",
        "no Lean Medium instance with an Axioms proof"])


def test_generated_organism_is_admitted():
    r = cgb_gate.gate("generated_s0")
    assert r["admitted"], r["failures"]
