"""Exact encoding/continuation audit of an existing finite Lean model.

This is not a synthetic learning study. No optimizer, language training or
capability estimate is used. Reference transitions/ports are exported by Lean.
"""

from __future__ import annotations

import argparse
from collections import deque
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
LEAN = ROOT / "lean" / "CGBAxioms"
EXPORT = LEAN / "reviews" / "export_sensory_record_20261025.lean"
KERNEL = ROOT / "information_boltzmann" / "core" / "cgb_sensory_record.py"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reachable(rows: list[dict], initial: int) -> dict[int, list[int]]:
    witnesses = {initial: []}
    pending = deque([initial])
    while pending:
        state = pending.popleft()
        for signal, successor in enumerate(rows[state]["transitions"]):
            if successor not in witnesses:
                witnesses[successor] = witnesses[state] + [signal]
                pending.append(successor)
    return witnesses


def partition_counts(rows: list[dict], states: list[int]) -> list[int]:
    labels = {state: rows[state]["action"] for state in states}
    counts = [len(set(labels.values()))]
    while True:
        signatures = {
            state: (labels[state], tuple(labels[t] for t in rows[state]["transitions"]))
            for state in states
        }
        classes: dict[tuple, int] = {}
        refined = {}
        for state in states:
            key = signatures[state]
            refined[state] = classes.setdefault(key, len(classes))
        if len(classes) == counts[-1]:
            return counts
        counts.append(len(classes))
        labels = refined


def distinguishing_inputs(rows: list[dict], left: int, right: int) -> list[int]:
    pending = deque([(left, right, [])])
    visited = {(left, right)}
    while pending:
        a, b, path = pending.popleft()
        if rows[a]["action"] != rows[b]["action"]:
            return path
        for signal, (a_next, b_next) in enumerate(zip(rows[a]["transitions"], rows[b]["transitions"])):
            if (a_next, b_next) not in visited:
                visited.add((a_next, b_next))
                pending.append((a_next, b_next, path + [signal]))
    raise AssertionError("a pair of purportedly distinct causal records is indistinguishable")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources = [Path(__file__).resolve(), KERNEL, EXPORT, LEAN / "CGBAxioms" / "LanguageLifeKernel.lean",
               LEAN / "CGBAxioms" / "SensoryMaintenance.lean",
               LEAN / "CGBAxioms" / "RenewableLearning.lean"]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    result = subprocess.run(
        ["lake", "env", "lean", str(EXPORT.relative_to(LEAN))], cwd=LEAN,
        capture_output=True, text=True, encoding="utf-8", timeout=180, check=True,
    )
    payloads = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
    assert len(payloads) == 1, result.stdout[-2000:]
    table = payloads[0]
    assert table["format"] == "CGB.SensoryRecord.v1"
    spec = importlib.util.spec_from_file_location("cgb_sensory_record", KERNEL)
    assert spec is not None and spec.loader is not None
    kernel = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = kernel
    spec.loader.exec_module(kernel)
    rows = table["rows"]
    assert len(rows) == 24 and [r["index"] for r in rows] == list(range(24))
    witnesses = reachable(rows, table["initial"])
    states = sorted(witnesses)
    assert set(states) == {r["index"] for r in rows if r["admissible"]}
    assert len(states) == 16
    labels = table["stimulus_labels"]
    assert tuple(labels) == kernel.STIMULI
    rejected = 0
    for row in rows:
        if not row["admissible"]:
            try:
                kernel.encode_record(table["cell_labels"][row["cell"]], row["reserve"])
            except ValueError:
                rejected += 1
            else:
                raise AssertionError("a state outside the proved region was accepted")
    for invalid in (-1, 16, True):
        try:
            kernel.read_record(invalid)
        except ValueError:
            rejected += 1
        else:
            raise AssertionError("invalid record accepted")
    for payload, count in ((b"", 1), (b"\xf0", 1), (b"", -1), (b"", True)):
        try:
            kernel.unpack_records(payload, count)
        except ValueError:
            rejected += 1
        else:
            raise AssertionError("noncanonical or mismatched packed record accepted")
    assert kernel.unpack_records(kernel.pack_records([]), 0) == []
    assert kernel.unpack_records(kernel.pack_records([15]), 1) == [15]
    checked = 0
    continuation_checks = 0
    model_hash = digest(LEAN / "CGBAxioms" / "LanguageLifeKernel.lean")
    with tempfile.TemporaryDirectory(dir=LEAN / ".lake") as directory:
        archive = Path(directory) / "record.bin"
        packed_bytes = kernel.write_archive(archive, list(range(16)), model_hash)
        assert kernel.read_archive(archive, model_hash) == list(range(16))
        payload_bytes = len(kernel.pack_records(list(range(16))))
        assert payload_bytes == 8
        try:
            kernel.read_archive(archive, "wrong-version")
        except ValueError:
            pass
        else:
            raise AssertionError("wrong model version accepted")
        corrupted = bytearray(archive.read_bytes())
        corrupted[-1] ^= 1
        archive.write_bytes(corrupted)
        try:
            kernel.read_archive(archive, model_hash)
        except ValueError:
            pass
        else:
            raise AssertionError("corrupted record accepted")
        for code, raw in enumerate(table["record_decode"]):
            row = rows[raw]
            assert row["record"] == code
            assert kernel.read_record(code) == (table["cell_labels"][row["cell"]], row["reserve"])
            assert kernel.encode_record(*kernel.read_record(code)) == code
            assert kernel.action(code) == row["action"]
            assert kernel.content(code) == row["content"]
            path = witnesses[raw]
            live = rows[table["initial"]]["record"]
            for signal in path:
                live = kernel.write_record(live, labels[signal])
            assert live == code
            kernel.write_archive(archive, [live], model_hash)
            restored = kernel.read_archive(archive, model_hash)[0]
            for signal, label in enumerate(labels):
                updated = kernel.write_record(code, label)
                assert updated == table["record_transitions"][code][signal]
                assert updated == rows[row["transitions"][signal]]["record"]
                assert list(kernel.ports(code, label)) == row["ports"][signal]
                incoming, outgoing, _, _, _ = kernel.ports(code, label)
                assert kernel.content(updated) + outgoing == kernel.content(code) + incoming
                assert kernel.write_record(restored, label) == updated
                continuation_checks += 1
                checked += 1
    partitions = partition_counts(rows, states)
    assert partitions[-1] == 16
    pairs = []
    for index, left in enumerate(states):
        for right in states[index + 1:]:
            path = distinguishing_inputs(rows, left, right)
            pairs.append({"left_record": rows[left]["record"], "right_record": rows[right]["record"],
                          "inputs": [labels[s] for s in path]})
    assert len(pairs) == 120
    assert hashes == {str(p.relative_to(ROOT)): digest(p) for p in sources}
    report = {
        "status": "exact_finite_record_verified_not_capability_evidence",
        "scope": "existing SensoryMaintenance body under registered boundary stimuli; world state not encoded",
        "original_state_cardinality": 24, "reachable_record_cardinality": 16,
        "reachability_domain": "all registered boundary-stimulus interventions, not natural-law support",
        "minimum_fixed_record_bits": 4,
        "current_action_class_count": len({rows[s]["action"] for s in states}),
        "action_content_class_count": len({(rows[s]["action"], rows[s]["content"]) for s in states}),
        "future_action_partition_counts": partitions,
        "transition_and_five_port_cases": checked,
        "restored_continuation_cases": continuation_checks,
        "distinguishing_pair_count": len(pairs),
        "max_distinguishing_input_length": max(len(p["inputs"]) for p in pairs),
        "all_record_codes_payload_bytes": payload_bytes,
        "all_record_codes_archive_bytes": packed_bytes,
        "archive_model_mismatch_rejected": True,
        "archive_payload_corruption_rejected": True,
        "invalid_domain_or_encoding_cases_rejected": rejected,
        "gpu_memory_or_speed_measured": False,
        "external_storage_energy_calibrated": False,
        "learner_reads_hidden_world": False,
        "training_started": False,
        "reachability_witnesses": {str(rows[s]["record"]): [labels[i] for i in path]
                                   for s, path in witnesses.items()},
        "distinguishing_input_witnesses": pairs,
        "source_sha256": hashes,
        "lean_export_sha256": hashlib.sha256(result.stdout.encode()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in (
        "status", "reachable_record_cardinality", "minimum_fixed_record_bits",
        "future_action_partition_counts", "transition_and_five_port_cases",
        "restored_continuation_cases", "max_distinguishing_input_length",
        "all_record_codes_payload_bytes", "all_record_codes_archive_bytes",
    )}))


if __name__ == "__main__":
    main()
