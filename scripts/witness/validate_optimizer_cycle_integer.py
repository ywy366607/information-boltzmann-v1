"""Cross-language arithmetic checks for the proved local optimizer kernel.

This runs no learner or capability task. Full optimizer admission stays closed.
Lean supplies all legal state/cue transitions; Python independently computes
the integer formula, exact rounding error, resource ledger and binary encoding.
"""
from __future__ import annotations

import argparse
import ast
from fractions import Fraction
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import re
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
PROOF = ROOT / "lean/CGBAxioms/reviews/optimizer_conservative_cycle.lean"


def encode(phase: int, reserve: int) -> int:
    if not (-32 <= phase <= 32 and 8 <= reserve <= 16):
        raise ValueError("illegal cycle state")
    return phase + 32 + 65 * (reserve - 8)


def decode(code: int) -> tuple[int, int]:
    if not 0 <= code < 585:
        raise ValueError("illegal cycle code")
    return code % 65 - 32, code // 65 + 8


def integer_step(phase: int, reserve: int, cue: int) -> tuple[int, int]:
    """Reference only: Python // agrees with positive-divisor Euclidean division."""
    encode(phase, reserve)
    if not -64 <= cue <= 64:
        raise ValueError("illegal cue code")
    numerator = 2 * (4 + reserve) * phase + reserve * cue
    next_phase = max(-32, min(32, (numerator + 8) // 16))
    supplied = (19 - reserve) // 4
    used = reserve * abs(phase) // 128
    return next_phase, reserve + supplied - used


def validate_observer(args) -> None:
    """Lean-first arithmetic admission of the online observation closure."""
    proof = ROOT / "lean/CGBAxioms/reviews/optimizer_v6_handoff.lean"
    runtime = ROOT / "information_boltzmann/conservative_cycle_optimizer.py"
    paths = [proof, PROOF, runtime, Path(__file__)]
    initial = {p: p.read_bytes() for p in paths}
    result = subprocess.run(
        [args.lake, "env", "lean", "-j1", "--run", "reviews/optimizer_v6_handoff.lean"],
        cwd=ROOT / "lean/CGBAxioms", capture_output=True, text=True, encoding="utf-8")
    output = result.stdout + result.stderr
    (ROOT / "lean/CGBAxioms/.lake/online_observer_oracle.log").write_text(output, encoding="utf-8")
    assert result.returncode == 0, output
    queried = set(re.findall(r"^#print axioms ([\w.]+)", initial[proof].decode("utf-8"), re.M))
    audited = set()
    for name, block in re.findall(r"'([\w.]+)' depends on axioms: \[(.*?)\]", output, re.S):
        assert set(re.findall(r"[^\s,]+", block)) <= {"propext", "Classical.choice", "Quot.sound"}
        audited.add(name)
    audited.update(re.findall(r"'([\w.]+)' does not depend on any axioms", output))
    assert queried == audited and len(audited) == 21, queried ^ audited
    assert not re.search(r"\b(?:sorry|admit)\b|^\s*axiom\s", initial[proof].decode("utf-8"), re.M)
    remaining = set(product((8, 12, 16), (-1, 1),
                            (Fraction(1, 8), Fraction(1, 2), Fraction(1)),
                            (Fraction(0), Fraction(1, 2), Fraction(1))))
    rows = []
    digest = hashlib.sha256()
    for line in output.splitlines():
        if not line.startswith("observer\t"):
            continue
        fields = line.split("\t")[1:]
        assert len(fields) == 15, fields
        reserve, sign = map(int, fields[:2])
        rate, mass = map(Fraction, fields[2:4])
        remaining.remove((reserve, sign, rate, mass))
        alpha = rate * Fraction(reserve + 16, 32)
        next_mass = (1 - alpha) * mass + alpha
        gain = alpha / next_mass
        old = [sign * Fraction(i % 4 + 1, 2) if mass else Fraction(0) for i in range(8)]
        old_energy = sum(x*x for x in old) / 4
        observed = list(map(Fraction, (1, 2, -3, -4, 2, 1, -4, -3)))
        mean = [(1 - gain) * m + gain * g for m, g in zip(old, observed)]
        energy = (1 - gain) * old_energy + gain * sum(g*g for g in observed) / 8
        assert list(map(Fraction, fields[4:7])) == [gain, next_mass, energy]
        assert list(map(Fraction, fields[7:])) == mean
        assert 0 < gain <= 1 and 0 < next_mass <= 1
        assert sum(m*m for m in mean) <= 8*energy
        codes, new_codes, numerators = [], [], []
        for b, (phase, stock) in enumerate(((-8, reserve), (8, 16))):
            g = mean[4*b:4*b+4]
            diagonal, cross = g[0]**2 + g[3]**2, g[1]**2 + g[2]**2
            cue = 2*(diagonal-cross)/(diagonal+cross) if diagonal+cross else Fraction(0)
            nk, nr = integer_step(phase, stock, (32*cue+Fraction(1, 2)).__floor__())
            shares = [Fraction(32+nk, 64), Fraction(32-nk, 64),
                      Fraction(32-nk, 64), Fraction(32+nk, 64)]
            codes.append(encode(phase, stock))
            new_codes.append(encode(nk, nr))
            numerators.extend(Fraction(1, 8)*Fraction(stock, 16)*a*m for a, m in zip(shares, g))
        rows.append((rate, mass, old, old_energy, codes, new_codes, mean,
                     energy, next_mass, numerators))
        digest.update(line.encode("utf-8"))
    assert len(rows) == 54 and not remaining
    # All theorem and executable-reference checks precede implementation execution.
    sys.path.insert(0, str(ROOT))
    import torch
    from information_boltzmann.conservative_cycle_optimizer import _resource_observer_chunk
    checks = []
    for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
        for dtype in (torch.float32, torch.float64):
            worst = dict(mean=0., rms=0., update=0., mass=0.)
            for rate in (Fraction(1, 8), Fraction(1, 2), Fraction(1)):
                selected = [r for r in rows if r[0] == rate]
                def tensor(values):
                    return torch.tensor(values, dtype=dtype, device=device)
                code = torch.tensor([r[4] for r in selected], dtype=torch.int16, device=device)
                mean = tensor([[float(x) for x in r[2]] for r in selected]).reshape(-1, 2, 4)
                rms = tensor([math.sqrt(float(r[3])) for r in selected]).reshape(-1, 1, 1)
                mass = tensor([float(r[1]) for r in selected]).reshape(-1, 1, 1)
                observed = tensor([1, 2, -3, -4, 2, 1, -4, -3]).reshape(1, 2, 4).expand_as(mean)
                actual = _resource_observer_chunk(code, mean, observed, rms, mass, float(rate), .125)
                assert torch.equal(actual[0], torch.tensor([r[5] for r in selected], dtype=torch.int16, device=device))
                expected = [tensor([[float(x) for x in r[6]] for r in selected]).reshape_as(mean),
                            tensor([math.sqrt(float(r[7])) for r in selected]).reshape_as(rms),
                            tensor([float(r[8]) for r in selected]).reshape_as(mass),
                            tensor([[float(x)/math.sqrt(float(r[7])) for x in r[9]]
                                    for r in selected]).reshape_as(mean)]
                tolerance = 2e-6 if dtype == torch.float32 else 2e-14
                for name, a, e in zip(("mean", "rms", "mass", "update"), actual[1:], expected):
                    torch.testing.assert_close(a, e, rtol=tolerance, atol=tolerance)
                    worst[name] = max(worst[name], float((a-e).abs().max()))
            checks.append(dict(device=device, dtype=str(dtype), vectors=len(rows), max_absolute_error=worst))
    assert all(p.read_bytes() == value for p, value in initial.items()), "source changed during audit"
    # Canonical AST fingerprints of the archived, previously audited routines.
    # Reproduction therefore does not depend on this machine's ignored archive.
    expected_ast = {
        "ConservativeCycle": "d0544ed7e318c733700a86f83907b2e648c6aa0aedb72ec6b318cfcc02a395bd",
    }
    after = ast.parse(runtime.read_text(encoding="utf-8"))
    for name, digest_ast in expected_ast.items():
        node = next(n for n in after.body if getattr(n, "name", None) == name)
        assert hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest() == digest_ast, name
    receipt = dict(status="online_observer_arithmetic_passed", full_optimizer_admitted=False,
                   scope="resource-clock empirical gradient observer; no model/training executed",
                   parameter_feedback_closed=False, bayesian_noise_or_curvature_calibration_claimed=False,
                   lean_exit_code=0, audited_declarations=sorted(audited),
                   sources={p.relative_to(ROOT).as_posix(): hashlib.sha256(v).hexdigest()
                            for p, v in initial.items()},
                   exact_rational_vectors=len(rows), vector_digest_sha256=digest.hexdigest(), checks=checks,
                   historical_training_class_ast_unchanged=True,
                   layout_cost=dict(fp32_mean_bytes_per_parameter=4, phase_bytes_per_parameter=.5,
                                    scalar_statistics_per_compartment=2, compartment_example_parameters=64,
                                    fp32_persistent_bytes_per_parameter_at_64=4.625,
                                    peak_training_memory_measured=False, minimality_claimed=False),
                   remaining="Joint mode generation from raw observations, functional retention/release, temporal service and real-stream benefit; autonomous arbitrary-weight repair is not claimed",
                   review_status="self_reviewed_independent_review_pending")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


def validate_raw_observer(args) -> None:
    """Exact recursive arithmetic, not a cue-probe capability experiment."""
    runtime = ROOT / "information_boltzmann/conservative_cycle_optimizer.py"
    sources = {p: p.read_bytes() for p in (PROOF, runtime, Path(__file__))}
    controlled = args.feedback_control
    result = subprocess.run(
        [args.lake, "env", "lean", "-j1", "--run", "reviews/optimizer_conservative_cycle.lean",
         "--feedback-control" if controlled else "--raw-observer"],
        cwd=ROOT / "lean/CGBAxioms", capture_output=True, text=True, encoding="utf-8")
    output = result.stdout + result.stderr
    (ROOT / "lean/CGBAxioms/.lake/raw_observation_oracle.log").write_text(output, encoding="utf-8")
    assert result.returncode == 0, output[:12000]
    queried = set(re.findall(r"^#print axioms ([\w.]+)", sources[PROOF].decode(), re.M))
    audited = set()
    for name, block in re.findall(r"'([\w.]+)' depends on axioms: \[(.*?)\]", output, re.S):
        assert set(re.findall(r"[^\s,]+", block)) <= {"propext", "Classical.choice", "Quot.sound"}
        audited.add(name)
    audited.update(re.findall(r"'([\w.]+)' does not depend on any axioms", output))
    assert queried == audited and len(audited) == 33, queried ^ audited
    assert not re.search(r"\b(?:sorry|admit)\b|^\s*axiom\s", sources[PROOF].decode(), re.M)
    rows, trace, digest = [], {}, hashlib.sha256()
    for line in output.splitlines():
        if not line.startswith("control\t" if controlled else "rawobserver\t"):
            continue
        f = line.split("\t")[1:]
        feedback = bool(int(f.pop(0))) if controlled else True
        assert len(f) == 23
        rate, scenario, tick = Fraction(f[0]), int(f[1]), int(f[2])
        key = feedback, rate, scenario
        shape = list(map(Fraction, [1, 0, 0, 1, 0, 1, 1, 0]))
        observed = [Fraction(0)]*8 if scenario == 2 else shape
        if tick == 0:
            assert key not in trace
            trace[key] = ([0, 0] if scenario == 0 else [-32, 32],
                          [16, 16] if scenario == 0 else [10, 10],
                          Fraction(0 if scenario == 0 else 1),
                          Fraction(1, 2) if scenario == 1 else Fraction(0),
                          [1-x for x in shape] if scenario == 1 else [Fraction(0)]*8, 0)
        k, r, mass, energy, mean, expected_tick = trace[key]
        assert tick == expected_tick
        alpha = rate*sum(r)/32
        nmass = (1-alpha)*mass + alpha
        gain = alpha/nmass
        nmean = [(1-gain)*m+gain*g for m, g in zip(mean, observed)]
        nenergy = (1-gain)*energy + gain*sum(g*g for g in observed)/8
        nk, nr, numerator = [], [], []
        for b in range(2):
            g = nmean[4*b:4*b+4]
            d, c = g[0]**2+g[3]**2, g[1]**2+g[2]**2
            cue = 2*(d-c)/(d+c) if d+c else Fraction(0)
            encoded_cue = (32*cue+Fraction(1, 2)).__floor__()
            kk, rr = integer_step(k[b], r[b], encoded_cue)
            if not feedback:
                kk = max(-32, min(32, (8*k[b]+r[b]*encoded_cue+8)//16))
            nk.append(kk); nr.append(rr)
            shares = [Fraction(32+kk, 64), Fraction(32-kk, 64),
                      Fraction(32-kk, 64), Fraction(32+kk, 64)]
            numerator.extend(Fraction(1, 8)*Fraction(r[b], 16)*a*x for a, x in zip(shares, g))
        assert list(map(int, f[3:5])) == [encode(a, b) for a, b in zip(nk, nr)]
        assert list(map(Fraction, f[5:])) == [nmass, nenergy]+nmean+numerator
        assert sum(x*x for x in nmean) <= 8*nenergy
        rows.append((key, tick, nk, nr, nmass, nenergy, nmean, numerator))
        trace[key] = nk, nr, nmass, nenergy, nmean, tick+1
        digest.update(line.encode())
    assert len(rows) == (960 if controlled else 480) and len(trace) == (12 if controlled else 6)
    assert all(t[-1] == 80 for t in trace.values())
    sys.path.insert(0, str(ROOT))
    import torch
    from information_boltzmann.conservative_cycle_optimizer import _resource_observer_chunk
    checks = []
    for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
        for dtype in (torch.float32, torch.float64):
            worst = dict(mean=0., rms=0., mass=0., update=0.)
            for feedback, rate, scenario in trace:
                def tensor(x):
                    return torch.tensor(x, dtype=dtype, device=device)
                code = torch.tensor([[552, 552] if scenario == 0 else [130, 194]],
                                    dtype=torch.int16, device=device)
                shape = tensor([1, 0, 0, 1, 0, 1, 1, 0]).reshape(1, 2, 4)
                mean = 1-shape if scenario == 1 else shape*0
                rms = tensor([math.sqrt(.5) if scenario == 1 else 0.]).reshape(1, 1, 1)
                mass = tensor([0. if scenario == 0 else 1.]).reshape(1, 1, 1)
                observed = shape*0 if scenario == 2 else shape
                for row in (r for r in rows if r[0] == (feedback, rate, scenario)):
                    code, mean, rms, mass, update = _resource_observer_chunk(
                        code, mean, observed, rms, mass, float(rate), .125, feedback=feedback)
                    _, tick, nk, nr, nmass, energy, nmean, numerator = row
                    assert code.tolist() == [[encode(a, b) for a, b in zip(nk, nr)]], (device, dtype, scenario, tick)
                    expected = (tensor([float(x) for x in nmean]).reshape_as(mean),
                                tensor([math.sqrt(float(energy))]).reshape_as(rms),
                                tensor([float(nmass)]).reshape_as(mass),
                                tensor([float(x)/math.sqrt(float(energy)) if energy else 0.
                                        for x in numerator]).reshape_as(update))
                    tol = 3e-6 if dtype == torch.float32 else 3e-14
                    for name, a, e in zip(worst, (mean, rms, mass, update), expected):
                        torch.testing.assert_close(a, e, rtol=tol, atol=tol)
                        worst[name] = max(worst[name], float((a-e).abs().max()))
            checks.append(dict(device=device, dtype=str(dtype), transitions=len(rows), max_absolute_error=worst))
    assert all(p.read_bytes() == value for p, value in sources.items()), "source changed during audit"
    receipt = dict(status="matched_feedback_refinement_passed" if controlled else "raw_observation_refinement_passed", full_optimizer_admitted=False,
                   training_executed=False, exact_recursive_transitions=len(rows),
                   scope="same-state feedback intervention arithmetic" if controlled else "recursive raw-observation arithmetic; no learner or task",
                   feedback_settings=[True, False] if controlled else [True],
                   intervention="only 2*R*K regenerative term; all state, rounding, observation and readout laws retained",
                   audited_declarations=sorted(audited), checks=checks, lean_exit_code=0,
                   sources={p.relative_to(ROOT).as_posix(): hashlib.sha256(v).hexdigest() for p, v in sources.items()},
                   oracle_digest_sha256=digest.hexdigest(),
                   remaining="Functional historical benefit, general optimizer interface and independent real-stream validation",
                   review_status="self_reviewed_independent_review_pending")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lake", default="lake")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--observer", action="store_true", help="audit resource-clock observation closure")
    parser.add_argument("--raw-observer", action="store_true", help="audit the same recursive raw-observation law")
    parser.add_argument("--feedback-control", action="store_true", help="audit the state-matched feedback intervention")
    args = parser.parse_args()
    if args.raw_observer or args.feedback_control:
        return validate_raw_observer(args)
    if args.observer:
        validate_observer(args)
        return
    source = PROOF.read_bytes()
    text = source.decode("utf-8")
    declarations = set(re.findall(r"^theorem (\w+)", text, re.MULTILINE))
    audited: set[str] = set()
    cases = product(range(-32, 33), range(8, 17), range(-64, 65))
    count = 0
    filtered_rows = []
    filtered_cases = set(product((-32, 0, 32), (8, 16), (-1, 1),
                                 (Fraction(0), Fraction(1, 4), Fraction(1))))
    digest = hashlib.sha256()
    filtered_digest = hashlib.sha256()
    diagnostics: list[str] = []
    process = subprocess.Popen(
        [args.lake, "env", "lean", "-j1", "--run", "reviews/optimizer_conservative_cycle.lean"],
        cwd=ROOT / "lean/CGBAxioms", stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, encoding="utf-8",
    )
    assert process.stdout is not None
    try:
        for line in process.stdout:
            if line.startswith("cycle\t"):
                values = tuple(map(int, line.split()[1:]))
                phase, reserve, cue = next(cases)
                new_phase, new_reserve = integer_step(phase, reserve, cue)
                old_code = encode(phase, reserve)
                new_code = encode(new_phase, new_reserve)
                expected = (phase, reserve, cue, new_phase, new_reserve, old_code, new_code)
                assert values == expected, (values, expected)
                assert decode(new_code) == (new_phase, new_reserve)
                assert decode(old_code) == (phase, reserve)
                assert struct.unpack("<H", struct.pack("<H", new_code))[0] == new_code
                assert new_reserve + reserve * abs(phase) // 128 == reserve + (19 - reserve) // 4
                raw = ((Fraction(1, 2) + Fraction(reserve, 8)) * Fraction(phase, 32)
                       + Fraction(reserve, 16) * Fraction(cue, 32))
                numerator = 2 * (4 + reserve) * phase + reserve * cue
                rounded = Fraction((numerator + 8) // 16, 32)
                disturbance = rounded - raw
                assert abs(disturbance) <= Fraction(1, 64)
                assert Fraction(new_phase, 32) == max(-1, min(1, raw + disturbance))
                assert -32768 <= numerator + 8 <= 32767
                assert 0 <= reserve * abs(phase) <= 512
                digest.update(line.strip().encode("utf-8"))
                count += 1
            elif line.startswith("filtered\t"):
                fields = line.strip().split("\t")[1:]
                assert len(fields) == 13, fields
                k, r, sign = map(int, fields[:3])
                gain, new_code = Fraction(fields[3]), int(fields[4])
                filtered_cases.remove((k, r, sign, gain))
                old = [sign * Fraction(i + 1, 2) for i in range(4)]
                observed = list(map(Fraction, (1, 2, -3, -4)))
                scale = [Fraction(1), Fraction(2), Fraction(1, 2), Fraction(4)]
                mean = [(1 - gain) * a + gain * b for a, b in zip(old, observed)]
                e = [m / s for m, s in zip(mean, scale)]
                diagonal, cross = e[0]**2 + e[3]**2, e[1]**2 + e[2]**2
                cue = 2 * (diagonal - cross) / (diagonal + cross) if diagonal + cross else Fraction(0)
                c = (32 * cue + Fraction(1, 2)).__floor__()
                nk, nr = integer_step(k, r, c)
                assert new_code == encode(nk, nr)
                shares = [Fraction(32 + nk, 64), Fraction(32 - nk, 64),
                          Fraction(32 - nk, 64), Fraction(32 + nk, 64)]
                update = [Fraction(1, 8) * Fraction(r, 16) * a * m / s**2
                          for a, m, s in zip(shares, mean, scale)]
                assert list(map(Fraction, fields[5:9])) == mean
                assert list(map(Fraction, fields[9:13])) == update
                filtered_rows.append((encode(k, r), old, gain, scale, new_code, mean, update))
                filtered_digest.update(line.strip().encode("utf-8"))
            else:
                match = re.match(
                    r"'CGB\.ConservativeCycleReview\.(\w+)' depends on axioms: \[(.*)\]", line,
                )
                if match:
                    dependencies = set(match[2].split(", "))
                    assert dependencies <= {"propext", "Classical.choice", "Quot.sound"}, line
                    audited.add(match[1])
                elif match := re.match(
                    r"'CGB\.ConservativeCycleReview\.(\w+)' does not depend on any axioms", line,
                ):
                    audited.add(match[1])
                elif line.strip():
                    diagnostics.append(line.strip())
                    assert len(diagnostics) < 100, "unexpected Lean output"
        assert process.wait() == 0, diagnostics
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait()
    assert count == 65 * 9 * 129 == 75465
    assert next(cases, None) is None
    # Long declaration names can wrap the axiom list across several lines.
    # Audit the complete block rather than silently dropping such results.
    for name, block in re.findall(
        r"'CGB\.ConservativeCycleReview\.(\w+)' depends on axioms: \[(.*?)\]",
        "\n".join(diagnostics), re.S,
    ):
        assert set(re.findall(r"[^\s,]+", block)) <= {"propext", "Classical.choice", "Quot.sound"}
        audited.add(name)
    assert declarations == audited, (declarations - audited, audited - declarations)
    assert len(filtered_rows) == 36 and not filtered_cases
    assert PROOF.read_bytes() == source, "proof changed while validating"
    # Execute the implementation only AFTER the complete Lean/axiom gate.
    # These are arithmetic vectors, not a trained task or capability study.
    sys.path.insert(0, str(ROOT))
    import torch
    from information_boltzmann.conservative_cycle_optimizer import _filtered_cycle_chunk
    filtered_checks = []
    for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
        for dtype in (torch.float32, torch.float64):
            def tensor(values):
                return torch.tensor(values, dtype=dtype, device=device)
            codes = torch.tensor([r[0] for r in filtered_rows], dtype=torch.int16, device=device)
            means = tensor([[float(x) for x in r[1]] for r in filtered_rows])
            gain = tensor([[float(r[2])] for r in filtered_rows])
            scales = tensor([[float(x) for x in r[3]] for r in filtered_rows])
            observed = tensor([[1, 2, -3, -4]]).expand_as(means)
            new_codes, new_means, update = _filtered_cycle_chunk(codes, means, observed, gain, scales, .125)
            expected_codes = torch.tensor([r[4] for r in filtered_rows], dtype=torch.int16, device=device)
            expected_means = tensor([[float(x) for x in r[5]] for r in filtered_rows])
            expected_update = tensor([[float(x) for x in r[6]] for r in filtered_rows])
            assert torch.equal(new_codes, expected_codes)
            tolerance = 1e-6 if dtype == torch.float32 else 1e-14
            torch.testing.assert_close(new_means, expected_means, rtol=tolerance, atol=tolerance)
            torch.testing.assert_close(update, expected_update, rtol=tolerance, atol=tolerance)
            filtered_checks.append(dict(device=device, dtype=str(dtype), vectors=len(filtered_rows),
                                        max_mean_error=float((new_means - expected_means).abs().max()),
                                        max_update_error=float((update - expected_update).abs().max())))
    receipt = {
        "status": "arithmetic_refinement_passed",
        "scope": "finite local kernel only; no model or training executed",
        "full_optimizer_admitted": False,
        "execution_enabled": False,
        "proof_source": str(PROOF.relative_to(ROOT)).replace("\\", "/"),
        "proof_sha256": hashlib.sha256(source).hexdigest(),
        "validator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "lean_exit_code": process.returncode,
        "named_lemmas_audited": sorted(audited),
        "legal_state_pairs": 585,
        "cue_codes": 129,
        "exact_transition_comparisons": count,
        "transition_digest_sha256": digest.hexdigest(),
        "filtered_bridge": {
            "exact_rational_vectors": len(filtered_rows),
            "oracle_digest_sha256": filtered_digest.hexdigest(),
            "implementation": "information_boltzmann/conservative_cycle_optimizer.py:_filtered_cycle_chunk",
            "implementation_sha256": hashlib.sha256((ROOT / "information_boltzmann/conservative_cycle_optimizer.py").read_bytes()).hexdigest(),
            "checks": filtered_checks,
            "online_gain_and_scale_estimation_admitted": False,
            "training_executed": False,
        },
        "representation": {"encoding_bits_sufficient": 10, "ordinary_storage_bytes_per_cycle": 2,
                           "integer_intermediate_bits_sufficient": 16,
                           "minimal_encoding_proved": False, "gpu_peak_measured": False},
        "phase_rounding_error_max": "1/64",
        "joint_phase_and_cue_error_budget": "1/32",
        "diagnostics": diagnostics,
        "review_status": "self_reviewed",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
