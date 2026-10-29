"""Exact finite-kernel algebra and the material optimizer numerical interface.

No model training, capability task or stochastic retention trial is run. This
checks source energy/rate refinement, solves absorbing linear equations and
validates the CPU optimizer interface. The
results condition on uninterrupted funded attempts; they do not admit a life
model or a memory-enabled optimizer. Metropolis is a mathematical comparison
within the SAME proposal/energy family, not a change to the runtime medium.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from information_boltzmann.core.cgb_local_medium import LocalLaw, LocalMedium, OpenMedium, ReservoirLaw
from information_boltzmann.conjugate_gradient_port import prepare_source_gradient_event


def check_lean_streamed(lake, relative_source, *, reuse_completed=False):
    """Keep diagnostics observable while a numeric certificate compiles."""
    project = ROOT / "lean/CGBAxioms"
    log_path = project / (relative_source + ".log")
    command = [lake, "env", "lean", "-j1", relative_source]
    source_hash = hashlib.sha256((project / relative_source).read_bytes()).hexdigest()
    record_path = log_path.with_suffix(".check.json")
    if reuse_completed and record_path.exists() and log_path.exists():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if (record.get("returncode") == 0 and record.get("source_sha256") == source_hash
                and record.get("log_sha256") == hashlib.sha256(log_path.read_bytes()).hexdigest()):
            return subprocess.CompletedProcess(command, 0, log_path.read_text(encoding="utf-8"), "")
    output = []
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        with subprocess.Popen(command, cwd=project, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding="utf-8") as child:
            for line in child.stdout:
                output.append(line)
                log.write(line); log.flush()
                if ".certificate' depends on axioms:" in line:
                    print(line.rstrip(), flush=True)
                elif re.search(r"\.lean:\d+:\d+: error:", line):
                    print(line.rstrip(), flush=True)
            code = child.wait()
    combined = "".join(output)
    log_path.with_suffix(".check.json").write_text(json.dumps(dict(
        command=command, source_sha256=source_hash, returncode=code,
        log_sha256=hashlib.sha256(log_path.read_bytes()).hexdigest()), indent=2) + "\n", encoding="utf-8")
    return subprocess.CompletedProcess(command, code, combined, "")


def audit_lean_declarations(output, namespace, expected):
    """Audit complete axiom blocks, including Lean's multiline pretty print.

    Benign warnings may precede these blocks. Callers must separately require
    compiler exit zero. Missing, duplicate, unexpected and nonfoundational
    declarations are rejected; a wrapped sorryAx must never disappear.
    """
    pattern = r"'" + re.escape(namespace) + r"\.([\w.]+)' depends on axioms: \[(.*?)\]"
    blocks = re.findall(pattern, output, re.S)
    names = [name for name, _ in blocks]
    assert len(names) == len(set(names)) and set(names) == set(expected), (names, expected)
    for name, deps in blocks:
        assert set(re.findall(r"[^\s,]+", deps)) <= {"propext", "Classical.choice", "Quot.sound"}, (name, deps)
    return names


def review_declarations(source):
    """Public results of the explicit review namespaces, without manual lists."""
    blocks = re.findall(r"^namespace (CGB\.[\w.]+)\n(.*?)^end \1\s*$", source, re.M | re.S)
    declared = {namespace: set(re.findall(r"^theorem (\w+)", body, re.M)) for namespace, body in blocks}
    assert declared and "CGB.OptimizerContentReview" in declared
    return declared


def audit_review(output, source, extra_optimizer=()):
    declared = review_declarations(source)
    declared["CGB.OptimizerContentReview"] |= set(extra_optimizer)
    return {namespace: audit_lean_declarations(output, namespace, expected)
            for namespace, expected in declared.items()}


def check_material_review(lake):
    """Check the integrated existing review, reusing only exact-source passes."""
    path = ROOT / "lean/CGBAxioms/reviews/optimizer_content_bridge.lean"
    source = path.read_bytes()
    generated = ROOT / "lean/CGBAxioms/.lake/material_combined_core.lean"
    if not generated.exists() or generated.read_bytes() != source:
        generated.write_bytes(source)
    checked = check_lean_streamed(lake, ".lake/material_combined_core.lean", reuse_completed=True)
    assert checked.returncode == 0, "Integrated material review failed; inspect its .lake log"
    audited = audit_review(checked.stdout, source.decode("utf-8").replace("\r\n", "\n"))
    assert path.read_bytes() == generated.read_bytes() == source
    return dict(path=str(path.relative_to(ROOT)).replace("\\", "/"),
        sha256=hashlib.sha256(source).hexdigest(), lean_exit_code=0, audited=audited,
        compile_record="lean/CGBAxioms/.lake/material_combined_core.lean.check.json",
        auto_implicit_variables=False, production_modules_added=0)


def power2(n):
    return F(2**n) if n >= 0 else F(1, 2**(-n))


def solve(matrix, rhs):
    """Independent exact Gaussian elimination for the six transient states."""
    rows = [list(row) + [value] for row, value in zip(matrix, rhs)]
    n = len(rows)
    for col in range(n):
        pivot = next(i for i in range(col, n) if rows[i][col])
        rows[col], rows[pivot] = rows[pivot], rows[col]
        scale = rows[col][col]
        rows[col] = [x / scale for x in rows[col]]
        for i in range(n):
            if i != col:
                scale = rows[i][col]
                rows[i] = [x - scale * y for x, y in zip(rows[i], rows[col])]
    return [row[-1] for row in rows]


def one_cell_protocol(a, coupling, cap=1):
    """Actual source events and reverse exponents, including disabled moves."""
    rows = []
    law = LocalLaw(1, 1, cap, cap, (0,) * (cap + 1), epsilon=0, coupling=coupling)
    def state_view(state):
        medium = OpenMedium(LocalMedium(law, state.to_bytes(law.state_bytes, "little")),
                            ReservoirLaw(3 * cap, content_cost=a, stimulus=0))
        # Enumerated states of ONE fixed total-material/initial configuration.
        # These are hypothetical source states, not recorded world exposure.
        content = sum(medium.body.get(0))
        medium.initial_content, medium.stock = 0, 3 * cap - content
        medium.intake = content
        return medium
    for state in range(law.states):
        medium = state_view(state)
        row = []
        for first, size in medium.proposal_families:
            for index in range(first, first + size):
                edits, forward, bound, sign = medium.prepare(index, 0)
                item = dict(proposal=F(1, len(medium.proposal_families) * size),
                            enabled=forward is not None, forward=forward or 0,
                            reverse=0, bound=bound, delta=sign, dest=state)
                if forward is not None:
                    p, q, w = edits[0][1]
                    dest = (p * (cap + 1) + q) * (cap + 1) + w
                    other = state_view(dest)
                    back, reverse, back_bound, back_sign = other.prepare(medium.reverse(index), 0)
                    assert back[0][1] == medium.body.get(0) and back_bound == bound
                    assert back_sign == -sign and reverse is not None
                    item.update(dest=dest, reverse=reverse)
                row.append(item)
        assert len(row) == 8 and sum(e["proposal"] for e in row) == 1
        rows.append(row)
    return rows


def event_rejection_power(event, force):
    work = force * event["delta"]
    f, r = event["forward"] - work, event["reverse"] + work
    return max(event["bound"], f, r) - f


def maximal_rejection_power(event, force):
    """Smallest common reverse-pair clock, derived in maximal_event_clock."""
    return max(F(0), event["reverse"] - event["forward"] + 2 * force * event["delta"])


def controlled_matrix(events, good, rates):
    q = [[F(int(i == j)) for j in range(len(good))] for i in range(len(good))]
    for i, state in enumerate(good):
        for event, alpha in zip(events[state], rates[i]):
            mass = event["proposal"] * alpha
            q[i][i] -= mass
            if event["dest"] in good:
                q[i][good.index(event["dest"])] += mass
    assert all(min(row) >= 0 and sum(row) <= 1 for row in q)
    return q


def robust_hitting_time(events, good, intervals, maximize):
    """Exact policy iteration over event-rate intervals; a certificate search.

    This observer changes no optimizer policy. Independent interval endpoints
    allow MORE adversarial kernels than the actual common force can produce.
    """
    rates = [[(lo + hi) / 2 for lo, hi in row] for row in intervals]
    for iteration in range(100):
        q = controlled_matrix(events, good, rates)
        n = len(good)
        t = solve([[F(int(i == j)) - q[i][j] for j in range(n)] for i in range(n)], [F(1)] * n)
        assert min(t) >= 1
        next_rates = []
        for i, state in enumerate(good):
            row = []
            for event, (lo, hi) in zip(events[state], intervals[i]):
                dest_value = t[good.index(event["dest"])] if event["dest"] in good else F(0)
                difference = dest_value - t[i]
                row.append(hi if (difference > 0 if maximize else difference < 0) else lo)
            next_rates.append(row)
        if next_rates == rates:
            return t, q, iteration + 1
        rates = next_rates
    raise AssertionError("finite kinetic certificate search did not converge")


def dyadic_power_bounds(base, exponent, bits=96):
    """Directed fixed-point powering, logarithmic in the funded horizon."""
    assert 0 <= base <= 1 and exponent >= 0
    unit = 1 << bits
    lo = base.numerator * unit // base.denominator
    hi = (base.numerator * unit + base.denominator - 1) // base.denominator
    out_lo = out_hi = unit
    while exponent:
        if exponent & 1:
            out_lo = out_lo * lo // unit
            out_hi = (out_hi * hi + unit - 1) // unit
        lo, hi = lo * lo // unit, (hi * hi + unit - 1) // unit
        exponent >>= 1
    return F(out_lo, unit), F(out_hi, unit)


def kinetic_budget_case(a, held, release):
    coupling = 3 * a
    events = one_cell_protocol(a, coupling)
    good = [i for i in range(8) if (i.bit_count() >= 2) == held]
    initial = 7 if held else 0
    nominal = (4 * coupling if held else -4 * coupling) if release else 0
    force_lo, force_hi = ((F(nominal, 2), F(2 * nominal)) if nominal > 0 else
                          (F(2 * nominal), F(nominal, 2)) if nominal < 0 else (F(-1, 4), F(1, 4)))
    intervals, exponent_bounds = [], []
    for state in good:
        row, powers = [], []
        for event in events[state]:
            if not event["enabled"]:
                row.append((F(0), F(0))); powers.append((0, 0))
                continue
            ends = [event_rejection_power(event, f) for f in [force_lo, force_hi]]
            high_k, low_k = max(ends), min(ends)
            ceil_k = -(-high_k.numerator // high_k.denominator)
            floor_k = low_k.numerator // low_k.denominator
            row.append((power2(-ceil_k), power2(-floor_k)))
            powers.append((-ceil_k, -floor_k))
        intervals.append(row); exponent_bounds.append(powers)
    t, worst_q, iterations = robust_hitting_time(events, good, intervals, maximize=release)
    v = [value / (min(t) if release else max(t)) for value in t]
    rho = 1 - 1 / (max(t) if release else min(t))
    assert 0 <= rho < 1
    for i in range(4):
        row = sum(worst_q[i][j] * v[j] for j in range(4))
        assert row <= rho * v[i] if release else rho * v[i] <= row
    # Exact nominal first-sign-change time and accepted material rewards.
    nominal_rates = [[power2(-event_rejection_power(e, nominal)) if e["enabled"] else F(0)
                      for e in events[state]] for state in good]
    q = controlled_matrix(events, good, nominal_rates)
    system = [[F(int(i == j)) - q[i][j] for j in range(4)] for i in range(4)]
    duration = solve(system, [F(1)] * 4)
    rewards = []
    for sign in [1, -1]:
        r = [sum(e["proposal"] * alpha * int(e["delta"] == sign)
                 for e, alpha in zip(events[state], nominal_rates[i])) for i, state in enumerate(good)]
        rewards.append(solve(system, r))
    start = good.index(initial)
    assert rewards[1][start] - rewards[0][start] == (2 if held else -2)
    horizons = []
    for horizon in [32, 256, 4096, 65536]:
        lower, upper = dyadic_power_bounds(rho, horizon)
        value = v[start] * (upper if release else lower)
        horizons.append(dict(funded_attempts=horizon, fee_units=horizon,
                             first_sign_change_probability_lower=str(max(F(0), 1 - value)) if release else None,
                             first_sign_change_probability_upper=str(min(F(1), 1 - value)) if not release else None))
    report = dict(a=a, coupling=coupling, held_pole="on" if held else "off", mode="release" if release else "hold",
                  force_interval=[str(force_lo), str(force_hi)], nominal_force=nominal,
                  mean_nominal_paid_attempts=str(duration[start]),
                  mean_nominal_intake=str(rewards[0][start]), mean_nominal_export=str(rewards[1][start]),
                  gross_nominal_turnover_per_attempt=str((rewards[0][start] + rewards[1][start]) / duration[start]),
                  policy_iterations=iterations, rho=str(rho), certificate_weights=[str(x) for x in v],
                  horizons=horizons, scope="first exit of midpoint readout pole, not permanent erasure or arbitrary parameter precision; funded attempts only; no imposed external wear")
    certificate = dict(events=events, good=good, intervals=intervals, powers=exponent_bounds,
                       weights=v, rho=rho, force_lo=force_lo, force_hi=force_hi, upper=release)
    return report, certificate


def emit_kinetic_audit(proof_source, certificates):
    """Generate numeric instantiations, not a new handwritten theorem module."""
    def rat(value):
        value = F(value)
        return f"(({value.numerator} : ℝ) / {value.denominator})"

    def table(name, data, typ="ℝ"):
        values = []
        for i, row in enumerate(data):
            for j, value in enumerate(row):
                atom = rat(value) if typ == "ℝ" else str(value)
                values.append(f"  | {i}, {j} => {atom}")
        return (f"noncomputable def {name} (x : Fin 4) (e : Fin 8) : {typ} :=\n"
                "  match x.val, e.val with\n" + "\n".join(values) + "\n  | _, _ => 0\n")

    chunks = [proof_source.replace("public section", "import Mathlib.Tactic.FinCases\n\npublic section", 1), """
namespace CGB.OptimizerContentReview.KineticAudit
open CGB.OptimizerContentReview
set_option maxHeartbeats 8000000
private theorem rateInterval (f r b w low high ell upper : ℝ)
    (hl : low ≤ w) (hh : w ≤ high)
    (he : ell ≤ eventLogRate f r b high) (hu : eventLogRate f r b low ≤ upper) :
    (2 : ℝ)^ell ≤ (2 : ℝ)^eventLogRate f r b w ∧
      (2 : ℝ)^eventLogRate f r b w ≤ (2 : ℝ)^upper := by
  have anti := (force_response_dissipative f r b).1
  exact ⟨(Real.rpow_le_rpow_of_exponent_le (by norm_num) he).trans (anti hh),
    (anti hl).trans (Real.rpow_le_rpow_of_exponent_le (by norm_num) hu)⟩

private theorem maximalRateInterval (f r _b w low high ell upper : ℝ)
    (hl : low ≤ w) (hh : w ≤ high)
    (he : ell ≤ min 0 (f-r-2*high)) (hu : min 0 (f-r-2*low) ≤ upper) :
    (2 : ℝ)^ell ≤ (2 : ℝ)^min 0 (f-r-2*w) ∧
      (2 : ℝ)^min 0 (f-r-2*w) ≤ (2 : ℝ)^upper := by
  have anti : Antitone (fun z : ℝ => (2 : ℝ)^min 0 (f-r-2*z)) := by
    intro x y hxy
    exact Real.rpow_le_rpow_of_exponent_le (by norm_num)
      (min_le_min (le_refl 0) (by linarith))
  exact ⟨(Real.rpow_le_rpow_of_exponent_le (by norm_num) he).trans (anti hh),
    (anti hl).trans (Real.rpow_le_rpow_of_exponent_le (by norm_num) hu)⟩

private theorem maximalRateUpper (f r w low upper : ℝ)
    (hl : low ≤ w) (hu : min 0 (f-r-2*low) ≤ upper) :
    (2 : ℝ)^min 0 (f-r-2*w) ≤ (2 : ℝ)^upper := by
  exact Real.rpow_le_rpow_of_exponent_le (by norm_num)
    ((min_le_min (le_refl 0) (by linarith)).trans hu)

noncomputable def eventKernel {X E : Type*} [Fintype E] [DecidableEq X]
    (proposal : E → ℝ) (rates : X → E → ℝ) (dest : X → E → Option X) (x y : X) : ℝ :=
  (if x = y then 1 - ∑ e, proposal e * rates x e else 0) +
    ∑ e, if dest x e = some y then proposal e * rates x e else 0

/-- Event form avoids expanding an |X| by |X| matrix for every certificate. -/
private theorem eventKernel_contract {X E : Type*} [Fintype X] [DecidableEq X] [Fintype E]
    (p : E → ℝ) (a : X → E → ℝ) (dest : X → E → Option X)
    (hp : ∀ e, 0 ≤ p e) (hsum : ∑ e, p e = 1)
    (ha : ∀ x e, 0 ≤ a x e ∧ a x e ≤ 1) :
    (∀ x y, 0 ≤ eventKernel p a dest x y) ∧
    (∀ v x, (∑ y, eventKernel p a dest x y * v y) =
      v x + ∑ e, p e * a x e * ((dest x e).elim 0 v - v x)) := by
  constructor
  · intro x y
    have hm : ∑ e, p e * a x e ≤ 1 := calc
      _ ≤ ∑ e, p e * 1 := Finset.sum_le_sum (by
        intro e _; exact mul_le_mul_of_nonneg_left (ha x e).2 (hp e))
      _ = 1 := by simpa using hsum
    unfold eventKernel
    apply add_nonneg
    · split_ifs
      · exact sub_nonneg.mpr hm
      · exact le_refl 0
    · apply Finset.sum_nonneg
      intro e _
      split_ifs
      · exact mul_nonneg (hp e) (ha x e).1
      · exact le_refl 0
  · intro v x
    unfold eventKernel
    simp_rw [add_mul, Finset.sum_add_distrib]
    have hd : (∑ y, (if x = y then 1 - ∑ e, p e * a x e else 0) * v y) =
        (1 - ∑ e, p e * a x e) * v x := by simp
    rw [hd]
    have hj : (∑ y, (∑ e, if dest x e = some y then p e * a x e else 0) * v y) =
        ∑ e, p e * a x e * (dest x e).elim 0 v := by
      simp_rw [Finset.sum_mul]
      rw [Finset.sum_comm]
      apply Finset.sum_congr rfl
      intro e _
      cases dest x e with
      | none => simp
      | some z => simp
    rw [hj]
    simp_rw [mul_sub, Finset.sum_sub_distrib]
    rw [← Finset.sum_mul]
    ring
"""
    ]
    for case_index, c in enumerate(certificates):
        case_start = len(chunks)
        good, events = c["good"], c["events"]
        rows = [events[state] for state in good]
        chunks.append(f"namespace Case{case_index}\n")
        for name, key in [("forward", "forward"), ("reverse", "reverse"), ("bound", "bound"), ("change", "delta")]:
            chunks.append(table(name, [[e[key] for e in row] for row in rows]))
        chunks.append(table("enabled", [[int(e["enabled"]) for e in row] for row in rows], "ℕ"))
        chunks.append("noncomputable def dest (x : Fin 4) (e : Fin 8) : Option (Fin 4) :=\n"
            "  match x.val, e.val with\n" + "\n".join(
                f"  | {i}, {j} => " + (f"some ⟨{good.index(e['dest'])}, by decide⟩" if e["dest"] in good else "none")
                for i, row in enumerate(rows) for j, e in enumerate(row)) + "\n  | _, _ => none\n")
        chunks.append(table("lower", [[p[0] for p in row] for row in c["intervals"]]))
        chunks.append(table("upper", [[p[1] for p in row] for row in c["intervals"]]))
        chunks.append("noncomputable def value (x : Fin 4) : ℝ :=\n  match x.val with\n" +
                      "\n".join(f"  | {i} => {rat(v)}" for i, v in enumerate(c["weights"])) + "\n  | _ => 0\n")
        chunks.append(f"noncomputable def rho : ℝ := {rat(c['rho'])}\n")
        chunks.append("""
noncomputable def rate (force : ℝ) (x : Fin 4) (e : Fin 8) : ℝ :=
  if enabled x e = 1 then (2 : ℝ)^eventLogRate (forward x e) (reverse x e) (bound x e) (force * change x e) else 0
""")
        for i, row in enumerate(rows):
            chunks.append(f"private theorem rateBounds{i} (force : ℝ)\n"
                f"    (hflo : {rat(c['force_lo'])} ≤ force) (hfhi : force ≤ {rat(c['force_hi'])}) :\n"
                f"    ∀ e, lower (⟨{i}, by decide⟩ : Fin 4) e ≤ rate force (⟨{i}, by decide⟩ : Fin 4) e ∧ "
                f"rate force (⟨{i}, by decide⟩ : Fin 4) e ≤ upper (⟨{i}, by decide⟩ : Fin 4) e := by\n"
                "  intro e; fin_cases e\n")
            for j, event in enumerate(row):
                lo, hi = c["intervals"][i][j]
                clipped = c.get("clock") == "maximal" and lo == 0 and event["enabled"]
                if not event["enabled"] or event["delta"] == 0:
                    chunks.append("  · norm_num [lower, upper, rate, enabled, forward, reverse, bound, change, eventLogRate]\n")
                    continue
                ell, top = c["powers"][i][j]
                wlo, whi = sorted([c["force_lo"] * event["delta"], c["force_hi"] * event["delta"]])
                f, r, b, d = (event[k] for k in ["forward", "reverse", "bound", "delta"])
                chunks.append("  · norm_num only [lower, upper, rate, enabled, forward, reverse, bound, change, ite_true]\n")
                if clipped:
                    chunks.append("    constructor\n    · positivity\n")
                    chunks.append(f"    · have h := maximalRateUpper ({f}) ({r}) (force * ({d})) {rat(wlo)} ({top})\n"
                                  "        (by linarith) (by norm_num)\n"
                                  "      norm_num at h\n      simpa only [mul_one, mul_zero, mul_neg_one, sub_neg_eq_add] using h\n")
                    continue
                helper = "maximalRateInterval" if c.get("clock") == "maximal" else "rateInterval"
                chunks.append(f"    have h := {helper} ({f}) ({r}) ({b}) (force * ({d})) {rat(wlo)} {rat(whi)} ({ell}) ({top})\n"
                              "      (by linarith) (by linarith) (by norm_num [eventLogRate]) (by norm_num [eventLogRate])\n"
                              "    norm_num at h\n    simpa only [mul_one, mul_zero, mul_neg_one, sub_neg_eq_add] using h\n")
        chunks.append("private theorem rateBounds (force : ℝ)\n"
            f"    (hflo : {rat(c['force_lo'])} ≤ force) (hfhi : force ≤ {rat(c['force_hi'])}) :\n"
            "    ∀ x e, lower x e ≤ rate force x e ∧ rate force x e ≤ upper x e := by\n"
            "  intro x; fin_cases x\n" + "\n".join(
                f"  · exact rateBounds{i} force hflo hfhi" for i in range(len(good))) + "\n")
        chunks.append("""
private theorem rateUnit (force : ℝ) (x : Fin 4) (e : Fin 8) :
    0 ≤ rate force x e ∧ rate force x e ≤ 1 := by
  unfold rate
  split_ifs
  · constructor
    · exact Real.rpow_nonneg (by norm_num) _
    · apply Real.rpow_le_one_of_one_le_of_nonpos (by norm_num)
""")
        if c.get("clock") == "maximal":
            chunks.append("      exact min_le_left _ _\n")
        else:
            chunks.append("""      unfold eventLogRate
      have h := (le_max_left (forward x e - force * change x e)
        (reverse x e + force * change x e)).trans
        (le_max_right (bound x e) (max (forward x e - force * change x e)
          (reverse x e + force * change x e)))
      linarith
""")
        chunks.append("  · exact ⟨le_refl 0, by norm_num⟩\n")
        relation = "(∑ y, kernel rates n x y * value y) ≤ rho * value x" if c["upper"] else "rho * value x ≤ ∑ y, kernel rates n x y * value y"
        chunks.append("""
private noncomputable def proposal (e : Fin 8) : ℝ := if e.val < 4 then 1/12 else 1/6
noncomputable def kernel (rates : ℕ → Fin 4 → Fin 8 → ℝ) (n : ℕ) (x y : Fin 4) : ℝ :=
  eventKernel proposal (rates n) dest x y
private theorem rows (rates : ℕ → Fin 4 → Fin 8 → ℝ)
    (hr : ∀ n x e, lower x e ≤ rates n x e ∧ rates n x e ≤ upper x e)
    (hp : ∀ n x e, 0 ≤ rates n x e ∧ rates n x e ≤ 1) :
    (∀ n x y, 0 ≤ kernel rates n x y) ∧ (∀ n x, """ + relation + ") := by\n" + """
  have hc (n : ℕ) := eventKernel_contract proposal (rates n) dest
    (by intro e; fin_cases e <;> norm_num [proposal])
    (by norm_num [proposal, Fin.sum_univ_succ])
    (hp n)
  constructor
  · intro n; exact (hc n).1
  · intro n x
    simp only [kernel]
    rw [(hc n).2 value x]
    clear hc hp
""")
        premises = "\n".join(f"    have h{e} := hr n x ({e} : Fin 8)" for e in range(8)) + "\n    clear hr\n"
        simp = "norm_num [proposal, dest, value, rho, lower, upper, Fin.sum_univ_succ] at * <;> linarith\n"
        chunks.append(premises + "    fin_cases x <;> " + simp)
        conclusion = "kernelSurvival (kernel (fun n x e => rate (force n x) x e)) T start x ≤ rho^T * value x" if c["upper"] else "rho^T * value x ≤ kernelSurvival (kernel (fun n x e => rate (force n x) x e)) T start x"
        chunks.append("theorem certificate (force : ℕ → Fin 4 → ℝ)\n" +
                      f"    (hf : ∀ n x, {rat(c['force_lo'])} ≤ force n x ∧ force n x ≤ {rat(c['force_hi'])}) :\n" +
                      "    ∀ T start x, " + conclusion + " := by\n" + """
  let rates := fun n x e => rate (force n x) x e
  have hr : ∀ n x e, lower x e ≤ rates n x e ∧ rates n x e ≤ upper x e := by
    intro n x e
    exact rateBounds (force n x) (hf n x).1 (hf n x).2 x e
  have hrows := rows rates hr (by intro n x e; exact rateUnit (force n x) x e)
""" + f"  exact (kinetic_survival_envelope (kernel rates) value rho hrows.1 (by norm_num [rho])).{'2' if c['upper'] else '1'}\n" +
                      "    (by intro x; fin_cases x <;> norm_num [value]) hrows.2\n")
        chunks.append(f"#print axioms certificate\nend Case{case_index}\n")
        rendered = "".join(chunks[case_start:]).replace("Fin 4", f"Fin {len(good)}")
        if c.get("clock") == "maximal":
            rendered = rendered.replace(
                "eventLogRate (forward x e) (reverse x e) (bound x e) (force * change x e)",
                "(min 0 (forward x e - reverse x e - 2 * force * change x e))")
        chunks[case_start:] = [rendered]
    chunks.append("end CGB.OptimizerContentReview.KineticAudit\n")
    return "\n".join(chunks)


def regeneration_budget_case(held, release):
    """Capacity two is the proved minimum for the registered defect margin.

    Equal endpoints require a=J*c^2/3, hence a=16 for the old J=12. No task
    data or parameter search is involved. Release controls are in [2Jc²,8Jc²]
    with the required sign; quiet controls remain [-1/4,1/4].
    """
    events = one_cell_protocol(16, 12, 2)
    count = lambda s: s // 9 + (s // 3) % 3 + s % 3
    initial, target = (26, 0) if held else (0, 26)
    good = ([s for s in range(27) if s != target] if release else
            [s for s in range(27) if (count(s) >= 5 if held else count(s) <= 1)])
    low, high = ((F(96), F(384)) if held else (F(-384), F(-96))) if release else (F(-1, 4), F(1, 4))
    intervals, powers = [], []
    for state in good:
        row, exponents = [], []
        for e in events[state]:
            ends = [maximal_rejection_power(e, f) for f in [low, high]]
            a, b = min(ends), max(ends)
            ell, upper = -(-(-b.numerator // b.denominator)), -(a.numerator // a.denominator)
            # Safe proof intervals only. The runtime's exact probabilities
            # are unchanged, including events far below this coarse bound.
            lower = power2(ell) if ell >= -32 else F(0)
            upper = max(upper, -32)
            row.append((lower, power2(upper)) if e["enabled"] else (F(0), F(0)))
            exponents.append((ell, upper) if e["enabled"] else (0, 0))
        intervals.append(row); powers.append(exponents)
    times, q, iterations = robust_hitting_time(events, good, intervals, maximize=release)
    if release:
        # A coarse rational super-solution avoids storing enormous exact
        # hitting-time fractions in the proof. It is checked against EVERY
        # permitted event rate, not just the searched extremal policy.
        rounded = [F(-(-16 * t.numerator // t.denominator), 16) for t in times]
        weights = [t / min(rounded) for t in rounded]
        rho = 1 - 1 / (2 * max(rounded))
        worst = []
        for i, state in enumerate(good):
            row = []
            for e, (lo, hi) in zip(events[state], intervals[i]):
                dest = weights[good.index(e["dest"])] if e["dest"] in good else F(0)
                row.append(hi if dest > weights[i] else lo)
            worst.append(row)
        q = controlled_matrix(events, good, worst)
    else:
        weights = [t / max(times) for t in times]
        rho = 1 - 1 / min(times)
    n = len(good)
    for i in range(n):
        row = sum(q[i][j] * weights[j] for j in range(n))
        assert row <= rho * weights[i] if release else rho * weights[i] <= row
    horizons = []
    for horizon in [256, 1024, 4096, 65536]:
        lo, hi = dyadic_power_bounds(rho, horizon)
        v = weights[good.index(initial)]
        horizons.append(dict(funded_opportunities=horizon,
            target_hit_probability_lower=str(max(F(0), 1 - v * hi)) if release else None,
            one_defect_shell_exit_probability_upper=str(min(F(1), 1 - v * lo)) if not release else None))
    return (dict(held="on" if held else "off", mode="full_endpoint_release" if release else "one_defect_shell_hold",
        cap=2, a=16, J=12, initial=initial, target=target if release else None,
        force_interval=[str(low), str(high)], good_states=good,
        horizons=horizons, policy_iterations=iterations),
        dict(events=events, good=good, intervals=intervals, powers=powers,
             weights=weights, rho=rho, force_lo=low, force_hi=high, upper=release, clock="maximal"))


def validate_regeneration(lake, out):
    """Five generated same-law certificates; no model execution or training."""
    pairs = [regeneration_budget_case(held, release) for held in [False, True] for release in [False, True]]
    proof = ROOT / "lean/CGBAxioms/reviews/optimizer_content_bridge.lean"
    proof_bytes = proof.read_bytes()
    source = emit_kinetic_audit(proof_bytes.decode("utf-8").replace("\r\n", "\n"), [c for _, c in pairs])
    good, events = [17, 23, 25, 26], one_cell_protocol(16, 12, 2)
    physical = controlled_matrix(events, good, [[power2(-maximal_rejection_power(e, 0)) if e["enabled"] else F(0)
                for e in events[state]] for state in good])
    eta = F(1, 1024)
    wear = [[F(0)] * 4 for _ in good]
    for i, state in enumerate(good):
        wear[i][i] += 1 - eta
        for unit in [1, 3, 9]:
            dest = state - unit  # all fields are positive in this shell
            if dest in good:
                wear[i][good.index(dest)] += eta / 3
    kernel = [[sum(wear[i][k] * physical[k][j] for k in range(4)) for j in range(4)] for i in range(4)]
    times = solve([[F(int(i == j)) - kernel[i][j] for j in range(4)] for i in range(4)], [F(1)] * 4)
    weights, rho = [t / max(times) for t in times], 1 - 1 / min(times)
    assert all(rho * weights[i] <= sum(kernel[i][j] * weights[j] for j in range(4)) for i in range(4))
    repair_system = [[F(int(i == j)) - physical[i][j] for j in range(3)] for i in range(3)]
    repair = solve(repair_system, [physical[i][3] for i in range(3)])
    repair_times = solve(repair_system, [F(1)] * 3)
    def rat(x):
        return f"(({x.numerator} : ℝ) / {x.denominator})"
    def table(name, data):
        return (f"noncomputable def {name} (x y : Fin 4) : ℝ :=\n  match x.val, y.val with\n" +
            "\n".join(f"  | {i}, {j} => {rat(x)}" for i, row in enumerate(data) for j, x in enumerate(row)) +
            "\n  | _, _ => 0\n")
    source += "\nnamespace CGB.OptimizerContentReview.RegenerationWear\nset_option maxHeartbeats 8000000\n"
    source += table("physical", physical) + table("wear", wear)
    source += """
noncomputable def kernel (_ : ℕ) (x y : Fin 4) : ℝ := ∑ z, wear x z * physical z y
noncomputable def value (x : Fin 4) : ℝ :=
  match x.val with
""" + "\n".join(f"  | {i} => {rat(x)}" for i, x in enumerate(weights))
    source += f"\n  | _ => 0\nnoncomputable def rho : ℝ := {rat(rho)}\n"
    source += """
theorem certificate : ∀ T start x,
    rho^T * value x ≤ kernelSurvival kernel T start x := by
  apply (kinetic_survival_envelope kernel value rho ?_ ?_).1
  · intro x; fin_cases x <;> norm_num [value]
  · intro n x
    fin_cases x <;> norm_num [kernel, wear, physical, value, rho, Fin.sum_univ_succ]
  · intro n x y
    fin_cases x <;> fin_cases y <;> norm_num [kernel, wear, physical, Fin.sum_univ_succ]
  · norm_num [rho]
#print axioms certificate
end CGB.OptimizerContentReview.RegenerationWear
"""
    generated = ROOT / "lean/CGBAxioms/.lake/material_regeneration_audit.lean"
    generated.write_text(source, encoding="utf-8", newline="\n")
    print("Checking minimal-capacity regeneration: two hold, two release, one wear certificate...", flush=True)
    checked = check_lean_streamed(lake, ".lake/material_regeneration_audit.lean")
    assert checked.returncode == 0, checked.stdout + checked.stderr
    results = audit_review(checked.stdout, proof_bytes.decode("utf-8").replace("\r\n", "\n"),
        {f"KineticAudit.Case{i}.certificate" for i in range(len(pairs))} | {"RegenerationWear.certificate"})
    audited = results["CGB.OptimizerContentReview"]
    assert proof.read_bytes() == proof_bytes
    wear_horizons = []
    for horizon in [4096, 65536]:
        lo, hi = dyadic_power_bounds(rho, horizon)
        assert lo <= rho**horizon <= hi
        wear_horizons.append(dict(funded_opportunities=horizon,
            shell_exit_probability_upper=str(min(F(1), 1 - weights[3] * lo))))
    receipt = dict(scope="generated finite-kernel algebra; conditional budgets, not executed exposure or task capability",
        full_optimizer_admitted=False, runtime_changed=False, review_status="self_reviewed",
        law=dict(cap=2, a=16, J=12, epsilon=0, potential="zero", total_material=6,
            clock="smallest reverse-pair ceiling; maximal reversible acceptance",
            provenance="minimum capacity from one_defect_regeneration_margin; equal endpoints fix a=J*c^2/3",
            physical_states=27, packed_bits_per_cell=5, readout_levels=7, certified_endpoint_codes=2),
        lean_exit_code=checked.returncode, audited=audited, profiles=[r for r, _ in pairs],
        wear=dict(probability=str(eta), force=0, order="wear then physical attempt",
            damage="uniform field; export one quantum to the existing stock",
            good_states=good, observation="first exit of N>=5 shell; stronger than first midpoint-pole change",
            mean_opportunities_to_shell_exit=str(times[3]), horizons=wear_horizons,
            environmental_work="externally imposed workload; maintenance and energy units remain separate"),
        one_defect_no_further_wear=dict(states=good[:3],
            repair_before_shell_exit=[str(x) for x in repair],
            mean_attempts_to_repair_or_exit=[str(x) for x in repair_times]),
        numerical_projection="directed 96-bit rational powers; all-horizon kernel bounds are Lean-checked",
        proof_sha256=hashlib.sha256(proof_bytes).hexdigest(),
        generated_source_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256((ROOT / "information_boltzmann/core/cgb_local_medium.py").read_bytes()).hexdigest())
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(lean_exit_code=0, law=receipt["law"],
        wear_exit_upper_4096=float(F(wear_horizons[0]["shell_exit_probability_upper"])),
        defect_repair_probability_lower=float(min(repair)), defect_attempts_upper=float(max(repair_times)))))


def validate_resource_energy(lake, out, *, with_wear=False):
    """All-size ground characterization plus exact four-cell energy refinement.

    The observer enumerates configurations, not trajectories or learning tasks.
    The Lean source snapshot preserves this review until it can be integrated
    without invalidating the concurrently running primary kinetic audit.
    """
    import numpy as np
    from information_boltzmann.conjugate_gradient_port import prepare_gradient_event

    core_review = check_material_review(lake)
    proof = ROOT / "lean/CGBAxioms/.lake/optimizer_resource_energy.lean"
    if not proof.exists() and out.exists():
        old = json.loads(out.read_text(encoding="utf-8"))
        proof.write_text(old["proof_source_snapshot"], encoding="utf-8", newline="\n")
    source = proof.read_bytes()
    proof_hash = hashlib.sha256(source).hexdigest()
    cache_path = proof.with_suffix(".checked.json")
    cached = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    reused_check = cached.get("proof_sha256") == proof_hash and cached.get("exit_code") == 0
    checked = (subprocess.CompletedProcess([], 0, cached["stdout"], "") if reused_check else
               check_lean_streamed(lake, ".lake/optimizer_resource_energy.lean"))
    if checked.returncode:
        raise RuntimeError("Resource-energy proof failed; see .lake/optimizer_resource_energy.lean.log")
    audited = audit_lean_declarations(checked.stdout, "CGB.ResourceEnergyReview",
        {"resource_ground_bound", "resource_ground_witness", "resource_monopoly_gap"})
    assert proof.read_bytes() == source
    if not reused_check:
        cache_path.write_text(json.dumps(dict(proof_sha256=proof_hash, exit_code=0, stdout=checked.stdout), indent=2) + "\n", encoding="utf-8")
    law = LocalLaw(2, 2, 2, 2, (0, 1, 4, 9, 16), epsilon=0, coupling=12)
    reservoir = ReservoirLaw(12, content_cost=16, chemical_base=2, stimulus=0)
    unit = LocalMedium(law, bytes(law.state_bytes))
    fields = np.array([(p, q, w) for p in range(3) for q in range(3) for w in range(3)], dtype=np.int16)
    local = np.array([unit.unit_energy(tuple(map(int, s))) + 14 * int(s.sum()) for s in fields], dtype=np.int16)
    assert 4 * max(abs(int(x)) for x in local) + 4 * max(law.potential) < 2**15
    potential = np.array(law.potential, dtype=np.int16)
    valid_count, minimum, ground_ids = 0, 2**30, []
    for start in range(0, 27**4, 4096):
        ids = np.arange(start, min(start + 4096, 27**4), dtype=np.int32)
        digits = np.stack([ids // 27**k % 27 for k in [3, 2, 1, 0]], axis=1)
        values = fields[digits]
        total = values.sum(axis=(1, 2), dtype=np.int16)
        p, q = values[:, :, 0].reshape(-1, 2, 2), values[:, :, 1].reshape(-1, 2, 2)
        energy = local[digits].sum(axis=1, dtype=np.int16)
        energy += potential[p.sum(axis=2, dtype=np.int16)].sum(axis=1, dtype=np.int16)
        energy += potential[q.sum(axis=1, dtype=np.int16)].sum(axis=1, dtype=np.int16)
        mask = total <= 12
        valid_count += int(mask.sum())
        if not mask.any():
            continue
        candidate = int(energy[mask].min())
        if candidate < minimum:
            minimum, ground_ids = candidate, []
        if candidate == minimum:
            ground_ids.extend(map(int, ids[mask & (energy == minimum)]))
    expected = {26 * 27**3 + 26, 26 * 27**2 + 26 * 27}
    assert minimum == -8 and set(ground_ids) == expected

    def view(state_id=None, states=None):
        if states is None:
            states = [tuple(map(int, fields[state_id // 27**k % 27])) for k in [3, 2, 1, 0]]
        word = sum(((p * 3 + q) * 3 + w) << (5 * e) for e, (p, q, w) in enumerate(states))
        m = OpenMedium(LocalMedium(law, word.to_bytes(law.state_bytes, "little")), reservoir)
        count = sum(map(sum, states))
        m.initial_content, m.stock, m.intake = 0, 12 - count, count
        assert m.stock >= 0
        return m

    def actual_energy(m):
        b = m.body
        return (sum(b.unit_energy(b.get(e)) + 14 * sum(b.get(e)) for e in range(4))
                + sum(law.potential[b.total(0, i)] for i in range(2))
                + sum(law.potential[b.total(1, j)] for j in range(2)))

    samples = set(range(0, 27**4, 6553)) | expected | {26 * (27**3 + 27**2), 26 * (27**3 + 27)}
    pairs_checked, configurations_checked = 0, 0
    for state_id in sorted(samples):
        state = [tuple(map(int, fields[state_id // 27**k % 27])) for k in [3, 2, 1, 0]]
        if sum(map(sum, state)) > 12:
            continue
        m = view(state_id)
        configurations_checked += 1
        for force in [(F(0),) * 4, (F(1, 4), F(-1, 4), F(1, 4), F(-1, 4))]:
            for event in range(m.events):
                edits, f, bound, sign = prepare_gradient_event(m, event, 0, force.__getitem__)
                if f is None:
                    continue
                successor = list(state)
                for edge, changed in edits:
                    successor[edge] = changed
                after = view(states=successor)
                assert after.stock == m.stock - sign
                _, reverse, reverse_bound, _ = prepare_gradient_event(after, m.reverse(event), 0, force.__getitem__)
                work = sum(force[e] * (sum(s) - sum(state[e])) for e, s in edits)
                assert f - reverse == -2 * (actual_energy(after) - actual_energy(m) + work)
                assert bound == reverse_bound == max(f, reverse)
                pairs_checked += 1
    # Source-event refinement at a dispersed minimum. These are algebraic
    # paths, not a repair policy: the actual optimizer still samples its fixed
    # proposal families and never reads the ground-code observer below.
    repair_paths = []
    for state_id in sorted(expected):
        ground = [tuple(map(int, fields[state_id // 27**k % 27])) for k in [3, 2, 1, 0]]
        for edge, cell in enumerate(ground):
            if cell != (2, 2, 2):
                continue
            for field in range(3):
                state = list(ground)
                defect = list(cell)
                defect[field] -= 1
                state[edge] = tuple(defect)
                route = ([(2, 1, 2), (2, 2, 2)] if field != 1 else [(2, 2, 2)])
                energies, selected = [], []
                for target in route:
                    before = view(states=state)
                    matches = []
                    for event in range(before.events):
                        edits, f, bound, sign = prepare_gradient_event(before, event, 0, lambda _e: F(0))
                        if edits == ((edge, target),):
                            matches.append((event, f, bound, sign))
                    assert matches
                    event, f, bound, sign = matches[0]
                    assert f is not None and f == bound  # acceptance is exactly one
                    state[edge] = target
                    after = view(states=state)
                    delta = actual_energy(after) - actual_energy(before)
                    assert delta <= 0 and after.stock == before.stock - sign
                    _, reverse, reverse_bound, _ = prepare_gradient_event(after, before.reverse(event), 0, lambda _e: F(0))
                    assert f - reverse == -2 * delta and bound == reverse_bound
                    energies.append(delta)
                    selected.append(event)
                assert state == ground
                assert energies == ([0, -31] if field == 0 else [-31] if field == 1 else [-3, -31])
                repair_paths.append(dict(ground_id=state_id, edge=edge, damaged_field="pqw"[field],
                    event_indices=selected, energy_changes=energies))
    assert len(repair_paths) == 12
    receipt = dict(scope="exact source-energy refinement and conditional Lean ground characterization; no model, task or executed exposure",
        review_status="self_reviewed", full_optimizer_admitted=False, training_enabled=False,
        hypothesis=dict(cap=2, coupling=12, content_cost=16, chemical_base=2, epsilon=0,
            potential="U(t)=t^2 on actual row-P and column-Q compartments", material_budget="6*n on an n by n body"),
        lean=dict(exit_code=0, audited=audited, reused_unchanged_proof=reused_check,
            original_check=cached if reused_check else None,
            conclusion="E>=-4n; every state attaining that bound is entirely off/full and has exactly one full code in every row and column; diagonal codes attain it for all n",
            scope="ground-state shape, not mixing, forced temporal participation or future predictive utility"),
        numerical=dict(all_admissible_configurations_checked=valid_count, all_microconfigurations=27**4,
            ground_energy=minimum, ground_state_ids=sorted(ground_ids), ground_states=len(ground_ids),
            source_event_pairs_checked=pairs_checked, source_configurations_checked=configurations_checked,
            dispersed_ground_single_defect_paths=repair_paths,
            path_scope="12 deterministic native-event paths, at most two events each, zero task force, unchanged other cells; each selected event has acceptance one. Proposal selection, ongoing wear, mixing and temporal service are not certified by these paths.",
            block_size=4096, energy_dtype="int16 with prechecked range; candidate IDs int32"),
        proof_path=str(proof.relative_to(ROOT)).replace("\\", "/"), proof_sha256=hashlib.sha256(source).hexdigest(),
        proof_source_snapshot=source.decode("utf-8"),
        integration_status="merged into the existing private review; standalone source snapshot retained for provenance",
        integrated_review=core_review,
        source_sha256=hashlib.sha256((ROOT / "information_boltzmann/core/cgb_local_medium.py").read_bytes()).hexdigest(),
        gradient_port_sha256=hashlib.sha256((ROOT / "information_boltzmann/conjugate_gradient_port.py").read_bytes()).hexdigest(),
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        joint_scope="This nonzero-potential chemical-base=2 family differs from the zero-potential chemical-base=0 kinetic audit. Its time bounds are not inherited.")
    if with_wear:
        receipt["shared_resource_wear"] = resource_wear_audit(lake, law, view, actual_energy)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(lean_exit_code=0, audited=audited, numerical=receipt["numerical"]), ensure_ascii=False))


def resource_wear_audit(lake, law, view, energy, *, readout_fiber=False):
    """A source-derived subkernel: other-cell events are NEVER frozen.

    The good set is a coarse observation, not an intervention. All proposals
    of the selected registered clock keep their actual weights. Any departure from
    the set is killed, including changes in unoccupied cells. Independent
    external wear is applied and observed before each funded opportunity.
    This is exact finite-kernel algebra, not an executed experience or task.
    """
    from itertools import product
    from math import lcm
    from information_boltzmann.conjugate_gradient_port import prepare_gradient_event, prepare_readout_fiber

    shell = [(1, 2, 2), (2, 1, 2), (2, 2, 1), (2, 2, 2)]
    good = [(a, (0, 0, 0), (0, 0, 0), b) for a, b in product(shell, repeat=2)]
    lookup, count, eta = {s: i for i, s in enumerate(good)}, len(good), F(1, 1024)
    physical = [[F(int(i == j)) for j in range(count)] for i in range(count)]
    wear = [[F(0)] * count for _ in good]
    checked_pairs = 0
    for i, state in enumerate(good):
        before = view(states=state)
        assert before.body.law == law and before.stock == 12 - sum(map(sum, state))
        probability_sum = F(0)
        for first, size in (() if readout_fiber else before.proposal_families):
            probability = F(1, len(before.proposal_families) * size)
            for event in range(first, first + size):
                probability_sum += probability
                edits, forward, bound, sign = prepare_gradient_event(before, event, 0, lambda _: F(0))
                if forward is None:
                    continue
                dest = list(state)
                for edge, changed in edits:
                    dest[edge] = changed
                after = view(states=dest)
                _, reverse, reverse_bound, _ = prepare_gradient_event(after, before.reverse(event), 0, lambda _: F(0))
                assert forward - reverse == -2 * (energy(after) - energy(before))
                assert bound == reverse_bound == max(forward, reverse)
                assert after.stock == before.stock - sign
                assert F(forward - bound).denominator == 1
                rate = probability * power2(int(forward - bound))
                physical[i][i] -= rate
                if tuple(dest) in lookup:
                    physical[i][lookup[tuple(dest)]] += rate
                checked_pairs += 1
        if readout_fiber:
            # SAME uniform cell/direction/candidate clock as the runtime.
            # Blocked directions remain self-loops; every other cell runs.
            for edge, direction in product(range(4), (-1, 1)):
                probability_sum += F(1, 8)
                block = prepare_readout_fiber(before, edge, direction, lambda _: F(0))
                if not block:
                    continue
                candidate_energies = []
                for state_after, power, sign in block:
                    dest = list(state)
                    dest[edge] = state_after
                    after = view(states=dest)
                    assert after.stock == before.stock-sign and power.denominator == 1
                    candidate_energies.append(energy(after)-energy(before))
                    rate = F(1, 8*len(block))*power2(-power.numerator)
                    physical[i][i] -= rate
                    if tuple(dest) in lookup:
                        physical[i][lookup[tuple(dest)]] += rate
                    checked_pairs += 1
                floor = min(candidate_energies)
                assert [power for _, power, _ in block] == [2*(E-floor) for E in candidate_energies]
        assert probability_sum == 1
        wear[i][i] += 1 - eta
        for edge, field in product(range(4), range(3)):
            dest = [list(s) for s in state]
            if dest[edge][field]:
                dest[edge][field] -= 1
            dest = tuple(tuple(s) for s in dest)
            if dest in lookup:
                wear[i][lookup[dest]] += eta / 12
    kernel = [[sum(wear[i][k] * physical[k][j] for k in range(count))
               for j in range(count)] for i in range(count)]
    assert all(min(row) >= 0 and sum(row) <= 1 for matrix in [physical, wear, kernel] for row in matrix)
    times = solve([[F(int(i == j)) - kernel[i][j] for j in range(count)]
                   for i in range(count)], [F(1)] * count)
    # Project only the proof witness. No probability in the actual kernel is
    # rounded. Every projected row inequality is checked with exact rationals.
    weight_scale, rho_scale = 2**32, 2**64
    weights = [F((t / max(times) * weight_scale).__floor__(), weight_scale) for t in times]
    assert min(weights) > 0 and weights[-1] == 1
    ratios = [sum(kernel[i][j] * weights[j] for j in range(count)) / weights[i] for i in range(count)]
    rho = F((min(ratios) * rho_scale).__floor__(), rho_scale)
    assert 0 <= rho < 1
    assert all(rho * weights[i] <= sum(kernel[i][j] * weights[j] for j in range(count)) for i in range(count))

    proof_path = ROOT / "lean/CGBAxioms/reviews/optimizer_content_bridge.lean"
    proof_bytes = proof_path.read_bytes()
    original = proof_bytes.decode("utf-8").replace("\r\n", "\n")
    envelope = original[original.index("noncomputable def kernelSurvival"):original.index("/-! The majority decoder")]
    denominator = lcm(*(x.denominator for row in kernel for x in row))
    chunks = ["""module
public import Mathlib.Basic.Real.Basic
public import Mathlib.Data.Fintype.Defs
public import Mathlib.Algebra.Order.BigOperators.Group.Finset
public import Mathlib.Algebra.BigOperators.Ring.Finset
public import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring
import Mathlib.Tactic.FinCases
public section
open scoped BigOperators
namespace CGB.OptimizerContentReview
""", envelope, "\nnamespace ResourceWear\nset_option maxHeartbeats 4000000\n"]
    for i, row in enumerate(kernel):
        chunks.append(f"private def row{i} (y : Fin {count}) : ℕ :=\n  match y.val with\n" +
                      "\n".join(f"  | {j} => {int(value * denominator)}" for j, value in enumerate(row) if value) +
                      "\n  | _ => 0\n")
    chunks.append(f"private def raw (x y : Fin {count}) : ℕ :=\n  match x.val with\n" +
                  "\n".join(f"  | {i} => row{i} y" for i in range(count)) + "\n  | _ => 0\n")
    chunks.append(f"private def rawValue (x : Fin {count}) : ℕ :=\n  match x.val with\n" +
                  "\n".join(f"  | {i} => {int(value * weight_scale)}" for i, value in enumerate(weights)) + "\n  | _ => 0\n")
    chunks.append(f"""
noncomputable def kernel (_ : ℕ) (x y : Fin {count}) : ℝ := raw x y / {denominator}
noncomputable def value (x : Fin {count}) : ℝ := rawValue x / {weight_scale}
noncomputable def rho : ℝ := ({int(rho * rho_scale)} : ℝ) / {rho_scale}
theorem certificate : ∀ T start x,
    rho^T * value x ≤ kernelSurvival kernel T start x := by
  apply (kinetic_survival_envelope kernel value rho ?_ ?_).1
  · intro x; fin_cases x <;> norm_num [value, rawValue]
  · intro n x
    fin_cases x <;> norm_num [kernel, raw, value, rawValue, rho,
""" + "      " + ", ".join(f"row{i}" for i in range(count)) + ", Fin.sum_univ_succ]\n" + """
  · intro n x y
    exact div_nonneg (Nat.cast_nonneg _) (by norm_num)
  · norm_num [rho]
end ResourceWear
#print axioms kinetic_survival_envelope
#print axioms ResourceWear.certificate
end CGB.OptimizerContentReview
""")
    source = "".join(chunks)
    relative = (".lake/material_fiber_wear.lean" if readout_fiber else ".lake/material_resource_wear.lean")
    if readout_fiber:
        source = source.replace("ResourceWear", "FiberWear")
    generated = ROOT / "lean/CGBAxioms" / relative
    generated.write_text(source, encoding="utf-8", newline="\n")
    checked = check_lean_streamed(lake, relative, reuse_completed=True)
    assert checked.returncode == 0, checked.stdout + checked.stderr
    audited = audit_lean_declarations(checked.stdout, "CGB.OptimizerContentReview",
        {"kinetic_survival_envelope", ("FiberWear.certificate" if readout_fiber else "ResourceWear.certificate")})
    assert proof_path.read_bytes() == proof_bytes
    horizons = []
    for horizon in [256, 1024, 4096, 65536]:
        lo, hi = dyadic_power_bounds(rho, horizon)
        horizons.append(dict(funded_opportunities=horizon, first_exit_probability_upper=str(1 - lo),
                             base_operation_fee_units=horizon*(13 if readout_fiber else 1)))
    transient = range(count - 1)
    system = [[F(int(i == j)) - physical[i][j] for j in transient] for i in transient]
    repair = solve(system, [physical[i][-1] for i in transient])
    repair_times = solve(system, [F(1)] * (count - 1))
    single = [i for i, s in enumerate(good) if sum(map(sum, s)) == 11]
    return dict(scope="Same quadratic-resource domain as ground theorem. Exact ideal-kernel algebra; no task, RNG trial or executed exposure.",
        workload=dict(wear_probability=str(eta), force=0, total_material=12,
            damage="uniform among all four cells and three fields; remove one unit if present, returning it to existing stock",
            order="wear then physical opportunity; first exit observed at either substep",
            funding="every counted opportunity funded; maintenance and external work separate"),
        observation="original diagonal cells each have N>=5; other cells remain zero; all other native events continue and any departure counts as failure",
        good_states=good, source_event_pairs_checked=checked_pairs,
        proposal_families=("all four cells, both directions, all adjacent candidates; no targeted repair selection"
                           if readout_fiber else "all five native reciprocal families at original family weights; no targeted repair selection"),
        kinetic_clock=("readout_fiber_v1" if readout_fiber else "maximal_reverse_pair_v1"),
        rho=str(rho), weights=[str(w) for w in weights], horizons=horizons,
        exact_rational_diagnostics=dict(mean_opportunities_to_first_exit=str(times[-1]),
            single_defect_indices=single, repair_before_exit_without_further_wear=[str(repair[i]) for i in single],
            mean_attempts_to_repair_or_exit_without_further_wear=[str(repair_times[i]) for i in single]),
        lean_exit_code=0, declarations=audited,
        proof_path=str(generated.relative_to(ROOT)).replace("\\", "/"),
        proof_sha256=hashlib.sha256(source.encode()).hexdigest(), proof_source_snapshot=source,
        reused_envelope_source_sha256=hashlib.sha256(envelope.encode()).hexdigest(),
        primary_proof_sha256=hashlib.sha256(proof_bytes).hexdigest(),
        scope_remaining="constant zero task force, first exit of a strict neighborhood; not forced release, changing-force robustness, long-term resource service or useful learning",
        full_optimizer_admitted=False, review_status="self_reviewed")


def validate_resource_release(lake, out):
    """Exact primitive/event bookkeeping checks for the proved release drift.

    This exercises local fields and clique sums, not a learning task. The
    concrete ideal kernel proves bookkeeping and includes continuing wear;
    executable update checks remain separate from full probability-sampler
    and external-world runtime refinement.
    """
    from itertools import product
    import torch
    from information_boltzmann.conjugate_gradient_port import MaterialOptimizer, prepare_gradient_event

    core = check_material_review(lake)
    law = LocalLaw(2, 2, 2, 2, (0, 1, 4, 9, 16), epsilon=0, coupling=12)
    reservoir = ReservoirLaw(12, content_cost=16, chemical_base=2, stimulus=0)
    def view(states):
        word = sum(((p*3+q)*3+w) << (law.bits*e) for e, (p, q, w) in enumerate(states))
        medium = OpenMedium(LocalMedium(law, word.to_bytes(law.state_bytes, "little")), reservoir)
        medium.initial_content, medium.intake = 0, sum(map(sum, states))
        medium.stock = 12 - medium.intake
        return medium
    def value(states):
        return sum(13*(p+w) + 9*q for p, q, w in states)
    def packed(states):
        return sum(((p*3+q)*3+w) << (law.bits*e) for e, (p, q, w) in enumerate(states))
    def cell_drift(medium, edge, force):
        p, q, w = medium.body.get(edge)
        P, Q = medium.body.total(0, edge//2), medium.body.total(1, edge%2)
        energies = (-12*w*(p-q-1)-2*P+2*Q+2, -12*w*(q-p-1)+2*P-2*Q+2,
                    -12*p*(w-q-1)+2*Q+1, -12*p*(q-w-1)-2*Q+1,
                    15-12*p*w+2*Q, -13+12*p*w-2*Q)
        enabled = (p>0 and q<2, q>0 and p<2, w>0 and q<2, q>0 and w<2,
                   q<2 and medium.stock>0, q>0)
        work = (0, 0, 0, 0, force, -force)
        rates = [power2(min(0, -2*(energy+task))) if allowed else F(0)
                 for energy, task, allowed in zip(energies, work, enabled)]
        return sum(c*a for c, a in zip((-4, 4, -4, 4, 9, -9), rates))
    samples = set()
    for edge in range(4):
        for cell in product(range(3), repeat=3):
            for extra_p, extra_q in product(range(3), repeat=2):
                states = [(0, 0, 0)] * 4
                states[edge], states[edge ^ 1], states[edge ^ 2] = cell, (extra_p, 0, 0), (0, extra_q, 0)
                samples.add(tuple(states))
    for i, j in product(range(4), repeat=2):
        if i != j:
            states = [(0, 0, 0)] * 4
            states[i] = states[j] = (2, 2, 2)
            samples.add(tuple(states))
    profiles = [(96, 96, 96, 96), (96, 384, 97, 511)]
    checked, nonzero_changes, cross_invariants = 0, 0, 0
    reference_inputs, reference_actions, reference_damage = [], [], []
    for states in sorted(samples):
        medium = view(states)
        assert medium.stock >= 0
        before_value = value(states)
        reference_inputs.append(packed(states))
        parameter = torch.nn.Parameter(torch.empty(4, dtype=torch.float64))
        owner = MaterialOptimizer([parameter], view(states), quantum=F(1), seed=0)
        snapshot = owner.state_dict()
        damage = []
        for index in range(12):
            edge, field = divmod(index, 3)
            owner.load_state_dict(snapshot)
            removed = owner.apply_external_loss(edge, field)
            assert removed == bool(states[edge][field])
            damaged = [owner.medium.body.get(i) for i in range(4)]
            assert parameter.tolist() == list(map(sum, damaged))
            assert owner.medium.stock == medium.stock + int(removed)
            assert owner.medium.released == int(removed)
            assert owner.costs == snapshot["material_optimizer"]["costs"]
            assert owner._rng.getstate() == snapshot["material_optimizer"]["rng"]
            damage.append(int.from_bytes(owner.medium.body.payload, "little")+1)
        reference_damage.append(damage)
        for profile_index, forces in enumerate(profiles):
            expected = F(0)
            actions = []
            for first, size in medium.proposal_families:
                proposal = F(1, len(medium.proposal_families)*size)
                for event in range(first, first+size):
                    edits, forward, bound, sign = prepare_gradient_event(medium, event, 0, lambda e: F(forces[e]))
                    if forward is None:
                        actions.append(0)
                        continue
                    dest = list(states)
                    for edge, state in edits:
                        dest[edge] = state
                    jump = value(dest) - before_value
                    actions.append(packed(dest)+1)
                    assert sum(map(sum, dest)) + medium.stock - sign == 12
                    if len(edits) == 2:
                        assert jump == 0
                        cross_invariants += 1
                    if jump:
                        assert F(forward-bound).denominator == 1
                        expected += proposal * power2(int(forward-bound)) * jump
                        nonzero_changes += 1
                    checked += 1
            unscaled = sum(cell_drift(medium, edge, forces[edge]) for edge in range(4))
            if profile_index == 0:
                reference_actions.append(actions)
            else:
                assert reference_actions[-1] == actions
            assert expected == unscaled/40
            if sum(map(sum, states)):
                assert 9 <= before_value <= 156
                assert unscaled <= -1 + F(36, 2**126)
                assert expected <= F(-39, 1600)
    horizons = []
    for T in [4096, 65536, 100000]:
        _, upper = dyadic_power_bounds(F(6399, 6400), T)
        horizons.append(dict(funded_opportunities=T,
            first_empty_hit_probability_lower=str(max(F(0), 1-F(140, 9)*upper))))
    receipt = dict(scope="full-state native+wear ideal-kernel release budget and executable update refinement; no learning or executed exposure",
        review_status="self_reviewed", full_optimizer_admitted=False, training_enabled=False,
        integrated_review=core,
        hypothesis=dict(cap=2, a=16, J=12, chemical_base=2, epsilon=0, potential="quadratic",
            local_force="each cell force >=96, possibly different and time/state varying",
            finite_budget="2x2 time corollary uses <=12 material units and all counted opportunities funded"),
        general_drift="For n cells the unscaled drift is <=-1+9*n/2^126; actual local proposal scale is 1/(2*families*n). Cross transfers preserve V=13(P+W)+9Q.",
        numerical=dict(source_configurations=len(samples), force_profiles=len(profiles),
            enabled_event_checks=checked, nonzero_observer_changes=nonzero_changes,
            cross_transfer_invariants=cross_invariants, largest_layout_tested="2x2, not a billion-cell allocation"),
        native_update_reference=check_native_updates(lake, core, reference_inputs, reference_actions, reference_damage),
        time_corollary=dict(rho="6399/6400", initial="two full diagonal codes, V=140", horizons=horizons,
            scope="Concrete full-state ideal native kernel: all legal nonempty bodies, stock<=12, forty proposals and wear probability1/1024 before each opportunity; hrow is proved, not assumed. Sampler/runtime probability refinement remains separate."),
        external_loss_runtime=dict(operation="apply_external_loss(edge, field)",
            before_attempt_order="funding, paid ledger, optional external loss, native proposal",
            proved_damage_successors_checked=len(reference_damage)*12,
            cache_stock_release_cost_and_rng_checked=True,
            external_world_rng_and_work="caller-owned and separately checkpointed"),
        pending="Link the separately checked ideal-bit rational sampler receipt; selective release, useful learning, temporal resource service and practical GPU execution remain open.",
        source_sha256=hashlib.sha256((ROOT/"information_boltzmann/core/cgb_local_medium.py").read_bytes()).hexdigest(),
        gradient_port_sha256=hashlib.sha256((ROOT/"information_boltzmann/conjugate_gradient_port.py").read_bytes()).hexdigest(),
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(receipt["numerical"]))


def check_native_updates(lake, core, inputs, expected_actions, expected_damage):
    """Execute the proved Lean updates; compare complete encoded cells, not V alone."""
    source = (ROOT/core["path"]).read_text(encoding="utf-8").replace("\r\n", "\n")
    source = source.replace("public section", "import Lean\n\npublic section", 1)
    # The installed Mathlib cache has logical declarations but no native
    # Multiset.map implementation. Specialize ONLY the finite-four fold for
    # execution and prove it equal to the mathematical sum below.
    mass = "def nativeMass (b : NativeBody) : ℕ := ∑ i, nativeCellMass (b i)"
    assert source.count(mass) == 1
    source = source.replace(mass, """def nativeMassExec (b : NativeBody) : ℕ :=
  nativeCellMass (b 0) + nativeCellMass (b 1) + nativeCellMass (b 2) + nativeCellMass (b 3)
@[implemented_by nativeMassExec]
""" + mass, 1)
    source += """
namespace CGB.NativeExecutionCheck
theorem finite_mass_codegen_refinement (b : CGB.ResourceReleaseReview.NativeBody) :
    CGB.ResourceReleaseReview.nativeMassExec b = CGB.ResourceReleaseReview.nativeMass b := by
  norm_num [CGB.ResourceReleaseReview.nativeMassExec, CGB.ResourceReleaseReview.nativeMass, Fin.sum_univ_succ]
  omega
#print axioms finite_mass_codegen_refinement
end CGB.NativeExecutionCheck
"""
    source += """
namespace CGB.ResourceReleaseReview
private def evalDecode (code : ℕ) : NativeBody := fun i =>
  let c := code / 2^(5*i.val) % 32
  (⟨c/9%3, Nat.mod_lt _ (by decide)⟩, ⟨c/3%3, Nat.mod_lt _ (by decide)⟩,
    ⟨c%3, Nat.mod_lt _ (by decide)⟩)
private def evalCellCode (c : NativeCell) : ℕ := (c.1.val*3+c.2.1.val)*3+c.2.2.val
private def evalEncode (b : NativeBody) : ℕ :=
  evalCellCode (b 0) + 32*evalCellCode (b 1) + 1024*evalCellCode (b 2) + 32768*evalCellCode (b 3)
private def evalEvent (n : Fin 40) : NativeEvent := by
  have hn := n.isLt
  exact if h0 : n.val < 16 then (⟨n.val/4, by omega⟩, ⟨n.val%4, by omega⟩)
    else if h1 : n.val < 24 then (⟨(n.val-16)/2, by omega⟩, ⟨4+(n.val-16)%2, by omega⟩)
    else if h2 : n.val < 28 then (⟨n.val-24, by omega⟩, 6)
    else if h3 : n.val < 32 then (⟨n.val-28, by omega⟩, 7)
    else (⟨(n.val-32)/2, by omega⟩, ⟨8+(n.val-32)%2, by omega⟩)
private def evalAction (code : ℕ) (index : Fin 40) : ℕ :=
  let b := evalDecode code
  let e := evalEvent index
  if nativeEnabled b e then evalEncode (nativeCandidate b e)+1 else 0
private def evalDamage (code : ℕ) (index : Fin 12) : ℕ :=
  let b := evalDecode code
  let e : Fin 4 × Fin 3 := (⟨index.val/3, by omega⟩, ⟨index.val%3, by omega⟩)
  evalEncode (nativeDamageCandidate b e)+1
#eval do
  let inputs : Array ℕ := #[""" + ",".join(map(str, inputs)) + """ ]
  let actions := inputs.map (fun code => Array.ofFn (fun i : Fin 40 => evalAction code i))
  let damage := inputs.map (fun code => Array.ofFn (fun i : Fin 12 => evalDamage code i))
  IO.println (Lean.Json.compress (Lean.Json.mkObj [("actions", Lean.toJson actions), ("damage", Lean.toJson damage)]))
end CGB.ResourceReleaseReview
"""
    generated = ROOT/"lean/CGBAxioms/.lake/material_native_conformance.lean"
    generated.write_text(source, encoding="utf-8", newline="\n")
    checked = check_lean_streamed(lake, ".lake/material_native_conformance.lean", reuse_completed=True)
    assert checked.returncode == 0, "Native Lean update reference failed; inspect its .lake log"
    audit_review(checked.stdout, (ROOT/core["path"]).read_text(encoding="utf-8"))
    audit_lean_declarations(checked.stdout, "CGB.NativeExecutionCheck", {"finite_mass_codegen_refinement"})
    rows = [json.loads(line) for line in checked.stdout.splitlines() if line.startswith('{"actions":')]
    assert len(rows) == 1
    assert rows[0]["actions"] == expected_actions and rows[0]["damage"] == expected_damage
    return dict(lean_exit_code=0, configurations=len(inputs),
        physical_event_results=sum(map(len, expected_actions)), damage_event_results=sum(map(len, expected_damage)),
        all_encoded_successors_and_disabled_flags_equal=True, rng_trials_or_learning=False,
        execution_refinement_lemma="CGB.NativeExecutionCheck.finite_mass_codegen_refinement",
        generated_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        log="lean/CGBAxioms/.lake/material_native_conformance.lean.log",
        scope="Complete bounded-cell successors and legality on the deterministic interface set; not full sampler/PRNG distribution refinement")


def validate_continuous_wear(lake, out):
    """Exact external-workload counterexample; no optimizer rule is changed.

    Before each funded attempt, independently with probability 1/1024, the
    environment chooses one of p/q/w uniformly and exports one unit if present.
    Export returns to the existing stock. The first midpoint-pole crossing is
    observed at BOTH substeps, so a late repair cannot hide an earlier crossing.
    This registers a driven damage workload, not a derived spontaneous law or
    a thermodynamic energy budget. Funding/maintenance and environmental work
    remain explicit conditions; this is algebra, not executed experience.
    """
    good = [3, 5, 6, 7]
    events = one_cell_protocol(4, 12)
    eta = F(1, 1024)
    wear = [[F(0)] * 4 for _ in good]
    for i, state in enumerate(good):
        wear[i][i] += 1 - eta
        for mask in [1, 2, 4]:
            dest = state & ~mask
            removed = int(bool(state & mask))
            # Same total material; the exported quantum enters existing stock.
            assert state.bit_count() - dest.bit_count() == removed
            assert 3 - dest.bit_count() == 3 - state.bit_count() + removed
            if dest in good:
                wear[i][good.index(dest)] += eta / 3
    rates = [[power2(-event_rejection_power(e, 0)) if e["enabled"] else F(0)
              for e in events[state]] for state in good]
    physical = controlled_matrix(events, good, rates)
    kernel = [[sum(wear[i][k] * physical[k][j] for k in range(4))
               for j in range(4)] for i in range(4)]
    assert all(min(row) >= 0 and sum(row) <= 1 for row in kernel)
    times = solve([[F(int(i == j)) - kernel[i][j] for j in range(4)]
                   for i in range(4)], [F(1)] * 4)
    weights = [t / min(times) for t in times]
    rho = 1 - 1 / max(times)
    assert 0 <= rho < 1 and min(weights) >= 1
    assert all(sum(kernel[i][j] * weights[j] for j in range(4))
               <= rho * weights[i] for i in range(4))
    repair_system = [[F(int(i == j)) - physical[i][j] for j in range(3)]
                     for i in range(3)]
    repair = solve(repair_system, [physical[i][3] for i in range(3)])
    repair_times = solve(repair_system, [F(1)] * 3)

    def rat(x):
        return f"(({x.numerator} : ℝ) / {x.denominator})"

    def table(name, data):
        return (f"noncomputable def {name} (x y : Fin 4) : ℝ :=\n"
                "  match x.val, y.val with\n" + "\n".join(
                    f"  | {i}, {j} => {rat(x)}"
                    for i, row in enumerate(data) for j, x in enumerate(row))
                + "\n  | _, _ => 0\n")

    proof = ROOT / "lean/CGBAxioms/reviews/optimizer_content_bridge.lean"
    proof_bytes = proof.read_bytes()
    source = proof_bytes.decode("utf-8").replace("\r\n", "\n").replace(
        "public section", "import Mathlib.Tactic.FinCases\n\npublic section", 1)
    source += "\nnamespace CGB.OptimizerContentReview.WearAudit\nset_option maxHeartbeats 8000000\n"
    source += table("physical", physical) + table("wear", wear)
    source += """
noncomputable def kernel (_ : ℕ) (x y : Fin 4) : ℝ :=
  ∑ z, wear x z * physical z y
noncomputable def value (x : Fin 4) : ℝ :=
  match x.val with
""" + "\n".join(f"  | {i} => {rat(x)}" for i, x in enumerate(weights))
    source += f"\n  | _ => 0\nnoncomputable def rho : ℝ := {rat(rho)}\n"
    source += """
theorem certificate : ∀ T start x,
    kernelSurvival kernel T start x ≤ rho^T * value x := by
  apply (kinetic_survival_envelope kernel value rho ?_ ?_).2
  · intro x; fin_cases x <;> norm_num [value]
  · intro n x
    fin_cases x <;> norm_num [kernel, wear, physical, value, rho, Fin.sum_univ_succ]
  · intro n x y
    fin_cases x <;> fin_cases y <;>
      norm_num [kernel, wear, physical, Fin.sum_univ_succ]
  · norm_num [rho]
#print axioms certificate
end CGB.OptimizerContentReview.WearAudit
"""
    generated = ROOT / "lean/CGBAxioms/.lake/material_wear_audit.lean"
    generated.write_text(source, encoding="utf-8", newline="\n")
    checked = subprocess.run([lake, "env", "lean", "-j1", ".lake/material_wear_audit.lean"],
                             cwd=ROOT / "lean/CGBAxioms", capture_output=True,
                             text=True, encoding="utf-8")
    assert checked.returncode == 0, checked.stdout + checked.stderr
    results = audit_review(checked.stdout, proof_bytes.decode("utf-8").replace("\r\n", "\n"), {"WearAudit.certificate"})
    audited = results["CGB.OptimizerContentReview"]
    assert proof.read_bytes() == proof_bytes
    horizons = []
    for horizon in [1024, 4096, 65536]:
        lo, hi = dyadic_power_bounds(rho, horizon)
        # Independent exact-rational check of the reported directed powering.
        assert lo <= rho ** horizon <= hi
        horizons.append(dict(funded_attempts=horizon,
            first_pole_change_probability_lower=str(max(F(0), 1 - weights[3] * hi))))
    receipt = dict(scope="source-derived exact external-damage kernel algebra; no training or executed exposure",
        review_status="self_reviewed", full_optimizer_admitted=False,
        optimizer_rule_changed=False, additional_optimizer_state=False,
        workload=dict(wear_probability=str(eta), order="wear, then original funded physical attempt",
            damage="uniform p/q/w; remove one unit if present; return it to original stock",
            observation="first midpoint-pole exit at either substep", force=0,
            a=4, J=12, total_material=3, initial_state="111",
            funding="all physical attempts funded; maintenance separate",
            environmental_work="externally imposed; energy cost not certified in joules"),
        physical_kernel=[[str(x) for x in row] for row in physical],
        wear_kernel=[[str(x) for x in row] for row in wear],
        composed_kernel=[[str(x) for x in row] for row in kernel],
        mean_funded_attempts_to_first_pole_change=str(times[3]),
        one_defect_no_further_wear=dict(states=["011", "101", "110"],
            restoration_before_pole_loss=[str(x) for x in repair],
            mean_attempts_to_restore_or_pole_loss=[str(x) for x in repair_times]),
        rho=str(rho), certificate_weights=[str(x) for x in weights], horizons=horizons,
        lean_scope="exact composed rational kernel weighted-row certificate; all horizons",
        numerical_projection="96-bit directed rational powers, independently checked against exact rational powers; projection not a new Lean declaration",
        lean_exit_code=checked.returncode, named_certificate="WearAudit.certificate",
        proof_sha256=hashlib.sha256(proof_bytes).hexdigest(),
        generated_source_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256((ROOT / "information_boltzmann/core/cgb_local_medium.py").read_bytes()).hexdigest(),
        runtime_sha256=hashlib.sha256((ROOT / "information_boltzmann/conjugate_gradient_port.py").read_bytes()).hexdigest())
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(lean_exit_code=0, named_certificate=receipt["named_certificate"],
        mean_funded_attempts=float(times[3]), horizons=[dict(funded_attempts=h["funded_attempts"],
        first_pole_change_probability_lower=float(F(h["first_pole_change_probability_lower"])))
        for h in horizons])))


def analyze(a, coupling, mu, metropolis, force_code=None):
    states = [(bool(i & 4), bool(i & 2), bool(i & 1)) for i in range(8)]
    force_value = 0 if force_code is None else force_code
    energy = [(a - mu + force_value) * sum(s) - coupling * all(s) for s in states]
    weights = [power2(-2 * value) for value in energy]
    transition = [[F(int(i == j)) for j in range(8)] for i in range(8)]
    rows_checked = 0
    for i, bits in enumerate(states):
        body = LocalMedium(LocalLaw(1, 1, 1, 1, (0, 0), epsilon=0, coupling=coupling), bytes([i]))
        medium = OpenMedium(body, ReservoirLaw(3 - sum(bits), content_cost=a,
                                              chemical_base=mu, stimulus=0))
        proposal = [F(0)] * medium.events
        for first, size in medium.proposal_families:
            for index in range(first, first + size):
                proposal[index] = F(1, len(medium.proposal_families) * size)
        assert sum(proposal) == 1
        for index in range(medium.events):
            edits, exponent, bound, sign = medium.prepare(index, 0)
            if exponent is None:
                continue
            assert len(edits) == 1 and edits[0][0] == 0
            p, q, w = edits[0][1]
            j = 4 * p + 2 * q + w
            delta = energy[j] - energy[i]
            assert exponent == -delta + force_value * (sum(states[j]) - sum(bits))
            assert sum(states[j]) - sum(bits) == sign
            if force_code is not None:
                driven = prepare_source_gradient_event(medium, index, 0, lambda _: force_code)
                assert driven[0] == edits and driven[3] == sign
                edits, exponent, bound, sign = driven
            probability = (min(F(1), power2(-2 * delta)) if metropolis
                           else power2(exponent - bound))
            assert 0 < probability <= 1
            transition[i][j] += proposal[index] * probability
            transition[i][i] -= proposal[index] * probability
            rows_checked += 1
    for i in range(8):
        assert sum(transition[i]) == 1
        assert min(transition[i]) >= 0
        for j in range(8):
            assert weights[i] * transition[i][j] == weights[j] * transition[j][i]
    x, k = power2(-2 * (a - mu + force_value)), power2(2 * coupling)
    off_mass = sum(weights[i] for i, s in enumerate(states) if sum(s) <= 1)
    on_mass = sum(weights[i] for i, s in enumerate(states) if sum(s) >= 2)
    assert off_mass == 1 + 3 * x and on_mass == k * x**3 + 3 * x**2
    transient = list(range(1, 7))
    system = [[F(int(i == j)) - transition[i][j] for j in transient] for i in transient]
    committor = [F(0)] + solve(system, [transition[i][7] for i in transient]) + [F(1)]
    duration = [F(0)] + solve(system, [F(1)] * 6) + [F(0)]
    for i in transient:
        assert 0 < committor[i] < 1
        assert committor[i] == sum(transition[i][j] * committor[j] for j in range(8))
        assert duration[i] == 1 + sum(transition[i][j] * duration[j] for j in range(8))
    return dict(
        a=a, coupling=coupling, mu=mu,
        force_code=force_code,
        acceptance=("maximal_reversible_metropolis" if metropolis else
                    "paired_conjugate_force" if force_code is not None else "existing_prepare_dyadic"),
        proposal="existing uniform-family then uniform-event sampler",
        transition_rows_checked=rows_checked,
        stationary_on_basin_probability=str(on_mass / (on_mass + off_mass)),
        endpoints_equal_energy=energy[0] == energy[7],
        on_before_off_probabilities=[str(v) for v in committor],
        worst_off_defect_repair_probability=float(min(1 - committor[i] for i in [1, 2, 4])),
        worst_on_defect_repair_probability=float(min(committor[i] for i in [3, 5, 6])),
        worst_expected_attempts_to_either_code=float(max(duration)),
        conditional_on_uninterrupted_funding=True,
    )


def validate_probability_sampler(lake, out, *, baseline_source=None):
    """Execute rational Lean references against actual sampler branches.

    All bit strings in each finite prefix are checked, rather than estimating
    an event probability by Monte Carlo. Optional timings exercise numerical
    coins only; no model, training stream or retention trial is executed.
    """
    import importlib.util
    from itertools import product
    import statistics
    import time
    import tracemalloc
    import torch
    from information_boltzmann.conjugate_gradient_port import MaterialOptimizer, ACCEPTANCE_SAMPLER

    core = check_material_review(lake)
    proof = ROOT / core["path"]
    probabilities = sorted({F(n, d) for d in range(1, 10) for n in range(d+1)} |
        {F(1, 17), F(16, 17), F(1, 2**200), 1-F(1, 2**200),
         F(1, 2**1074), 1-F(1, 2**1074)})
    inputs = ", ".join(f"(({q.numerator}:ℚ)/{q.denominator})" for q in probabilities)
    source = proof.read_text(encoding="utf-8") + f"""
namespace CGB.SamplerExecutionCheck
def inputs : List ℚ := [{inputs}]
def encode (q : ℚ) : Lean.Json := .arr #[.str (toString q.num), .str (toString q.den)]
def rows : Lean.Json := .arr (inputs.toArray.map fun q => .arr #[encode q,
  .arr ((List.range 25).toArray.map fun n => encode (CGB.RationalSamplerReview.coefficientQ q n)),
  .arr ((List.range 9).toArray.map fun n => encode (CGB.RationalSamplerReview.coinAcceptedQ n q))])
#eval IO.println ("SAMPLER_REFERENCE=" ++ rows.compress)
end CGB.SamplerExecutionCheck
"""
    relative = ".lake/material_probability_reference.lean"
    generated = ROOT / "lean/CGBAxioms" / relative
    generated.write_text(source, encoding="utf-8", newline="\n")
    checked = check_lean_streamed(lake, relative, reuse_completed=True)
    assert checked.returncode == 0, "Sampler reference failed; inspect its .lake log"
    audit_review(checked.stdout, proof.read_text(encoding="utf-8"))
    lines = [line for line in checked.stdout.splitlines() if line.startswith("SAMPLER_REFERENCE=")]
    assert len(lines) == 1
    vectors = json.loads(lines[0].split("=", 1)[1])
    def decimal(value):
        # Trusted local Lean numerals can exceed Python's decimal string cap.
        # Decode bounded chunks instead of changing a process-wide setting.
        negative = value.startswith("-")
        digits = value[1:] if negative else value
        result = 0
        for i in range(0, len(digits), 9):
            chunk = digits[i:i+9]
            result = result * 10**len(chunk) + int(chunk)
        return -result if negative else result
    rational = lambda pair: F(decimal(pair[0]), decimal(pair[1]))

    def make(cls, seed=17):
        law = LocalLaw(1, 1, 2, 2, (0, 0, 0), epsilon=0, coupling=12)
        medium = OpenMedium(LocalMedium(law, b"\x00"), ReservoirLaw(1, stimulus=0))
        return cls([torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))],
                   medium, quantum=F(1), seed=seed)

    optimizer = make(MaterialOptimizer)
    original_coin = optimizer._rational_ticket
    coefficient_checks = prefix_checks = completed_bit_strings = 0
    for q, (encoded, coefficients, accepted) in zip(probabilities, vectors, strict=True):
        assert rational(encoded) == q
        if 0 < q < 1:
            for n, expected in enumerate(coefficients):
                captured = []
                optimizer._rational_ticket = lambda c: (captured.append(c) or False, 0)
                bits = iter([0]*n + [1])
                optimizer._rng.getrandbits = lambda _: next(bits)
                assert optimizer._probability_ticket(q) == (True, n+1)
                assert captured == [rational(expected)]
                coefficient_checks += 1
        optimizer._rational_ticket = original_coin
        for depth, expected in enumerate(accepted):
            count = pending = 0
            for bits in product((0, 1), repeat=depth):
                stream = iter(bits)
                optimizer._rng.getrandbits = lambda _: next(stream)
                try:
                    result, used = optimizer._rational_ticket(q)
                    assert used <= depth
                    count += int(result)
                except StopIteration:
                    pending += 1
                completed_bit_strings += 1
            assert F(count, 2**depth) == rational(expected)
            assert 0 <= q-F(count, 2**depth) <= F(pending, 2**depth) <= F(1, 2**depth)
            prefix_checks += 1

    benchmark = None
    if baseline_source is not None:
        spec = importlib.util.spec_from_file_location("information_boltzmann._root_sampler_v3", baseline_source)
        baseline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(baseline)
        profiles = [F(1, 2), F(1, 3), F(1, 2**80), 1-F(1, 2**80), F(128)+F(1, 3)]
        rows = []
        for power in profiles:
            timings = {}
            for name, cls in [("root_v3", baseline.MaterialOptimizer), ("rational_v4", MaterialOptimizer)]:
                elapsed = []
                for repeat in range(3):
                    opt = make(cls, seed=17+repeat)
                    start = time.perf_counter()
                    for _ in range(1500):
                        opt._probability_ticket(power)
                    elapsed.append((time.perf_counter()-start)/1500)
                memory_opt = make(cls)
                tracemalloc.start()
                for _ in range(1500):
                    memory_opt._probability_ticket(power)
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                timings[name] = dict(median_seconds_per_coin=statistics.median(elapsed),
                    python_peak_transient_bytes=peak, requested_bits_per_coin=opt.costs["random_bits"]/1500,
                    executed_work_ledger=opt.costs)
            rows.append(dict(power=str(power), results=timings))
        benchmark = dict(scope="Single-thread CPU acceptance microbenchmark; 3x1500 timed coins plus separate Python allocation measurement; no model or GPU speed claim",
            baseline_source=str(baseline_source), baseline_sha256=hashlib.sha256(baseline_source.read_bytes()).hexdigest(),
            profiles=rows, logical_bits_are_not_PRNG_word_cost_or_joules=True)
    receipt = dict(scope="Exact ideal-fair-bit probability identity, a.s. finite rational decision, expected work, executed reference and CPU interface; no useful-learning claim",
        integrated_review=core, sampler=ACCEPTANCE_SAMPLER, checkpoint_version=4,
        review_status="self_reviewed; independent review pending", full_optimizer_admitted=False,
        training_enabled=False, physical_event_law_changed=False,
        ideal_bit_assumption="Fresh independent uniform bits; finite Python PRNG is an implementation source, not a theorem of independence",
        fractional_expected_requested_bits_upper=4, fractional_expected_coefficient_updates=1,
        fractional_rational_coin_expected_bits_upper=2,
        probability_identity="P(accept)=2^(-power); N=n has mass2^(-n-1), c0=x,c(n+1)=c(n)*(n+1-x)/(n+2)",
        reference=dict(source=str(generated.relative_to(ROOT)).replace("\\", "/"),
            sha256=hashlib.sha256(generated.read_bytes()).hexdigest(), lean_exit_code=0,
            configurations=len(probabilities), coefficient_checks=coefficient_checks,
            finite_prefix_means=prefix_checks, complete_bit_strings=completed_bit_strings,
            maximum_coefficient_index=24, maximum_prefix_depth=8,
            scope="Actual branch coefficients and all supplied bit prefixes match executable rational Lean definitions; recursive rational reference refines the real semantics for every depth"),
        costs="No roots, no added per-parameter state; rational precision and individual stopping time have no fixed cap. Requested bits, Python PRNG work, allocations and time are distinct.",
        benchmark=benchmark, source_sha256=hashlib.sha256((ROOT/"information_boltzmann/conjugate_gradient_port.py").read_bytes()).hexdigest(),
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k: receipt[k] for k in ["sampler", "checkpoint_version", "reference", "fractional_expected_requested_bits_upper", "training_enabled"]}, indent=2))
    if benchmark:
        for row in benchmark["profiles"]:
            print(json.dumps(dict(power=row["power"], results={k: v["median_seconds_per_coin"] for k, v in row["results"].items()})))


def validate_scaled_pools(lake, out):
    """Exact density-scaling references and source compatibility, no training."""
    import importlib.util
    import types
    import torch
    from information_boltzmann.conjugate_gradient_port import MaterialOptimizer, prepare_gradient_event
    core = check_material_review(lake)
    proof = ROOT / core["path"]
    cases = [(1, k) for k in range(2)] + [(2, k) for k in range(16)]
    cases += [(m, k) for m in (3, 4) for k in (0, 1, (1 << (m*m))-1, 0x15)]
    sizes = list(range(1, 17)) + [31, 31623]
    queries = [(d, t) for d in sizes for t in
               (range(2*d+1) if d <= 16 else (0, 1, d, 2*d-1, 2*d))]
    query_code = ",".join(f"({d},{t})" for d, t in queries)
    case_code = ",".join(f"({m},{word})" for m, word in cases)
    source = proof.read_text(encoding="utf-8") + f"""
namespace CGB.ResourceEnergyReview
private def spreadExec (m : ℕ) (mask : Fin m → Fin m → Bool) (i j : Fin (8*m)) : Field :=
  fieldCode (band8 (mask ⟨i.val%m, i.modNat.isLt⟩ ⟨j.val%m, j.modNat.isLt⟩)
    ⟨i.val/m, i.divNat.isLt⟩ ⟨j.val/m, j.divNat.isLt⟩)
private theorem spread_codegen_refinement (m : ℕ) (mask : Fin m → Fin m → Bool)
    (i j : Fin (8*m)) : spreadExec m mask i j = spread m mask i j := by rfl
attribute [implemented_by spreadExec] spread
#print axioms spread_codegen_refinement
private def executionMask (m word : ℕ) (i j : Fin m) : Bool :=
  decide (word / 2^(i.val*m+j.val) % 2 = 1)
private def executionFields (m word : ℕ) : Array Lean.Json :=
  ((List.ofFn fun i : Fin (8*m) => List.ofFn fun j : Fin (8*m) =>
    (spread m (executionMask m word) i j).val).flatten.toArray).map Lean.toJson
private def executionPool (D total : ℕ) : ℚ := 2*(total:ℚ)^2/D
private theorem scaled_pool_codegen_refinement (D total : ℕ) :
    (executionPool D total : ℝ) = scaledPool D total := by
  simp [executionPool, scaledPool]
#print axioms scaled_pool_codegen_refinement
private def executionRows : Lean.Json := .arr #[
  .arr (([{query_code}] : List (ℕ × ℕ)).toArray.map fun (d,t) =>
    let q := executionPool d t
    .arr #[.str (toString q.num), .str (toString q.den)]),
  .arr (([{case_code}] : List (ℕ × ℕ)).toArray.map fun (m,w) =>
    .arr (executionFields m w))]
#eval IO.println ("SCALED_REFERENCE=" ++ executionRows.compress)
end CGB.ResourceEnergyReview
"""
    relative = ".lake/material_scaled_reference.lean"
    generated = ROOT / "lean/CGBAxioms" / relative
    generated.write_text(source, encoding="utf-8", newline="\n")
    checked = check_lean_streamed(lake, relative, reuse_completed=True)
    assert checked.returncode == 0, "Scaled reference failed; inspect its .lake log"
    # The code-generation equality is private and receives its own axiom
    # check; the integrated public contract audit remains automatic.
    audit_review(checked.stdout, proof.read_text(encoding="utf-8"))
    for name in ("scaled_pool_codegen_refinement", "spread_codegen_refinement"):
        match = re.search(name + r"' depends on axioms: \[(.*?)\]", checked.stdout, re.S)
        assert match and set(re.findall(r"[^\s,]+", match[1])) <= {"propext", "Classical.choice", "Quot.sound"}
    line, = [x for x in checked.stdout.splitlines() if x.startswith("SCALED_REFERENCE=")]
    pool_values, fields = json.loads(line.split("=", 1)[1])
    laws = {}
    for (d, total), pair in zip(queries, pool_values, strict=True):
        expected = F(int(pair[0]), int(pair[1]))
        for channel in (0, 1):
            if (d, channel) not in laws:
                shape = (1, d) if channel == 0 else (d, 1)
                laws[d, channel] = LocalLaw(*shape, 2, 2,
                    tuple(t*t for t in range(2*d+1)), epsilon=0, coupling=12, pool_reference_size=2)
            assert laws[d, channel].pool_energy(channel, total) == expected
    witness_reports, distinct = [], {}
    for (m, mask), values in zip(cases, fields, strict=True):
        n = 8*m
        expected = [2 if ((i//m+j//m+((mask >> ((i%m)*m+j%m)) & 1)) % 8 < 3)
                    else 0 for i in range(n) for j in range(n)]
        assert values == expected
        word = sum(((v*3+v)*3+v) << (5*e) for e, v in enumerate(values))
        law = LocalLaw(n, n, 2, 2, tuple(t*t for t in range(2*n+1)),
                       epsilon=0, coupling=12, pool_reference_size=2)
        body = LocalMedium(law, word.to_bytes(law.state_bytes, "little"))
        E = sum(body.unit_energy(body.get(e)) + 14*sum(body.get(e)) for e in range(law.edges))
        E += F(2, n)*(sum(body.total(0, i)**2 for i in range(n)) +
                      sum(body.total(1, j)**2 for j in range(n)))
        active = sum(v == 2 for v in values)
        assert E == -144*m*m and active == 24*m*m
        assert all(body.total(ch, i) == 6*m for ch in (0, 1) for i in range(n))
        distinct[m, mask] = tuple(values)
        witness_reports.append(dict(m=m, mask=mask, active_codes=active, energy=str(E)))
    assert len({distinct[2, k] for k in range(16)}) == 16
    margins = []
    for d in (1, 2, 3, 8, 9, 64, 31623):
        law = laws.get((d, 1)) or LocalLaw(d, 1, 2, 2,
            tuple(t*t for t in range(2*d+1)), epsilon=0, coupling=12, pool_reference_size=2)
        body = LocalMedium(law, bytes(law.state_bytes))
        for e in range(d):
            body._put(e, (2, 2, 2))
        medium = OpenMedium(body, ReservoirLaw(0, content_cost=16, chemical_base=2, stimulus=0))
        edits, exponent, bound, sign = prepare_gradient_event(medium, law.events+1, 0, lambda _: 0)
        assert exponent-bound == -2*(26+F(2, d)) and sign == -1
        medium._commit(edits, sign)
        _, reverse, ceiling, back_sign = prepare_gradient_event(medium, law.events, 0, lambda _: 0)
        assert reverse-ceiling == 0 and back_sign == 1
        margins.append(dict(column_size=d, export_cost=str(26+F(2, d)), repair_acceptance=1))

    # Frozen v4 predecessor: isolate BOTH modules so the old implementation
    # cannot accidentally import the newly changed core through relative imports.
    archive = ROOT / "lean/CGBAxioms/.lake/material_optimizer_pre_scaled_v4"
    frozen = None
    if (archive/"source_manifest.json").exists():
        manifest = json.loads((archive/"source_manifest.json").read_text())
        for rel in ["information_boltzmann/core/cgb_local_medium.py", "information_boltzmann/conjugate_gradient_port.py"]:
            assert hashlib.sha256((archive/Path(rel).name).read_bytes()).hexdigest() == manifest[rel]
        for name in ("_cgb_frozen", "_cgb_frozen.core"):
            pkg = types.ModuleType(name); pkg.__path__ = [str(archive)]; sys.modules[name] = pkg
        def load(name, filename):
            spec = importlib.util.spec_from_file_location(name, archive/filename)
            mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
            spec.loader.exec_module(mod)
            return mod
        old_core = load("_cgb_frozen.core.cgb_local_medium", "cgb_local_medium.py")
        old_port = load("_cgb_frozen.conjugate_gradient_port", "conjugate_gradient_port.py")
        def make(core_module, cls):
            law = core_module.LocalLaw(2, 3, 2, 2, tuple(t*t for t in range(7)), epsilon=0, coupling=12)
            initial = [(2, 1, 2), (0, 0, 0), (1, 2, 0), (0, 2, 1), (0, 0, 0), (2, 2, 2)]
            word = sum(((p*3+q)*3+w) << (5*e) for e, (p, q, w) in enumerate(initial))
            opened = core_module.OpenMedium(core_module.LocalMedium(law, word.to_bytes(law.state_bytes, "little")),
                core_module.ReservoirLaw(5, content_cost=16, chemical_base=2, stimulus=0), opportunities_per_tick=24)
            p = torch.nn.Parameter(torch.zeros(6, dtype=torch.float64))
            return p, cls([p], opened, quantum=F(1, 8), offset=F(-3, 8), seed=13)
        import information_boltzmann.core.cgb_local_medium as current_core
        old_p, old = make(old_core, old_port.MaterialOptimizer)
        new_p, new = make(current_core, MaterialOptimizer)
        def same(a, b, pa, pb):
            assert torch.equal(pa, pb) and a.costs == b.costs and a._rng.getstate() == b._rng.getstate()
            assert a.medium.body.payload == b.medium.body.payload
            assert (a.medium.stock, a.medium.intake, a.medium.released) == (b.medium.stock, b.medium.intake, b.medium.released)
        for step in range(6):
            old_p.grad = torch.tensor([step, -step, 2.**-80, 1., 0., -1.], dtype=old_p.dtype)
            new_p.grad = old_p.grad.clone()
            old.step(); new.step(); same(old, new, old_p, new_p)
        restored_p, restored = make(current_core, MaterialOptimizer)
        restored.load_state_dict(old.state_dict())
        old_p.grad = torch.ones_like(old_p); restored_p.grad = old_p.grad.clone()
        old.step(); restored.step(); same(old, restored, old_p, restored_p)
        frozen = dict(steps=7, paid_attempts=168, legacy_body_v2_loaded_as_unscaled_v3=True,
            source_manifest=str((archive/"source_manifest.json").relative_to(ROOT)).replace("\\", "/"),
            physical_state_cache_rng_and_work_identical=True)
    receipt = dict(scope="Same-carrier density-scaled quadratic pools: all-size energy/generation and local Q margins; numerical refinement only, no learning trial",
        integrated_review=core, full_optimizer_admitted=False, training_enabled=False,
        review_status="self_reviewed; independent review pending",
        hypothesis="Registered cap2,J12,a16,mu2,epsilon0; U_D(N)=2*N^2/D. Same-density energy replication is a declared construction condition, not a general axiom or calibrated biological law.",
        derived=dict(pool_marginal_range="0<DeltaU<8 for legal cap2 intake, independent of D>=1",
            Q_quiet_margin="|force|<=1: empty intake cost>13 and one-Q-defect repair cost<-25",
            energy_lower_bound="-9*n^2/4", equality="all cells000/222; each row/column3*n/8 full codes",
            capacity_witness="For n=8m an injective m*m-bit family attains the bound: >=2^(m^2) configurations, density3/8",
            scope="Ground configurations, not equilibrium expected density, functional diversity, kinetic service or task-learning capacity"),
        reference=dict(path=str(generated.relative_to(ROOT)).replace("\\", "/"), sha256=hashlib.sha256(generated.read_bytes()).hexdigest(),
            lean_exit_code=0, pool_values_checked=2*len(queries), generated_bodies=len(cases), body_reports=witness_reports,
            codegen_refinement="Explicit quotient/remainder witness execution equals the original finite-product equivalence for every input; kernel and dynamics unchanged"),
        full_column_margins=margins, frozen_unscaled_equivalence=frozen,
        interface="One static pool-reference integer per law; same packed fields/caches. Body checkpointv3 distinguishes scaling; legacyv2 loads only as unscaled. Legacy dyadic/event-driven executors reject scaling before effects.",
        old_two_by_two_kernel="D=2 makes U_D(N)=N^2 exactly; larger-body time bounds do not follow from the old small-body certificate",
        medium_sha256=hashlib.sha256((ROOT/"information_boltzmann/core/cgb_local_medium.py").read_bytes()).hexdigest(),
        gradient_port_sha256=hashlib.sha256((ROOT/"information_boltzmann/conjugate_gradient_port.py").read_bytes()).hexdigest(),
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        next="Same-kernel temporal resource service and task-evidence selectivity; then practical execution and independent real-data quality")
    out.write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(dict(pool_values_checked=2*len(queries), generated_bodies=len(cases),
        frozen_unscaled_equivalence=frozen, full_column_margins=margins), indent=2))


def validate_readout_fiber(lake, out, *, baseline_source=None):
    """Execute the new kernel's exact local reference; no model/learning trial."""
    from copy import deepcopy
    import importlib.util
    import torch
    from information_boltzmann.conjugate_gradient_port import (
        MaterialOptimizer, prepare_readout_fiber,
    )

    core = check_material_review(lake)
    proof = ROOT / core["path"]
    contexts = [(a, b, R, C, f) for a, b in ((1, 1), (3, 2), (2, 3), (17, 1), (1, 17))
                for R in sorted({0, 2*a-2}) for C in sorted({0, 2*b-2})
                for f in (F(-35), F(-23), F(0), F(2, 7), F(1, 2**200))]
    inputs = ",".join(f"({a},{b},{R},{C},(({f.numerator}:ℚ)/{f.denominator}))"
                      for a, b, R, C, f in contexts)
    source = proof.read_text(encoding="utf-8") + f"""
namespace CGB.FiberExecutionAudit
open CGB.FiberOptimizerReview
private def cells : List Cell :=
  (List.ofFn fun p : Fin 3 => (List.ofFn fun q : Fin 3 =>
    List.ofFn fun w : Fin 3 => (p,q,w)).flatten).flatten
private def blockCells (k : Fin 6) : List Cell :=
  cells.filter (fun s => cellMass s = k.val) ++
    cells.filter (fun s => cellMass s = k.val+1)
private theorem block_execution_refines (k : Fin 6) :
    (blockCells k).toFinset = readoutBlock k ∧ (blockCells k).Nodup := by
  fin_cases k <;> decide
private def energyQ (a b R C : ℕ) (f : ℚ) (s : Cell) : ℚ :=
  14*((s.1.val:ℚ)+s.2.1.val+s.2.2.val) - 12*s.1.val*s.2.1.val*s.2.2.val +
    2*((R:ℚ)+s.1.val)^2/a + 2*((C:ℚ)+s.2.1.val)^2/b + f*cellMass s
private theorem energy_execution_refines (a b R C : ℕ) (f : ℚ) (s : Cell) :
    (energyQ a b R C f s : ℝ) = tiltedEnergy a b R C f s := by
  simp [energyQ, tiltedEnergy, cellEnergy, rawEnergy, cellMass]
private def encodeQ (q : ℚ) : Lean.Json :=
  .arr #[.str (toString q.num), .str (toString q.den)]
private def encodeCell (s : Cell) : Lean.Json :=
  .arr #[Lean.toJson s.1.val, Lean.toJson s.2.1.val, Lean.toJson s.2.2.val]
private def inputs : List (ℕ × ℕ × ℕ × ℕ × ℚ) := [{inputs}]
private def rows : Lean.Json := .arr #[
  .arr ((List.ofFn fun k : Fin 6 => .arr ((blockCells k).toArray.map encodeCell)).toArray),
  .arr (inputs.toArray.map fun (a,b,R,C,f) =>
    .arr (cells.toArray.map fun s => encodeQ (energyQ a b R C f s)))]
#print axioms block_execution_refines
#print axioms energy_execution_refines
#eval IO.println ("FIBER_REFERENCE=" ++ rows.compress)
end CGB.FiberExecutionAudit
"""
    relative = ".lake/material_readout_fiber_reference.lean"
    generated = ROOT / "lean/CGBAxioms" / relative
    generated.write_text(source, encoding="utf-8", newline="\n")
    checked = check_lean_streamed(lake, relative, reuse_completed=True)
    assert checked.returncode == 0, "Readout reference failed; inspect its .lake log"
    audit_review(checked.stdout, proof.read_text(encoding="utf-8"))
    for name in ("block_execution_refines", "energy_execution_refines"):
        match = re.search(name + r"' depends on axioms: \[(.*?)\]", checked.stdout, re.S)
        assert match and set(re.findall(r"[^\s,]+", match[1])) <= {
            "propext", "Classical.choice", "Quot.sound"}
    line, = [x for x in checked.stdout.splitlines() if x.startswith("FIBER_REFERENCE=")]
    blocks, values = json.loads(line.split("=", 1)[1])
    cells = sorted({tuple(s) for block in blocks for s in block})
    assert len(cells) == 27
    comparisons = 0
    for (a, b, R, C, force), encoded in zip(contexts, values, strict=True):
        E = {s: F(int(q[0]), int(q[1])) for s, q in zip(cells, encoded, strict=True)}
        law = LocalLaw(b, a, 2, 2, tuple(t*t for t in range(2*max(a, b)+1)),
                       epsilon=0, coupling=12, pool_reference_size=2)
        for cell in cells:
            body = LocalMedium(law, bytes(law.state_bytes))
            for i in range(1, a):
                body._put(i, (min(2, max(0, R-2*(i-1))), 0, 0))
            for i in range(1, b):
                body._put(i*a, (0, min(2, max(0, C-2*(i-1))), 0))
            body._put(0, cell)
            medium = OpenMedium(body, ReservoirLaw(1, content_cost=16, chemical_base=2, stimulus=0))
            for direction in (-1, 1):
                actual = prepare_readout_fiber(medium, 0, direction, lambda _: force)
                level = sum(cell)
                if not 0 <= level+direction <= 6:
                    assert actual == ()
                    continue
                states = [tuple(s) for s in blocks[min(level, level+direction)]]
                floor = min(E[s] for s in states)
                expected = [(s, 2*(E[s]-floor), sum(s)-level) for s in states]
                assert list(actual) == expected
                comparisons += len(actual)
    frozen = dict(status="not_requested")
    if baseline_source is not None:
        module_spec = importlib.util.spec_from_file_location(
            "information_boltzmann._pre_fiber_native_v4", baseline_source)
        predecessor = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(predecessor)
        def make(cls):
            law = LocalLaw(2, 2, 2, 2, (0, 1, 4, 9, 16),
                           epsilon=0, coupling=12, pool_reference_size=2)
            codes = (26, 0, 0, 26)
            word = sum(code << (5*i) for i, code in enumerate(codes))
            medium = OpenMedium(LocalMedium(law, word.to_bytes(law.state_bytes, "little")),
                                ReservoirLaw(2, content_cost=16, chemical_base=2, stimulus=0),
                                opportunities_per_tick=24)
            p = torch.nn.Parameter(torch.zeros(4))
            return p, cls([p], medium, quantum=F(1, 8), seed=27)
        p, old = make(predecessor.MaterialOptimizer)
        q, current = make(MaterialOptimizer)
        for step in range(7):
            p.grad = torch.tensor([0.25, -0.5, step/8, 1.0])
            q.grad = p.grad.clone()
            old.step(before_attempt=lambda: old.apply_external_loss(0, 0))
            current.step(before_attempt=lambda: current.apply_external_loss(0, 0))
            assert old.state_dict() == current.state_dict() and torch.equal(p, q)
        frozen = dict(status="identical", steps=7, paid_attempts=168,
                      source=str(baseline_source),
                      source_sha256=hashlib.sha256(baseline_source.read_bytes()).hexdigest())
    tests = ROOT / "tests/test_conjugate_gradient_port.py"
    interface = subprocess.run([sys.executable, "-m", "pytest", str(tests), "-q"],
                               cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    assert interface.returncode == 0, interface.stdout+interface.stderr
    passed = re.search(r"(\d+) passed in ([\d.]+)s", interface.stdout)
    assert passed, interface.stdout
    port = ROOT / "information_boltzmann/conjugate_gradient_port.py"
    receipt = dict(scope="Adjacent-readout kernel, same physical carrier and energy; numerical/interface only",
        integrated_review=core, full_optimizer_admitted=False, training_enabled=False,
        reference=dict(path=str(generated.relative_to(ROOT)).replace("\\", "/"),
            sha256=hashlib.sha256(generated.read_bytes()).hexdigest(), lean_exit_code=0,
            contexts=len(contexts), rational_energies=27*len(contexts), candidate_comparisons=comparisons),
        native_v4_continuation=frozen,
        tests=dict(source=str(tests.relative_to(ROOT)).replace("\\", "/"),
                   sha256=hashlib.sha256(tests.read_bytes()).hexdigest(),
                   passed=int(passed[1]), elapsed_seconds=float(passed[2]), exit_code=0),
        source_sha256=hashlib.sha256(port.read_bytes()).hexdigest(),
        medium_source_sha256=hashlib.sha256((ROOT/"information_boltzmann/core/cgb_local_medium.py").read_bytes()).hexdigest(),
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        operator=dict(class_name="FiberMaterialOptimizer", checkpoint=5, clock="readout_fiber_v1",
            max_candidate_energies=13, fee_factor=13, new_per_parameter_arrays=0,
            force_service="f>34 or f<-22: a descent-level successor has probability>=1/26 per selected cell under scaled H",
            regeneration="N=5, available stock, f<=1: full222 repair probability>=1/8 per selected cell under scaled H",
            scope="Bounds conditional on specified H, frozen force and funded execution; no arbitrary useful-memory or biologically calibrated claim"),
        open=["same-kernel continued-wear and holding/release time bounds",
              "functional descent with noisy/estimated forces and resource competition",
              "practical GPU execution and independent real-data continual quality"],
        review_status="self_reviewed; independent review pending",
        model_experiments=0)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(dict(contexts=len(contexts), candidate_comparisons=comparisons,
                         tests=receipt["tests"], native_v4_continuation=frozen), indent=2))


def validate_readout_temporal(lake, out):
    """Same-clock wear/empty-hit certificates and bounded runtime refinement."""
    from itertools import product
    from information_boltzmann.conjugate_gradient_port import prepare_readout_fiber

    core = check_material_review(lake)
    law = LocalLaw(2, 2, 2, 2, (0, 1, 4, 9, 16), epsilon=0, coupling=12, pool_reference_size=2)
    reservoir = ReservoirLaw(12, content_cost=16, chemical_base=2, stimulus=0)
    def packed(states):
        return sum(((p*3+q)*3+w) << (5*e) for e, (p, q, w) in enumerate(states))
    def view(states):
        mass = sum(map(sum, states))
        medium = OpenMedium(LocalMedium(law, packed(states).to_bytes(law.state_bytes, "little")), reservoir)
        medium.initial_content, medium.intake, medium.stock = 0, mass, 12-mass
        return medium
    def energy(medium):
        body = medium.body
        return sum(body.unit_energy(body.get(e))+14*sum(body.get(e)) for e in range(4)) + sum(
            law.pool_energy(channel, body.total(channel, i)) for channel in (0, 1) for i in range(2))
    wear = resource_wear_audit(lake, law, view, energy, readout_fiber=True)
    # The CLI regenerates this source from the actual kernel. Do not duplicate
    # its full generated text in each curated receipt.
    wear.pop("proof_source_snapshot", None)
    samples = set()
    for edge in range(4):
        for cell in product(range(3), repeat=3):
            for extra_p, extra_q in product(range(3), repeat=2):
                state = [(0, 0, 0)]*4
                state[edge] = cell
                state[edge ^ 1] = (extra_p, 0, 0)
                state[edge ^ 2] = (0, extra_q, 0)
                samples.add(tuple(state))
    for i, j in product(range(4), repeat=2):
        if i != j:
            state = [(0, 0, 0)]*4
            state[i] = state[j] = (2, 2, 2)
            samples.add(tuple(state))
    samples = sorted(samples)
    proof = ROOT/core["path"]
    source = proof.read_text(encoding="utf-8")
    # Install the finite-four execution replacement BEFORE compiled callers.
    # A late implemented_by attribute leaves already-inlined missing cache code.
    mass_decl = "def nativeMass (b : NativeBody) : ℕ := ∑ i, nativeCellMass (b i)"
    assert source.count(mass_decl) == 1
    source = source.replace(mass_decl, """def nativeMassExec (b : NativeBody) : ℕ :=
  nativeCellMass (b 0)+nativeCellMass (b 1)+nativeCellMass (b 2)+nativeCellMass (b 3)
@[implemented_by nativeMassExec]
"""+mass_decl, 1)
    source += """
namespace CGB.FiberBodyExecutionAudit
open CGB.ResourceReleaseReview CGB.FiberOptimizerReview
private theorem mass_execution_refines (b : NativeBody) : nativeMassExec b = nativeMass b := by
  norm_num [nativeMassExec, nativeMass, Fin.sum_univ_succ]
  omega
private def decode (word : ℕ) : NativeBody := fun i =>
  let c := word/2^(5*i.val)%32
  (⟨c/9%3, Nat.mod_lt _ (by decide)⟩, ⟨c/3%3, Nat.mod_lt _ (by decide)⟩,
    ⟨c%3, Nat.mod_lt _ (by decide)⟩)
private def cellCode (s : Cell) : ℕ := (s.1.val*3+s.2.1.val)*3+s.2.2.val
private def encode (b : NativeBody) : ℕ :=
  cellCode (b 0)+32*cellCode (b 1)+1024*cellCode (b 2)+32768*cellCode (b 3)
private def cells : List Cell :=
  (List.ofFn fun p : Fin 3 => (List.ofFn fun q : Fin 3 =>
    List.ofFn fun w : Fin 3 => (p,q,w)).flatten).flatten
private def row (word : ℕ) : Lean.Json :=
  let b := decode word
  if h : 0 < nativeMass b ∧ nativeMass b ≤ 12 then
    let x : NativeState := ⟨b,h⟩
    .arr (Array.ofFn fun i : Fin 4 =>
      .arr #[Lean.toJson (bodyBudget x i).val,
        Lean.toJson (b (partner false i)).1.val,
        Lean.toJson (b (partner true i)).2.1.val,
        .arr (cells.toArray.map fun c =>
          let dest := Function.update b i c
          if 0 < nativeMass dest ∧ nativeMass dest ≤ 12 then
            Lean.toJson (encode dest+1) else Lean.toJson (0:ℕ))])
  else .null
#print axioms mass_execution_refines
#eval IO.println ("FIBER_BODY_REFERENCE=" ++ (Lean.Json.arr (([""" + ",".join(str(packed(s)) for s in samples) + """] : List ℕ).toArray.map row)).compress)
end CGB.FiberBodyExecutionAudit
"""
    relative = ".lake/material_fiber_body_conformance.lean"
    generated = ROOT/"lean/CGBAxioms"/relative
    generated.write_text(source, encoding="utf-8", newline="\n")
    checked = check_lean_streamed(lake, relative, reuse_completed=True)
    assert checked.returncode == 0, "Body reference failed; inspect its .lake log"
    audit_review(checked.stdout, proof.read_text(encoding="utf-8"))
    match = re.search(r"mass_execution_refines' depends on axioms: \[(.*?)\]", checked.stdout, re.S)
    assert match and set(re.findall(r"[^\s,]+", match[1])) <= {"propext", "Classical.choice", "Quot.sound"}
    line, = [s for s in checked.stdout.splitlines() if s.startswith("FIBER_BODY_REFERENCE=")]
    rows = json.loads(line.split("=", 1)[1])
    profiles = [(96, 96, 96, 96), (96, 384, 97, 511)]
    rho, epsilon = F(1279, 1280), F(1, 2**118)
    candidate_checks, action_checks, budget_checks = 0, 0, 0
    for states, row in zip(samples, rows, strict=True):
        medium = view(states)
        mass = sum(map(sum, states))
        if not mass:
            assert row is None
            continue
        for edge in range(4):
            budget, R, C, actions = row[edge]
            assert budget == min(6, sum(states[edge])+medium.stock)
            assert R == states[edge ^ 1][0] and C == states[edge ^ 2][1]
            for cell, action in zip(product(range(3), repeat=3), actions, strict=True):
                dest = list(states); dest[edge] = cell
                dest_mass = sum(map(sum, dest))
                assert action == (packed(dest)+1 if 0 < dest_mass <= 12 else 0)
                action_checks += 1
            budget_checks += 1
        before = energy(medium)
        for forces in profiles:
            drift = F(0)
            for edge, direction in product(range(4), (-1, 1)):
                block = prepare_readout_fiber(medium, edge, direction, lambda i: forces[i])
                diffs = []
                for cell, power, delta in block:
                    dest = list(states); dest[edge] = cell
                    after = view(dest)
                    assert after.stock == medium.stock-delta >= 0
                    diffs.append(energy(after)-before+forces[edge]*delta)
                    assert power.denominator == 1
                    drift += F(1, 8*len(block))*power2(-power.numerator)*delta
                    candidate_checks += 1
                if block:
                    floor = min(diffs)
                    assert [power for _, power, _ in block] == [2*(v-floor) for v in diffs]
            assert drift <= -F(1, 104)+epsilon
            assert mass+drift <= rho*mass
    horizons = []
    for T in (4096, 8192, 16384, 65536):
        _, upper = dyadic_power_bounds(rho, T)
        bound = max(F(0), 1-12*upper)
        horizons.append(dict(funded_opportunities=T, base_operation_fee_units=13*T,
                             first_empty_hit_probability_lower=str(bound), lower_float=float(bound)))
    receipt = dict(scope="Same readout kernel: continued wear holding, full-state first empty hit and executable body refinement; no learning trial",
        integrated_review=core, full_optimizer_admitted=False, training_enabled=False,
        model_experiments=0, review_status="self_reviewed; independent review pending",
        holding=wear,
        release=dict(declarations=["CGB.FiberOptimizerReview.readout_release_drift", "CGB.FiberOptimizerReview.readout_release_budget"],
            rho=str(rho), initial_mass=12, force="every cell>=96, possibly time/state dependent",
            body_scope="all four-cell nonempty states with mass<=12; stock=12-mass; same1/1024 external wear before each paid opportunity, observed at both substeps",
            kernel_nonnegative_and_row_sum_at_most_one=True, hrow_assumed=False, horizons=horizons),
        execution_reference=dict(path=str(generated.relative_to(ROOT)).replace("\\", "/"),
            source_sha256=hashlib.sha256(generated.read_bytes()).hexdigest(), lean_exit_code=0,
            source_bodies=len(samples), budget_and_pool_views=budget_checks,
            complete_candidate_successors=action_checks, whole_energy_candidate_checks=candidate_checks,
            force_profiles=len(profiles)),
        source_sha256=hashlib.sha256((ROOT/"information_boltzmann/conjugate_gradient_port.py").read_bytes()).hexdigest(),
        medium_source_sha256=hashlib.sha256((ROOT/"information_boltzmann/core/cgb_local_medium.py").read_bytes()).hexdigest(),
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        no_optimizer_transition_change=True,
        open=["selective useful retention and actual-risk response under noisy/estimated forces",
              "temporal resource service and functional anti-monopoly at general body sizes",
              "practical GPU budget and independent continual real-data quality"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(dict(execution_reference=receipt["execution_reference"], release=receipt["release"],
                         holding_horizons=wear["horizons"]), indent=2))


def validate_readout_task_risk(lake, out):
    """Exact analytic risk/port checks; no model training or capability trial."""
    from itertools import product
    import torch
    from information_boltzmann.conjugate_gradient_port import FiberMaterialOptimizer, prepare_readout_fiber

    core = check_material_review(lake)
    law = LocalLaw(2, 2, 2, 2, (0, 1, 4, 9, 16), epsilon=0, coupling=12, pool_reference_size=2)
    def medium_for(cell, R, C, stock):
        states = (cell, (R, 0, 0), (0, C, 0), (0, 0, 0))
        word = sum(((p*3+q)*3+w) << (5*e) for e, (p, q, w) in enumerate(states))
        return OpenMedium(LocalMedium(law, word.to_bytes(law.state_bytes, "little")),
                          ReservoirLaw(stock, content_cost=16, chemical_base=2, stimulus=0))
    def row_for(medium, force):
        old = medium.body.get(0)
        row = {old: F(1)}
        for direction in (-1, 1):
            block = prepare_readout_fiber(medium, 0, direction, lambda _: force)
            for state, power, _ in block:
                assert power.denominator == 1
                rate = F(1, 2*len(block))*power2(-power.numerator)
                row[state] = row.get(state, F(0))+rate
                row[old] -= rate
        assert sum(row.values()) == 1 and min(row.values()) >= 0
        return row
    epsilon, exact_checks, row_checks, negative_checks = F(1, 2**118), 0, 0, 0
    for R, C in ((0, 0), (0, 2), (2, 0), (2, 2), (1, 1)):
        for cell in product(range(3), repeat=3):
            n = sum(cell)
            for stock in (0, 1):
                medium = medium_for(cell, R, C, stock)
                for sign in (-1, 1):
                    if (sign == 1 and n == 0) or (sign == -1 and (n == 6 or stock == 0)):
                        continue
                    row = row_for(medium, 96*sign)
                    good = sum(p for state, p in row.items() if sum(state) == n-sign)
                    bad = sum(p for state, p in row.items() if sum(state) == n+sign)
                    assert good >= F(1, 26) and bad <= epsilon
                    row_checks += 1
                    for quantum, estimate, L, error_ratio, error_sign in product(
                            (F(1, 8), F(1, 64)), (F(1), F(4)), (F(1), F(4)),
                            (F(0), F(1, 2)), (-1, 1)):
                        if error_ratio == 0 and error_sign == -1:
                            continue
                        error = estimate*error_ratio
                        actual = sign*estimate+error_sign*error
                        theta = quantum*n
                        target = theta-actual/L
                        loss_before = L/2*(theta-target)**2
                        expected = sum(p*(L/2*(quantum*sum(state)-target)**2-loss_before)
                                       for state, p in row.items())
                        D, c = quantum*estimate, quantum*error+L*quantum**2/2
                        bound = -(D-c)/26+(D+c)*epsilon
                        decomposition = (-actual*quantum+L*quantum**2/2)*sum(
                            p for state, p in row.items() if sum(state) == n-1)
                        decomposition += (actual*quantum+L*quantum**2/2)*sum(
                            p for state, p in row.items() if sum(state) == n+1)
                        assert abs(actual-sign*estimate) <= error
                        assert c <= D and L*quantum <= estimate/2 and error <= estimate/2
                        assert expected == decomposition <= bound <= -D/208
                        exact_checks += 1
                        negative_checks += int(expected < 0)
    port_checks = 0
    for quantum, estimate, sign in product((F(1, 8), F(1, 64)), (F(1), F(4)), (-1, 1)):
        p = torch.nn.Parameter(torch.zeros(4))
        owner = FiberMaterialOptimizer([p], medium_for((1, 1, 1), 1, 1, 1), quantum=quantum,
                                       energy_unit=quantum*estimate/96, seed=0)
        p.grad = torch.full_like(p, float(sign*estimate))
        assert owner._force(0) == 96*sign
        port_checks += 1
    small = LocalLaw(1, 1, 2, 2, (0, 1, 4), epsilon=0, coupling=12, pool_reference_size=2)
    neutral = OpenMedium(LocalMedium(small, b"\x00"), ReservoirLaw(1, content_cost=16, chemical_base=2, stimulus=0))
    neutral_row = row_for(neutral, 0)
    floor = sum(p*sum(state)**2 for state, p in neutral_row.items())
    assert floor == F(9, 2**34) >= F(1, 2**31) > 0
    receipt = dict(scope="Actual readout-kernel risk bound under explicit finite Taylor/error conditions and an analytic autonomous-noise counterexample; no capability trial",
        integrated_review=core, full_optimizer_admitted=False, training_enabled=False, model_experiments=0,
        declarations=["CGB.FiberOptimizerReview.readout_task_risk_contract", "CGB.FiberOptimizerReview.autonomous_task_noise_floor"],
        assumptions=dict(force="magnitude>=96 in the actual port units, signed with observed gradient",
            model="whole task objective along the one changed coordinate has the written Taylor upper bound",
            error="explicit envelope on the difference between true current gradient and supplied observation; no estimator is assumed valid",
            feasibility="positive signal:nonempty cell; negative signal:room in N+stock budget",
            robust_domain="gradient error<=50% observed magnitude and L*quantum<=50% observed magnitude"),
        derived=dict(D="quantum*observed_magnitude", c="quantum*gradient_error+L*quantum^2/2",
            one_opportunity_risk_upper="-(D-c)/26+(D+c)/2^118",
            robust_risk_upper="-D/208 per selected feasible cell", no_risk_sign_assumed=True),
        numeric=dict(exact_kernel_rows=row_checks, independent_quadratic_checks=exact_checks,
            negative_expected_increments=negative_checks, actual_force_port_checks=port_checks,
            scope="independent analytic loss differences, source row probabilities and exact rational arithmetic; no learner, epochs or empirical convergence"),
        noise_floor=dict(objective="theta^2 at theta0=0, supplied gradient0, available stock1",
            body="single cap2 cell, scaled quadratic H", exact_expected_increment=str(floor),
            proved_lower="1/2^31", significance="unconditional task descent at every state is false; finite quantum and autonomous dynamics require a noise/accuracy budget"),
        source_sha256=hashlib.sha256((ROOT/"information_boltzmann/conjugate_gradient_port.py").read_bytes()).hexdigest(),
        medium_source_sha256=hashlib.sha256((ROOT/"information_boltzmann/core/cgb_local_medium.py").read_bytes()).hexdigest(),
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        no_optimizer_transition_change=True, review_status="self_reviewed; independent review pending",
        open=["actual online noise/staleness envelopes and useful selective retention over full trajectories",
              "weak-signal/finite-precision task-risk floor and general-size temporal resource service/functional anti-monopoly",
              "practical GPU budget and independent continual real-data convergence/quality"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(dict(numeric=receipt["numeric"],noise_floor=receipt["noise_floor"]),indent=2))


def validate_population_window(lake, out):
    """Uncut arbitrary-size proof, executed geometry, exact finite-window algebra."""
    from itertools import product
    from information_boltzmann.conjugate_gradient_port import prepare_readout_fiber

    core = check_material_review(lake)
    cells = list(product(range(3), repeat=3))
    def packed(states):
        return sum(((p*3+q)*3+w) << (5*e) for e,(p,q,w) in enumerate(states))
    queries, bodies = [], []
    for rows, cols in ((1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 2), (1, 17), (17, 1)):
        n = rows*cols
        patterns = (list(product(cells, repeat=n)) if n <= 2 else
                    [tuple([s]*n) for s in cells] +
                    [tuple(s if e == at else (0,0,0) for e in range(n)) for s in cells for at in (0,n-1)] +
                    [tuple(cells[(k+e)%27] for e in range(n)) for k in range(27)])
        for states in sorted(set(patterns)):
            mass = sum(map(sum,states))
            for stock in (0,1):
                queries.append((rows,cols,mass+stock,packed(states)))
                bodies.append(states)
    proof = ROOT/core["path"]
    source = proof.read_text(encoding="utf-8")
    replacements = {
        "def populationMass {r c : ℕ} (b : Population r c) : ℕ := ∑ i, cellMass (b i)":
        """private def populationMassExec {r c : ℕ} (b : Population r c) : ℕ :=
  (List.ofFn fun i : Fin r => (List.ofFn fun j : Fin c => cellMass (b (i,j))).sum).sum
@[implemented_by populationMassExec]
def populationMass {r c : ℕ} (b : Population r c) : ℕ := ∑ i, cellMass (b i)""",
        "def outsideRow {r c : ℕ} (b : Population r c) (i : Site r c) : ℕ :=":
        """private def outsideRowExec {r c : ℕ} (b : Population r c) (i : Site r c) : ℕ :=
  (List.ofFn fun j : Fin c => if j=i.2 then 0 else (b (i.1,j)).1.val).sum
@[implemented_by outsideRowExec]
def outsideRow {r c : ℕ} (b : Population r c) (i : Site r c) : ℕ :=""",
        "def outsideCol {r c : ℕ} (b : Population r c) (i : Site r c) : ℕ :=":
        """private def outsideColExec {r c : ℕ} (b : Population r c) (i : Site r c) : ℕ :=
  (List.ofFn fun j : Fin r => if j=i.1 then 0 else (b (j,i.2)).2.1.val).sum
@[implemented_by outsideColExec]
def outsideCol {r c : ℕ} (b : Population r c) (i : Site r c) : ℕ :="""}
    for old,new in replacements.items():
        assert source.count(old)==1
        source=source.replace(old,new,1)
    # Bound elaborator recursion independently of the number of vectors.  A
    # single several-thousand-element Lean List literal nests cons nodes and
    # exceeds its default recursion limit before the executable reference runs.
    chunks = [queries[i:i+64] for i in range(0, len(queries), 64)]
    inputs = "\n".join(
        f"private def populationInputs{k} : List (ℕ × ℕ × ℕ × ℕ) := ["
        + ",".join(f"({r},{c},{m},{w})" for r,c,m,w in chunk) + "]"
        for k, chunk in enumerate(chunks))
    input_names = ",".join(f"populationInputs{k}" for k in range(len(chunks)))
    source += """
namespace CGB.FiberOptimizerReview
private theorem erased_fin_execution (n : ℕ) (f : Fin n → ℕ) (i : Fin n) :
    (List.ofFn fun j => if j=i then 0 else f j).sum = ∑ j ∈ Finset.univ.erase i, f j := by
  rw [List.sum_ofFn]
  have hs := Finset.sum_erase_add Finset.univ (fun j => if j=i then 0 else f j) (Finset.mem_univ i)
  calc
    _ = (∑ j ∈ Finset.univ.erase i, if j=i then 0 else f j)+(if i=i then 0 else f i) := hs.symm
    _ = ∑ j ∈ Finset.univ.erase i, if j=i then 0 else f j := by simp
    _ = _ := Finset.sum_congr rfl (by intro j hj; simp [(Finset.mem_erase.mp hj).1])
private theorem population_mass_execution {r c : ℕ} (b : Population r c) : populationMassExec b=populationMass b := by
  simp [populationMassExec,populationMass,List.sum_ofFn,Fintype.sum_prod_type]
private theorem population_row_execution {r c : ℕ} (b : Population r c) (i : Site r c) : outsideRowExec b i=outsideRow b i := by
  exact erased_fin_execution c (fun j => (b (i.1,j)).1.val) i.2
private theorem population_col_execution {r c : ℕ} (b : Population r c) (i : Site r c) : outsideColExec b i=outsideCol b i := by
  exact erased_fin_execution r (fun j => (b (j,i.2)).2.1.val) i.1
private def populationDecode (r c word : ℕ) : Population r c := fun i =>
  let z := word/2^(5*(i.1.val*c+i.2.val))%32
  (⟨z/9%3,Nat.mod_lt _ (by decide)⟩,⟨z/3%3,Nat.mod_lt _ (by decide)⟩,⟨z%3,Nat.mod_lt _ (by decide)⟩)
private def populationCode {r c : ℕ} (b : Population r c) : ℕ :=
  (List.ofFn fun i : Fin r => (List.ofFn fun j : Fin c =>
    let s := b (i,j)
    ((s.1.val*3+s.2.1.val)*3+s.2.2.val)*2^(5*(i.val*c+j.val))).sum).sum
private def executionCells : List Cell :=
  (List.ofFn fun p : Fin 3 => (List.ofFn fun q : Fin 3 => List.ofFn fun w : Fin 3 => (p,q,w)).flatten).flatten
private def executionPopulation (r c supply word : ℕ) : Lean.Json :=
  let b := populationDecode r c word
  .arr #[Lean.toJson (populationMass b), .arr ((List.ofFn fun row : Fin r =>
    (List.ofFn fun col : Fin c =>
      let i : Site r c := (row,col)
      let budget := (populationBudget supply b i).val
      .arr #[Lean.toJson budget,Lean.toJson (outsideRow b i),Lean.toJson (outsideCol b i),
        .arr (executionCells.toArray.map fun z =>
          if cellMass z ≤ budget then Lean.toJson (populationCode (Function.update b i z)+1) else Lean.toJson (0:ℕ)),
        .arr ((List.ofFn fun field : Fin 3 => Lean.toJson (populationCode (populationDamage b (i,field))+1)).toArray)]
      )).flatten.toArray)]
#print axioms population_mass_execution
#print axioms population_row_execution
#print axioms population_col_execution
"""+inputs+"""
#eval IO.println ("POPULATION_REFERENCE=" ++ (Lean.Json.arr ((["""+input_names+"""] : List (List (ℕ × ℕ × ℕ × ℕ))).flatten.toArray.map fun (r,c,m,w) => executionPopulation r c m w)).compress)
end CGB.FiberOptimizerReview
"""
    relative=".lake/material_population_reference.lean"
    generated=ROOT/"lean/CGBAxioms"/relative
    generated.write_text(source,encoding="utf-8",newline="\n")
    checked=check_lean_streamed(lake,relative,reuse_completed=True)
    assert checked.returncode==0,"Population reference failed; inspect its .lake log"
    audit_review(checked.stdout,proof.read_text(encoding="utf-8"))
    for name in ("population_mass_execution","population_row_execution","population_col_execution"):
        match=re.search(name+r"' depends on axioms: \[(.*?)\]",checked.stdout,re.S)
        assert match and set(re.findall(r"[^\s,]+",match[1])) <= {"propext","Classical.choice","Quot.sound"}
    line,=[s for s in checked.stdout.splitlines() if s.startswith("POPULATION_REFERENCE=")]
    vectors=json.loads(line.split("=",1)[1])
    budget_views=successors=damage=energies=0
    def make(rows,cols,states,supply):
        law=LocalLaw(rows,cols,2,2,tuple(t*t for t in range(2*max(rows,cols)+1)),
                     epsilon=0,coupling=12,pool_reference_size=2)
        return OpenMedium(LocalMedium(law,packed(states).to_bytes(law.state_bytes,"little")),
                          ReservoirLaw(supply-sum(map(sum,states)),content_cost=16,chemical_base=2,stimulus=0))
    for (rows,cols,supply,word),states,vector in zip(queries,bodies,vectors,strict=True):
        mass,views=vector;assert mass==sum(map(sum,states)) and supply>=mass
        medium=make(rows,cols,states,supply)
        for edge,(budget,R,C,actions,damages) in enumerate(views):
            row,col=divmod(edge,cols);old=states[edge];level=sum(old)
            assert budget==min(6,level+medium.stock)
            assert R==medium.body.total(0,row)-old[0] and C==medium.body.total(1,col)-old[1]
            assert R+2<=2*cols and C+2<=2*rows
            budget_views+=1
            for cell,action in zip(cells,actions,strict=True):
                dest=list(states);dest[edge]=cell
                assert action==(packed(dest)+1 if sum(map(sum,dest))<=supply else 0)
                successors+=1
            for field,action in enumerate(damages):
                dest=[list(s) for s in states]
                dest[edge][field]=max(0,dest[edge][field]-1)
                assert action==packed(dest)+1
                assert 0<=mass-sum(map(sum,dest))<=1
                damage+=1
            for direction in (-1,1):
                block=prepare_readout_fiber(medium,edge,direction,lambda _: F(0))
                if not 0<=level+direction<=6 or (direction==1 and medium.stock==0):
                    assert not block;continue
                local=lambda z:14*sum(z)-12*z[0]*z[1]*z[2]+F(2,cols)*(R+z[0])**2+F(2,rows)*(C+z[1])**2
                floor=min(local(z) for z,_,_ in block)
                for z,power,delta in block:
                    assert power==2*(local(z)-floor) and delta==sum(z)-level
                    energies+=1
    temporal=population_window_algebra_checks()
    receipt=dict(scope="Uncut arbitrary-rectangle population, frozen-gradient window risk and displacement budget; numerical/interface only",
        integrated_review=core,full_optimizer_admitted=False,training_enabled=False,model_experiments=0,
        declarations=["CGB.FiberOptimizerReview.population_kernel_contract",
          "CGB.FiberOptimizerReview.population_task_window_contract","CGB.FiberOptimizerReview.population_frozen_gradient_budget"],
        reference=dict(path=str(generated.relative_to(ROOT)).replace("\\","/"),sha256=hashlib.sha256(generated.read_bytes()).hexdigest(),
            lean_exit_code=0,bodies=len(queries),shapes=sorted({(r,c) for r,c,_,_ in queries}),
            budget_and_pool_views=budget_views,complete_candidate_successors=successors,external_damage_successors=damage,
            local_energy_candidate_checks=energies,full_populations_enumerated_in_lean=False),
        frozen_budget="initial_error_i+sensitivity_i*quantum*min(2*funded_opportunities,2*total_supply)",
        window_scope="fixed task objective and frozen observed gradient; all weak/blocked coordinates, exact external-loss contribution and stochastic physical transitions remain in the bound; no risk sign assumed",
        algebra=temporal,source_sha256=hashlib.sha256((ROOT/"information_boltzmann/conjugate_gradient_port.py").read_bytes()).hexdigest(),
        medium_source_sha256=hashlib.sha256((ROOT/"information_boltzmann/core/cgb_local_medium.py").read_bytes()).hexdigest(),
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        no_optimizer_transition_change=True,review_status="self_reviewed; independent review pending",
        open=["objective sensitivity and initial observation-error calibration on actual tasks; conditional window envelope need not be negative",
              "general-size temporal resource service and functional anti-monopoly/useful selective retention",
              "practical GPU/precision budget and independent continual real-data convergence and quality"])
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(dict(reference=receipt["reference"],algebra=temporal),indent=2))


def population_window_algebra_checks():
    """Exact backward expectations, not a simulated learner or task benchmark."""
    from itertools import product
    from information_boltzmann.conjugate_gradient_port import prepare_readout_fiber
    cells=list(product(range(3),repeat=3));q=F(1,8);eta=F(1,1024)
    checks=rows_checked=positive_zero_birth=0
    reports=[]
    for supply in (0,1,3,6):
        legal=[s for s in cells if sum(s)<=supply];lookup={s:i for i,s in enumerate(legal)};n=len(legal)
        origins=[legal[0]]+([legal[-1]] if n>1 else [])
        for origin in origins:
            center=3*q;observed=q*sum(origin)-center
            force=F(768)*observed
            law=LocalLaw(1,1,2,2,(0,1,4),epsilon=0,coupling=12,pool_reference_size=2)
            P=[[F(int(i==j)) for j in range(n)] for i in range(n)]
            D=[[F(0) for _ in range(n)] for i in range(n)]
            allowance=[];loss=[(q*sum(s)-center)**2/2 for s in legal]
            for i,state in enumerate(legal):
                code=(state[0]*3+state[1])*3+state[2]
                medium=OpenMedium(LocalMedium(law,bytes([code])),ReservoirLaw(supply-sum(state),content_cost=16,chemical_base=2,stimulus=0))
                for direction in (-1,1):
                    block=prepare_readout_fiber(medium,0,direction,lambda _: force)
                    for dest,power,_ in block:
                        assert power.denominator==1
                        rate=F(1,2*len(block))*power2(-power.numerator)
                        P[i][i]-=rate;P[i][lookup[dest]]+=rate
                D[i][i]+=1-eta
                for field in range(3):
                    dest=list(state);dest[field]=max(0,dest[field]-1)
                    D[i][lookup[tuple(dest)]]+=eta/3
                assert sum(P[i])==sum(D[i])==1 and min(P[i])>=0 and min(D[i])>=0
                up=sum(P[i][j] for j,s in enumerate(legal) if sum(s)==sum(state)+1)
                down=sum(P[i][j] for j,s in enumerate(legal) if sum(s)+1==sum(state))
                error=q*abs(sum(state)-sum(origin))
                actual=q*sum(state)-center
                assert abs(actual-observed)<=error
                localA=q*observed*(up-down)+(q*error+q*q/2)*(up+down)
                assert sum(P[i][j]*(loss[j]-loss[i]) for j in range(n))<=localA
                allowance.append(localA);rows_checked+=1
            K=[[sum(D[i][u]*P[u][j] for u in range(n)) for j in range(n)] for i in range(n)]
            A=[sum(D[i][u]*(loss[u]-loss[i]+allowance[u]) for u in range(n)) for i in range(n)]
            assert all(sum(row)==1 and min(row)>=0 for row in K)
            if supply:
                assert sum(K[0][j] for j in range(n) if sum(legal[j])>0)>0
                positive_zero_birth+=1
            V,B=list(loss),[F(0)]*n
            for T in range(5):
                for i,state in enumerate(legal):
                    assert V[i]-loss[i]<=B[i]
                    checks+=1
                if T<4:
                    V=[sum(K[i][j]*V[j] for j in range(n)) for i in range(n)]
                    B=[A[i]+sum(K[i][j]*B[j] for j in range(n)) for i in range(n)]
            # Structural source support gives the error budget for every
            # reachable prefix, without preserving a shadow origin in runtime.
            reachable={lookup[origin]}
            for T in range(5):
                for i in reachable:
                    distance=abs(sum(legal[i])-sum(origin))
                    assert distance<=min(2*T,2*supply)
                    assert abs((q*sum(legal[i])-center)-observed)<=q*min(2*T,2*supply)
                    checks+=1
                reachable={j for i in reachable for j in range(n) if K[i][j]>0}
            reports.append(dict(supply=supply,legal_states=n,origin=list(origin),window_horizons=[0,1,2,3,4]))
    return dict(exact_one_coordinate_rows=rows_checked,window_and_staleness_checks=checks,
        nonabsorbing_zero_cases=positive_zero_birth,cases=reports,scope="exact finite algebra only; no optimizer training or empirical convergence")


def validate_floor_reduction(lake, out):
    """Proved floor, executed finite table and equal-choice preparation timing."""
    import random
    import statistics
    import time
    from information_boltzmann.conjugate_gradient_port import (
        _READOUT_MINIMA, _prepare_reduced_fiber, prepare_readout_fiber)

    core = check_material_review(lake)
    text = (ROOT/core["path"]).read_text(encoding="utf-8")
    start = text.index("private def reducedReadoutBlock")
    stop = text.index("private theorem reduced_geometry", start)
    definition = text[start:stop]
    source = "import Mathlib.Data.Finset.Basic\nimport Lean\nabbrev Cell := Fin 3 × Fin 3 × Fin 3\n" + definition
    source += """
private def allCells : List Cell :=
  (List.ofFn fun p : Fin 3 => (List.ofFn fun q : Fin 3 => List.ofFn fun w : Fin 3 => (p,q,w)).flatten).flatten
#eval IO.println ("REDUCED_FLOOR_CELLS=" ++ (Lean.Json.arr ((List.ofFn fun k : Fin 6 =>
  Lean.Json.arr ((allCells.filter fun s => decide (s ∈ reducedReadoutBlock k)).toArray.map fun s =>
    Lean.toJson [s.1.val,s.2.1.val,s.2.2.val])).toArray)).compress)
"""
    relative = ".lake/material_floor_table_reference.lean"
    generated = ROOT/"lean/CGBAxioms"/relative
    generated.write_text(source, encoding="utf-8", newline="\n")
    checked = check_lean_streamed(lake, relative, reuse_completed=True)
    assert checked.returncode == 0, "Exact floor-table execution failed; inspect its log"
    line, = [l for l in checked.stdout.splitlines() if l.startswith("REDUCED_FLOOR_CELLS=")]
    table = json.loads(line.split("=", 1)[1])
    for k, states in enumerate(table):
        assert set(map(tuple, states)) == set(_READOUT_MINIMA[k]+_READOUT_MINIMA[k+1])
        assert 2 <= len(states) <= 3

    law = LocalLaw(2,3,2,2,tuple(t*t for t in range(7)),epsilon=0,coupling=12,pool_reference_size=2)
    states = ((1,1,1),(1,1,2),(1,2,2),(2,1,2),(0,0,0),(2,2,2))
    word = sum(((p*3+q)*3+w) << (5*i) for i,(p,q,w) in enumerate(states))
    medium = OpenMedium(LocalMedium(law,word.to_bytes(law.state_bytes,"little")),
        ReservoirLaw(3,content_cost=16,chemical_base=2,stimulus=0))
    events = [(i%6, -1 if (i//6)%2 else 1) for i in range(1200)]
    def execute(reduced):
        rng = random.Random(593);trace=[];evaluations=0
        for edge,direction in events:
            force = lambda e: F(e*7-13,11)
            if reduced:
                state,power,sign,count = _prepare_reduced_fiber(medium,edge,direction,force,rng.randrange)
                answer = None if state is None else (state,power,sign)
            else:
                block = prepare_readout_fiber(medium,edge,direction,force)
                count = len(block);answer = block[rng.randrange(len(block))] if block else None
            trace.append(answer);evaluations += count
        return trace,evaluations,rng.getstate()
    baseline,old_count,old_rng = execute(False)
    reduced,new_count,new_rng = execute(True)
    assert baseline == reduced and old_rng == new_rng and new_count < old_count
    times = {"enumerated": [], "reduced": []}
    for repeat in range(5):
        for use_reduced in ((False,True) if repeat%2 == 0 else (True,False)):
            started=time.perf_counter();execute(use_reduced)
            times["reduced" if use_reduced else "enumerated"].append(time.perf_counter()-started)
    medians={k:statistics.median(v) for k,v in times.items()}
    tests=json.loads((ROOT/"lean/CGBAxioms/.lake/material_floor_tests.check.json").read_text(encoding="utf-8"))
    assert tests["returncode"] == 0 and tests["passed"] == 51
    for field,path in (("source_sha256",tests["source"]),
                       ("runtime_source_sha256","information_boltzmann/conjugate_gradient_port.py")):
        assert tests[field] == hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
    receipt=dict(scope="Lossless registered-H execution reduction; arithmetic/interface and frozen preparation timing only",
        integrated_review=core,full_optimizer_admitted=False,training_enabled=False,model_experiments=0,
        declaration="CGB.FiberOptimizerReview.readout_floor_reduction",
        reference=dict(path=str(generated.relative_to(ROOT)).replace("\\","/"),
            sha256=hashlib.sha256(generated.read_bytes()).hexdigest(),lean_exit_code=0,
            extracted_definition_sha256=hashlib.sha256(definition.encode()).hexdigest(),blocks=table),
        work=dict(previous_max_candidate_evaluations=13,new_max_candidate_evaluations=4,
            floor_evaluations_max=3,shared_minima_tuples=8,per_parameter_arrays_added=0,
            tariff_per_opportunity=13,tariff_changed=False,checkpoint_version=5,clock_changed=False,
            applicability="cap2,epsilon0,coupling12,quadratic numerator and pool_reference_size2; other laws retain enumeration"),
        tests=tests,predecessor_snapshot="lean/CGBAxioms/.lake/material_optimizer_pre_floor_reduction/manifest.json",
        benchmark=dict(events=len(events),repeats=5,scope="CPU exact preparation on a frozen2x3 body; no learner or empirical quality",
            same_choices_powers_and_rng=True,enumerated_evaluations=old_count,reduced_evaluations=new_count,
            seconds=times,median_seconds=medians,preparation_speedup=medians["enumerated"]/medians["reduced"]),
        source_sha256=hashlib.sha256((ROOT/"information_boltzmann/conjugate_gradient_port.py").read_bytes()).hexdigest(),
        medium_source_sha256=hashlib.sha256((ROOT/"information_boltzmann/core/cgb_local_medium.py").read_bytes()).hexdigest(),
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        update_probabilities_changed=False,review_status="self_reviewed; independent review pending",
        open=["calibrated task sensitivity/error and precision budget", "general-size temporal functional resource service and useful retention",
              "GPU implementation, whole-model resource costs and independent continual real-data quality"])
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(dict(work=receipt["work"],benchmark=receipt["benchmark"]),indent=2))


def validate_resource_service(lake, out):
    """Two-stage resource routes in the actual kernel; no learning experiment."""
    from copy import copy
    from itertools import product
    from information_boltzmann.conjugate_gradient_port import (
        _EditedBody, _prepare_reduced_fiber, prepare_readout_fiber)

    core = check_material_review(lake)
    cells = list(product(range(3), repeat=3))
    cases = one_step = two_step = exact_rows = path_checks = execution_checks = 0
    max_donors = 0
    minima = {}
    etas = (F(0), F(1,1024), F(1,2), F(1))
    for rows,cols in ((1,1),(1,2),(2,1),(2,2),(2,3),(3,2),(1,17),(17,1)):
        n = rows*cols
        if n <= 2:
            patterns = product(cells, repeat=n)
        else:
            targets = cells if n <= 6 else ((0,0,0),(1,1,1),(1,1,2),(2,2,2))
            patterns = [tuple([s]+[v]*(n-1)) for s in targets for v in ((0,0,0),(1,1,1),(2,2,2))]
            patterns += [tuple([s]+[(2,2,2) if e%2 else (0,0,0) for e in range(1,n)]) for s in targets]
        for states in patterns:
            target = 0
            level = sum(states[target])
            if level == 6:
                continue
            word = sum(((p*3+q)*3+w) << (5*i) for i,(p,q,w) in enumerate(states))
            for stock,(release_force,uptake_force) in product((0,1),((35,-23),(96,-96))):
                law=LocalLaw(rows,cols,2,2,tuple(t*t for t in range(2*max(rows,cols)+1)),
                    epsilon=0,coupling=12,pool_reference_size=2)
                body=LocalMedium(law,word.to_bytes(law.state_bytes,"little"))
                medium=OpenMedium(body,ReservoirLaw(stock,content_cost=16,chemical_base=2,stimulus=0))
                cases += 1
                def branch(owner,edge,direction,force):
                    block=prepare_readout_fiber(owner,edge,direction,lambda _: F(force))
                    usable=[(index,item) for index,item in enumerate(block)
                            if item[2] == direction and item[1] == 0]
                    assert usable
                    index,item=usable[0]
                    actual=_prepare_reduced_fiber(owner,edge,direction,lambda _: F(force),lambda _: index)
                    assert actual[:3] == item and actual[3] <= 4
                    return item[0],F(1,2*n*len(block))
                if stock:
                    z,rate=branch(medium,target,1,uptake_force)
                    execution_checks += 1
                    assert sum(z)==level+1
                    for eta in etas:
                        assert (1-eta)*rate >= (1-eta)/(26*n)
                        one_step += 1
                    continue
                assert not prepare_readout_fiber(medium,target,1,lambda _: F(uptake_force))
                donors=[i for i in range(n) if i != target and sum(states[i]) > 0]
                max_donors=max(max_donors,len(donors))
                total=F(0);middle_words=set()
                for donor in donors:
                    z,release=branch(medium,donor,-1,release_force)
                    after=copy(medium);after.body=_EditedBody(body,((donor,z),));after.stock=1
                    assert sum(z)+1 == sum(states[donor]) and after.body.get(target)==states[target]
                    middle_word=word+(((z[0]*3+z[1])*3+z[2])-((states[donor][0]*3+states[donor][1])*3+states[donor][2]))*2**(5*donor)
                    middle_words.add(middle_word)
                    y,uptake=branch(after,target,1,uptake_force)
                    assert sum(y)==level+1
                    total += release*uptake
                    path_checks += 1;execution_checks += 2
                assert len(middle_words)==len(donors)
                for eta in etas:
                    bound=len(donors)*(1-eta)**2/(676*n*n)
                    certificate=(1-eta)**2*total
                    assert certificate >= bound
                    two_step += 1
                    if bound:
                        key=f"{rows}x{cols}"
                        minima[key]=min(minima.get(key,certificate/bound),certificate/bound)
                # With integral powers, sum every donor-down/target-up path.
                # At zero stock these are exactly the physical two-step paths
                # ending with target mass=old+1. Other choices cannot create
                # the missing material. World wear adds nonnegative paths.
                if max(rows,cols) <= 2:
                    exact=F(0)
                    for donor in donors:
                        block=prepare_readout_fiber(medium,donor,-1,lambda _: F(release_force))
                        for z,power,delta in block:
                            if delta != -1:
                                continue
                            assert power.denominator==1
                            after=copy(medium);after.body=_EditedBody(body,((donor,z),));after.stock=1
                            recipient=prepare_readout_fiber(after,target,1,lambda _: F(uptake_force))
                            up=F(0)
                            for y,power2_,change in recipient:
                                if change == 1:
                                    assert power2_.denominator==1
                                    up += F(1,2*n*len(recipient)*2**power2_.numerator)
                            exact += F(1,2*n*len(block)*2**power.numerator)*up
                    assert exact >= total >= F(len(donors),676*n*n)
                    exact_rows += 1
    old=json.loads((ROOT/"results/published/optimizer_floor_reduction_20261028.json").read_text(encoding="utf-8"))
    tests=old["tests"]
    for field,path in (("source_sha256",tests["source"]),
                       ("runtime_source_sha256","information_boltzmann/conjugate_gradient_port.py")):
        assert tests[field] == hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
    receipt=dict(scope="Conditional population resource service in the same readout+wear kernel; arithmetic/interface only",
        integrated_review=core,declaration="CGB.FiberOptimizerReview.population_resource_service",
        full_optimizer_admitted=False,training_enabled=False,model_experiments=0,
        premises=dict(target="N<6 and next force<-22",stocked="initial stock>=1",
            empty_stock="zero stock; each counted donor differs from target, has material and current force>34",
            world="same uniform pre-opportunity loss kernel,0<=eta<=1; each opportunity funded",
            benefit="signed force usefulness and objective-error validity are not assumed proved"),
        derived=dict(one_opportunity="(1-eta)/(26*P)",two_opportunities="d*(1-eta)^2/(676*P^2)",
            event="target mass increases by one relative to the initial state",
            scope="d counts actual eligible donor sites; bound may be zero when none exists; not a lifetime or all-request fairness theorem"),
        numeric=dict(cases=cases,stocked_bounds=one_step,empty_stock_bounds=two_step,
            distinct_donor_routes=path_checks,exact_two_step_physical_rows=exact_rows,
            actual_reduced_executor_choices=execution_checks,max_donors=max_donors,
            minimum_witness_to_bound_ratio={k:str(v) for k,v in minima.items()}),
        source_sha256=hashlib.sha256((ROOT/"information_boltzmann/conjugate_gradient_port.py").read_bytes()).hexdigest(),
        medium_source_sha256=hashlib.sha256((ROOT/"information_boltzmann/core/cgb_local_medium.py").read_bytes()).hexdigest(),
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        no_optimizer_transition_change=True,unchanged_tests=tests,
        predecessor_snapshot="lean/CGBAxioms/.lake/material_optimizer_pre_resource_service/manifest.json",
        review_status="self_reviewed; independent review pending",
        open=["useful selective retention and task calibration; all-request temporal fairness is not claimed",
              "actual task precision/noise floor and safe efficient GPU execution",
              "independent continual real-data quality and convergence"])
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(dict(derived=receipt["derived"],numeric=receipt["numeric"]),indent=2))


def validate_gpu_matching(lake, out):
    """Actual CUDA matching, bounded-prefix work and device cost measurement."""
    from collections import Counter
    from decimal import Decimal, localcontext
    import statistics
    import time
    import torch
    from information_boltzmann.material_matching_cuda import CudaMatchingMaterial, COIN_LEVELS, _kernels
    from numba import cuda

    core = check_material_review(lake)
    # Private execution helper has a separate complete axiom block audit.
    log=(ROOT/"lean/CGBAxioms/.lake/material_combined_core.lean.log").read_text(encoding="utf-8")
    for helper in ('matching_coin_error','matching_coin_task_error'):
        match=re.search(helper+r"' depends on axioms: \[(.*?)\]",log,re.S)
        assert match and set(re.findall(r"[^\s,]+",match[1])) <= {"propext","Classical.choice","Quot.sound"}
    for size in (4,9,13):
        counts=Counter(i%size for i in range(468))
        assert len(counts)==size and set(counts.values())=={468//size}
    # Observe the ACTUAL CUDA preparation kernel's cutoffs, not a different
    # Torch exponential. Body remains frozen throughout this arithmetic audit.
    size=32
    parameter=torch.empty((size,size),device='cuda',dtype=torch.float32)
    codes=(torch.arange(size*size,device='cuda',dtype=torch.int32)%27).to(torch.uint8).reshape(size,size)
    owner=CudaMatchingMaterial(parameter,codes,stock=size*size,quantum=F(1,8),energy_unit=F(1,128))
    proposals=torch.empty(size,device='cuda',dtype=torch.int32);delta=torch.empty_like(proposals)
    diagnostics=torch.empty((size,2),device='cuda',dtype=torch.float64)
    tickets=torch.zeros((size,3),device='cuda',dtype=torch.int32)
    stream=cuda.external_stream(torch.cuda.current_stream().cuda_stream)
    def view(x):return cuda.as_cuda_array(x.detach(),sync=False)
    probability_checks=0;maximum_error=F(0);maximum_cutoff_disagreement=0
    for repeat in range(128):
        tickets[:,0]=repeat%2;tickets[:,1]=(repeat*37)%468
        gradient=(torch.arange(size*size,device='cuda',dtype=torch.float32)%31-repeat%31)/16
        _kernels()[0][1,128,stream](view(owner.codes),view(owner.row_p),view(owner.col_q),view(gradient),
            view(tickets),view(proposals),view(delta),view(diagnostics),repeat%size,size,size,16.,True)
        for power,cutoff in diagnostics.cpu().tolist():
            if power==0 and cutoff==0:continue  # blocked mass boundary
            with localcontext() as ctx:
                ctx.prec=90
                exact=Decimal(2)**(-Decimal.from_float(power))
                quantized=Decimal(int(cutoff))/Decimal(COIN_LEVELS)
                disagreement=abs(int(cutoff)-int(exact*COIN_LEVELS))
                assert disagreement<=1
                maximum_cutoff_disagreement=max(maximum_cutoff_disagreement,disagreement)
                maximum_error=max(maximum_error,F(abs(quantized-exact)))
                probability_checks+=1
    del owner,parameter,codes,proposals,delta,diagnostics,tickets,gradient
    tests=json.loads((ROOT/"lean/CGBAxioms/.lake/gpu_matching_tests.check.json").read_text(encoding="utf-8"))
    assert tests['returncode']==0 and tests['passed']>=21
    for field,path in [('source_sha256',tests['source']),('runtime_source_sha256',tests['runtime_source'])]:
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==tests[field]
    records=[]
    for size in (128,512):
        torch.cuda.empty_cache()
        parameter=torch.empty((size,size),device='cuda',dtype=torch.float32)
        gradient=torch.ones_like(parameter)
        codes=torch.full((size,size),13,device='cuda',dtype=torch.uint8)
        torch.cuda.synchronize()
        before=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
        owner=CudaMatchingMaterial(parameter,codes,stock=size*size,quantum=F(1,8),offset=F(-3,8),
                                    energy_unit=F(1,128),seed=46)
        torch.cuda.synchronize()
        initialization_peak=torch.cuda.max_memory_allocated()-before
        del codes
        owner.sweep(gradient);torch.cuda.synchronize()
        frozen=owner.state_dict()
        frozen_rows=owner.row_p.clone();frozen_cols=owner.col_q.clone()
        frozen_parameter=parameter.clone()
        def reset_sample():
            owner.codes.copy_(frozen['codes']);owner.stock.copy_(frozen['stock'])
            owner.row_p.copy_(frozen_rows);owner.col_q.copy_(frozen_cols)
            parameter.copy_(frozen_parameter);owner.generator.set_state(frozen['rng'])
            torch.cuda.synchronize()
        resident=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
        times=[];attempts=owner.attempts
        for _ in range(3):
            reset_sample()
            started=time.perf_counter();owner.sweep(gradient);torch.cuda.synchronize()
            times.append(time.perf_counter()-started)
        transient=torch.cuda.max_memory_allocated()-resident
        assert owner.attempts-attempts==3*size*size
        assert owner.fee_units==13*owner.attempts
        capture_resident=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
        reset_sample()
        owner.capture_sweep(gradient);torch.cuda.synchronize()
        graph_capture_peak=torch.cuda.max_memory_allocated()-capture_resident
        for _ in range(5):
            reset_sample();owner.graph_sweep(gradient);torch.cuda.synchronize()
        graph_resident=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
        graph_times=[]
        for _ in range(10):
            reset_sample()
            started=time.perf_counter();owner.graph_sweep(gradient);torch.cuda.synchronize()
            graph_times.append(time.perf_counter()-started)
        graph_transient=torch.cuda.max_memory_allocated()-graph_resident
        grid=owner.codes.view(size,size)
        assert torch.equal((grid//9).sum(1,dtype=torch.int32),owner.row_p)
        assert torch.equal(((grid//3)%3).sum(0,dtype=torch.int32),owner.col_q)
        assert int((grid//9+(grid//3)%3+grid%3).sum(dtype=torch.int64).item())+owner.stock.item()==owner.supply
        records.append(dict(rows=size,cols=size,parameters=size*size,colors_per_sweep=size,
            attempted_cells_per_sweep=size*size,wall_seconds=times,median_seconds=statistics.median(times),
            attempted_cells_per_second=size*size/statistics.median(times),torch_resident_bytes=resident,
            initialization_peak_additional_bytes=initialization_peak,sweep_peak_additional_bytes=transient,
            graph_wall_seconds=graph_times,graph_median_seconds=statistics.median(graph_times),
            graph_speedup=statistics.median(times)/statistics.median(graph_times),
            measurement='same frozen starting physical body and RNG per sample; resets outside timing;5 graph warmups',
            graph_capture_peak_additional_bytes=graph_capture_peak,graph_resident_bytes=graph_resident,
            graph_sweep_peak_additional_bytes=graph_transient,
            physical_bytes=owner.codes.numel(),shared_pool_bytes=(owner.row_p.numel()+owner.col_q.numel())*4,
            reusable_matching_workspace_bytes=sum(x.numel()*x.element_size() for x in owner._workspace.values()),
            stock_bytes=8,all_accounted_arrays_on_gpu=True))
        del grid,owner,parameter,gradient,frozen,frozen_rows,frozen_cols,frozen_parameter
    receipt=dict(scope="CUDA matching executor and standard multi-parameter optimizer; no capability claim",
        integrated_review=core,source=tests['runtime_source'],source_sha256=tests['runtime_source_sha256'],
        full_optimizer_admitted=False,training_enabled=False,model_experiments=0,
        structural_declaration='CGB.FiberOptimizerReview.matching_parallel_preparation',
        tests=tests,hardware=dict(name=torch.cuda.get_device_name(),total_memory=torch.cuda.get_device_properties(0).total_memory,
            torch_version=torch.__version__,cuda_version=torch.version.cuda),
        clock=dict(name='readout_matching_sweep_v1',changed_from_cpu=True,
            same_matching_serial_stock_and_pool_semantics=True,
            old_random_site_lifetime_bounds_inherited=False,
            external_wear='not automatically injected; existing wear-per-random-opportunity certificates do not transfer'),
        precision=dict(physical='all27 codes exact in uint8; byte aligned rather than5-bit packed',
            parameter_cache='FP32; exactly representable seven-level alphabet',candidate_choice='uniform word0..467 modulo4/9/13, exact cardinality',
            force_and_energy='FP64 represented gradient*force units and scaled-pool energies',
            coin='31-bit integer ticket, floor(2^31*computed_probability)',
            proved_error='coin-grid error<=2^-31+error in supplied probability; GPU exp2 error is not assumed zero',
            gpu_probability_checks=probability_checks,maximum_observed_probability_error=str(float(maximum_error)),
            maximum_actual_cuda_cutoff_disagreement=maximum_cutoff_disagreement,
            ideal_unlimited_bit_sampler_claimed=False),
        costs=dict(persistent_state='one uint8 per cell, shared row/column int32 totals, one global int64 stock and GPU RNG',
            temporary='O(min(rows,cols)) proposal/prefix/tickets plus finiteness-check workspace',
            measurement_excludes='CUDA context/compiler private memory; Torch allocator counters include only this process allocations',
            benchmark='frozen supplied gradient and complete physical sweeps; not training or speed/quality versus Adam',rows=records),
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cpu_source_sha256=hashlib.sha256((ROOT/'information_boltzmann/conjugate_gradient_port.py').read_bytes()).hexdigest(),
        predecessor_snapshot='lean/CGBAxioms/.lake/material_optimizer_pre_gpu_matching/manifest.json',
        review_status='self_reviewed; independent review pending',
        interface=dict(name='CudaMaterialOptimizer',base='torch.optim.Optimizer',
            independent_compartment_per_parameter=True,moment_or_shadow_arrays=False,
            initialization='nearest seven-level readout; uniform existing microstate within that readout',
            checkpoint='physical state,RNG,units,supply,fee and clock; caller owns pending gradient',
            lr='gradient force scale; zero retains autonomous motion'),
        open=['registered controlled real-language comparison and whole-model cost evaluation',
              'full healthy-lifetime sweep-clock behavior; old random-site lifetime bounds are not inherited',
              'registered independent continual real-language quality/resource validation'])
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(hardware=receipt['hardware'],costs=receipt['costs'],precision=receipt['precision']),indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lake", default="lake")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--floor-reduction-only", action="store_true")
    parser.add_argument("--resource-service-only", action="store_true")
    parser.add_argument("--gpu-matching-only", action="store_true")
    parser.add_argument("--continuous-wear-only", action="store_true",
                        help="Check a declared external-damage workload without rerunning the quiet certificates")
    parser.add_argument("--regeneration-only", action="store_true",
                        help="Check the derived minimal-capacity maximal clock before any runtime change")
    parser.add_argument("--resource-energy-only", action="store_true",
                        help="Check the existing compartment energy's generated ground structure")
    parser.add_argument("--with-resource-wear", action="store_true",
                        help="Also certify the same-resource sixteen-state damage subkernel")
    parser.add_argument("--resource-release-only", action="store_true",
                        help="Check the integrated release drift and source event bookkeeping")
    parser.add_argument("--sampler-only", action="store_true",
                        help="Execute Lean rational-probability references against actual sampler branches")
    parser.add_argument("--sampler-baseline-source", type=Path,
                        help="Optional archived v3 source for a paired CPU coin microbenchmark")
    parser.add_argument("--scaled-pool-only", action="store_true",
                        help="Execute all-size density-scaling energy and source references")
    parser.add_argument("--readout-fiber-only", action="store_true",
                        help="Execute adjacent-readout Lean reference and optimizer interface")
    parser.add_argument("--fiber-baseline-source", type=Path,
                        help="Optional archived native v4 port for exact continuation comparison")
    parser.add_argument("--readout-temporal-only", action="store_true",
                        help="Check same-readout-kernel wear and full-state empty-hit budgets")
    parser.add_argument("--readout-task-risk-only", action="store_true",
                        help="Check actual-risk bound, force units and analytic noise floor")
    parser.add_argument("--population-window-only", action="store_true",
                        help="Check uncut arbitrary-size population, whole window and frozen observations")
    args = parser.parse_args()
    if args.gpu_matching_only:
        validate_gpu_matching(args.lake, args.out)
        return
    if args.resource_service_only:
        validate_resource_service(args.lake, args.out)
        return
    if args.floor_reduction_only:
        validate_floor_reduction(args.lake, args.out)
        return
    if args.population_window_only:
        validate_population_window(args.lake, args.out)
        return
    if args.readout_task_risk_only:
        validate_readout_task_risk(args.lake, args.out)
        return
    if args.readout_temporal_only:
        validate_readout_temporal(args.lake, args.out)
        return
    if args.readout_fiber_only:
        validate_readout_fiber(args.lake, args.out, baseline_source=args.fiber_baseline_source)
        return
    if args.scaled_pool_only:
        validate_scaled_pools(args.lake, args.out)
        return
    if args.sampler_only:
        validate_probability_sampler(args.lake, args.out, baseline_source=args.sampler_baseline_source)
        return
    if args.resource_release_only:
        validate_resource_release(args.lake, args.out)
        return
    if args.resource_energy_only:
        validate_resource_energy(args.lake, args.out, with_wear=args.with_resource_wear)
        return
    if args.regeneration_only:
        validate_regeneration(args.lake, args.out)
        return
    if args.continuous_wear_only:
        validate_continuous_wear(args.lake, args.out)
        return
    proof = ROOT / "lean/CGBAxioms/reviews/optimizer_content_bridge.lean"
    proof_bytes = proof.read_bytes()
    declarations = review_declarations(proof_bytes.decode("utf-8").replace("\r\n", "\n"))["CGB.OptimizerContentReview"]
    kinetic_pairs = [kinetic_budget_case(a, held, release) for a in [1, 4]
                     for held in [False, True] for release in [False, True]]
    kinetic_reports = [pair[0] for pair in kinetic_pairs]
    kinetic_source = emit_kinetic_audit(proof_bytes.decode("utf-8").replace("\r\n", "\n"),
                                      [pair[1] for pair in kinetic_pairs])
    generated = ROOT / "lean/CGBAxioms/.lake/material_kinetic_audit.lean"
    generated.write_text(kinetic_source, encoding="utf-8", newline="\n")
    print("Checking eight source-derived, interval-controlled kinetic certificates in Lean...", flush=True)
    checked = subprocess.run([args.lake, "env", "lean", "-j1", ".lake/material_kinetic_audit.lean"],
                             cwd=ROOT / "lean/CGBAxioms", capture_output=True, text=True, encoding="utf-8")
    if checked.returncode or "sorryAx" in checked.stdout + checked.stderr:
        raise RuntimeError(checked.stdout + checked.stderr)
    results = audit_review(checked.stdout, proof_bytes.decode("utf-8").replace("\r\n", "\n"),
        {f"KineticAudit.Case{i}.certificate" for i in range(len(kinetic_pairs))})
    names = results["CGB.OptimizerContentReview"]
    audited = {name for name in names if not name.startswith("KineticAudit.")}
    kinetic_audited = set(names) - audited
    assert audited == declarations, (audited, declarations)
    assert kinetic_audited == {f"KineticAudit.Case{i}.certificate" for i in range(len(kinetic_pairs))}
    assert proof.read_bytes() == proof_bytes, "proof changed during validation"
    cases = [analyze(a, 3 * a, 0, alternate) for a in [1, 2, 3, 4]
             for alternate in [False, True]]
    assert all(case["endpoints_equal_energy"] for case in cases)
    assert all(F(case["stationary_on_basin_probability"]) < F(1, 2) for case in cases)
    gradient_cases = [analyze(1, 3, 0, False, force_code=g) for g in [-5, 0, 5]]
    assert (F(gradient_cases[0]["stationary_on_basin_probability"])
            > F(gradient_cases[1]["stationary_on_basin_probability"])
            > F(gradient_cases[2]["stationary_on_basin_probability"]))
    for field in ["on_before_off_probabilities", "worst_expected_attempts_to_either_code"]:
        assert gradient_cases[1][field] == cases[0][field], field
    port = ROOT / "information_boltzmann/conjugate_gradient_port.py"
    tests = ROOT / "tests/test_conjugate_gradient_port.py"
    port_bytes, test_bytes = port.read_bytes(), tests.read_bytes()
    interface = subprocess.run([sys.executable, "-m", "pytest", str(tests), "-q"],
                               cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    assert interface.returncode == 0, interface.stdout + interface.stderr
    passed = re.search(r"(\d+) passed", interface.stdout)
    assert passed is not None, interface.stdout
    assert port.read_bytes() == port_bytes and tests.read_bytes() == test_bytes
    alphabets = []
    for cap in [1, 3, 5, 15]:
        law = LocalLaw(1, 1, cap, cap, (0,) * (cap + 1))
        alphabets.append(dict(cap=cap, physical_states=law.states, readout_levels=3 * cap + 1,
                              stored_bits_per_cell=law.bits, certified_endpoint_codes=2))
    receipt = dict(
        scope="exact kernel algebra and CPU optimizer interface; no model training, capability task or stochastic retention trial",
        full_optimizer_admitted=False, training_enabled=False,
        proof_path=str(proof.relative_to(ROOT)).replace("\\", "/"),
        proof_sha256=hashlib.sha256(proof_bytes).hexdigest(), lean_exit_code=checked.returncode,
        named_lemmas_audited=sorted(audited),
        kinetic_certificates=dict(instances=len(kinetic_pairs),
            generated_source_sha256=hashlib.sha256(kinetic_source.encode("utf-8")).hexdigest(),
            generated_source_path="lean/CGBAxioms/.lake/material_kinetic_audit.lean",
            scope="Actual fixed-total one-cell protocol; all horizons and adaptive forces within registered intervals. Automatic numeric witnesses, no new optimizer rule or training.",
            audited=sorted(kinetic_audited), profiles=kinetic_reports),
        lean_scope="energy barriers, repair support, gradient/content conjugacy, real-valued paired force port, certified fractional-rate bounds and dissipative learning response",
        absorption_probability_scope="exact rational linear-system verification, not yet a Lean first-hit theorem",
        source_sha256=hashlib.sha256((ROOT / "information_boltzmann/core/cgb_local_medium.py").read_bytes()).hexdigest(),
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        gradient_port_sha256=hashlib.sha256(port_bytes).hexdigest(),
        gradient_port_scope="Source-clock algebra is a historical reference; the current CPU MaterialOptimizer uses the maximal reverse-pair clock. No real-task or GPU admission.",
        runtime_interface=dict(
            optimizer_class="information_boltzmann.conjugate_gradient_port.MaterialOptimizer",
            test_source_sha256=hashlib.sha256(test_bytes).hexdigest(), tests_passed=int(passed[1]),
            device="cpu", dtypes=["float32", "float64"],
            initialization="medium owns weights; floating tensors are reconstructed readout caches",
            cached_alphabet="finite exactly representable normal-or-zero values, exhaustively checked at construction",
            lr_semantics="common conjugate-force unit across coupled cells; local response depends on physical state/rate; zero lr does not stop physical events",
            checkpoint="physical weights, material stock, RNG and complete cost ledger; future continuation tested",
            random_work="proposal and acceptance bits counted separately from PRNG word/CPU cost; fractional work records coefficient_steps and maximum rational denominator bit length",
            numerical_revision=dict(checkpoint_version=4, kinetic_clock="maximal_reverse_pair_v1", sampler="rational_geometric_mixture_v1", nonzero_forces_not_rounded_away=True,
                                    rare_outcome_beyond_200_bits_checked=True,
                                    probability_certificates="exact rational binomial mixture and executable rational finite-prefix semantics; roots are historical only",
                                    full_probability_runtime_refinement="ideal fair-bit identity, a.s. stopping and expected work proved; bounded executable conformance checked in the separate sampler receipt; PRNG independence is assumed"),
            physical_alphabets=alphabets,
            direct_cycle_proof_transfer=False, real_data_training=False,
            world_funding_contract="optional existing PrefixReplayWorld.pay_write callback; refused funding stops before random draws/material changes; caller owns world checkpoint and continuously advancing publication/maintenance"),
        review_status="self_reviewed", cases=cases, gradient_cases=gradient_cases,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in receipt.items() if k not in {"cases", "gradient_cases", "kinetic_certificates"}}, indent=2))
    for r in kinetic_reports:
        h = r["horizons"][1 if r["mode"] == "release" else 3]
        print(json.dumps(dict(a=r["a"], pole=r["held_pole"], mode=r["mode"],
                              mean_attempts=float(F(r["mean_nominal_paid_attempts"])),
                              budget=h["funded_attempts"],
                              sign_change_bound=float(F(h["first_sign_change_probability_lower"]
                                  if r["mode"] == "release" else h["first_sign_change_probability_upper"])))))
    for case in cases + gradient_cases:
        print(json.dumps({k: case[k] for k in ["a", "acceptance", "worst_off_defect_repair_probability",
                         "force_code", "worst_on_defect_repair_probability", "worst_expected_attempts_to_either_code"]}))


if __name__ == "__main__":
    main()
