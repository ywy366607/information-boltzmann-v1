# CGBAxioms

Lean formalization of candidate CGB constraints, models, life targets, and foundation
audits. Pinned Lean/mathlib versions are in lean-toolchain and lakefile.toml.

From this directory:

```text
lake build
python scripts/audit.py
python scripts/audit.py --self-test
```

The full audit builds current sources and checks imported project definitions/theorems.
It writes .lake/audit_inventory.json with signatures, dependencies and source hashes,
and rejects missing modules, incomplete output and non-foundational axioms. Source and
compiler-generated declarations are counted separately; these are not brain mechanism
counts. scripts/audit.sh is a wrapper; Audit.lean is the old manual subset.

The auditor resolves Lake's library path once and launches Lean directly, avoiding a
large resident Lake parent. Records stream to stdout instead of accumulating in Lean's
message log; dependency names use sets. All coverage/trust checks remain unchanged.
For low-memory machines, set `LEAN_NUM_THREADS=1` in the audit process.

Foundation.lean checks finite-state specialization, clocks, probability bounds and the
distinction between independence, universal guarantees and existence. It adds no physical
axiom. The audit does not establish physical validity, minimality, or a life existence theorem.

RuleEncoding.lean provides a complete finite rational rulebook, fixed interpreter,
roundtrip/injectivity and total-instance bit accounting. WorldRecovery.lean derives
actual first-hit probabilities from the existing joint dynamics, with time-dependent
and loop-cut versions. OperationalLife.lean is the versioned partial target using
both interfaces; Life.lean keeps the historical favourable-stimulus-menu target.
RecoveryControls.lean proves a semantic accessibility counterexample. Neither target
is an established life existence theorem; material intake is not yet energy intake.

Interaction.lean gives finite probabilistic interaction coalgebras, common-output
couplings, coinductive soundness for all finite observation cylinders, composable
state refinement, event first-hit preservation and a separate world-clock certificate.
InteractionBridge.lean reuses the existing normal/cut kernels and proves that
first-hit observation is exactly the prior actual-recovery semantics. Semantic
controls are in InteractionControls.lean; this adds no physical axiom or generator.

ConstraintAudit.lean factors the six compound clauses into 14 equivalent audit
conditions, proves finite-state redundancy and structural vacuity regimes, joint
kernel consequences, conductance parametrization and rate-scale freedom.
ConstraintFreedom.lean admits arbitrary finite driven tables/processes in an
exposed-site representation. These are model-class results, not life or encoded
budget witnesses, and not a certificate that all 14 conditions are independent.

Scope and findings:
- ../../docs/information_boltzmann/CGB_AXIOMS_V0_20261022.md
- ../../docs/information_boltzmann/CGB_FOUNDATION_AUDIT_20261023.md
- ../../docs/information_boltzmann/CGB_OPERATIONAL_FOUNDATIONS_20261023.md
- ../../docs/information_boltzmann/CGB_COALGEBRA_SEMANTICS_20261023.md
- ../../docs/information_boltzmann/CGB_MINIMALITY_GENERATIVITY_AUDIT_20261023.md
