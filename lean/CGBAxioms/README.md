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

Foundation.lean checks finite-state specialization, clocks, probability bounds and the
distinction between independence, universal guarantees and existence. It adds no physical
axiom. The audit does not establish physical validity, minimality, or a life existence theorem.

Scope and findings:
- ../../docs/information_boltzmann/CGB_AXIOMS_V0_20261022.md
- ../../docs/information_boltzmann/CGB_FOUNDATION_AUDIT_20261023.md
