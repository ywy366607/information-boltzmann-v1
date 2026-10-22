#!/usr/bin/env bash
# CGB axiom audit: no `sorry`, no Lean `axiom`, and every printed theorem depends
# only on Lean's three foundational axioms.
set -euo pipefail
cd "$(dirname "$0")/.."
bad=$(grep -rnE '^\s*(axiom|sorry)\b|\bsorry\b|\badmit\b' CGBAxioms --include=*.lean || true)
if [ -n "$bad" ]; then echo "FORBIDDEN:"; echo "$bad"; exit 1; fi
out=$(lake env lean CGBAxioms/Audit.lean 2>&1)
echo "$out"
extra=$(echo "$out" | grep -oE "depends on axioms: \[[^]]*\]" | tr -d '[]' | sed 's/depends on axioms: //' | tr ',' '\n' | sed 's/ //g' | grep -vE '^(propext|Classical.choice|Quot.sound)$' | grep -v '^$' || true)
if [ -n "$extra" ]; then echo "NON-FOUNDATIONAL AXIOMS: $extra"; exit 1; fi
echo "AUDIT OK"
