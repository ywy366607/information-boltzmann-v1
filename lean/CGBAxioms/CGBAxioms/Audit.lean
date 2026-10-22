import CGBAxioms.Symmetry

/-! Every theorem may depend only on Lean's foundational axioms
(`propext`, `Classical.choice`, `Quot.sound`); CGB clauses appear as hypotheses.
`scripts/audit.sh` checks this output. -/

#print axioms CGB.JumpProcess.expect_const_iff
#print axioms CGB.JumpProcess.noether
#print axioms CGB.JumpProcess.energy_of_reversible
#print axioms CGB.JumpProcess.ring3_not_reversible
