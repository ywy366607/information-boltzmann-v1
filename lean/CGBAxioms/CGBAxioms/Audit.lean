import CGBAxioms.Models
import CGBAxioms.Life

/-! Every theorem may depend only on Lean's foundational axioms
(`propext`, `Classical.choice`, `Quot.sound`); CGB clauses appear as hypotheses.
`scripts/audit.sh` checks this output. -/

-- jump-process layer: Noether, energy, the one-way 3-cycle
#print axioms CGB.JumpProcess.expect_const_iff
#print axioms CGB.JumpProcess.noether
#print axioms CGB.JumpProcess.energy_of_reversible
#print axioms CGB.JumpProcess.ring3_not_reversible
-- the medium language: finiteness, conservation shells, energy, reversibility
#print axioms CGB.Medium.finite_reachable
#print axioms CGB.Medium.finite_states
#print axioms CGB.Medium.total_of_reachable
#print axioms CGB.Medium.energy_of_p2b
#print axioms CGB.Medium.p2b_of_micro
-- consistency: closed, and open/driven/growing
#print axioms CGB.Models.twinFlip_axioms
#print axioms CGB.Models.twinFlip_swap_isAut
#print axioms CGB.Models.seed_axioms
#print axioms CGB.Models.seed_swap_isAut
#print axioms CGB.Models.seed_driven
#print axioms CGB.Models.seed_grows
-- a rock satisfies all six clauses and never moves: decay is not yet a theorem
#print axioms CGB.Models.rock_axioms
#print axioms CGB.Models.rock_frozen
-- independence: one witness per clause (per part for P1a and P3)
#print axioms CGB.Models.twinTape_witness
#print axioms CGB.Models.twinBag_witness
#print axioms CGB.Models.twinBlank_witness
#print axioms CGB.Models.farFlip_witness
#print axioms CGB.Models.twinToggle_witness
#print axioms CGB.Models.twinCycle_witness
#print axioms CGB.Models.triLookup_witness
#print axioms CGB.Models.pairPreset_witness
#print axioms CGB.Models.innerClock_witness
-- the genome bottleneck: P2c makes parameters depend only on identity pages
#print axioms CGB.Models.xhop_pages
-- life statement: the semantics is a probability law
#print axioms CGB.Medium.kernel_nonneg
#print axioms CGB.Medium.kernel_sum
#print axioms CGB.Medium.law_sum
#print axioms CGB.Medium.lawCut_sum
-- self-maintenance through the loop: exposure and metabolism are necessary
#print axioms CGB.Medium.not_selfMaintains_of_unexposed
#print axioms CGB.Medium.not_selfMaintains_of_closedMatter
#print axioms CGB.Medium.Lives.exposed
#print axioms CGB.Medium.Lives.metabolizes
