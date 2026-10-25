import CGBAxioms.PortSeparation

open CGB CGB.PortSeparation
open CGB.RenewableLearning (Cell Signal Env)

namespace PortSeparationReview20261024

example : body.Axioms ∧ body.EventClocked 1 := admission

example (u : Input) (x : State) := local_budgets u x

example (s : SensoryMaintenance.Stim) (x : State) :
    rank x ≤ 3 ∧ (rank x = 0 ↔ x = empty) := by
  exact ⟨(starvation_progress s x).1, (starvation_progress s x).2.1⟩

example (s : SensoryMaintenance.Stim) (x : State) (hx : x ≠ empty) :
    rank (next (s,false) x) < rank x ∧ 0 < outgoing (s,false) x :=
  by exact ⟨(starvation_progress s x).2.2.1 hx, (starvation_progress s x).2.2.2 hx⟩

example (u : Input) : next u empty = empty := next_empty u

example : incoming (none,true) (.search,2) = 1 ∧
    outgoing (none,true) (.search,2) = 1 ∧ next (none,true) (.search,2) = (.search,2) := by decide

example : incoming (none,false) (.search,2) = 0 ∧
    rank (next (none,false) (.search,2)) < rank (.search,2) := by decide

end PortSeparationReview20261024
