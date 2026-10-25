import CGBAxioms.ActiveMaintenanceCarrier

/- Initial and energy/material binding checks; kept outside the production
   declaration inventory. Run: lake env lean reviews/active_maintenance_20261024.lean -/

open CGB CGB.ActiveMaintenance CGB.ActiveMaintenanceEnergy
open CGB.ActiveMaintenanceCarrier CGB.MaterialSource

example (s : Unit → Stim) (e : Unit) : body.rate e s (fun _ => 0) = 0 := by
  simp [body]

example : project ((initial, 0) : R) = initial ∧
    count (owner ((initial, 0) : R)) .body = 1 ∧
    count (owner ((initial, 0) : R)) .source = 1 ∧
    count (owner ((initial, 0) : R)) .sink = 1 := by
  constructor
  · rfl
  · change count (owners 1 0) .body = 1 ∧ count (owners 1 0) .source = 1 ∧
      count (owners 1 0) .sink = 1
    decide

example (cut : Bool) (t : ℕ) (r r' : R) (e : Option Unit)
    (hp : 0 < (physical cut).step t r e r') :
    port incoming t (project r) e (project r') =
        (moved (owner r) (owner r') .source .body : ℝ) ∧
    port outgoing t (project r) e (project r') =
        (moved (owner r) (owner r') .body .sink : ℝ) := by
  have ht := (realized cut).transport t r e r' hp
  constructor
  · rw [ht.2.1]
    cases e <;> simp [port, declared]
  · rw [ht.2.2]
    cases e <;> simp [port, declared]
