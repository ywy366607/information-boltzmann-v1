import CGBAxioms.SensoryMaintenanceTwoTickGlobal

/-!
# Audit of state-only repairs after the two-tick refutation

The existing material ports make the `aSpent`/`miss` failure structurally
forced.  A repair that only changes the successor state, while keeping zero
incoming material, the registered outgoing port and the local balance law,
cannot retain a nonempty state.
-/

namespace CGB.SensoryMaintenanceRepairAudit

open RenewableLearning

private theorem cost_zero_iff_empty (x : SensoryMaintenance.State) :
    SensoryMaintenance.cost x = 0 ↔ x = SensoryMaintenance.empty := by
  rcases x with ⟨q, r⟩
  fin_cases r <;> cases q <;>
    norm_num [SensoryMaintenance.cost, SensoryMaintenance.empty,
      RenewableLearning.cost, RenewableLearning.buffer, RenewableLearning.credit]

theorem spent_miss_budget_forces_empty (y : SensoryMaintenance.State)
    (hbal : SensoryMaintenance.cost y + SensoryMaintenance.outgoing
        (some .miss) (.aSpent, 0) =
      SensoryMaintenance.cost (.aSpent, 0) + SensoryMaintenance.incoming
        (some .miss) (.aSpent, 0)) :
    y = SensoryMaintenance.empty := by
  have ho : SensoryMaintenance.outgoing (some .miss) (.aSpent, 0) = 1 := by
    norm_num [SensoryMaintenance.outgoing, RenewableLearning.cost,
      RenewableLearning.buffer, RenewableLearning.credit,
      RenewableLearning.outgoing, RenewableLearning.next]
  have hi : SensoryMaintenance.incoming (some .miss) (.aSpent, 0) = 0 := by
    norm_num [SensoryMaintenance.incoming, RenewableLearning.incoming,
      RenewableLearning.next]
  have hx : SensoryMaintenance.cost (.aSpent, 0) = 1 := by
    norm_num [SensoryMaintenance.cost, RenewableLearning.cost,
      RenewableLearning.buffer, RenewableLearning.credit]
  rw [ho, hi, hx] at hbal
  exact cost_zero_iff_empty y |>.mp (by omega)

theorem no_state_only_two_tick_repair
    (repair : SensoryMaintenance.Stim → SensoryMaintenance.State → SensoryMaintenance.State)
    (hbalance : ∀ s x, SensoryMaintenance.cost (repair s x) +
      SensoryMaintenance.outgoing s x = SensoryMaintenance.cost x +
      SensoryMaintenance.incoming s x)
    (hnonempty : repair (some .miss) (.aSpent, 0) ≠ SensoryMaintenance.empty) : False := by
  apply hnonempty
  apply spent_miss_budget_forces_empty
  exact hbalance (some .miss) (.aSpent, 0)

end CGB.SensoryMaintenanceRepairAudit
