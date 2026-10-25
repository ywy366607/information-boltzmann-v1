import CGBAxioms.SensoryMaintenanceRepairAudit

open CGB

namespace SensoryRepairAuditReview20261025

example (y : SensoryMaintenance.State)
    (h : SensoryMaintenance.cost y + SensoryMaintenance.outgoing
        (some .miss) (.aSpent, 0) =
      SensoryMaintenance.cost (.aSpent, 0) + SensoryMaintenance.incoming
        (some .miss) (.aSpent, 0)) :
    y = SensoryMaintenance.empty :=
  SensoryMaintenanceRepairAudit.spent_miss_budget_forces_empty y h

example (repair : SensoryMaintenance.Stim → SensoryMaintenance.State → SensoryMaintenance.State)
    (hb : ∀ s x, SensoryMaintenance.cost (repair s x) +
      SensoryMaintenance.outgoing s x = SensoryMaintenance.cost x +
      SensoryMaintenance.incoming s x)
    (hn : repair (some .miss) (.aSpent, 0) ≠ SensoryMaintenance.empty) : False :=
  SensoryMaintenanceRepairAudit.no_state_only_two_tick_repair repair hb hn

end SensoryRepairAuditReview20261025
