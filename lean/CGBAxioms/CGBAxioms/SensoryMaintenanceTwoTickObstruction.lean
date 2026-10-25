import CGBAxioms.SensoryMaintenanceBoundaries

namespace CGB.SensoryMaintenanceTwoTickObstruction

open RenewableLearning SensoryMaintenance

private theorem policy_table (q : Cell) : policy q = match q with
    | .aSpent | .aReady | .aHeld => 0
    | .bSpent | .bReady | .bHeld => 1
    | _ => 2 := by cases q <;> decide

/-- A concrete world branch that remains physically possible through two lost
ticks while the projected cell is `aSpent`. -/
theorem two_tick_world_branch :
    GraceRecovery.moves .feedA (fun _ => .aSpent) .feedB = (1 : ℝ) / 2 ∧
    GraceRecovery.moves .feedB (fun _ => .aSpent) .missB = 1 := by
  norm_num [GraceRecovery.moves, RenewableLearning.moves, policy_table]

/-- The same branch drains the two-unit reserve and the next sensed miss sends
the cell to vacuum. This is a local obstruction to lifting the one-tick
certificate unchanged to a two-tick first-hit certificate. -/
theorem two_tick_local_failure :
    SensoryMaintenance.next (some (.miss : Signal))
        (SensoryMaintenance.burn (.aSpent, 2) 2) = SensoryMaintenance.empty := by
  norm_num [SensoryMaintenance.burn, SensoryMaintenance.next,
    SensoryMaintenance.empty, RenewableLearning.next]

end CGB.SensoryMaintenanceTwoTickObstruction
