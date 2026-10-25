import CGBAxioms.SensoryMaintenanceCarrier
import CGBAxioms.PortSeparation
import CGBAxioms.SensoryMaintenanceTwoTickGlobal

/-! Algorithm verification only: export an executable integer representation of
the EXISTING GraceRecovery world. No production medium or rule is added. -/
open Lean CGB CGB.RenewableLearning

namespace UnifiedReferenceReview20261025

private def cells : List Cell :=
  [.vacuum, .search, .aSpent, .bSpent, .aReady, .bReady, .aHeld, .bHeld]

private def worlds : List Env :=
  [.feedA, .feedB, .missA, .missB, .probeA, .probeB, .answerA, .answerB, .trap]

private def stimuli : List SensoryMaintenance.Stim :=
  [none, some .calm, some .answerA, some .answerB,
    some .feed, some .scarce, some .miss, some .probe]

private def worldMass (q : Cell) (w v : Env) : Nat :=
  if q = .vacuum then (if v = .trap then 2 else 0) else
  match w with
  | .feedA => if policy q = 2 then (if v = .answerA then 2 else 0)
      else if policy q = 0 then (if v = .feedA ∨ v = .feedB then 1 else 0)
      else if v = .missA then 2 else 0
  | .feedB => if policy q = 2 then (if v = .answerB then 2 else 0)
      else if policy q = 1 then (if v = .feedA ∨ v = .feedB then 1 else 0)
      else if v = .missB then 2 else 0
  | .missA => if v = .probeA then 2 else 0
  | .missB => if v = .probeB then 2 else 0
  | .probeA => if v = (if policy q = 2 then .answerA else .missA) then 2 else 0
  | .probeB => if v = (if policy q = 2 then .answerB else .missB) then 2 else 0
  | .answerA => if v = .feedA then 2 else 0
  | .answerB => if v = .feedB then 2 else 0
  | .trap => if v = .trap then 2 else 0

private theorem policy_table (q : Cell) : policy q = match q with
    | .aSpent | .aReady | .aHeld => 0
    | .bSpent | .bReady | .bHeld => 1
    | _ => 2 := by cases q <;> decide

private theorem worldMass_correct (q : Cell) (w v : Env) :
    (worldMass q w v : Real) = 2 * GraceRecovery.moves w (fun _ => q) v := by
  cases q <;> cases w <;> cases v <;>
    norm_num [worldMass, GraceRecovery.moves, RenewableLearning.moves, policy_table]

-- The actual medium, clock and transport statements all refer to ONE body.
example : SensoryMaintenance.body.Axioms ∧ SensoryMaintenance.body.EventClocked 1 :=
  SensoryMaintenance.admission

-- Simultaneous claims about exactly the same body, world, ledger and clock.
example :
    (∀ n, Exchange.survival SensoryMaintenance.body SensoryMaintenance.world
      SensoryMaintenance.ec SensoryMaintenance.material false
      (fun _ => SensoryMaintenance.world.sense) n = 1) ∧
    (∀ start n, Exchange.survival SensoryMaintenance.body SensoryMaintenance.world
      SensoryMaintenance.ec SensoryMaintenance.material false
      (SensoryMaintenance.sensoryProtocol start 1) n = 1) ∧
    (∀ n, 7 ≤ n → Exchange.survival SensoryMaintenance.body SensoryMaintenance.world
      SensoryMaintenance.ec SensoryMaintenance.material true
      (fun _ => SensoryMaintenance.world.sense) n ≤ 3 / 4) ∧
    ¬ SensoryMaintenanceRecovery.ToleratesSensoryLoss 0 2 :=
  ⟨SensoryMaintenanceRecovery.natural_survival,
    SensoryMaintenanceRecovery.one_tick_survival,
    SensoryMaintenanceClosure.cut_survival,
    SensoryMaintenanceTwoTickGlobal.not_two_tick_tolerance⟩

example (cut : Bool) (sched : Nat → Env → Unit → SensoryMaintenance.Stim) :=
  SensoryMaintenanceCarrier.realized cut sched
example (cut : Bool) (sched : Nat → Env → Unit → SensoryMaintenance.Stim)
    (t k : Nat) (r : SensoryMaintenanceCarrier.Carrier cut sched) :=
  SensoryMaintenanceCarrier.recovery_preserved cut sched t k r
example (cut : Bool) (sched : Nat → Env → Unit → SensoryMaintenance.Stim) (n : Nat) :=
  SensoryMaintenanceCarrier.source_energy_budget cut sched n

-- This is only local equality, not an unrestricted global port-refinement claim.
example (s : SensoryMaintenance.Stim) (x : SensoryMaintenance.State) :=
  PortSeparation.legacy_rules s x

#print axioms worldMass_correct
#print axioms CGB.SensoryMaintenanceCarrier.realized
#print axioms CGB.SensoryMaintenanceCarrier.recovery_preserved

#eval IO.println (Json.mkObj [
  ("format", toJson "CGB.UnifiedReference.world.v1"),
  ("denominator", toJson (2 : Nat)),
  ("rows", toJson (cells.map fun q => worlds.map fun w => worlds.map fun v =>
    worldMass q w v)),
  ("carrier_parts", toJson (cells.flatMap fun q => (List.finRange 3).map fun r =>
    stimuli.map fun s => [false, true].map fun b =>
      [SensoryMaintenanceCarrier.partContent b (q,r),
       SensoryMaintenanceCarrier.partIn b s (q,r),
       SensoryMaintenanceCarrier.partOut b s (q,r)]))]).compress

end UnifiedReferenceReview20261025
