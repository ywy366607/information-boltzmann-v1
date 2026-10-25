import CGBAxioms.SensoryMaintenanceCarrier
import CGBAxioms.PortSeparation
import CGBAxioms.SensoryMaintenanceTwoTickGlobal
import CGBAxioms.CausalLearning

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

-- Full-support byte models cannot learn by Boolean exclusion alone.
private theorem full_support_frozen {H : Type*} [Fintype H] [DecidableEq H]
    {B : Nat} (m : CausalLearning.Memory H B) (o : Fin 256) :
    CausalLearning.advance (fun (_ : H) (_ : Unit) (_ : Unit) (_ : Fin 256) => true)
      m () () o = m ∧
    CausalLearning.fee (fun (_ : H) (_ : Unit) (_ : Unit) (_ : Fin 256) => true)
      m () () o = 0 :=
  CausalLearning.agreement_no_write _ m () () o (by intros; rfl)

-- Algorithm reference only: Hamilton projection with stable index ties.
private def projectTickets (counts likelihood : List Nat) : List Nat := Id.run do
  let tickets := counts.sum
  let masses := counts.zipWith (· * ·) likelihood
  let total := masses.sum
  let floors := masses.map (fun m => tickets * m / total)
  let order := (List.range counts.length).mergeSort (fun a b =>
    let ra := tickets * masses[a]! % total
    let rb := tickets * masses[b]! % total
    ra > rb || (ra == rb && a ≤ b))
  let selected := order.take (tickets - floors.sum)
  return (List.range counts.length).map (fun h => floors[h]! + if h ∈ selected then 1 else 0)

private def ticketVectors : List (List Nat × List Nat) :=
  [([1,1], [1,3]), ([3,4,8], [7,2,5]), ([0,127,128], [99,1,1]),
   ([1,1,1,1,1,1,1], [1,2,3,4,5,6,7]), ([255], [32767])]

example : 255 * 255 * 32767 < (2 : Nat)^31 := by decide

#print axioms full_support_frozen
#print axioms CGB.Interaction.Machine.conditioned_next

-- A pairwise integer acceptance rule has the required weighted reverse flow.
private theorem boundary_accept_balance {a b : Real} (ha : 0 < a) (hb : 0 < b) :
    a * min 1 (b / a) = b * min 1 (a / b) := by
  rw [mul_min_of_nonneg _ _ ha.le, mul_min_of_nonneg _ _ hb.le]
  simp only [mul_one, mul_div_cancel₀ _ ha.ne', mul_div_cancel₀ _ hb.ne']
  exact min_comm _ _

private theorem boundary_accept_minorization {a b m : Real}
    (ha : 0 < a) (hb : 0 < b) (ham : a ≤ m) (hbm : b ≤ m) :
    b / m ≤ min 1 (b / a) := by
  have hm : 0 < m := lt_of_lt_of_le ha ham
  apply le_min
  · exact (div_le_one hm).mpr hbm
  · exact (div_le_div_iff₀ hm ha).mpr (mul_le_mul_of_nonneg_left ham hb.le)

-- Independent binary charges: normalized mean is NOT the one-particle law.
example : ((1 / 2 : Rat) / (1 / 2 + 3 / 4),
    (3 / 4 : Rat) / (1 / 2 + 3 / 4)) = (2 / 5, 3 / 5) := by norm_num
example : (2 / 5 : Rat) ≠ 1 / (1 + 3) := by norm_num

private def boundaryMass (weights : List Nat) (i j : Nat) : Nat :=
  if i = j then weights.length * weights[i]! -
    ((List.range weights.length).filter (· ≠ i)).foldl
      (fun acc k => acc + min weights[i]! weights[k]!) 0
  else min weights[i]! weights[j]!

private def boundaryVectors : List (List Nat) := [[1,3], [2,5,7], [1,1,1], [65535,1]]

#print axioms boundary_accept_balance
#print axioms boundary_accept_minorization

-- G8's crossed channels, with the local unit factor split as r*r.
private theorem local_cross_balance {r p q : Real}
    (hr : 0 < r) (hp : 0 < p) (hq : 0 < q) :
    r * p + r * q = (r * r * p * q) * (1 / r / p + 1 / r / q) := by
  field_simp [hr.ne', hp.ne', hq.ne']
  ring

example (p q w : Nat) (hp : 0 < p) : (p - 1) + (q + 1) + w = p + q + w := by omega
example (p q w : Nat) (hq : 0 < q) : p + (q - 1) + (w + 1) = p + q + w := by omega
#print axioms local_cross_balance

-- Consuming one charge to form a slot changes both factors of w*p*q.
private theorem local_plastic_delta (ε J p q w : Real) :
    (ε * (w + 1) - J * (w + 1) * p * (q - 1)) - (ε * w - J * w * p * q) =
      ε - J * p * (q - w - 1) := by ring
#print axioms local_plastic_delta
#eval IO.println (Json.mkObj [
  ("format", toJson "CGB.UnifiedReference.boundary.v1"),
  ("vectors", toJson (boundaryVectors.map fun w => Json.mkObj [
    ("weights", toJson w),
    ("denominators", toJson (w.map (w.length * ·))),
    ("rows", toJson ((List.range w.length).map fun i =>
      (List.range w.length).map fun j => boundaryMass w i j))]))]).compress

#eval IO.println (Json.mkObj [
  ("format", toJson "CGB.UnifiedReference.tickets.v1"),
  ("vectors", toJson (ticketVectors.map fun (counts, likelihood) =>
    [counts, likelihood, projectTickets counts likelihood]))]).compress

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
