import CGBAxioms.SensoryMaintenanceRepairCandidates

namespace CGB.SensoryMaintenanceProtocolRepairRefutation

open RenewableLearning

private theorem policy_table (q : Cell) : policy q = match q with
    | .aSpent | .aReady | .aHeld => 0
    | .bSpent | .bReady | .bHeld => 1
    | _ => 2 := by cases q <;> decide

private noncomputable def path (n : ℕ) : SensoryMaintenance.S :=
  match n with
  | 0 => SensoryMaintenance.initial
  | 1 => (fun _ => (.search, 2), .answerA)
  | 2 => (fun _ => (.aSpent, 2), .feedA)
  | 3 => (fun _ => (.aSpent, 1), .feedB)
  | 4 => (fun _ => (.aSpent, 2), .missB)
  | 5 => (fun _ => SensoryMaintenance.empty, .probeB)
  | _ => SensoryMaintenance.initial

private def event (_ : ℕ) : Option Unit := some ()

private noncomputable abbrev K := fun t =>
  SensoryMaintenance.body.jointEventKernel SensoryMaintenance.world
    SensoryMaintenance.ec false
    (SensoryMaintenanceRepairCandidates.protectedTwoTickProtocol 2) t

private theorem path_step (t : ℕ) (ht : t < 5) :
    0 < (K t).step (path t) (event t) (path (t + 1)) := by
  have ht' : t = 0 ∨ t = 1 ∨ t = 2 ∨ t = 3 ∨ t = 4 := by omega
  rcases ht' with rfl | rfl | rfl | rfl | rfl <;>
    norm_num [K, path, event,
      SensoryMaintenanceRepairCandidates.protectedTwoTickProtocol,
      Medium.jointEventKernel, Medium.eventWorldStep, Medium.eventStep,
      SensoryMaintenance.body, SensoryMaintenance.world,
      SensoryMaintenance.next, SensoryMaintenance.incoming,
      SensoryMaintenance.outgoing, SensoryMaintenance.initial,
      SensoryMaintenance.empty, RenewableLearning.next,
      RenewableLearning.incoming, RenewableLearning.outgoing,
      RenewableLearning.sense, GraceRecovery.moves, RenewableLearning.moves,
      policy_table]

theorem protected_protocol_empty_mass_positive :
    0 < SensoryMaintenance.body.law SensoryMaintenance.world
      (SensoryMaintenanceRepairCandidates.protectedTwoTickProtocol 2) 5
      (fun _ => SensoryMaintenance.empty) .probeB := by
  have hpath : path 0 = SensoryMaintenance.initial := by rfl
  have hp := SensoryMaintenanceTwoTickGlobal.stateLaw_pos_of_path K
    SensoryMaintenance.initial path event hpath 5 path_step
  have heq : (fun z : SensoryMaintenance.S =>
      if z = SensoryMaintenance.initial then (1 : ℝ) else 0) =
      (fun z => if z.1 = SensoryMaintenance.body.init ∧
        z.2 = SensoryMaintenance.world.start then 1 else 0) := by
    funext z
    simp [SensoryMaintenance.initial, Prod.ext_iff]
  rw [heq] at hp
  rw [SensoryMaintenance.body.jointEventKernel_stateLaw
    SensoryMaintenance.world SensoryMaintenance.ec false
    (SensoryMaintenanceRepairCandidates.protectedTwoTickProtocol 2) 5 (path 5)] at hp
  simpa [path] using hp

theorem protected_protocol_survival_lt_one :
    Exchange.survival SensoryMaintenance.body SensoryMaintenance.world
      SensoryMaintenance.ec SensoryMaintenance.material false
      (SensoryMaintenanceRepairCandidates.protectedTwoTickProtocol 2) 5 < 1 := by
  let sched := SensoryMaintenanceRepairCandidates.protectedTwoTickProtocol 2
  let d : ℝ := ∑ w : Env,
    SensoryMaintenance.body.law SensoryMaintenance.world sched 5
      (fun _ => SensoryMaintenance.empty) w
  have hd : 0 < d := by
    dsimp [d, sched]
    have hle := Finset.single_le_sum (s := (Finset.univ : Finset Env))
      (f := fun w => SensoryMaintenance.body.law SensoryMaintenance.world
        (SensoryMaintenanceRepairCandidates.protectedTwoTickProtocol 2) 5
        (fun _ => SensoryMaintenance.empty) w)
      (fun w _ => SensoryMaintenance.body.law_nonneg_of_clocked
        SensoryMaintenance.world
        (SensoryMaintenance.body.eventClocked_implies_clocked SensoryMaintenance.ec)
        (SensoryMaintenanceRepairCandidates.protectedTwoTickProtocol 2) 5
        (fun _ => SensoryMaintenance.empty) w)
      (Finset.mem_univ Env.probeB)
    exact lt_of_lt_of_le protected_protocol_empty_mass_positive hle
  have point (x : Unit → SensoryMaintenance.State) (w : Env) :
      Exchange.alive SensoryMaintenance.body SensoryMaintenance.world
          SensoryMaintenance.ec SensoryMaintenance.material false sched 5 (x, w) ≤
        if x = (fun _ => SensoryMaintenance.empty) then 0 else 1 := by
    split_ifs with hx
    · rw [hx]
      simp [Exchange.alive, Viability.alive, SensoryMaintenance.body,
        Medium.total, SensoryMaintenance.cost, SensoryMaintenance.empty,
        RenewableLearning.cost, RenewableLearning.buffer, RenewableLearning.credit]
    · exact (Viability.alive_bounds _ _ _).2
  have hs := SensoryMaintenance.body.law_sum SensoryMaintenance.world sched 5
  have bound : Exchange.survival SensoryMaintenance.body SensoryMaintenance.world
      SensoryMaintenance.ec SensoryMaintenance.material false sched 5 ≤
      ∑ x : Unit → SensoryMaintenance.State, ∑ w : Env,
        (if x = (fun _ => SensoryMaintenance.empty) then 0 else 1) *
          SensoryMaintenance.body.law SensoryMaintenance.world sched 5 x w := by
    apply Finset.sum_le_sum
    intro x _
    apply Finset.sum_le_sum
    intro w _
    exact mul_le_mul_of_nonneg_right (point x w)
      (SensoryMaintenance.body.law_nonneg_of_clocked SensoryMaintenance.world
        (SensoryMaintenance.body.eventClocked_implies_clocked SensoryMaintenance.ec)
        sched 5 x w)
  have heq :
      (∑ x : Unit → SensoryMaintenance.State, ∑ w : Env,
        (if x = (fun _ => SensoryMaintenance.empty) then 0 else 1) *
          SensoryMaintenance.body.law SensoryMaintenance.world sched 5 x w) =
        1 - d := by
    dsimp [d]
    have pointid (x : Unit → SensoryMaintenance.State) (w : Env) :
        (if x = (fun _ => SensoryMaintenance.empty) then 0 else 1) *
            SensoryMaintenance.body.law SensoryMaintenance.world sched 5 x w =
          SensoryMaintenance.body.law SensoryMaintenance.world sched 5 x w -
            (if x = (fun _ => SensoryMaintenance.empty) then 1 else 0) *
              SensoryMaintenance.body.law SensoryMaintenance.world sched 5 x w := by
      split_ifs <;> ring
    simp_rw [pointid, Finset.sum_sub_distrib]
    rw [hs]
    simp [ite_mul, Finset.sum_ite_irrel]
  rw [heq] at bound
  linarith

end CGB.SensoryMaintenanceProtocolRepairRefutation
