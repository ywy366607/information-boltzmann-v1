import CGBAxioms.SensoryMaintenanceRepairCandidates

open CGB

namespace SensoryRepairCandidatesReview20261025

example (n₀ : ℕ) (w : RenewableLearning.Env) (u : Unit) :
    SensoryMaintenanceRepairCandidates.protectedTwoTickProtocol n₀ n₀ w u = none :=
  SensoryMaintenanceRepairCandidates.protected_first_tick n₀ w u

example (x : SensoryMaintenance.State) (hq : x.1 ≠ .vacuum)
    (hr : 1 ≤ x.2.val) :
    SensoryMaintenance.next (some .probe)
        (SensoryMaintenance.next none x) ≠ SensoryMaintenance.empty :=
  SensoryMaintenanceRepairCandidates.protected_branch_excludes_local_blackout x hq hr

example (x : SensoryMaintenance.State) (hq : x.1 ≠ .vacuum)
    (hr : 1 ≤ x.2.val) :
    SensoryMaintenance.outgoing none x +
        SensoryMaintenance.outgoing (some .probe)
          (SensoryMaintenance.next none x) = 2 :=
  SensoryMaintenanceRepairCandidates.protected_probe_path_outgoing x hq hr

example (x : SensoryMaintenance.State) (hq : x.1 ≠ .vacuum)
    (hr : 2 ≤ x.2.val) :
    SensoryMaintenance.incoming (some .probe)
        (SensoryMaintenance.next none x) = 2 :=
  SensoryMaintenanceRepairCandidates.protected_probe_path_incoming x hq hr

example (x : SensoryMaintenance.State)
    (hx : SensoryMaintenanceRepairCandidates.eligibleTwoTickStart x) :
    x.1 ≠ .aSpent ∧ x.1 ≠ .bSpent :=
  SensoryMaintenanceRepairCandidates.eligible_excludes_spent x hx

example (x : SensoryMaintenance.State) (w : RenewableLearning.Env)
    (hx : SensoryMaintenanceRepairCandidates.eligibleTwoTickStart x) :
    ¬(x.1 = .aSpent ∧ w = .feedA) :=
  SensoryMaintenanceRepairCandidates.eligible_excludes_verified_dangerous_branch x w hx

example :
    SensoryMaintenanceRepairCandidates.guardedTwoTickStart .search .probeA :=
  SensoryMaintenanceRepairCandidates.initial_guarded_two_tick_start

example (n₀ : ℕ) (q : RenewableLearning.Cell) (w : RenewableLearning.Env)
    (hg : SensoryMaintenanceRepairCandidates.guardedTwoTickStart q w) :
    let sched := SensoryMaintenanceRepairCandidates.guardedLossProtocol n₀
    let K := fun t => SensoryMaintenance.body.jointEventKernel
      SensoryMaintenance.world SensoryMaintenance.ec false sched t
    let mark := Exchange.intake SensoryMaintenance.body SensoryMaintenance.world
      SensoryMaintenance.material sched
    EventChannels.Kernel.firstHit K mark n₀ 3
      ((fun _ => (q, 2)), w) = 1 :=
  SensoryMaintenanceRepairCandidates.guarded_firstHit n₀ q w hg

example (n₀ : ℕ) (q : RenewableLearning.Cell) (w : RenewableLearning.Env)
    (hg : SensoryMaintenanceRepairCandidates.guardedTwoTickStart q w) :
    Exchange.alive SensoryMaintenance.body SensoryMaintenance.world
      SensoryMaintenance.ec SensoryMaintenance.material false
      (SensoryMaintenanceRepairCandidates.guardedLossProtocol n₀) n₀
      ((fun _ => (q, 2)), w) = 1 :=
  SensoryMaintenanceRepairCandidates.guarded_alive n₀ q w hg

example :
    SensoryMaintenanceRepairCandidates.GuardedTwoTickLocalTolerance 0 :=
  SensoryMaintenanceRepairCandidates.guarded_two_tick_local_tolerance

example : SensoryMaintenanceRepairCandidates.GuardedTwoTickSupport 0 :=
  SensoryMaintenanceRepairCandidates.guarded_support_zero

example : ¬ SensoryMaintenanceRepairCandidates.GuardedTwoTickSupport 2 :=
  SensoryMaintenanceRepairCandidates.not_guarded_support_two

example :
    Exchange.survival SensoryMaintenance.body SensoryMaintenance.world
      SensoryMaintenance.ec SensoryMaintenance.material false
      (SensoryMaintenanceRepairCandidates.guardedLossProtocol 0) 0 = 1 :=
  SensoryMaintenanceRepairCandidates.guarded_initial_survival

example (s : SensoryMaintenance.Stim) (x : SensoryMaintenance.State) :
    SensoryMaintenance.cost (SensoryMaintenance.next s x) +
        SensoryMaintenance.outgoing s x =
      SensoryMaintenance.cost x + SensoryMaintenance.incoming s x :=
  SensoryMaintenanceRepairCandidates.eligible_keeps_existing_budget_contract s x

end SensoryRepairCandidatesReview20261025
