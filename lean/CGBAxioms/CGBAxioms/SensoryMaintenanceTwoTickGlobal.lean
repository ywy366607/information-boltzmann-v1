import CGBAxioms.SensoryMaintenanceTwoTickObstruction

open Finset

namespace CGB.SensoryMaintenanceTwoTickGlobal

open RenewableLearning SensoryMaintenance

private theorem policy_table (q : Cell) : policy q = match q with
    | .aSpent | .aReady | .aHeld => 0
    | .bSpent | .bReady | .bHeld => 1
    | _ => 2 := by cases q <;> decide

private theorem stateLaw_nonneg {S E : Type*} [Fintype S] [DecidableEq S] [Fintype E]
    (K : ℕ → EventChannels.Kernel S E) (initial : S) (n : ℕ) (s : S) :
    0 ≤ EventChannels.Kernel.stateLaw K
      (fun s => if s = initial then (1 : ℝ) else 0) n s := by
  induction n generalizing s with
  | zero => simp [EventChannels.Kernel.stateLaw]; split_ifs <;> norm_num
  | succ n ih =>
    change 0 ≤ ∑ x, EventChannels.Kernel.stateLaw K
      (fun s => if s = initial then (1 : ℝ) else 0) n x *
      (∑ e, (K n).step x e s)
    apply Finset.sum_nonneg
    intro x _
    exact mul_nonneg (ih x) (Finset.sum_nonneg fun e _ => (K n).nonneg _ _ _)

theorem stateLaw_pos_of_path {S E : Type*} [Fintype S] [DecidableEq S] [Fintype E]
    (K : ℕ → EventChannels.Kernel S E) (initial : S) (path : ℕ → S)
    (event : ℕ → E) (hpath : path 0 = initial) (n : ℕ)
    (hstep : ∀ t < n, 0 < (K t).step (path t) (event t) (path (t + 1))) :
    0 < EventChannels.Kernel.stateLaw K
      (fun s => if s = initial then (1 : ℝ) else 0) n (path n) := by
  induction n with
  | zero => simpa [EventChannels.Kernel.stateLaw, hpath]
  | succ n ih =>
    change 0 < ∑ s, EventChannels.Kernel.stateLaw K
      (fun s => if s = initial then (1 : ℝ) else 0) n s *
      (∑ e, (K n).step s e (path (n + 1)))
    have hinner : 0 < ∑ e, (K n).step (path n) e (path (n + 1)) := by
      have hle := Finset.single_le_sum (s := (Finset.univ : Finset E))
        (f := fun e => (K n).step (path n) e (path (n + 1)))
        (fun e _ => (K n).nonneg _ _ _) (Finset.mem_univ (event n))
      exact lt_of_lt_of_le (hstep n (by omega)) hle
    have ih' : 0 < EventChannels.Kernel.stateLaw K
        (fun s => if s = initial then (1 : ℝ) else 0) n (path n) :=
      ih (fun t ht => hstep t (by omega))
    have hterm : 0 < EventChannels.Kernel.stateLaw K
        (fun s => if s = initial then (1 : ℝ) else 0) n (path n) *
        (∑ e, (K n).step (path n) e (path (n + 1))) := mul_pos ih' hinner
    have hle := Finset.single_le_sum (s := (Finset.univ : Finset S))
      (f := fun s => EventChannels.Kernel.stateLaw K
        (fun s => if s = initial then (1 : ℝ) else 0) n s *
        (∑ e, (K n).step s e (path (n + 1))))
      (fun s _ => mul_nonneg (by
        exact stateLaw_nonneg K initial n s)
        (Finset.sum_nonneg fun e _ => (K n).nonneg _ _ _))
      (Finset.mem_univ (path n))
    exact lt_of_lt_of_le hterm hle

private noncomputable def path (n : ℕ) : SensoryMaintenance.S :=
  match n with
  | 0 => SensoryMaintenance.initial
  | 1 => (fun _ => (.search, 2), .answerA)
  | 2 => (fun _ => (.aSpent, 2), .feedA)
  | 3 => (fun _ => (.aSpent, 1), .feedB)
  | 4 => (fun _ => (.aSpent, 0), .missB)
  | 5 => (fun _ => SensoryMaintenance.empty, .probeB)
  | _ => SensoryMaintenance.initial

private def event (_ : ℕ) : Option Unit := some ()

private def protocol : ℕ → Env → Unit → Stim := sensoryProtocol 2 2

private noncomputable abbrev K := fun t =>
  SensoryMaintenance.body.jointEventKernel SensoryMaintenance.world
    SensoryMaintenance.ec false protocol t

private theorem path_step (t : ℕ) (ht : t < 5) :
    0 < (K t).step (path t) (event t) (path (t + 1)) := by
  have ht' : t = 0 ∨ t = 1 ∨ t = 2 ∨ t = 3 ∨ t = 4 := by omega
  rcases ht' with rfl | rfl | rfl | rfl | rfl <;>
    norm_num [K, path, event, protocol, SensoryMaintenance.sensoryProtocol,
      Medium.jointEventKernel, Medium.eventWorldStep, Medium.eventStep,
      SensoryMaintenance.body, SensoryMaintenance.world, SensoryMaintenance.next,
      SensoryMaintenance.initial, SensoryMaintenance.empty, RenewableLearning.sense,
      SensoryMaintenance.incoming, SensoryMaintenance.outgoing,
      RenewableLearning.next, RenewableLearning.incoming,
      RenewableLearning.outgoing, GraceRecovery.moves, RenewableLearning.moves,
      policy_table]

theorem empty_mass_positive :
    0 < SensoryMaintenance.body.law SensoryMaintenance.world protocol 5
      (fun _ => SensoryMaintenance.empty) .probeB := by
  have hpath : path 0 = SensoryMaintenance.initial := by rfl
  have hp := stateLaw_pos_of_path K SensoryMaintenance.initial path event hpath 5 path_step
  have heq : (fun z : SensoryMaintenance.S =>
      if z = SensoryMaintenance.initial then (1 : ℝ) else 0) =
      (fun z => if z.1 = SensoryMaintenance.body.init ∧
        z.2 = SensoryMaintenance.world.start then 1 else 0) := by
    funext z
    simp [SensoryMaintenance.initial, Prod.ext_iff]
  rw [heq] at hp
  rw [SensoryMaintenance.body.jointEventKernel_stateLaw
    SensoryMaintenance.world SensoryMaintenance.ec false protocol 5 (path 5)] at hp
  simpa [path] using hp

theorem two_tick_survival_lt_one :
    Exchange.survival SensoryMaintenance.body SensoryMaintenance.world
      SensoryMaintenance.ec SensoryMaintenance.material false protocol 5 < 1 := by
  let d : ℝ := ∑ w : Env,
    SensoryMaintenance.body.law SensoryMaintenance.world protocol 5
      (fun _ => SensoryMaintenance.empty) w
  have hd : 0 < d := by
    dsimp [d]
    have hle := Finset.single_le_sum (s := (Finset.univ : Finset Env))
      (f := fun w => SensoryMaintenance.body.law SensoryMaintenance.world protocol 5
        (fun _ => SensoryMaintenance.empty) w)
      (fun w _ => SensoryMaintenance.body.law_nonneg_of_clocked
        SensoryMaintenance.world
        (SensoryMaintenance.body.eventClocked_implies_clocked SensoryMaintenance.ec)
        protocol 5 (fun _ => SensoryMaintenance.empty) w)
      (Finset.mem_univ Env.probeB)
    exact lt_of_lt_of_le empty_mass_positive hle
  have point (x : Unit → SensoryMaintenance.State) (w : Env) :
      Exchange.alive SensoryMaintenance.body SensoryMaintenance.world
          SensoryMaintenance.ec SensoryMaintenance.material false protocol 5 (x, w) ≤
        if x = (fun _ => SensoryMaintenance.empty) then 0 else 1 := by
    split_ifs with hx
    · rw [hx]
      simp [Exchange.alive, Viability.alive, SensoryMaintenance.body,
        Medium.total,
        SensoryMaintenance.cost, SensoryMaintenance.empty,
        RenewableLearning.cost, RenewableLearning.buffer, RenewableLearning.credit]
    · exact (Viability.alive_bounds _ _ _).2
  have hs := SensoryMaintenance.body.law_sum SensoryMaintenance.world protocol 5
  have bound : Exchange.survival SensoryMaintenance.body SensoryMaintenance.world
      SensoryMaintenance.ec SensoryMaintenance.material false protocol 5 ≤
      ∑ x : Unit → SensoryMaintenance.State, ∑ w : Env,
        (if x = (fun _ => SensoryMaintenance.empty) then 0 else 1) *
          SensoryMaintenance.body.law SensoryMaintenance.world protocol 5 x w := by
    apply Finset.sum_le_sum
    intro x _
    apply Finset.sum_le_sum
    intro w _
    exact mul_le_mul_of_nonneg_right (point x w)
      (SensoryMaintenance.body.law_nonneg_of_clocked SensoryMaintenance.world
        (SensoryMaintenance.body.eventClocked_implies_clocked SensoryMaintenance.ec)
        protocol 5 x w)
  have heq :
      (∑ x : Unit → SensoryMaintenance.State, ∑ w : Env,
        (if x = (fun _ => SensoryMaintenance.empty) then 0 else 1) *
          SensoryMaintenance.body.law SensoryMaintenance.world protocol 5 x w) =
        1 - d := by
    dsimp [d]
    have pointid (x : Unit → SensoryMaintenance.State) (w : Env) :
        (if x = (fun _ => SensoryMaintenance.empty) then 0 else 1) *
            SensoryMaintenance.body.law SensoryMaintenance.world protocol 5 x w =
          SensoryMaintenance.body.law SensoryMaintenance.world protocol 5 x w -
            (if x = (fun _ => SensoryMaintenance.empty) then 1 else 0) *
              SensoryMaintenance.body.law SensoryMaintenance.world protocol 5 x w := by
      split_ifs <;> ring
    simp_rw [pointid, Finset.sum_sub_distrib]
    rw [hs]
    simp [ite_mul, Finset.sum_ite_irrel]
  rw [heq] at bound
  linarith

theorem not_two_tick_tolerance :
    ¬ SensoryMaintenanceRecovery.ToleratesSensoryLoss 0 2 := by
  intro h
  have hb := h.2 2 5
  change 1 - (0 : ℝ) ≤ Exchange.survival SensoryMaintenance.body
    SensoryMaintenance.world SensoryMaintenance.ec SensoryMaintenance.material
    false (sensoryProtocol 2 2) 5 at hb
  have hs := two_tick_survival_lt_one
  change Exchange.survival SensoryMaintenance.body SensoryMaintenance.world
    SensoryMaintenance.ec SensoryMaintenance.material false
    (sensoryProtocol 2 2) 5 < 1 at hs
  linarith

end CGB.SensoryMaintenanceTwoTickGlobal
