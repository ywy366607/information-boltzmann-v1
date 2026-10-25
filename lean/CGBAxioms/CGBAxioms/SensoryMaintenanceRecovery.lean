import CGBAxioms.SensoryMaintenance

open Finset

namespace CGB.SensoryMaintenanceRecovery
open RenewableLearning SensoryMaintenance

abbrev S := (Unit → State) × Env
def project (z : S) : RenewableLearning.S := ((fun u => (z.1 u).1), z.2)
def decode (sched : ℕ → Env → Unit → Stim) : ℕ → Env → Unit → Signal :=
  fun t w _ => (sched t w ()).getD .calm
def oldEvent (s : Stim) : Option Unit := if s = none ∨ s = some .calm then none else some ()

private theorem intake_same (s : Stim) (x : State) :
    decide (0 < SensoryMaintenance.incoming s x) =
      decide (0 < RenewableLearning.incoming true (s.getD .calm) x.1) := by
  rcases x with ⟨q,r⟩
  fin_cases r <;> cases s with
  | none => cases q <;> decide
  | some a => cases a <;> cases q <;> decide

private theorem old_event (s : Stim) (x : State) (hq : x.1 ≠ .vacuum) (hr : 0 < x.2.val) :
    GraceRecovery.body.eventStep 1 (fun _ => s.getD .calm) (fun _ => x.1)
      (oldEvent s) (fun _ => (SensoryMaintenance.next s x).1) = 1 := by
  rcases x with ⟨q,r⟩
  fin_cases r <;> norm_num at hr
  all_goals cases s with
  | none => cases q <;> norm_num [oldEvent, Medium.eventStep, GraceRecovery.body,
      plant, SensoryMaintenance.next, empty, RenewableLearning.next, funext_iff] at *
  | some a => cases a <;> cases q <;> norm_num [oldEvent, Medium.eventStep,
      GraceRecovery.body, plant, SensoryMaintenance.next, empty, RenewableLearning.next,
      funext_iff] at *

private theorem supported (sched : ℕ → Env → Unit → Stim)
    (t : ℕ) (z z' : S) (e : Option Unit)
    (hs : sched t z.2 () ≠ some .calm) (hq : (z.1 ()).1 ≠ .vacuum)
    (hr : 0 < (z.1 ()).2.val)
    (hp : 0 < (body.jointEventKernel world ec false sched t).step z e z') :
    (z'.1 ()) = SensoryMaintenance.next (sched t z.2 ()) (z.1 ()) ∧
    0 < (GraceRecovery.body.jointEventKernel GraceRecovery.world GraceRecovery.ec false
      (decode sched) t).step (project z) (oldEvent (sched t z.2 ())) (project z') ∧
    Exchange.intake body world material sched t z e z' =
      Exchange.intake GraceRecovery.body GraceRecovery.world GraceRecovery.material
        (decode sched) t (project z) (oldEvent (sched t z.2 ())) (project z') := by
  have hb : 0 < body.eventStep 1 (sched t z.2) z.1 e z'.1 := by
    by_contra hn
    exact (not_lt_of_ge (mul_nonpos_of_nonneg_of_nonpos
      (body.eventWorldStep_nonneg world ec false sched t z z'.2) (le_of_not_gt hn))) hp
  have hw : 0 < GraceRecovery.moves z.2 (fun u => (z.1 u).1) z'.2 := by
    by_contra hn
    exact (not_lt_of_ge (mul_nonpos_of_nonpos_of_nonneg (le_of_not_gt hn)
      (body.eventStep_nonneg ec _ _ _ _))) hp
  have hx : z.1 () ≠ empty := by intro h; exact hq (congrArg Prod.fst h)
  cases e with
  | none => simp [Medium.eventStep, body, hs, hx] at hb
  | some u =>
    have he : body.act u (sched t z.2) z.1 = z'.1 := by
      by_contra hn; simp [Medium.eventStep, hn] at hb
    have hn : z'.1 () = SensoryMaintenance.next (sched t z.2 ()) (z.1 ()) :=
      (congrFun he ()).symm
    refine ⟨hn, ?_, ?_⟩
    · change 0 < GraceRecovery.moves z.2 (fun u => (z.1 u).1) z'.2 * _
      have eqx : (fun u => (z.1 u).1) = fun _ => (z.1 ()).1 := by
        funext u; cases u; rfl
      have eqy : (fun u => (z'.1 u).1) =
          fun _ => (SensoryMaintenance.next (sched t z.2 ()) (z.1 ())).1 := by
        funext u; cases u; rw [hn]
      change 0 < _ * GraceRecovery.body.eventStep 1
        (decode sched t z.2) (fun u => (z.1 u).1) (oldEvent (sched t z.2 ()))
          (fun u => (z'.1 u).1)
      rw [eqx, eqy]
      change 0 < _ * GraceRecovery.body.eventStep 1
        (fun _ => (sched t z.2 ()).getD .calm) _ _ _
      rw [old_event _ _ hq hr, mul_one]
      simpa only [eqx] using hw
    · change decide (0 < SensoryMaintenance.incoming (sched t z.2 ()) (z.1 ())) = _
      rw [intake_same]
      by_cases h0 : sched t z.2 () = none
      · simp [oldEvent, h0, Exchange.intake, GraceRecovery.material, ledger, project,
          decode, RenewableLearning.incoming]
      · by_cases hc : sched t z.2 () = some .calm
        · simp [oldEvent, hc, Exchange.intake, GraceRecovery.material, ledger, project,
            decode, RenewableLearning.incoming]
        · simp [oldEvent, h0, hc, Exchange.intake, GraceRecovery.material, ledger, project, decode]
          rfl

private theorem sensed_not_calm (w : Env) : some (sense w ()) ≠ (some Signal.calm : Stim) := by
  cases w <;> decide

theorem decode_protocol (n₀ : ℕ) : decode (sensoryProtocol n₀ 1) = GraceRecovery.interrupted n₀ := by
  funext t w u
  cases u
  simp only [decode, sensoryProtocol, GraceRecovery.interrupted, Medium.interrupt]
  split_ifs <;> rfl

private theorem positive_region (n₀ t : ℕ) (z : S)
    (hg : (GraceRecovery.certificate n₀).region t (project z)) : (z.1 ()).1 ≠ .vacuum := by
  change GraceRecovery.region n₀ t (project z) at hg
  unfold GraceRecovery.region at hg
  split_ifs at hg
  all_goals cases hw : z.2 <;> cases hq : (z.1 ()).1 <;>
    simp_all [project, GraceRecovery.Good, RenewableLearningLife.Good, GraceRecovery.Safe]

private theorem reserve_sensed (a : Signal) (x : State) (hr : 1 ≤ x.2.val)
    (hq : (SensoryMaintenance.next (some a) x).1 ≠ .vacuum) :
    1 ≤ (SensoryMaintenance.next (some a) x).2.val ∧
      (x.2.val = 2 → (SensoryMaintenance.next (some a) x).2.val = 2) := by
  rcases x with ⟨q,r⟩
  fin_cases r <;> cases a <;> cases q <;>
    norm_num [SensoryMaintenance.next, empty, RenewableLearning.next,
      RenewableLearning.incoming] at *

noncomputable def certificate (n₀ : ℕ) : ScheduledRecovery.Ranked
    (body.jointEventKernel world ec false (sensoryProtocol n₀ 1))
    (Exchange.intake body world material (sensoryProtocol n₀ 1)) 3 where
  region t z := (GraceRecovery.certificate n₀).region t (project z) ∧
    1 ≤ (z.1 ()).2.val ∧ (t ≤ n₀ → (z.1 ()).2.val = 2)
  rank t z := (GraceRecovery.certificate n₀).rank t (project z)
  bounded t z hg := (GraceRecovery.certificate n₀).bounded t (project z) hg.1
  closed t z e z' hg hp := by
    have hs : sensoryProtocol n₀ 1 t z.2 () ≠ some .calm := by
      unfold sensoryProtocol; split_ifs
      · decide
      · exact sensed_not_calm _
    obtain ⟨hx, hw, _⟩ := supported _ t z z' e hs (positive_region n₀ t z hg.1) (by omega) hp
    rw [decode_protocol] at hw
    have hG := (GraceRecovery.certificate n₀).closed t (project z) _ (project z') hg.1 hw
    refine ⟨hG, ?_⟩
    by_cases ht : t = n₀
    · have hr := hg.2.2 (by omega)
      constructor
      · rw [hx]
        simp [sensoryProtocol, ht, SensoryMaintenance.next, positive_region n₀ t z hg.1, hr]
      · intro h; omega
    · have hs' : sensoryProtocol n₀ 1 t z.2 () = some (sense z.2 ()) := by
        have h : ¬ (n₀ ≤ t ∧ t < n₀ + 1) := by omega
        simp only [sensoryProtocol, if_neg h]
      rw [hs'] at hx
      have hq' : (SensoryMaintenance.next (some (sense z.2 ())) (z.1 ())).1 ≠ .vacuum := by
        rw [← hx]; exact positive_region n₀ (t + 1) z' hG
      have hr := reserve_sensed _ _ hg.2.1 hq'
      rw [hx]
      exact ⟨hr.1, fun htt => hr.2 (hg.2.2 (by omega))⟩
  progresses t z e z' hg hp hm := by
    have hs : sensoryProtocol n₀ 1 t z.2 () ≠ some .calm := by
      unfold sensoryProtocol; split_ifs
      · decide
      · exact sensed_not_calm _
    obtain ⟨_, hw, hi⟩ := supported _ t z z' e hs (positive_region n₀ t z hg.1) (by omega) hp
    rw [decode_protocol] at hw hi
    exact (GraceRecovery.certificate n₀).progresses t (project z) _ (project z') hg.1 hw
      (hi.symm.trans hm)

theorem one_tick_survival (n₀ n : ℕ) :
    Exchange.survival body world ec material false (sensoryProtocol n₀ 1) n = 1 := by
  apply ScheduledRecovery.survival_of_ranked body world ec material (sensoryProtocol n₀ 1)
    (by norm_num) (certificate n₀)
  · simp [certificate, project, GraceRecovery.certificate, GraceRecovery.region,
      GraceRecovery.Good, RenewableLearningLife.Good, body, plant]
  · intro t z hg
    have hq := positive_region n₀ t z hg.1
    change 0 < SensoryMaintenance.cost (z.1 ())
    rcases hx : z.1 () with ⟨q,r⟩
    cases q <;> simp_all [SensoryMaintenance.cost, RenewableLearning.cost, buffer, credit]

noncomputable def naturalCertificate : ScheduledRecovery.Ranked
    (body.jointEventKernel world ec false (fun _ => world.sense))
    (Exchange.intake body world material (fun _ => world.sense)) 3 where
  region _ z := GraceRecovery.naturalCertificate.region (project z) ∧ (z.1 ()).2.val = 2
  rank _ z := GraceRecovery.naturalCertificate.rank (project z)
  bounded _ z hg := GraceRecovery.naturalCertificate.bounded (project z) hg.1
  closed t z e z' hg hp := by
    have hq : (z.1 ()).1 ≠ .vacuum := positive_region 0 0 z hg.1
    obtain ⟨hx, hw, _⟩ := supported _ t z z' e (sensed_not_calm _) hq (by omega) hp
    have hG := GraceRecovery.naturalCertificate.closed t (project z) _ (project z') hg.1 hw
    refine ⟨hG, ?_⟩
    have hq' : (SensoryMaintenance.next (some (sense z.2 ())) (z.1 ())).1 ≠ .vacuum := by
      rw [← hx]; exact positive_region 0 0 z' hG
    rw [hx]
    exact (reserve_sensed _ _ (by omega) hq').2 hg.2
  progresses t z e z' hg hp hm := by
    have hq : (z.1 ()).1 ≠ .vacuum := positive_region 0 0 z hg.1
    obtain ⟨_, hw, hi⟩ := supported _ t z z' e (sensed_not_calm _) hq (by omega) hp
    exact GraceRecovery.naturalCertificate.progresses t (project z) _ (project z') hg.1 hw
      (hi.symm.trans hm)

theorem natural_survival (n : ℕ) : Exchange.survival body world ec material false
    (fun _ => world.sense) n = 1 := by
  apply ScheduledRecovery.survival_of_ranked body world ec material (fun _ => world.sense)
    (by norm_num) naturalCertificate
  · simp [naturalCertificate, project, GraceRecovery.naturalCertificate, body,
      GraceRecovery.Good, RenewableLearningLife.Good]
  · intro t z hg
    change 0 < SensoryMaintenance.cost (z.1 ())
    unfold SensoryMaintenance.cost
    have hr : (z.1 ()).2.val = 2 := hg.2
    omega

/-- Explicit sensory-loss target; the legacy equilibrium-interruption target
is preserved and must not be silently substituted for this protocol. -/
def ToleratesSensoryLoss (δ : ℝ) (T : ℕ) : Prop :=
  Viability.ToleratesSensoryInterruption
    (fun n₀ n => Exchange.survival body world ec material false (sensoryProtocol n₀ T) n) δ T

theorem one_tick_tolerance : ToleratesSensoryLoss 0 1 := by
  refine ⟨by norm_num, fun n₀ n => ?_⟩
  change 1 - (0 : ℝ) ≤ Exchange.survival body world ec material false (sensoryProtocol n₀ 1) n
  rw [one_tick_survival]
  norm_num

end CGB.SensoryMaintenanceRecovery
