import CGBAxioms.NoetherBridge
import CGBAxioms.RecoveryControls

/-!
# Definition controls prompted by passive absorption and frozen-copy objections

The old reflex/self-model advantages remain withdrawn. These theorems connect
intake exclusion to the CURRENT actual-world target, not the historical menu.
They add neither physical axioms nor a definition of cognitive superiority.
-/

open Finset

namespace CGB.FirstHit

variable {S : Type*} [Fintype S]

/-- Protocol-dependent extension of zero_on_closed, also valid for the future
marginal-ghost cut. An observer cannot create a marked transition. -/
theorem during_zero_on_closed (K : ℕ → Kernel S) (mark : S → S → Bool)
    (C : S → Prop)
    (closed : ∀ t z z', C z → 0 < (K t).step z z' → C z')
    (unmarked : ∀ t z z', C z → 0 < (K t).step z z' → mark z z' = false)
    (t k : ℕ) (z : S) (hz : C z) : during K mark t k z = 0 := by
  induction k generalizing t z with
  | zero => rfl
  | succ k ih =>
    apply Finset.sum_eq_zero
    intro z' _
    rcases ((K t).nonneg z z').lt_or_eq with hp | hp
    · simp only [unmarked t z z' hz hp, Bool.false_eq_true, ↓reduceIte,
        ih (t + 1) z' (closed t z z' hz hp), mul_zero]
    · simp [hp.symm]

end CGB.FirstHit

namespace CGB.Operational

variable {U Λ A I : Type*} [Fintype U] [DecidableEq U] [Fintype Λ] [DecidableEq Λ]
  [Fintype A] (M : Medium U Λ A I) (Wd : World U Λ I) (hc : M.Clocked Wd.clock)

/-- No possible positive-rate intake excludes recovery on BOTH registered
future protocols, for every current joint state and every horizon. -/
theorem recovery_zero_without_intake (h : ∀ x, ¬ M.CanTakeIn x)
    (cut : Bool) (sched : ℕ → Wd.W → U → I) (t k : ℕ)
    (z : (U → Λ) × Wd.W) : recovery M Wd hc cut sched t k z = 0 := by
  apply FirstHit.during_zero_on_closed _ _ (fun _ => True) (by simp) _ t k z trivial
  intro n y y' _ hp
  apply Bool.eq_false_iff.2
  intro hm
  have hlt : M.total y.1 < M.total y'.1 := of_decide_eq_true hm
  have hk := M.kernel_up_zero h Wd.clock (sched n y.2) hlt
  cases cut
  · change 0 < Wd.moves y.2 y.1 y'.2 * M.kernel Wd.clock (sched n y.2) y.1 y'.1 at hp
    rw [hk, mul_zero] at hp
    exact lt_irrefl _ hp
  · change 0 < Medium.cutMove Wd (M.lawCut Wd sched n) y.1 y.2 y'.2 *
      M.kernel Wd.clock (sched n y.2) y.1 y'.1 at hp
    rw [hk, mul_zero] at hp
    exact lt_irrefl _ hp

theorem alive_zero_without_intake (h : ∀ x, ¬ M.CanTakeIn x)
    (cut : Bool) (sched : ℕ → Wd.W → U → I) (t : ℕ)
    (z : (U → Λ) × Wd.W) : alive M Wd hc cut sched t z = 0 := by
  classical
  have hn : ¬ Wd.recovery ≤ 0 := not_le_of_gt Wd.recovery_pos
  simp [alive, recovery_zero_without_intake M Wd hc h, hn]

theorem survival_zero_without_intake (h : ∀ x, ¬ M.CanTakeIn x)
    (cut : Bool) (sched : ℕ → Wd.W → U → I) (n : ℕ) :
    survival M Wd hc cut sched n = 0 := by
  simp [survival, alive_zero_without_intake M Wd hc h]

theorem not_persists_without_intake (h : ∀ x, ¬ M.CanTakeIn x) (δ : ℝ) :
    ¬ Persists M Wd hc δ := by
  rintro ⟨hδ, hp⟩
  have hn := hp 0
  rw [survival_zero_without_intake M Wd hc h] at hn
  linarith

theorem not_selfMaintains_without_intake (h : ∀ x, ¬ M.CanTakeIn x) :
    ¬ SelfMaintains M Wd hc := by
  rintro ⟨n₀, ε, hε, hp⟩
  have hn := hp n₀ le_rfl
  rw [survival_zero_without_intake M Wd hc h,
    survival_zero_without_intake M Wd hc h] at hn
  linarith

/-- This migrates the fully-frozen-copy exclusion to the actual-world target.
Freezing only under quiet, or only freezing a subset of state, is a different
intervention and is not excluded by this sufficient hypothesis. -/
theorem not_persists_of_frozen
    (hf : ∀ a s x, 0 < M.rate a s x → M.act a s x = x) (δ : ℝ) :
    ¬ Persists M Wd hc δ := by
  apply not_persists_without_intake M Wd hc _ δ
  intro x h
  obtain ⟨a, s, hp, hlt⟩ := h
  rw [hf a s x hp] at hlt
  exact lt_irrefl _ hlt

/-- Direct downstream use of the new Noether bridge: if closed moment drifts
vanish and there are no boundary participants, actual intake is impossible.
With an open boundary the flux term must instead be accounted for. -/
theorem not_persists_of_closed_moments_and_empty_boundary
    (hm : ∀ s,
      (∀ x, (M.closedJump s).gen M.contentObservable x = 0) ∧
      (∀ x, (M.closedJump s).gen (fun y => M.contentObservable y ^ 2) x = 0))
    (hb : M.boundary = ∅) (δ : ℝ) : ¬ Persists M Wd hc δ := by
  have hp2a := M.p2a_iff_closed_moments.mpr hm
  apply not_persists_without_intake M Wd hc _ δ
  intro x h
  obtain ⟨a, s, hp, hlt⟩ := h
  have hclosed : M.Closed a := by simp [Medium.Closed, hb]
  rw [hp2a a s x hclosed hp] at hlt
  exact lt_irrefl _ hlt

end CGB.Operational

namespace CGB.Medium

variable {U Λ A I : Type*} [Fintype U] [DecidableEq U] [Fintype Λ] [DecidableEq Λ]
  [Fintype A] (M : Medium U Λ A I)

/-- The full initial state need only be absorbing for all possible stimuli.
Other states of the same medium may still possess intake events. -/
theorem kernel_from_frozen_state (x : U → Λ)
    (hf : ∀ a s, 0 < M.rate a s x → M.act a s x = x)
    (R : ℝ) (s : U → I) (y : U → Λ) :
    M.kernel R s x y = if x = y then 1 else 0 := by
  have hj : ∀ y, (M.jump s).rate x y = 0 := by
    intro y
    apply Finset.sum_eq_zero
    intro a _
    change (if M.act a s x = y ∧ x ≠ y then M.rate a s x else 0) = 0
    split_ifs with he
    · by_cases hp : 0 < M.rate a s x
      · exact False.elim (he.2 ((hf a s hp).symm.trans he.1))
      · exact le_antisymm (le_of_not_gt hp) (M.rate_nonneg a s x)
    · rfl
  simp [kernel, hj]

end CGB.Medium

namespace CGB.Operational

variable {U Λ A I : Type*} [Fintype U] [DecidableEq U] [Fintype Λ] [DecidableEq Λ]
  [Fintype A] (M : Medium U Λ A I) (Wd : World U Λ I) (hc : M.Clocked Wd.clock)

theorem recovery_zero_from_frozen_state (x : U → Λ)
    (hf : ∀ a s, 0 < M.rate a s x → M.act a s x = x)
    (cut : Bool) (sched : ℕ → Wd.W → U → I) (t k : ℕ) (w : Wd.W) :
    recovery M Wd hc cut sched t k (x, w) = 0 := by
  let K : ℕ → FirstHit.Kernel ((U → Λ) × Wd.W) :=
    if cut then M.scheduledCutKernel Wd hc sched else M.scheduledKernel Wd hc sched
  have hstay : ∀ n z z', z.1 = x → 0 < (K n).step z z' → z'.1 = x := by
    intro n z z' hz hp
    by_contra hn
    have hne : x ≠ z'.1 := Ne.symm hn
    cases cut
    · change 0 < Wd.moves z.2 z.1 z'.2 * M.kernel Wd.clock (sched n z.2) z.1 z'.1 at hp
      rw [hz, M.kernel_from_frozen_state x hf, ite_eq_right hne, mul_zero] at hp
      exact lt_irrefl _ hp
    · change 0 < Medium.cutMove Wd (M.lawCut Wd sched n) z.1 z.2 z'.2 *
        M.kernel Wd.clock (sched n z.2) z.1 z'.1 at hp
      rw [hz, M.kernel_from_frozen_state x hf, ite_eq_right hne, mul_zero] at hp
      exact lt_irrefl _ hp
  apply FirstHit.during_zero_on_closed K _ (fun z => z.1 = x) hstay _ t k (x, w) rfl
  intro n z z' hz hp
  have hz' := hstay n z z' hz hp
  simp [Medium.materialIntake, hz, hz']

theorem alive_zero_from_frozen_state (x : U → Λ)
    (hf : ∀ a s, 0 < M.rate a s x → M.act a s x = x)
    (cut : Bool) (sched : ℕ → Wd.W → U → I) (t : ℕ) (w : Wd.W) :
    alive M Wd hc cut sched t (x, w) = 0 := by
  classical
  have hn : ¬ Wd.recovery ≤ 0 := not_le_of_gt Wd.recovery_pos
  simp [alive, recovery_zero_from_frozen_state M Wd hc x hf, hn]

/-- Even holding material at birth cannot rescue an absorbing initial body. -/
theorem not_persists_of_frozen_initial
    (hf : ∀ a s, 0 < M.rate a s M.init → M.act a s M.init = M.init) (δ : ℝ) :
    ¬ Persists M Wd hc δ := by
  rintro ⟨hδ, hp⟩
  have hz : survival M Wd hc false (fun _ => Wd.sense) 0 = 0 := by
    classical
    simp [survival, Medium.law, mul_ite, ite_and,
      alive_zero_from_frozen_state M Wd hc M.init hf]
  have hn := hp 0
  rw [hz] at hn
  linarith

end CGB.Operational

namespace CGB.RecoveryControls

/-- The existing passive-absorption control also satisfies all six physical
clauses; its failure of the life target is not a violation of those clauses. -/
theorem passive_body_axioms : body.Axioms := by
  refine ⟨⟨fun _ => Set.toFinite _, ⟨1, by decide⟩, ⟨false, rfl, by decide⟩⟩,
    ?_, ?_, ?_, ?_, ?_⟩
  · refine ⟨?_, by intros; trivial, ?_, ?_⟩
    · intro u v
      exact ⟨Equiv.refl Unit, by intros; rfl, Subsingleton.elim _ _⟩
    · intro a s x u hu
      simp [body] at hu
    · intro a s s' x y hh
      have hx : x = y := funext fun u => (hh u (by simp [body])).1
      have hs : s = s' := funext fun u => (hh u (by simp [body])).2
      subst y
      subst s'
      exact ⟨rfl, fun _ _ => rfl⟩
  · intro a s x hc _
    simp [Medium.Closed, body] at hc
  · refine ⟨fun _ => 1, fun _ => one_pos, ?_⟩
    intro x y
    simp [Medium.jump, body]
  · intro σ _
    have he : σ = Equiv.refl Unit := by
      apply Equiv.ext
      intro u
      exact Subsingleton.elim _ _
    subst σ
    exact ⟨Equiv.refl Unit, fun _ _ _ => ⟨rfl, rfl⟩⟩
  · refine ⟨?_, fun _ _ => rfl⟩
    intro a s s' x hh
    have hs : s = s' := funext fun u => hh u (by simp [body])
    subst s'
    exact ⟨rfl, rfl⟩


/-- Passive absorption really occurs, so this is stronger than a frozen body
with no intake transition anywhere. It still does not certify ongoing life. -/
theorem passive_body_can_take_in : body.CanTakeIn (fun _ => false) := by
  refine ⟨(), fun _ => true, ?_, ?_⟩
  · norm_num [body]
  · norm_num [Medium.total, body]

/-- Once full, the passive body's absorption stops in every environment. -/
theorem passive_full_frozen : ∀ a s, 0 < body.rate a s (fun _ => true) →
    body.act a s (fun _ => true) = (fun _ => true) := by
  intro a s hp
  simp [body] at hp

/-- The current actual-world alive test excludes the full passive state for
both normal and cut futures, at every horizon and in every world. -/
theorem passive_full_not_alive (Wd : World Unit Bool Bool) (hc : body.Clocked Wd.clock)
    (cut : Bool) (sched : ℕ → Wd.W → Unit → Bool) (t : ℕ) (w : Wd.W) :
    Operational.alive body Wd hc cut sched t ((fun _ => true), w) = 0 :=
  Operational.alive_zero_from_frozen_state body Wd hc _ passive_full_frozen cut sched t w

/-- Start the same passive mechanism already full, so failure cannot be blamed
on an initially empty body. Its transition repertoire still permits intake in
other states. All rates and actions are inherited unchanged. -/
def filledPassive : Medium Unit Bool Unit Bool :=
  { body with init := fun _ => true }

theorem filledPassive_axioms : filledPassive.Axioms := by
  have h := passive_body_axioms
  exact ⟨h.p1a, h.p1b, h.p2a, h.p2b, h.p2c, h.p3.1, fun _ _ => rfl⟩

theorem filledPassive_clock : filledPassive.Clocked 1 := body_clock

theorem filledPassive_not_persists (Wd : World Unit Bool Bool)
    (hc : filledPassive.Clocked Wd.clock) (δ : ℝ) :
    ¬ Operational.Persists filledPassive Wd hc δ :=
  Operational.not_persists_of_frozen_initial filledPassive Wd hc passive_full_frozen δ

/-- Explicit definition control: physical admissibility, initial material, and
an intake event somewhere do not imply actual persistence in this world. -/
theorem passive_intake_not_persistence (δ : ℝ) :
    filledPassive.Axioms ∧ filledPassive.Clocked 1 ∧
      0 < filledPassive.total filledPassive.init ∧
      (∃ x, filledPassive.CanTakeIn x) ∧
      ¬ Operational.Persists filledPassive (world true) filledPassive_clock δ := by
  refine ⟨filledPassive_axioms, filledPassive_clock, ?_,
    ⟨fun _ => false, passive_body_can_take_in⟩,
    filledPassive_not_persists (world true) filledPassive_clock δ⟩
  norm_num [Medium.total, filledPassive, body]

end CGB.RecoveryControls

namespace CGB.NoetherControls

/-- Existing admissible seed: its closed-event content symmetry is certified. -/
theorem seed_closed_symmetry : ∀ s,
    (Models.seed.closedJump s).ObservableSymmetry Models.seed.contentObservable :=
  Models.seed.p2a_iff_closed_observable_symmetry.mp Models.seed_p2a

/-- Existing nonconserving closed toggle is rejected by the new moment bridge. -/
theorem toggle_fails_closed_moments : ¬ (∀ s,
    (∀ x, (Models.twinToggle.closedJump s).gen Models.twinToggle.contentObservable x = 0) ∧
    (∀ x, (Models.twinToggle.closedJump s).gen
      (fun y => Models.twinToggle.contentObservable y ^ 2) x = 0)) :=
  fun h => Models.twinToggle_witness.1 (Models.twinToggle.p2a_iff_closed_moments.mpr h)

/-- A finite probabilistic cancellation control: 0 branches to -1 and +1.
This checks the two-moment requirement, not learning or organism capability. -/
def meanOnly : JumpProcess (Fin 3) where
  rate x y := if x = 0 ∧ y ≠ 0 then 1 else 0
  rate_nonneg _ _ := by split_ifs <;> norm_num

def signedValue (x : Fin 3) : ℝ := if x = 1 then -1 else if x = 2 then 1 else 0

theorem meanOnly_first_drift : ∀ x, meanOnly.gen signedValue x = 0 := by
  intro x
  fin_cases x <;> norm_num [JumpProcess.gen, Fin.sum_univ_three, meanOnly, signedValue]

theorem meanOnly_second_drift : meanOnly.gen (fun x => signedValue x ^ 2) 0 = 2 := by
  norm_num [JumpProcess.gen, Fin.sum_univ_three, meanOnly, signedValue]

theorem meanOnly_not_conserved : ¬ meanOnly.Conserved signedValue := by
  intro h
  have he := ((JumpProcess.noether meanOnly signedValue).mpr h).2 0
  rw [meanOnly_second_drift] at he
  norm_num at he

end CGB.NoetherControls
