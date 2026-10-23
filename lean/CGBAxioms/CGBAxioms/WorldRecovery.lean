import CGBAxioms.Foundation

/-!
# First-hit probabilities in the actual body--world chain

No best-stimulus oracle is used. The finite-horizon recurrence follows the
registered transition kernel from the CURRENT joint state. A marked event is an
observer's question, not a new reward, goal or physical law. The content-increase
specialization measures material intake; identifying usable free-energy intake
requires a separate thermodynamic ledger and is not asserted here.
-/

open Finset

namespace CGB.FirstHit

variable {S : Type*} [Fintype S]

structure Kernel (S : Type*) [Fintype S] where
  step : S → S → ℝ
  nonneg : ∀ z z', 0 ≤ step z z'
  sum_one : ∀ z, ∑ z', step z z' = 1

/-- Probability of at least one marked transition, counted only on its first
occurrence. Future steps follow the same registered chain, with no interventions. -/
noncomputable def probability (K : Kernel S) (mark : S → S → Bool) : ℕ → S → ℝ
  | 0 => fun _ => 0
  | k + 1 => fun z => ∑ z', K.step z z' *
      (if mark z z' then 1 else probability K mark k z')

theorem bounds (K : Kernel S) (mark : S → S → Bool) (k : ℕ) (z : S) :
    0 ≤ probability K mark k z ∧ probability K mark k z ≤ 1 := by
  induction k generalizing z with
  | zero => simp [probability]
  | succ k ih =>
    constructor
    · exact Finset.sum_nonneg fun z' _ => mul_nonneg (K.nonneg z z') (by
        split_ifs <;> simp_all [(ih z').1])
    · calc
        _ ≤ ∑ z', K.step z z' * 1 := Finset.sum_le_sum fun z' _ =>
          mul_le_mul_of_nonneg_left (by split_ifs <;> simp_all [(ih z').2]) (K.nonneg z z')
        _ = 1 := by simp [K.sum_one]

theorem monotone_horizon (K : Kernel S) (mark : S → S → Bool) (k : ℕ) (z : S) :
    probability K mark k z ≤ probability K mark (k + 1) z := by
  induction k generalizing z with
  | zero => exact (bounds K mark 1 z).1
  | succ k ih =>
    apply Finset.sum_le_sum
    intro z' _
    apply mul_le_mul_of_nonneg_left _ (K.nonneg z z')
    split_ifs
    · exact le_rfl
    · exact ih z'

/-- An equivalent absorbing 'already happened' observer. Its flag cannot change
the body/world transition, so observing success does not create that success. -/
noncomputable def flagged (K : Kernel S) (mark : S → S → Bool) : ℕ → S → Bool → ℝ
  | 0 => fun _ seen => if seen then 1 else 0
  | k + 1 => fun z seen => ∑ z', K.step z z' * flagged K mark k z' (seen || mark z z')

theorem flagged_true (K : Kernel S) (mark : S → S → Bool) (k : ℕ) (z : S) :
    flagged K mark k z true = 1 := by
  induction k generalizing z with
  | zero => rfl
  | succ k ih => simp [flagged, ih, K.sum_one]

theorem flagged_false (K : Kernel S) (mark : S → S → Bool) (k : ℕ) (z : S) :
    flagged K mark k z false = probability K mark k z := by
  induction k generalizing z with
  | zero => rfl
  | succ k ih =>
    simp only [flagged, Bool.false_or, probability]
    apply Finset.sum_congr rfl
    intro z' _
    by_cases h : mark z z' = true
    · simp [h, flagged_true]
    · simp [h, ih]

/-- A closed region with no marked positive-probability transition has zero
success probability at every finite horizon, even if success exists elsewhere. -/
theorem zero_on_closed (K : Kernel S) (mark : S → S → Bool) (C : S → Prop)
    (closed : ∀ z z', C z → 0 < K.step z z' → C z')
    (unmarked : ∀ z z', C z → 0 < K.step z z' → mark z z' = false)
    (k : ℕ) (z : S) (hz : C z) : probability K mark k z = 0 := by
  induction k generalizing z with
  | zero => rfl
  | succ k ih =>
    apply Finset.sum_eq_zero
    intro z' _
    rcases (K.nonneg z z').lt_or_eq with hp | hp
    · simp [unmarked z z' hz hp, ih z' (closed z z' hz hp)]
    · simp [hp.symm]

/-- Topological possibility uses the SAME actual kernel. It is kept distinct
from a finite-horizon probability threshold and from an arbitrary stimulus path. -/
def Possible (K : Kernel S) (mark : S → S → Bool) (z : S) : Prop :=
  ∃ y, Relation.ReflTransGen (fun a b => 0 < K.step a b) z y ∧
    ∃ y', 0 < K.step y y' ∧ mark y y' = true

theorem impossible_absorbing (K : Kernel S) (mark : S → S → Bool) {z z' : S}
    (h : ¬ Possible K mark z) (hs : 0 < K.step z z') : ¬ Possible K mark z' := by
  rintro ⟨y, hy, hm⟩
  exact h ⟨y, Relation.ReflTransGen.head hs hy, hm⟩

theorem zero_of_impossible (K : Kernel S) (mark : S → S → Bool) {z : S}
    (h : ¬ Possible K mark z) (k : ℕ) : probability K mark k z = 0 := by
  apply zero_on_closed K mark (fun y => ¬ Possible K mark y) _ _ k z h
  · intro y y' hy hs
    exact impossible_absorbing K mark hy hs
  · intro y y' hy hs
    apply Bool.eq_false_iff.2
    intro hm
    exact hy ⟨y, Relation.ReflTransGen.refl, y', hs, hm⟩

end CGB.FirstHit

namespace CGB.FirstHit

variable {S : Type*} [Fintype S]

/-- Time-dependent extension, needed for externally specified interruptions and
the pre-existing distribution-dependent loop-cut control. Time t is world time. -/
noncomputable def during (K : ℕ → Kernel S) (mark : S → S → Bool) : ℕ → ℕ → S → ℝ
  | _, 0 => fun _ => 0
  | t, k + 1 => fun z => ∑ z', (K t).step z z' *
      (if mark z z' then 1 else during K mark (t + 1) k z')

theorem during_bounds (K : ℕ → Kernel S) (mark : S → S → Bool) (t k : ℕ) (z : S) :
    0 ≤ during K mark t k z ∧ during K mark t k z ≤ 1 := by
  induction k generalizing t z with
  | zero => simp [during]
  | succ k ih =>
    constructor
    · exact Finset.sum_nonneg fun z' _ => mul_nonneg ((K t).nonneg z z') (by
        split_ifs <;> simp_all [(ih (t + 1) z').1])
    · calc
        _ ≤ ∑ z', (K t).step z z' * 1 := Finset.sum_le_sum fun z' _ =>
          mul_le_mul_of_nonneg_left (by split_ifs <;> simp_all [(ih (t + 1) z').2])
            ((K t).nonneg z z')
        _ = 1 := by simp [(K t).sum_one]

theorem during_constant (K : Kernel S) (mark : S → S → Bool) (t k : ℕ) (z : S) :
    during (fun _ => K) mark t k z = probability K mark k z := by
  induction k generalizing t z with
  | zero => rfl
  | succ k ih => simp only [during, probability, ih]

end CGB.FirstHit

namespace CGB.Medium

variable {U Λ A I : Type*} [Fintype U] [DecidableEq U] [Fintype Λ] [DecidableEq Λ] [Fintype A]
  (M : Medium U Λ A I) (Wd : World U Λ I)

/-- Same old-state update ordering as law: world reacts to x, body senses w.
No unannounced change to the world's clock or to simultaneous-update semantics. -/
noncomputable def jointStep (sched : ℕ → Wd.W → U → I) (t : ℕ)
    (z z' : (U → Λ) × Wd.W) : ℝ :=
  Wd.moves z.2 z.1 z'.2 * M.kernel Wd.clock (sched t z.2) z.1 z'.1

theorem jointStep_sum (sched : ℕ → Wd.W → U → I) (t : ℕ) (z : (U → Λ) × Wd.W) :
    ∑ z', M.jointStep Wd sched t z z' = 1 := by
  simp only [jointStep, Fintype.sum_prod_type]
  rw [Finset.sum_comm]
  simp only [← Finset.mul_sum, M.kernel_sum, mul_one, Wd.moves_sum]

theorem law_uses_jointStep (sched : ℕ → Wd.W → U → I) (n : ℕ) (y : U → Λ) (w' : Wd.W) :
    M.law Wd sched (n + 1) y w' =
      ∑ z, M.law Wd sched n z.1 z.2 * M.jointStep Wd sched n z (y, w') := by
  simp only [law, jointStep, Fintype.sum_prod_type, mul_assoc]

/-- The actual autonomous body--world kernel. Clock admission supplies probability
nonnegativity; it is not omitted merely because the rows algebraically sum to one. -/
noncomputable def actualKernel (hc : M.Clocked Wd.clock) :
    FirstHit.Kernel ((U → Λ) × Wd.W) where
  step := M.jointStep Wd (fun _ => Wd.sense) 0
  nonneg z z' := mul_nonneg (Wd.moves_nonneg _ _ _) (M.kernel_nonneg hc _ _ _)
  sum_one := M.jointStep_sum Wd _ 0

noncomputable def scheduledKernel (hc : M.Clocked Wd.clock)
    (sched : ℕ → Wd.W → U → I) (t : ℕ) : FirstHit.Kernel ((U → Λ) × Wd.W) where
  step := M.jointStep Wd sched t
  nonneg z z' := mul_nonneg (Wd.moves_nonneg _ _ _) (M.kernel_nonneg hc _ _ _)
  sum_one := M.jointStep_sum Wd sched t

/-- The existing marginal-ghost cut, INCLUDING its future world transitions.
Using actualKernel here instead would silently reconnect the future at scoring. -/
noncomputable def scheduledCutKernel (hc : M.Clocked Wd.clock)
    (sched : ℕ → Wd.W → U → I) (t : ℕ) : FirstHit.Kernel ((U → Λ) × Wd.W) where
  step z z' := cutMove Wd (M.lawCut Wd sched t) z.1 z.2 z'.2 *
    M.kernel Wd.clock (sched t z.2) z.1 z'.1
  nonneg z z' := mul_nonneg (Finset.sum_nonneg fun x _ => mul_nonneg
    (Finset.sum_nonneg fun w _ => M.lawCut_nonneg_of_clocked Wd hc sched t x w)
    (Wd.moves_nonneg _ _ _)) (M.kernel_nonneg hc _ _ _)
  sum_one z := by
    simp only [Fintype.sum_prod_type]
    rw [Finset.sum_comm]
    simp only [← Finset.mul_sum, M.kernel_sum, mul_one]
    exact cutMove_sum Wd (M.lawCut_sum Wd sched t) _ _

theorem lawCut_uses_scheduledCutKernel (hc : M.Clocked Wd.clock)
    (sched : ℕ → Wd.W → U → I) (n : ℕ) (y : U → Λ) (w' : Wd.W) :
    M.lawCut Wd sched (n + 1) y w' =
      ∑ z, M.lawCut Wd sched n z.1 z.2 *
        (M.scheduledCutKernel Wd hc sched n).step z (y, w') := by
  simp only [lawCut, scheduledCutKernel, Fintype.sum_prod_type, mul_assoc]

/-- Material intake only; no thermodynamic interpretation is smuggled into this label. -/
def materialIntake (z z' : (U → Λ) × Wd.W) : Bool := decide (M.total z.1 < M.total z'.1)

noncomputable def actualRecover (hc : M.Clocked Wd.clock) (k : ℕ)
    (x : U → Λ) (w : Wd.W) : ℝ :=
  FirstHit.probability (M.actualKernel Wd hc) (M.materialIntake Wd) k (x, w)

theorem actualRecover_bounds (hc : M.Clocked Wd.clock) (k : ℕ) (x : U → Λ) (w : Wd.W) :
    0 ≤ M.actualRecover Wd hc k x w ∧ M.actualRecover Wd hc k x w ≤ 1 :=
  FirstHit.bounds _ _ _ _

theorem actualRecover_mono (hc : M.Clocked Wd.clock) (k : ℕ) (x : U → Λ) (w : Wd.W) :
    M.actualRecover Wd hc k x w ≤ M.actualRecover Wd hc (k + 1) x w :=
  FirstHit.monotone_horizon _ _ _ _

/-- No possible material intake implies zero actual recovery, without assuming
that a favourable stimulus can be summoned from some other world state. -/
theorem actualRecover_zero_without_intake (hc : M.Clocked Wd.clock)
    (h : ∀ x, ¬ M.CanTakeIn x) (k : ℕ) (x : U → Λ) (w : Wd.W) :
    M.actualRecover Wd hc k x w = 0 := by
  apply FirstHit.zero_on_closed _ _ (fun _ => True) (by simp) _ k (x, w) trivial
  intro z z' _ hp
  apply Bool.eq_false_iff.2
  intro hm
  have hlt : M.total z.1 < M.total z'.1 := of_decide_eq_true hm
  have hk := M.kernel_up_zero h Wd.clock (Wd.sense z.2) hlt
  change 0 < Wd.moves z.2 z.1 z'.2 * M.kernel Wd.clock (Wd.sense z.2) z.1 z'.1 at hp
  rw [hk, mul_zero] at hp
  exact (lt_irrefl _ hp)

end CGB.Medium
