import CGBAxioms.NoetherBridge
import CGBAxioms.OperationalLife

/-!
# Event-resolved observations with unchanged body/world marginals

State-preserving events can carry gross exchange. Resolving them needs an event
clock bound, stronger than the old bound on state-changing exit rates. The same
world clock is retained. Labels are emitted by registered channels and are not
additional organism state or a feedback mechanism.
-/

open Finset

namespace CGB.FirstHit.Kernel

theorem eq_of_step {S : Type*} [Fintype S] (K K' : FirstHit.Kernel S)
    (h : K.step = K'.step) : K = K' := by
  cases K
  cases K'
  cases h
  rfl

end CGB.FirstHit.Kernel

namespace CGB.EventChannels

structure Kernel (S E : Type*) [Fintype S] [Fintype E] where
  step : S → E → S → ℝ
  nonneg : ∀ s e s', 0 ≤ step s e s'
  sum_one : ∀ s, ∑ e, ∑ s', step s e s' = 1

namespace Kernel

variable {S E : Type*} [Fintype S] [Fintype E]

noncomputable def forget (K : Kernel S E) : FirstHit.Kernel S where
  step s s' := ∑ e, K.step s e s'
  nonneg _ _ := Finset.sum_nonneg fun _ _ => K.nonneg _ _ _
  sum_one s := by rw [Finset.sum_comm]; exact K.sum_one s

/-- First marked channel occurrence under an explicitly time-dependent
protocol; t is the external world clock, not a stored body variable. -/
noncomputable def firstHit (K : ℕ → Kernel S E)
    (mark : ℕ → S → E → S → Bool) : ℕ → ℕ → S → ℝ
  | _, 0 => fun _ => 0
  | t, k + 1 => fun s => ∑ e, ∑ s', (K t).step s e s' *
      (if mark t s e s' then 1 else firstHit K mark (t + 1) k s')

theorem firstHit_bounds (K : ℕ → Kernel S E) (mark : ℕ → S → E → S → Bool)
    (t k : ℕ) (s : S) :
    0 ≤ firstHit K mark t k s ∧ firstHit K mark t k s ≤ 1 := by
  induction k generalizing t s with
  | zero => simp [firstHit]
  | succ k ih =>
    constructor
    · apply Finset.sum_nonneg
      intro e _
      apply Finset.sum_nonneg
      intro s' _
      exact mul_nonneg ((K t).nonneg s e s') (by
        split_ifs
        · norm_num
        · exact (ih (t + 1) s').1)
    · calc
        _ ≤ ∑ e, ∑ s', (K t).step s e s' := by
          apply Finset.sum_le_sum
          intro e _
          apply Finset.sum_le_sum
          intro s' _
          apply mul_le_of_le_one_right ((K t).nonneg s e s')
          split_ifs
          · exact le_rfl
          · exact (ih (t + 1) s').2
        _ = 1 := (K t).sum_one s

/-- A state-only mark is the specialization of the single channel observer
that ignores event labels. This also covers the legacy net-growth question. -/
theorem state_mark_firstHit (K : ℕ → Kernel S E) (mark : S → S → Bool)
    (t k : ℕ) (s : S) :
    firstHit K (fun _ x _ y => mark x y) t k s =
      FirstHit.during (fun n => (K n).forget) mark t k s := by
  induction k generalizing t s with
  | zero => rfl
  | succ k ih =>
    simp only [firstHit, FirstHit.during, forget, ih]
    rw [Finset.sum_comm]
    apply Finset.sum_congr rfl
    intro s' _
    rw [Finset.sum_mul]

theorem firstHit_unmarked (K : ℕ → Kernel S E)
    (t k : ℕ) (s : S) : firstHit K (fun _ _ _ _ => false) t k s = 0 := by
  induction k generalizing t s with
  | zero => rfl
  | succ k ih => simp [firstHit, ih]

/-- State law after forgetting labels. -/
noncomputable def stateLaw (K : ℕ → Kernel S E) (initial : S → ℝ) : ℕ → S → ℝ
  | 0 => initial
  | n + 1 => fun s' => ∑ s, stateLaw K initial n s * (K n).forget.step s s'

open Classical in
/-- Forward law with an observer's last channel attached. -/
noncomputable def labelledLaw (K : ℕ → Kernel S E) (initial : S → ℝ) (previous : E) :
    ℕ → S → E → ℝ
  | 0 => fun s e => if e = previous then initial s else 0
  | n + 1 => fun s' e => ∑ s, ∑ old, labelledLaw K initial previous n s old * (K n).step s e s'

theorem labelledLaw_marginal (K : ℕ → Kernel S E) (initial : S → ℝ) (previous : E)
    (n : ℕ) (s : S) : ∑ e, labelledLaw K initial previous n s e = stateLaw K initial n s := by
  classical
  induction n generalizing s with
  | zero => simp [labelledLaw, stateLaw]
  | succ n ih =>
    simp only [labelledLaw, stateLaw]
    rw [Finset.sum_comm]
    apply Finset.sum_congr rfl
    intro x _
    rw [Finset.sum_comm]
    simp only [← Finset.mul_sum]
    rw [← Finset.sum_mul, ih]
    rfl

end Kernel
end CGB.EventChannels

namespace CGB.Medium

variable {U Λ A I : Type*} [Fintype U] [DecidableEq U] [Fintype Λ] [DecidableEq Λ]
  [Fintype A] (M : Medium U Λ A I)

/-- Clock admission for all registered event channels, including self-events.
This is an observation/representation obligation, not a new physical axiom. -/
def EventClocked (R : ℝ) : Prop :=
  0 < R ∧ ∀ s x, ∑ a, M.rate a s x ≤ R

theorem exit_rate_events (s : U → I) (x : U → Λ) :
    (∑ y, (M.jump s).rate x y) =
      ∑ a, if M.act a s x = x then 0 else M.rate a s x := by
  simp only [jump]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro a _
  by_cases he : M.act a s x = x
  · simp [he]
  · rw [Finset.sum_eq_single (M.act a s x)]
    · simp [he, Ne.symm he]
    · intro y _ hy
      simp [Ne.symm hy]
    · simp

theorem eventClocked_implies_clocked {R : ℝ} (h : M.EventClocked R) : M.Clocked R := by
  refine ⟨h.1, fun s x => ?_⟩
  rw [M.exit_rate_events]
  exact (Finset.sum_le_sum (fun a _ => by
    split_ifs
    · exact M.rate_nonneg a s x
    · exact le_rfl)).trans (h.2 s x)

/-- None is idle; some a is an actual occurrence of registered channel a. -/
noncomputable def eventStep (R : ℝ) (s : U → I) (x : U → Λ) :
    Option A → (U → Λ) → ℝ
  | none, y => if x = y then 1 - (∑ a, M.rate a s x) / R else 0
  | some a, y => if M.act a s x = y then M.rate a s x / R else 0

omit [DecidableEq U] [Fintype Λ] in
theorem eventStep_nonneg {R : ℝ} (h : M.EventClocked R)
    (s : U → I) (x : U → Λ) (e : Option A) (y : U → Λ) :
    0 ≤ M.eventStep R s x e y := by
  cases e with
  | none =>
    change 0 ≤ if x = y then 1 - (∑ a, M.rate a s x) / R else 0
    split_ifs
    · linarith [(div_le_one h.1).mpr (h.2 s x)]
    · exact le_rfl
  | some a =>
    change 0 ≤ if M.act a s x = y then M.rate a s x / R else 0
    split_ifs
    · exact div_nonneg (M.rate_nonneg a s x) h.1.le
    · exact le_rfl

theorem eventStep_sum (R : ℝ) (s : U → I) (x : U → Λ) :
    ∑ e, ∑ y, M.eventStep R s x e y = 1 := by
  simp only [Fintype.sum_option, eventStep, Finset.sum_ite_eq, Finset.mem_univ, ite_true]
  simp only [div_eq_mul_inv, Finset.sum_mul]
  ring

/-- Forgetting the event labels recovers the ORIGINAL kernel, even when some
positive-rate events leave the body state unchanged. -/
theorem eventStep_forget (R : ℝ) (s : U → I) (x y : U → Λ) :
    ∑ e, M.eventStep R s x e y = M.kernel R s x y := by
  rw [Fintype.sum_option]
  simp only [eventStep, Medium.kernel]
  by_cases hxy : x = y
  · subst y
    have hj : (M.jump s).rate x x = 0 := by simp [jump]
    have hs : (∑ a, M.rate a s x) =
        (∑ a, if M.act a s x = x then M.rate a s x else 0) +
        (∑ a, if M.act a s x = x then 0 else M.rate a s x) := by
      rw [← Finset.sum_add_distrib]
      apply Finset.sum_congr rfl
      intro a _
      split_ifs <;> simp
    simp only [hj, zero_div, ite_true, zero_add]
    have hself : (∑ a, if M.act a s x = x then M.rate a s x / R else 0) =
        (∑ a, if M.act a s x = x then M.rate a s x else 0) / R := by
      simp only [div_eq_mul_inv, Finset.sum_mul, ite_mul, zero_mul]
    rw [hself, hs, M.exit_rate_events]
    ring
  · simp [jump, hxy, div_eq_mul_inv, Finset.sum_mul, ite_mul]

noncomputable def eventKernel {R : ℝ} (h : M.EventClocked R) (s : U → I) :
    EventChannels.Kernel (U → Λ) (Option A) where
  step := M.eventStep R s
  nonneg := M.eventStep_nonneg h s
  sum_one := M.eventStep_sum R s

variable (Wd : World U Λ I) (he : M.EventClocked Wd.clock)

/-- The cut uses its own original marginal-ghost future. Body marginal equality
ensures recording channel labels does not alter that pre-existing cut process. -/
noncomputable def eventWorldStep (cut : Bool) (sched : ℕ → Wd.W → U → I)
    (t : ℕ) (z : (U → Λ) × Wd.W) (w' : Wd.W) : ℝ :=
  if cut then cutMove Wd (M.lawCut Wd sched t) z.1 z.2 w'
  else Wd.moves z.2 z.1 w'

include he in
theorem eventWorldStep_nonneg (cut : Bool) (sched : ℕ → Wd.W → U → I)
    (t : ℕ) (z : (U → Λ) × Wd.W) (w' : Wd.W) :
    0 ≤ M.eventWorldStep Wd cut sched t z w' := by
  cases cut
  · exact Wd.moves_nonneg _ _ _
  · exact Finset.sum_nonneg fun x _ => mul_nonneg
      (Finset.sum_nonneg fun w _ =>
        M.lawCut_nonneg_of_clocked Wd (M.eventClocked_implies_clocked he) sched t x w)
      (Wd.moves_nonneg _ _ _)

theorem eventWorldStep_sum (cut : Bool) (sched : ℕ → Wd.W → U → I)
    (t : ℕ) (z : (U → Λ) × Wd.W) :
    ∑ w', M.eventWorldStep Wd cut sched t z w' = 1 := by
  cases cut
  · exact Wd.moves_sum _ _
  · exact cutMove_sum Wd (M.lawCut_sum Wd sched t) z.1 z.2

noncomputable def jointEventKernel (cut : Bool) (sched : ℕ → Wd.W → U → I) (t : ℕ) :
    EventChannels.Kernel ((U → Λ) × Wd.W) (Option A) where
  step z e z' := M.eventWorldStep Wd cut sched t z z'.2 *
    M.eventStep Wd.clock (sched t z.2) z.1 e z'.1
  nonneg z e z' := mul_nonneg (M.eventWorldStep_nonneg Wd he cut sched t z z'.2)
    (M.eventStep_nonneg he _ _ _ _)
  sum_one z := by
    simp only [Fintype.sum_prod_type, ← Finset.sum_mul,
      M.eventWorldStep_sum, one_mul]
    exact M.eventStep_sum Wd.clock _ _

theorem jointEventKernel_forget (cut : Bool) (sched : ℕ → Wd.W → U → I) (t : ℕ)
    (z z' : (U → Λ) × Wd.W) :
    ((M.jointEventKernel Wd he cut sched t).forget).step z z' =
      (if cut then M.scheduledCutKernel Wd (M.eventClocked_implies_clocked he) sched t
      else M.scheduledKernel Wd (M.eventClocked_implies_clocked he) sched t).step z z' := by
  change (∑ e, M.eventWorldStep Wd cut sched t z z'.2 *
    M.eventStep Wd.clock (sched t z.2) z.1 e z'.1) = _
  rw [← Finset.mul_sum, M.eventStep_forget]
  cases cut <;> rfl

theorem event_state_mark_matches_operational (cut : Bool)
    (sched : ℕ → Wd.W → U → I) (t k : ℕ) (z : (U → Λ) × Wd.W) :
    EventChannels.Kernel.firstHit (M.jointEventKernel Wd he cut sched)
      (fun _ x _ y => M.materialIntake Wd x y) t k z =
      Operational.recovery M Wd (M.eventClocked_implies_clocked he) cut sched t k z := by
  rw [EventChannels.Kernel.state_mark_firstHit]
  have hK : (fun n => (M.jointEventKernel Wd he cut sched n).forget) =
      if cut then M.scheduledCutKernel Wd (M.eventClocked_implies_clocked he) sched
      else M.scheduledKernel Wd (M.eventClocked_implies_clocked he) sched := by
    funext n
    apply FirstHit.Kernel.eq_of_step
    funext x y
    cases cut
    · exact M.jointEventKernel_forget Wd he false sched n x y
    · exact M.jointEventKernel_forget Wd he true sched n x y
  rw [hK]
  rfl

/-- The entire observed forward process has the original normal/cut state law,
not just an equal one-step kernel. The observer is not an extra body state. -/
theorem jointEventKernel_stateLaw (cut : Bool) (sched : ℕ → Wd.W → U → I)
    (n : ℕ) (z : (U → Λ) × Wd.W) :
    EventChannels.Kernel.stateLaw (M.jointEventKernel Wd he cut sched)
      (fun z => if z.1 = M.init ∧ z.2 = Wd.start then 1 else 0) n z =
      if cut then M.lawCut Wd sched n z.1 z.2 else M.law Wd sched n z.1 z.2 := by
  classical
  induction n generalizing z with
  | zero => cases cut <;> rfl
  | succ n ih =>
    simp only [EventChannels.Kernel.stateLaw, ih, M.jointEventKernel_forget,
      Fintype.sum_prod_type]
    cases cut <;> simp [law, lawCut, scheduledKernel, scheduledCutKernel, jointStep, mul_assoc]

end CGB.Medium
