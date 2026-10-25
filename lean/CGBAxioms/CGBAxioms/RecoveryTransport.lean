import CGBAxioms.RecoveryCriteria

/-!
# Recovery and state-score transport through existing exact refinements

State-dependent intake must match on supported transitions, in addition to
the joint event/state law and body content. Time is the same external input.
-/

open Finset

namespace CGB.RecoveryTransport
open Interaction Energetics

private theorem mean_next {T E : Type*} [Fintype T] [DecidableEq T] [Fintype E]
    (M : Machine T ℕ E) (z : T) (n : ℕ) (v : T → ℝ) :
    mean (law M z (n + 1)) v =
      mean (law M z n) (fun x => ∑ e, ∑ y, M.step n x e y * v y) := by
  simp only [mean, law, EventChannels.Kernel.stateLaw, EventChannels.Kernel.forget,
    events, Finset.sum_mul, Finset.mul_sum, mul_assoc]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro x _
  rw [Finset.sum_comm]

variable {R S E : Type*} [Fintype R] [DecidableEq R]
  [Fintype S] [DecidableEq S] [Fintype E]
  {P : Machine R ℕ E} {Q : Machine S ℕ E} {f : R → S}

/-- Supported actual-intake correspondence preserves every recovery horizon,
including in a distribution-dependent cut supplied as a separate kernel. -/
theorem firstHit_eq (h : Refines P Q f)
    (a : ℕ → R → E → R → Bool) (b : ℕ → S → E → S → Bool)
    (hm : ∀ t r e r', 0 < P.step t r e r' → a t r e r' = b t (f r) e (f r'))
    (t k : ℕ) (r : R) :
    EventChannels.Kernel.firstHit (events P) a t k r =
      EventChannels.Kernel.firstHit (events Q) b t k (f r) := by
  induction k generalizing t r with
  | zero => rfl
  | succ k ih =>
    change (∑ e, ∑ r', P.step t r e r' *
      (if a t r e r' then 1 else EventChannels.Kernel.firstHit (events P) a (t + 1) k r')) =
      ∑ e, ∑ s', Q.step t (f r) e s' *
      (if b t (f r) e s' then 1 else EventChannels.Kernel.firstHit (events Q) b (t + 1) k s')
    apply Finset.sum_congr rfl
    intro e _
    calc
      _ = ∑ r', P.step t r e r' * (if b t (f r) e (f r') then 1 else
          EventChannels.Kernel.firstHit (events Q) b (t + 1) k (f r')) := by
        apply Finset.sum_congr rfl
        intro r' _
        rcases (P.nonneg t r e r').lt_or_eq with hp | hz
        · rw [hm t r e r' hp, ih]
        · simp [← hz]
      _ = _ := by
        obtain ⟨c⟩ := Refines.bisimulation h r (f r) rfl
        exact c.expectation_at_output t e _ _ (by intro r' s' he; rw [he])

/-- Exact state-score expectations at every time, without assuming that the
score is a probability or replacing physical state by output histories. -/
theorem mean_eq (h : Refines P Q f) (initial : R) (n : ℕ) (g : S → ℝ) :
    mean (law P initial n) (fun r => g (f r)) = mean (law Q (f initial) n) g := by
  induction n generalizing g with
  | zero => simp [mean, law, EventChannels.Kernel.stateLaw]
  | succ n ih =>
    have transfer (r : R) :
        (∑ e, ∑ r', P.step n r e r' * g (f r')) =
          ∑ e, ∑ s', Q.step n (f r) e s' * g s' := by
      obtain ⟨c⟩ := Refines.bisimulation h r (f r) rfl
      apply Finset.sum_congr rfl
      intro e _
      exact c.expectation_at_output n e _ _ (by intro r' s' he; rw [he])
    rw [mean_next P, mean_next Q]
    simp_rw [transfer]
    exact ih (fun s => ∑ e, ∑ s', Q.step n s e s' * g s')

end CGB.RecoveryTransport
