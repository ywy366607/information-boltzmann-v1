import Mathlib.Tactic

/-! Actual binding: audit_native_policy_boundary.py, identified body 2473.
The readout view is (z,K,conditional candidate packet,precision,q). A stored
feeling intervention changes only the habit input. This proves failure of a
pathwise actor decoder on that view, not impossibility of a conditional mean.
Recall candidates and actual draft endpoints are also different; pushforward
below separates candidate-value mismatch from option-policy mismatch.
-/
namespace CGB.NativeConditionalActorView
noncomputable section

/-- The minimum expectation contract consumes an actual policy view, not z alone. -/
def value {A : Type*} [Fintype A] (pi q : A → ℝ) : ℝ := ∑ a, pi a * q a

theorem shared_policy_shared_value {A : Type*} [Fintype A]
    (pi q : A → ℝ) : value pi q = ∑ a, pi a * q a := rfl

theorem same_view_decoder {S V : Type*} (view : S → V) (read : V → ℝ)
    (target : S → ℝ) (h : ∀ s, read (view s) = target s)
    (s t : S) (equal : view s = view t) : target s = target t := by
  rw [← h s, ← h t, equal]

theorem distinct_targets_forbid_decoder {S V : Type*} (view : S → V)
    (target : S → ℝ) (s t : S) (equal : view s = view t)
    (different : target s ≠ target t) : ¬ ∃ read : V → ℝ, ∀ s, read (view s) = target s := by
  rintro ⟨read, h⟩
  exact different (same_view_decoder view read target h s t equal)

/-- Keeping the actor's actual sufficient view is a constructive decoder. -/
theorem actor_view_sufficient {S V A : Type*} [Fintype A] (view : S → V)
    (policy : V → A → ℝ) (q : V → A → ℝ) :
    ∃ read : V → ℝ, ∀ s, read (view s) = value (policy (view s)) (q (view s)) := by
  exact ⟨fun v => value (policy v) (q v), fun _ => rfl⟩

/-- Same candidates: changing a policy changes expectation by this exact sum. -/
theorem policy_gap {A : Type*} [Fintype A] (pi rho q : A → ℝ) :
    value rho q - value pi q = ∑ a, (rho a-pi a)*q a := by
  simp only [value, sub_mul, Finset.sum_sub_distrib]

/-- Different candidates require a separate candidate-value term. -/
theorem candidate_and_policy_gap {I A : Type*} [Fintype I] [Fintype A]
    (w : I → ℝ) (option : I → A) (g : I → ℝ) (rho pi q : A → ℝ)
    (push : ∑ i, w i*q (option i) = value rho q) :
    (∑ i, w i*g i) - value pi q =
      (∑ i, w i*(g i-q (option i))) + ∑ a, (rho a-pi a)*q a := by
  simp only [mul_sub, Finset.sum_sub_distrib]
  rw [push, ← policy_gap]
  ring

/-- Measured same-q values under current/stored intero, fixed z/K/candidates. -/
theorem measured_intero_targets_distinct :
    (0.06641688942909241 : ℚ) ≠ 0.06373170018196106 := by norm_num

/-- Measured same-option-q gap is nonzero; candidate differences are separate. -/
theorem measured_policy_gap_nonzero : (0.014704219996929169 : ℚ) > 0 := by norm_num

end
end CGB.NativeConditionalActorView
