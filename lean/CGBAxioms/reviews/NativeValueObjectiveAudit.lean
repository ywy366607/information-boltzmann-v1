import Mathlib.Tactic
import Mathlib.Analysis.SpecialFunctions.Sqrt

/-! Production binding, without simulating a learner:
* finite_use_value.score_scope/observed scores before/after relation revision
  with the same encoder and current query set. F t includes those coordinates,
  queries, scope and predictive score. r t is the persistent relation state.
* parameter_update_gap uses the actual settlement-time parameters for payment;
  the other term is the logged representation_shift at the unrevised relation.
* native_amortized_decision.predict uses source-balanced squared residual fitting
  and ridge; kind labels change weights, not the features or output target.
* relation_retention_use.advance/biases uses bias-corrected EMA moments. The
  one-observation identities below apply to its actual first/second moment law.
These identities determine objective semantics, not observed benefit sizes.
-/
namespace CGB.NativeValueObjectiveAudit
noncomputable section

def revisionPayment {R : Type*} (F : R → ℝ) (before after : R) : ℝ :=
  F after - F before

theorem changing_score_sum {R : Type*} (F : ℕ → R → ℝ) (r : ℕ → R) (n : ℕ) :
    (∑ t ∈ Finset.range n, revisionPayment (F t) (r t) (r (t+1))) =
      F n (r n) - F 0 (r 0) +
      ∑ t ∈ Finset.range n, (F t (r (t+1)) - F (t+1) (r (t+1))) := by
  induction n with
  | zero => simp
  | succ n ih =>
    simp only [Finset.sum_range_succ]
    rw [ih]
    unfold revisionPayment
    ring

theorem fixed_score_sum {R : Type*} (F : R → ℝ) (r : ℕ → R) (n : ℕ) :
    (∑ t ∈ Finset.range n, revisionPayment F (r t) (r (t+1))) =
      F (r n) - F (r 0) := by
  simpa using changing_score_sum (fun _ => F) r n

theorem parameter_update_gap {T R : Type*} (score : T → R → ℝ)
    (beforeT afterT : T) (beforeR afterR : R) :
    score afterT afterR - score beforeT beforeR =
      revisionPayment (score afterT) beforeR afterR +
      (score afterT beforeR - score beforeT beforeR) := by
  unfold revisionPayment
  ring

def fitLoss (wn ww ridge native world prior x : ℝ) : ℝ :=
  wn*(x-native)^2 + ww*(x-world)^2 + ridge*(x-prior)^2

theorem source_mixture_excess (wn ww ridge native world prior z x : ℝ)
    (stationary : wn*(z-native) + ww*(z-world) + ridge*(z-prior) = 0) :
    fitLoss wn ww ridge native world prior x -
      fitLoss wn ww ridge native world prior z = (wn+ww+ridge)*(x-z)^2 := by
  unfold fitLoss
  linear_combination 2*(x-z)*stationary

theorem equal_sources_no_ridge (native world x : ℝ) :
    fitLoss 1 1 0 native world 0 x -
      fitLoss 1 1 0 native world 0 ((native+world)/2) =
      2*(x-(native+world)/2)^2 := by
  unfold fitLoss
  ring

theorem first_corrected_moment (beta reward : ℝ) (h : beta ≠ 1) :
    ((1-beta)*reward)/(1-beta) = reward := by
  have hb : 1-beta ≠ 0 := sub_ne_zero.mpr (Ne.symm h)
  field_simp

theorem first_positive_ratio (reward epsilon : ℝ) (h : 0 < reward) :
    reward / (Real.sqrt (reward^2) + epsilon) = reward/(reward+epsilon) := by
  rw [Real.sqrt_sq_eq_abs, abs_of_pos h]

theorem first_positive_zero_epsilon (reward : ℝ) (h : 0 < reward) :
    reward / (Real.sqrt (reward^2) + 0) = 1 := by
  rw [first_positive_ratio reward 0 h]
  simp [ne_of_gt h]

end
end CGB.NativeValueObjectiveAudit
