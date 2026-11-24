import Mathlib.Tactic

/-! Actual body mapping:
X = causal_value_learning.prediction inputs (z_source, action_delta, option).
K = bounded identified mother facts R and held-out queries Q, encoded by the
current encoder. Required value = finite_use_value.expected under that K.
Native witness numbers below come from the identity-correct 2437 support
report. They certify differences of the BODY'S EXPECTED UTILITY, not a claim
that its expected utility equals the world's true conditional mean.
Implementation of the lifted evaluator: epistemic_value_reference.py.
-/
namespace CGB.NativeEpistemicValueState
noncomputable section

theorem same_input_requires_same_value {S X : Type*}
    (input : S → X) (required : S → ℝ) (read : X → ℝ)
    (exact : ∀ s, read (input s) = required s)
    (a b : S) (same : input a = input b) : required a = required b := by
  rw [← exact a, ← exact b, same]

theorem pair_loss_decomposition (p m n : ℝ) :
    ((p-m)^2+(p-n)^2)/2 = (p-(m+n)/2)^2+(m-n)^2/4 := by ring

theorem pair_loss_floor (p m n : ℝ) :
    (m-n)^2/4 ≤ ((p-m)^2+(p-n)^2)/2 := by
  rw [pair_loss_decomposition]; exact le_add_of_nonneg_left (sq_nonneg _)

/-- Each input retains the actual (R,Q) context. A read with that context
can directly execute the same existing U/scorer. -/
theorem lift_read {X K : Type*} (operator : X → K → ℝ) :
    ∃ read : X × K → ℝ, ∀ x k, read (x,k) = operator x k := by
  exact ⟨fun p => operator p.1 p.2, fun _ _ => rfl⟩

/-- Transport the learning context through U at each imagined step. This is
the commutation condition checked against the production first-step U; it
also gives recursive equivalence for a sequence of hypothetical arrivals. -/
theorem transport_steps {S K E : Type*} (encode : S → K)
    (step : S → E → S) (advance : K → E → K)
    (commute : ∀ s e, encode (step s e) = advance (encode s) e)
    (s : S) (events : List E) :
    encode (events.foldl step s) = events.foldl advance (encode s) := by
  induction events generalizing s with
  | nil => rfl
  | cons e es ih =>
    simp only [List.foldl_cons]
    rw [ih, commute]

def risk (base cross second alpha : ℝ) :=
  base - 2*alpha*cross + alpha^2*second

/-- With the measured nonpositive direction/target cross moment, every
nonnegative amplitude worsens or ties this batch's baseline risk. -/
theorem nonpositive_cross (b c v a : ℝ) (hc : c ≤ 0) (hv : 0 ≤ v) (ha : 0 ≤ a) :
    b ≤ risk b c v a := by
  have h1 : 0 ≤ -2*a*c := by nlinarith [mul_nonpos_of_nonneg_of_nonpos ha hc]
  have h2 : 0 ≤ a^2*v := mul_nonneg (sq_nonneg a) hv
  unfold risk; nlinarith

/-- Full support vs withdrawal episode:2066; same X/Q, different R.
Exact dyadic rational values from the actual float32 mean * 10. -/
def muFull : ℝ := 10 * (13527829 / 134217728 : ℝ)
def muWithdraw : ℝ := 10 * (11450606 / 536870912 : ℝ)

theorem actual_expected_values_differ : muFull ≠ muWithdraw := by
  norm_num [muFull, muWithdraw]

end
end CGB.NativeEpistemicValueState
