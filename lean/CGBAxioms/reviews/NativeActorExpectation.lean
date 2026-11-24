import Mathlib.Tactic

/-! Actual binding: conditional_actor_reference.py and audit_native_policy_boundary.py.
Actions are the original silence/generate/reuse endpoints at ONE real boundary.
pi is that actor's habit + precision-scaled finite-U logits, q is its already
computed 10*g per actual endpoint. No future drafts or feelings are supplied as
known observations. The encoded read uses the same query/read/history tensors.
The risk identity separates value bias from actual action-sampling variance.
-/
namespace CGB.NativeActorExpectation
noncomputable section

def mean {A : Type*} [Fintype A] (pi q : A → ℝ) := ∑ a, pi a * q a
def risk {A : Type*} [Fintype A] (pi q : A → ℝ) (v : ℝ) :=
  ∑ a, pi a * (v - q a)^2
def variance {A : Type*} [Fintype A] (pi q : A → ℝ) := risk pi q (mean pi q)

theorem risk_expansion {A : Type*} [Fintype A] (pi q : A → ℝ) (v : ℝ) :
    risk pi q v = v^2 * (∑ a, pi a) - 2*v*mean pi q + ∑ a, pi a*(q a)^2 := by
  unfold risk mean
  calc
    _ = ∑ a, (v^2*pi a - (2*v)*(pi a*q a) + pi a*(q a)^2) := by
      apply Finset.sum_congr rfl
      intro a _
      ring
    _ = _ := by simp [Finset.sum_add_distrib, Finset.sum_sub_distrib, Finset.mul_sum]

theorem risk_decomposition {A : Type*} [Fintype A] (pi q : A → ℝ)
    (normalized : ∑ a, pi a = 1) (v : ℝ) :
    risk pi q v = (v - mean pi q)^2 + variance pi q := by
  unfold variance
  rw [risk_expansion, risk_expansion, normalized]
  ring

theorem actor_mean_minimizes_sampling_risk {A : Type*} [Fintype A] (pi q : A → ℝ)
    (normalized : ∑ a, pi a = 1) (v : ℝ) :
    variance pi q ≤ risk pi q v := by
  rw [risk_decomposition pi q normalized v]
  nlinarith [sq_nonneg (v - mean pi q)]

theorem excess_risk_is_value_bias {A : Type*} [Fintype A] (pi q : A → ℝ)
    (normalized : ∑ a, pi a = 1) (v : ℝ) :
    risk pi q v - variance pi q = (v - mean pi q)^2 := by
  rw [risk_decomposition pi q normalized v]
  ring

/-- CausalProgram._value gives critic units 100*g from Wood units 10*g.
Centering cancels in every available-action logit difference. -/
theorem actual_centered_logit_difference (hi hj qi qj center precision : ℝ) :
    (hi + precision*(10*qi-center)) - (hj + precision*(10*qj-center)) =
      hi-hj + 10*precision*(qi-qj) := by ring

/-- Actual binding: learning_scope_cross in audit_native_policy_boundary.py.
v00 uses persisted previous R/Q, v11 uses the newly bound R/Q. The other two
hold Q or R fixed at the SAME current body, candidates, encoder and policy. -/
theorem retained_state_query_interaction (v00 v10 v01 v11 : ℝ) :
    v11-v00 = (v10-v00)+(v01-v00)+(v11-v10-v01+v00) := by ring

end
end CGB.NativeActorExpectation
