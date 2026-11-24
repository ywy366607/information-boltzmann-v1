import Mathlib.Tactic

/-! Production binding: native_amortized_decision reads the ORIGINAL schema5
representative points/weights. `p` is that original normalized mass, `v` the
original prospective_decision.value, and `mu` a frozen conditional prediction.
It evaluates residuals at privately sampled original indices. The predictor
and its finite training bank stay fixed through the complete decision; arrived
world settlement commits teacher samples and the actual on-policy label.
-/
namespace CGB.NativeAmortizedDecision
noncomputable section

theorem kernel_entry_feature_identity (dot : ℝ) :
    (1+dot)^2 = 1+2*dot+dot^2 := by ring

theorem correction_expectation {I : Type*} [Fintype I]
    (p mu v : I → ℝ) (normal : ∑ i, p i = 1) :
    (∑ i, p i * ∑ j, p j * mu j) +
      (∑ i, p i * (v i-mu i)) = ∑ i, p i*v i := by
  rw [← Finset.sum_mul, normal, one_mul]
  simp_rw [mul_sub]
  rw [Finset.sum_sub_distrib]
  ring

theorem full_native_recovery {I : Type*} [Fintype I] (p mu v : I → ℝ) :
    (∑ i, p i*mu i) + ∑ i, p i*(v i-mu i) = ∑ i, p i*v i := by
  simp_rw [mul_sub]
  rw [Finset.sum_sub_distrib]
  ring

/- The production A is the weighted polynomial kernel plus ridge*I,
   a is the Cholesky-solved coefficient, y the frozen residual labels. -/
theorem dual_normal_minimum {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]
    (A : V →ₗ[ℝ] V) (symmetric : ∀ u v, inner ℝ u (A v) = inner ℝ v (A u))
    (positive : ∀ v, 0 ≤ inner ℝ v (A v))
    (a y d : V) (normal : A a = y) :
    inner ℝ a (A a) - 2*inner ℝ y a ≤
      inner ℝ (a+d) (A (a+d)) - 2*inner ℝ y (a+d) := by
  have expand : inner ℝ (a+d) (A (a+d)) - 2*inner ℝ y (a+d) -
      (inner ℝ a (A a) - 2*inner ℝ y a) = inner ℝ d (A d) := by
    simp only [map_add, inner_add_left, inner_add_right]
    rw [symmetric a d, normal, real_inner_comm y d]
    ring
  linarith [positive d]

theorem squared_residual_error {I : Type*} [Fintype I]
    (p error : I → ℝ) (nonnegative : ∀ i, 0 ≤ p i) :
    0 ≤ ∑ i, p i * (error i)^2 :=
  Finset.sum_nonneg fun i _ => mul_nonneg (nonnegative i) (sq_nonneg _)

end
end CGB.NativeAmortizedDecision
