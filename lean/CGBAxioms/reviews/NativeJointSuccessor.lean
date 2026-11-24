import Mathlib.Tactic

/-! Actual binding: joint_successor.coordinate and
successor_joint_reference.residual_scatter. V is the finite evidence space,
Z the real encoder output (128 coordinates in the measured body). Y maps an
output test direction to residuals; J is (XXᵀ+ridge I)⁻¹. Positivity of its
inverse is the regularized dual solve contract. The production numerical
test checks the full matrix's diagonal against the original posterior and
Cholesky checks its positivity on the actual body before execution.
-/
namespace CGB.NativeJointSuccessor
noncomputable section
variable {V Z : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]
    [NormedAddCommGroup Z] [InnerProductSpace ℝ Z]

theorem dual_kernel_strict (ridge : ℝ) (positive : 0<ridge)
    (X : V →ₗ[ℝ] Z) (v : V) (nonzero : v≠0) :
    0 < ridge*‖v‖^2+‖X v‖^2 := by
  have hv : 0<‖v‖ := norm_pos_iff.mpr nonzero
  have h : 0<ridge*‖v‖^2 := mul_pos positive (sq_pos_of_pos hv)
  nlinarith [sq_nonneg ‖X v‖]

theorem retained_joint_scatter_nonnegative
    (J : V →ₗ[ℝ] V) (inverse_positive : ∀ x : V, 0 ≤ inner ℝ x (J x))
    (Y : Z →ₗ[ℝ] V) (ridge : ℝ) (nonnegative : 0≤ridge) (v : Z) :
    0 ≤ ridge * inner ℝ (Y v) (J (Y v)) := by
  exact mul_nonneg nonnegative (inverse_positive (Y v))

theorem common_program_disagreement_nonnegative {K : Type*} [Fintype K]
    (w : K → ℝ) (nonnegative : ∀ k, 0≤w k) (error : K → Z) (v : Z) :
    0≤∑ k, w k * (inner ℝ (error k) v)^2 := by
  apply Finset.sum_nonneg
  intro k _
  exact mul_nonneg (nonnegative k) (sq_nonneg _)

end
end CGB.NativeJointSuccessor
