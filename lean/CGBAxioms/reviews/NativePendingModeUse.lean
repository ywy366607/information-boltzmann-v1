import Mathlib.Tactic
import Mathlib.Analysis.Calculus.FDeriv.Add

namespace CGB.NativePendingModeUse
open Matrix

/-- C is the actual H^T M diag(alpha) map from the eight native fact rows.
    CP=C says P covers the reachable measured mode. It does not assert that
    the rest of the full edge field can be held fixed by this input lift. -/
theorem targeted_residual_erases (C : Matrix (Fin 64) (Fin 8) ℝ)
    (P : Matrix (Fin 8) (Fin 8) ℝ) (Y : Matrix (Fin 8) (Fin 128) ℝ)
    (cover : C * P = C) : C * (Y-P*Y) = 0 := by
  rw [Matrix.mul_sub, ← Matrix.mul_assoc, cover, sub_self]

theorem matched_residual_preserves (C : Matrix (Fin 64) (Fin 8) ℝ)
    (Q : Matrix (Fin 8) (Fin 8) ℝ) (Y : Matrix (Fin 8) (Fin 128) ℝ)
    (zero : C * Q = 0) (a : ℝ) : C * (Y+a • (Q*Y)) = C*Y := by
  simp only [Matrix.mul_add, Matrix.mul_smul, ← Matrix.mul_assoc, zero,
    Matrix.zero_mul, smul_zero, add_zero]

theorem matched_norm {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]
    (target control : V) (positive : 0 < ‖control‖) :
    ‖(‖target‖/‖control‖) • control‖ = ‖target‖ := by
  rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg (div_nonneg (norm_nonneg _) (norm_nonneg _))]
  exact div_mul_cancel₀ _ (ne_of_gt positive)

/-- Runtime uses a detached offset; gradients still flow through the old
    fact encoder/other original inputs with the original input Jacobian. -/
theorem fixed_offset_derivative (delta x : (Fin 8 → Fin 128 → ℝ)) :
    HasFDerivAt (fun y => y+delta) (ContinuousLinearMap.id ℝ _) x := by
  simpa using (hasFDerivAt_id x).add_const delta
end CGB.NativePendingModeUse
