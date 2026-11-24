import Mathlib.Tactic

namespace CGB.NativeLargestGroupFactOrder
open Matrix

/-- Exact original-to-intervened positions of the eight native facts. -/
example : ([0, 7, 6, 1, 2, 3, 4, 5] : List Nat).Perm (List.range 8) := by decide

/-- Once solve coefficients transform with the actual fact rows, the native
    query-X-transpose-coefficient mean is independent of their presentation. -/
theorem native_mean_permutation
    (q : Matrix (Fin 1) (Fin 4) ℝ)
    (X : Matrix (Fin 8) (Fin 4) ℝ)
    (C : Matrix (Fin 8) (Fin 128) ℝ)
    (P : Matrix (Fin 8) (Fin 8) ℝ)
    (orthogonal : P.transpose * P = 1) :
    q * (P * X).transpose * (P * C) = q * X.transpose * C := by
  simp only [Matrix.transpose_mul, Matrix.mul_assoc]
  rw [← Matrix.mul_assoc P.transpose P C, orthogonal, Matrix.one_mul]

/-- The conditional quadrature must permute the observation-noise rows too.
    Permuting only X and its solve is not the same finite sample path. -/
theorem native_draw_permutation
    (X : Matrix (Fin 8) (Fin 4) ℝ)
    (W : Matrix (Fin 4) (Fin 128) ℝ)
    (E : Matrix (Fin 8) (Fin 128) ℝ)
    (P : Matrix (Fin 8) (Fin 8) ℝ) :
    P * (X * W + E) = (P * X) * W + P * E := by
  rw [Matrix.mul_add, Matrix.mul_assoc]

end CGB.NativeLargestGroupFactOrder

namespace CGB.NativeLargestGroupFactOrder
open Matrix

/-- The actual 8-fact, rank-4, 128-output conditional operator is unchanged
    when the keyed noise and dual solution follow the same fact permutation. -/
theorem native_conditional_operator_permutation
    (W : Matrix (Fin 4) (Fin 128) ℝ)
    (X : Matrix (Fin 8) (Fin 4) ℝ)
    (L : Matrix (Fin 8) (Fin 128) ℝ)
    (P : Matrix (Fin 8) (Fin 8) ℝ)
    (orthogonal : P.transpose * P = 1) :
    W - (P * X).transpose * (P * L) = W - X.transpose * L := by
  simp only [Matrix.transpose_mul, Matrix.mul_assoc]
  rw [← Matrix.mul_assoc P.transpose P L, orthogonal, Matrix.one_mul]

end CGB.NativeLargestGroupFactOrder
