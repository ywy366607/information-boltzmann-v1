import Mathlib.Tactic
import Mathlib.Analysis.Calculus.FDeriv.Add
namespace CGB.NativeTransportCurlUse
open Matrix
/-- C = P_c E signed-M diag(alpha) on the admitted 981-edge body carrier.
The before/after curl difference T is lifted through eight actual fact rows.
N is joint-null for C and the original posterior mean map A. -/
theorem paired_curl (C : Matrix (Fin 981) (Fin 8) ℝ)
    (Y L N K : Matrix (Fin 8) (Fin 128) ℝ)
    (T : Matrix (Fin 981) (Fin 128) ℝ)
    (lift : C * L = -T) (null : C * N = 0) (control : C * K = 0) :
    C * (Y + (L + N)) = C * Y - T ∧ C * (Y + K) = C * Y := by
  constructor <;> simp [Matrix.mul_add, lift, null, control, sub_eq_add_neg]
theorem paired_mean (A : Matrix (Fin 1) (Fin 8) ℝ)
    (Y L N K : Matrix (Fin 8) (Fin 128) ℝ)
    (null : A * N = 0) (matched : A * L = A * K) :
    A * (Y + (L + N)) = A * (Y + K) := by
  simp [Matrix.mul_add, null, matched]
theorem fixed_offset_derivative (delta x : (Fin 8 → Fin 128 → ℝ)) :
    HasFDerivAt (fun y => y + delta) (ContinuousLinearMap.id ℝ _) x := by
  simpa using (hasFDerivAt_id x).add_const delta
end CGB.NativeTransportCurlUse
