import Mathlib.Tactic

/-! Body binding: mechanism adds momentum_to_boundary(P_read) before the
mother_from_boundary map. Old fitted residuals are unchanged. The same base
is now used by epistemic_value_reference.response(query_shift=...). Historical
held-out scores retain the original baseline; only prospective outcomes use P.
Actual certificate: cgb_actual_actor_reference_v5_2489_20261119.json,
candidate_condition_decomposition; source cycle2489, common R/Q/parameters.
-/
namespace CGB.NativeMomentumValueContext
noncomputable section

def prediction (mother : ℝ → ℝ) (f residual shift : ℝ) :=
  f + shift + mother (f+shift) + residual

theorem zero_recovers_original (mother : ℝ → ℝ) (f residual : ℝ) :
    prediction mother f residual 0 = f + mother f + residual := by
  simp [prediction]

theorem same_residual_exact_shift (mother : ℝ → ℝ) (f residual shift : ℝ) :
    prediction mother f residual shift - prediction mother f residual 0 =
      shift + mother (f+shift) - mother f := by
  simp only [prediction, add_zero]
  ring

theorem encoded_projection_matches_raw_base (mother : ℝ → ℝ)
    (f residual rawShift encodedShift : ℝ) (same : rawShift=encodedShift) :
    prediction mother f residual rawShift = prediction mother f residual encodedShift := by
  rw [same]

theorem measured_condition_value_gap :
    (6309569627046585 : ℚ)/10^17 > 6/100 := by norm_num

end
end CGB.NativeMomentumValueContext
