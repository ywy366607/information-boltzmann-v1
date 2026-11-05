import CGBAxioms.GenerationActiveNavigation

/-! Output-preserving affine normalization of an existing scalar critic.
This is a numerical reparameterization admission, not a theorem of better
prediction or of invariance of an Adam trajectory. Recalled value stays raw.
-/

example (s s' m m' w b h recalled : ℝ) (hs : s' ≠ 0) :
    s' * ((s / s' * w) * h + (s * b + m - m') / s') + m' + recalled =
      s * (w * h + b) + m + recalled := by
  field_simp
  ring

-- Normalized loss is the same instantaneous raw residual, in different units.
example (s m prediction target : ℝ) (hs : s ≠ 0) :
    (prediction - (target - m) / s)^2 =
      ((s * prediction + m - target) / s)^2 := by
  congr 1
  field_simp
  ring

-- A positive change of units cancels between a value difference and its RMS.
example (c difference rms : ℝ) (hc : c ≠ 0) (hr : rms ≠ 0) :
    (c * difference) / (c * rms) = difference / rms := by
  field_simp

-- Keeping the normalized network fixed while changing only the outer affine
-- map cannot preserve all outputs: agreement at 0 and 1 forces equal maps.
example (s s' m m' : ℝ)
    (h0 : s * 0 + m = s' * 0 + m')
    (h1 : s * 1 + m = s' * 1 + m') : s = s' ∧ m = m' := by
  constructor <;> linarith
