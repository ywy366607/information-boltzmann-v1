import CGBAxioms.GenerationColumnMemory

/-! Local admission for v4p, not a theorem of task performance. Baselines and
confidence are frozen during one forecast; the next observation changes only
the next forecast. Reuse the existing innovation rule for an action address.
-/
open CGB.Generation

example (base correction target : ℝ) :
    (base + correction - target)^2 = (correction - (target-base))^2 := by ring

example (base correction : ℝ) : base + 0 * correction = base := by ring

-- Calibration gives a convex interpolation between the mean and raw forecast.
-- If raw really has lower squared error on an outcome, the interpolation does
-- too. Past error estimates do NOT guarantee this premise on future outcomes.
example (base raw target c : ℝ) (h0 : 0 ≤ c) (h1 : c ≤ 1)
    (hbetter : (raw-target)^2 ≤ (base-target)^2) :
    ((1-c)*base+c*raw-target)^2 ≤ (base-target)^2 := by
  have h := mul_nonneg (mul_nonneg h0 (sub_nonneg.mpr h1)) (sq_nonneg (base-raw))
  have hh := mul_nonneg h0 (sub_nonneg.mpr hbetter)
  nlinarith [sq_nonneg (base-target)]

example {n : ℕ} (slot : Fin n) (row : Fin n → ℝ) (beta target : ℝ) :
    tiedWrite (fun j => if j = slot then 1 else 0) beta target row slot =
      row slot + beta * (target-row slot) := by
  simp [tiedWrite, dot]
