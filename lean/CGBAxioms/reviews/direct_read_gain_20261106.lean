import CGBAxioms.GenerationGradientFilter

/- Local implementation review. C and V are measured, prequential moments of
   (target-baseline)*read_delta and read_delta^2. This does not assume that a
   noisy estimated moment remains valid in a changing world. No production
   theorem count is added by this review. -/
namespace CGB.DirectReadReview

def error (B C V w : ℝ) : ℝ := B - 2 * C * w + V * w ^ 2

example (B C V w : ℝ) (hV : 0 < V) :
    error B C V w = B - C ^ 2 / V + V * (w - C / V) ^ 2 := by
  unfold error
  field_simp
  ring

example (B C V w : ℝ) (hV : 0 < V) :
    error B C V (C / V) ≤ error B C V w := by
  have square (x : ℝ) : error B C V x = B - C ^ 2 / V + V * (x - C / V) ^ 2 := by
    unfold error
    field_simp
    ring
  rw [square, square, sub_self]
  nlinarith [mul_nonneg hV.le (sq_nonneg (w - C / V))]

-- Negative alignment selects zero gain on the nonnegative interval.
example (B C V w : ℝ) (hC : C ≤ 0) (hV : 0 ≤ V) (hw : 0 ≤ w) :
    error B C V 0 ≤ error B C V w := by
  unfold error
  nlinarith [mul_nonpos_of_nonpos_of_nonneg hC hw,
    mul_nonneg hV (sq_nonneg w)]

-- If the unconstrained gain exceeds one, the interval optimum is one.
example (B C V w : ℝ) (hV : 0 ≤ V) (hC : V ≤ C) (hw : w ≤ 1) :
    error B C V 1 ≤ error B C V w := by
  have h := mul_nonneg_of_nonpos_of_nonpos (sub_nonpos.mpr hC) (sub_nonpos.mpr hw)
  have hs := mul_nonneg hV (sq_nonneg (w - 1))
  unfold error
  nlinarith

-- A unit change cancels from the covariance/second-moment gain.
example (C V k : ℝ) (hk : k ≠ 0) :
    (k ^ 2 * C) / (k ^ 2 * V) = C / V := by
  exact mul_div_mul_left C V (pow_ne_zero 2 hk)

end CGB.DirectReadReview
