import CGBAxioms.GenerationActiveNavigation

/-! v4q local algebra. The existing branch_residual_common_invariant motivates
separating a common level from action contrasts. These examples verify the
executed normalized innovation, not identification of causal action effects.
No claim of future accuracy follows from sharing an innovation. -/

-- Changing the gauge does not change any forecast.
example (c d shift : ℝ) : (c + shift) + (d - shift) = c + d := by ring

-- Updating one of three centered contrasts preserves their zero sum.
example (a b c u : ℝ) (h : a + b + c = 0) :
    (a + u - u / 3) + (b - u / 3) + (c - u / 3) = 0 := by linarith

-- Feature [1, one_hot - 1/3] has squared norm 5/3. Selected forecast
-- therefore receives exactly gain * innovation, as in the old EMA.
example (base target gain : ℝ) :
    base + gain * (target-base) / (5/3) * (1 + 2/3) =
      base + gain * (target-base) := by ring

-- Other forecasts move by 2/5 of the selected increment.
example (err gain : ℝ) :
    gain * err / (5/3) * (1 - 1/3) = (2/5) * gain * err := by ring

-- Replay the residual with its own decision-time baseline, not today's one.
example (target oldBase newBase : ℝ) :
    (target-newBase) - (target-oldBase) = oldBase-newBase := by ring
