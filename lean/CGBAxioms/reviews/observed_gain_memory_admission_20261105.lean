import CGBAxioms.GenerationFutureValue

/-! Refinement of an observed-gain memory, not a ban on TD return estimation.
Observed gains and TD estimates have different contracts. Keep delayed credit
while preventing estimates from replacing historical observations. No theorem
here establishes stability of the remaining critic/policy learning loop. -/

#check CGB.GDNLife.Process.observed_continuation_improvement_is_empirically_paid

-- An observation slot's update is independent of any proposed return target.
example (g y y' : ℝ) : (fun _ : ℝ => g) y = (fun _ : ℝ => g) y' := rfl

-- Merging measurements preserves their weighted mean, not a fresh TD target.
example (n oldMean g y : ℝ) (h : n + 1 ≠ 0) :
    (n * oldMean + g) / (n + 1) - y =
      (n * (oldMean - y) + (g - y)) / (n + 1) := by
  field_simp
  ring

-- The delayed return remains a separate valid algebraic training target.
example (g gamma nextValue scale : ℝ) (hs : scale ≠ 0) :
    scale * (g + gamma * nextValue / scale) = scale * g + gamma * nextValue := by
  field_simp
