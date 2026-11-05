import CGBAxioms.GenerationUnifiedOptimizer

open CGB.Unified

/-! Local admission for the existing outcome channel interpreted as a
Bernoulli probability. Proper expected scores do not prove that the trained
body is calibrated or that acceptance by this world is beneficial. -/

example (r : ℝ) : 0 < Real.exp r / (1 + Real.exp r) := by positivity

example (r : ℝ) : Real.exp r / (1 + Real.exp r) < 1 := by
  apply (div_lt_one (by positivity : 0 < 1 + Real.exp r)).mpr
  linarith

-- Actual maximization score is negative squared prediction loss.
example (p y : ℝ) : bernoulliBrierScore p y = -(p-y)^2 := by
  unfold bernoulliBrierScore
  ring

example (p q : ℝ) :
    bernoulliExpectedBrier q q - bernoulliExpectedBrier p q = (p-q)^2 :=
  calibrated_evidence_penalty_is_square p q

-- A convex mixture of valid probabilities remains a valid probability.
example (b p w : ℝ) (hb0 : 0 ≤ b) (hb1 : b ≤ 1)
    (hp0 : 0 ≤ p) (hp1 : p ≤ 1) (hw0 : 0 ≤ w) (hw1 : w ≤ 1) :
    0 ≤ b+w*(p-b) ∧ b+w*(p-b) ≤ 1 := by
  constructor
  · nlinarith [mul_nonneg (sub_nonneg.mpr hw1) hb0, mul_nonneg hw0 hp0]
  · nlinarith [mul_nonneg (sub_nonneg.mpr hw1) (sub_nonneg.mpr hb1),
               mul_nonneg hw0 (sub_nonneg.mpr hp1)]

-- Averaging binary labels preserves the p-dependent part of Brier loss,
-- provided all those labels are evaluated at the SAME prediction p.
example (p q : ℝ) :
    q*(p-1)^2 + (1-q)*p^2 = (p-q)^2+q*(1-q) := by ring
