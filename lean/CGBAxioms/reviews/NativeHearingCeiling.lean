import Mathlib.Basic.Real.Basic
import Mathlib.Tactic.Linarith

namespace CGB.NativeHearingCeiling

/-- Two equiprobable operator draws. Guess a draw, then take its preferred
fixed-menu action. A Bayes response rule does at least as well as this decoder.
If its errors are rare, perfect identification has little value left to add. -/
theorem ceiling_gap (v₀ v₁ e₀ e₁ loss₀ loss₁ cap adaptive : ℝ)
    (p₀ : 0 ≤ e₀) (p₁ : 0 ≤ e₁) (b₀ : loss₀ ≤ cap) (b₁ : loss₁ ≤ cap)
    (feasible : ((v₀ - e₀ * loss₀) + (v₁ - e₁ * loss₁)) / 2 ≤ adaptive) :
    (v₀ + v₁) / 2 - adaptive ≤ (e₀ * cap + e₁ * cap) / 2 := by
  linarith [mul_le_mul_of_nonneg_left b₀ p₀, mul_le_mul_of_nonneg_left b₁ p₁]

/-- Same uniform-prior continuation value can hide different preferences in
individual draws: hearing is useful only for a payoff-relevant distinction. -/
theorem common_best (v₀ v₁ : ℝ) (q₀ q₁ : ℝ) (h₀ : q₀ ≤ v₀) (h₁ : q₁ ≤ v₁) :
    (q₀ + q₁) / 2 ≤ (v₀ + v₁) / 2 := by linarith

end CGB.NativeHearingCeiling
