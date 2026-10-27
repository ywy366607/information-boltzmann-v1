import CGBAxioms.Generation

/-!
# An optimizer as a learner of its gradient stream (side branch, 2026-10-27)

The outer loop's world is the stream of noisy gradients. Its harvest is the loss decrease, so
what it must carry from history is what predicts the next true gradient (G10, G11b). This file
records the scalar facts that order the known optimizers; it does not claim any optimizer is
best on every problem.

* `memoryless_gain_excess` (from G10a `fixed_gain_error`): a learner without memory (gain
  `K = 1`, plain SGD on the current gradient) exceeds the optimal tracking error by exactly
  `r^2 / (P + r)`, strictly positive whenever the gradient is noisy (`r > 0`).
* `shrink_error_square`, `shrink_optimum`: estimating a signal of power `S` from an estimate
  with independent noise of power `N` by a factor `w` costs the optimum plus `(S + N)(w - w*)^2`,
  `w* = S / (S + N)` (Wiener). `unshrunk_excess`: taking the estimate at face value (`w = 1`,
  as a normalized step such as Adam's or an orthogonalized one such as Muon's does) costs exactly
  `N^2 / (S + N)` more.
* `no_shared_shrink_for_two_snr`: one shared factor cannot be optimal for two directions with
  different signal fractions (the G10c pattern, now over directions of the update).
-/

namespace CGB.GradientFilter

/-- A learner without memory (gain 1) exceeds the optimal error `P` by `r^2 / (P + r)`. -/
theorem memoryless_gain_excess (r P : ℝ) (hr : 0 < r) (hP : 0 < P) :
    (P ^ 2 / (P + r) + 1 ^ 2 * r) / (1 * (2 - 1)) - P = r ^ 2 / (P + r) := by
  have h := CGB.Medium.fixed_gain_error r P 1 hr hP one_pos (by norm_num)
  rw [h]
  have hPr : P + r ≠ 0 := by positivity
  field_simp
  ring

theorem memoryless_strictly_worse (r P : ℝ) (hr : 0 < r) (hP : 0 < P) :
    P < (P ^ 2 / (P + r) + 1 ^ 2 * r) / (1 * (2 - 1)) := by
  have h := memoryless_gain_excess r P hr hP
  have : 0 < r ^ 2 / (P + r) := by positivity
  linarith

/-- Mean-square error of the scaled estimate `w * (s + n)` of `s`, with `E s^2 = S`,
`E n^2 = N`, `E s n = 0`. -/
def shrinkError (S N w : ℝ) : ℝ := w ^ 2 * (S + N) - 2 * w * S + S

theorem shrink_error_square (S N w : ℝ) (h : 0 < S + N) :
    shrinkError S N w = S * N / (S + N) + (S + N) * (w - S / (S + N)) ^ 2 := by
  unfold shrinkError
  field_simp
  ring

theorem shrink_optimum (S N w : ℝ) (h : 0 < S + N) :
    shrinkError S N (S / (S + N)) ≤ shrinkError S N w := by
  rw [shrink_error_square S N w h, shrink_error_square S N _ h, sub_self]
  nlinarith [sq_nonneg (w - S / (S + N)), h.le]

/-- Taking the noisy estimate at face value costs exactly `N^2 / (S + N)` more. -/
theorem unshrunk_excess (S N : ℝ) (h : 0 < S + N) :
    shrinkError S N 1 - shrinkError S N (S / (S + N)) = N ^ 2 / (S + N) := by
  rw [shrink_error_square S N 1 h, shrink_error_square S N _ h, sub_self]
  field_simp
  ring

/-- One shared factor is strictly suboptimal in at least one of two directions whose signal
fractions differ. -/
theorem no_shared_shrink_for_two_snr (S₁ N₁ S₂ N₂ w : ℝ) (h₁ : 0 < S₁ + N₁) (h₂ : 0 < S₂ + N₂)
    (hne : S₁ / (S₁ + N₁) ≠ S₂ / (S₂ + N₂)) :
    shrinkError S₁ N₁ (S₁ / (S₁ + N₁)) < shrinkError S₁ N₁ w ∨
      shrinkError S₂ N₂ (S₂ / (S₂ + N₂)) < shrinkError S₂ N₂ w := by
  rw [shrink_error_square S₁ N₁ w h₁, shrink_error_square S₁ N₁ _ h₁,
    shrink_error_square S₂ N₂ w h₂, shrink_error_square S₂ N₂ _ h₂]
  simp only [sub_self, ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true, zero_pow, mul_zero,
    add_zero]
  by_cases hw : w = S₁ / (S₁ + N₁)
  · right
    have : w - S₂ / (S₂ + N₂) ≠ 0 := by rw [hw]; exact sub_ne_zero.mpr hne
    have := pow_pos (abs_pos.mpr this) 2
    rw [sq_abs] at this
    nlinarith
  · left
    have : w - S₁ / (S₁ + N₁) ≠ 0 := sub_ne_zero.mpr hw
    have := pow_pos (abs_pos.mpr this) 2
    rw [sq_abs] at this
    nlinarith

end CGB.GradientFilter
