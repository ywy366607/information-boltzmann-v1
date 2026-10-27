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

/-! ## Premises of usual data and the components they force (2026-10-27)

W0 (the optimizer is an outer living layer: finite local state, local reads, an infinite
stream without reset, costly memory) constrains realizations; it adds no component.
W1 noisy gradients force memory (`memoryless_gain_excess`). The theorems below cover:
W2 drift forbids a vanishing gain (`vanishing_gain_loses_under_drift`); W3 different
occurrence rates force an occurrence clock (`occurrence_clock_beats_time_clock`); W4
correlated curvature forces a non-diagonal relation basis (`diagonal_cannot_whiten`);
W5 anti-dominance is implied by a per-direction gain falling with that direction's
accumulated use (`usage_normalized_gain_anti_dominant`), so it needs no separate component.
-/

/-- **W2 (drift forbids a vanishing gain)**: the tracking error variance obeys
`V (t+1) = (1 - K t)^2 V t + q + K t^2 r`. With drift `q > 0`, if every gain is at most
`q / (4 M)` the variance exceeds `M`: a gain that decays to zero (1/t, AdaGrad) loses
under drift, whatever bound `M` is asked for. -/
theorem vanishing_gain_loses_under_drift (V K : ℕ → ℝ) (q r M : ℝ) (hq : 0 < q) (hr : 0 ≤ r)
    (hM : 0 < M) (hV0 : 0 ≤ V 0) (hK0 : ∀ t, 0 ≤ K t) (hK : ∀ t, K t ≤ q / (4 * M))
    (hrec : ∀ t, V (t + 1) = (1 - K t) ^ 2 * V t + q + K t ^ 2 * r) :
    ∃ t, M < V t := by
  have hnn : ∀ t, 0 ≤ V t := by
    intro t
    induction t with
    | zero => exact hV0
    | succ n ih =>
      rw [hrec n]
      have := mul_nonneg (sq_nonneg (1 - K n)) ih
      have := mul_nonneg (sq_nonneg (K n)) hr
      linarith
  by_contra h
  push Not at h
  have hstep : ∀ n, V n + q / 2 ≤ V (n + 1) := by
    intro n
    rw [hrec n]
    have hKM : K n * M ≤ q / 4 := by
      have h0 := mul_le_mul_of_nonneg_right (hK n) hM.le
      have he : q / (4 * M) * M = q / 4 := by field_simp
      linarith
    have h1 : K n * V n ≤ K n * M := mul_le_mul_of_nonneg_left (h n) (hK0 n)
    have h2 := mul_nonneg (sq_nonneg (K n)) (hnn n)
    have h3 := mul_nonneg (sq_nonneg (K n)) hr
    nlinarith
  have hgrow : ∀ n : ℕ, V 0 + n * (q / 2) ≤ V n := by
    intro n
    induction n with
    | zero => simp
    | succ n ih =>
      have := hstep n
      push_cast
      linarith
  obtain ⟨n, hn⟩ := exists_nat_gt (M / (q / 2))
  have hq2 : 0 < q / 2 := by linarith
  have := hgrow n
  have : M < n * (q / 2) := by rwa [div_lt_iff₀ hq2] at hn
  linarith [h n, hV0]

/-- Stationary variance of an exponential average with factor `x` per observation of a
constant observed with noise variance `r`. -/
noncomputable def emaVar (r x : ℝ) : ℝ := r * (1 - x) / (1 + x)

/-- It is the fixed point of `v = x^2 v + (1 - x)^2 r`. -/
theorem emaVar_fixed (r x : ℝ) (hx : 0 ≤ x) :
    x ^ 2 * emaVar r x + (1 - x) ^ 2 * r = emaVar r x := by
  unfold emaVar
  have : (1 + x) ≠ 0 := by linarith
  field_simp
  ring

/-- **W3 (an occurrence clock)**: a direction observed once every `k ≥ 2` steps, averaged
with the per-step factor `β` of the frequent directions, effectively uses `β^k` per
observation and has a strictly larger stationary variance than averaging per occurrence. -/
theorem occurrence_clock_beats_time_clock (r β : ℝ) (k : ℕ) (hr : 0 < r) (hβ0 : 0 < β)
    (hβ1 : β < 1) (hk : 2 ≤ k) :
    emaVar r β < emaVar r (β ^ k) := by
  have hpow : β ^ k < β := by
    have := pow_lt_pow_right_of_lt_one₀ hβ0 hβ1 (show 1 < k by omega)
    simpa using this
  have hpk : 0 ≤ β ^ k := pow_nonneg hβ0.le k
  unfold emaVar
  rw [div_lt_div_iff₀ (by linarith) (by linarith)]
  nlinarith

/-- **W4 (a relation basis)**: if the curvature `[[a, b], [b, d]]` has a nonzero
off-diagonal term, no diagonal preconditioner `diag(d₁, d₂)` turns it into a nonzero
multiple of the identity (one-step whitening); a basis-aware preconditioner does. -/
theorem diagonal_cannot_whiten (a b d d₁ d₂ c : ℝ) (hb : b ≠ 0) (hc : c ≠ 0) :
    ¬ (d₁ * a = c ∧ d₁ * b = 0 ∧ d₂ * b = 0 ∧ d₂ * d = c) := by
  rintro ⟨h1, h2, _h3, _h4⟩
  have hd1 : d₁ = 0 := (mul_eq_zero.mp h2).resolve_right hb
  rw [hd1, zero_mul] at h1
  exact hc h1.symm

/-- **W5 is implied by W3**: a gain normalized by that direction's accumulated use
(`s / sqrt (c + m)`, the AdaGrad / Adam form) strictly falls when the direction's own demand
grows: it is locally anti-dominant without a separate anti-winner-take-all component. -/
theorem usage_normalized_gain_anti_dominant (s c m₁ m₂ : ℝ) (hs : 0 < s) (hc : 0 ≤ c)
    (hm₁ : 0 < m₁) (hm : m₁ < m₂) :
    s / Real.sqrt (c + m₂) < s / Real.sqrt (c + m₁) := by
  have h1 : 0 < Real.sqrt (c + m₁) := Real.sqrt_pos.mpr (by linarith)
  have h2 : Real.sqrt (c + m₁) < Real.sqrt (c + m₂) :=
    Real.sqrt_lt_sqrt (by linarith) (by linarith)
  exact div_lt_div_of_pos_left hs h1 h2

end CGB.GradientFilter
