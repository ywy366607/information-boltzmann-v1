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

/-! ## The step form (2026-10-27)

Along one direction take the local quadratic `L = h e^2 / 2` (`e` the distance to the optimum)
and an observed gradient `g = h e + ξ` with noise variance `σ2`. One step `e' = e - η g` has
`E e'^2 = (1 - η h)^2 e^2 + η^2 σ2` (`stepErr`).

* `step_error_square`: the error is the optimum plus `(h^2 e^2 + σ2) (η - η*)^2`,
  `η* = h e^2 / (h^2 e^2 + σ2)`.
* `optimal_step_newton_times_signal`: `η* = (1 / h) * S / (S + N)` with `S = (h e)^2` the power
  of the true gradient and `N = σ2`: a curvature scale times the signal fraction.
* `no_fixed_step_two_distances`: with noise, the optimal step differs at two distances, so a
  fixed step needs a schedule; a schedule tied to the total length is excluded by W0, so the
  signal fraction must be estimated online.
* `no_optimal_step_negative_curvature` (W6, nonconvex data): with `h < 0` the loss along the
  step is unbounded below, so the quadratic step does not exist and the step must be bounded.
-/

/-- Mean-square distance to the optimum after one step of size `η`. -/
def stepErr (h e σ2 η : ℝ) : ℝ := (1 - η * h) ^ 2 * e ^ 2 + η ^ 2 * σ2

theorem step_error_square (h e σ2 η : ℝ) (hA : 0 < h ^ 2 * e ^ 2 + σ2) :
    stepErr h e σ2 η = e ^ 2 * σ2 / (h ^ 2 * e ^ 2 + σ2) +
      (h ^ 2 * e ^ 2 + σ2) * (η - h * e ^ 2 / (h ^ 2 * e ^ 2 + σ2)) ^ 2 := by
  unfold stepErr
  field_simp
  ring

theorem optimal_step_minimizes (h e σ2 η : ℝ) (hA : 0 < h ^ 2 * e ^ 2 + σ2) :
    stepErr h e σ2 (h * e ^ 2 / (h ^ 2 * e ^ 2 + σ2)) ≤ stepErr h e σ2 η := by
  rw [step_error_square h e σ2 η hA, step_error_square h e σ2 _ hA, sub_self]
  nlinarith [sq_nonneg (η - h * e ^ 2 / (h ^ 2 * e ^ 2 + σ2)), hA.le]

theorem optimal_step_newton_times_signal (h e σ2 : ℝ) (hh : h ≠ 0) :
    h * e ^ 2 / (h ^ 2 * e ^ 2 + σ2) = (1 / h) * ((h * e) ^ 2 / ((h * e) ^ 2 + σ2)) := by
  rw [mul_pow]
  field_simp

/-- With noise, the optimal step differs at two different distances to the optimum. -/
theorem no_fixed_step_two_distances (h σ2 x y : ℝ) (hh : 0 < h) (hσ : 0 < σ2)
    (hx : 0 ≤ x) (hy : 0 ≤ y) (hxy : x ≠ y) :
    h * x / (h ^ 2 * x + σ2) ≠ h * y / (h ^ 2 * y + σ2) := by
  intro heq
  have hdx : 0 < h ^ 2 * x + σ2 := by positivity
  have hdy : 0 < h ^ 2 * y + σ2 := by positivity
  rw [div_eq_div_iff hdx.ne' hdy.ne'] at heq
  have : h * σ2 * (x - y) = 0 := by linear_combination heq
  rcases mul_eq_zero.mp this with h0 | h0
  · exact absurd h0 (mul_pos hh hσ).ne'
  · exact hxy (sub_eq_zero.mp h0)

/-- **W6 (nonconvex data)**: with negative curvature the loss along the step is unbounded
below, so no optimal quadratic step exists and the step must be bounded. -/
theorem no_optimal_step_negative_curvature (h e g M : ℝ) (hh : h < 0) (hg : g ≠ 0) :
    ∃ η : ℝ, h * (e - η * g) ^ 2 / 2 < M := by
  obtain ⟨n, hn⟩ := exists_nat_gt ((|e| + Real.sqrt (2 * |M| / (-h)) + 1) / |g|)
  refine ⟨n * (if 0 < g then 1 else -1), ?_⟩
  have hgabs : 0 < |g| := abs_pos.mpr hg
  have hlarge : Real.sqrt (2 * |M| / (-h)) + |e| + 1 < |n * (if 0 < g then 1 else -1) * g| := by
    have : |n * (if 0 < g then (1 : ℝ) else -1) * g| = n * |g| := by
      split_ifs with hpos
      · rw [mul_one, abs_mul, Nat.abs_cast]
      · rw [abs_mul, abs_mul, Nat.abs_cast, abs_neg, abs_one, mul_one]
    rw [this]
    rw [div_lt_iff₀ hgabs] at hn
    linarith
  have hd : Real.sqrt (2 * |M| / (-h)) + 1 < |e - n * (if 0 < g then 1 else -1) * g| := by
    have := abs_sub_abs_le_abs_sub (n * (if 0 < g then (1 : ℝ) else -1) * g) e
    rw [abs_sub_comm] at this
    linarith
  have hs0 : 0 ≤ Real.sqrt (2 * |M| / (-h)) := Real.sqrt_nonneg _
  have hsq : 2 * |M| / (-h) < (e - n * (if 0 < g then 1 else -1) * g) ^ 2 := by
    have h1 : Real.sqrt (2 * |M| / (-h)) < |e - n * (if 0 < g then 1 else -1) * g| := by linarith
    have h2 := Real.sq_sqrt (show 0 ≤ 2 * |M| / (-h) by
      apply div_nonneg (by positivity) (by linarith))
    rw [← sq_abs (e - _)]
    nlinarith
  have hneg : 0 < -h := by linarith
  rw [div_lt_iff₀ hneg] at hsq
  nlinarith [le_abs_self M, neg_abs_le M]

end CGB.GradientFilter
