import CGBAxioms.GenerationGrowth

/-!
# A bank of two-state pools keeps a power-law trace (generation chain, cascade memory, 2026-10-27)

G20a (`two_state_geometric`) makes the trace of one two-state pool exponential. A memory that must
stay readable over a wide range of delays cannot be one pool:

* `single_pool_falls_below_power`: one pool's trace `exp(-t/τ)/τ` drops below any `c / t` once
  `t > 2 τ / c` (an exponential beats every power).
* `bank_trace_ge_one_term`, `bank_power_law_lower_bound`: a bank of pools with time constants
  `τ k` and weights `1 / τ k` keeps a trace of at least `1 / (e γ t)` at every `t` for which some
  pool has `t ≤ τ k ≤ γ t`.
* `geometric_bank_covers`: with geometric time constants `τ k = τ₀ γ^k` (`γ > 1`) every
  `t ∈ [τ₀, τ₀ γ^m]` has such a pool, so the bank keeps the `1/t` bound over the whole band, whose
  top grows as `γ^m`: the readable lifetime grows exponentially with the number of levels.

This is the depth-by-age half of a beneficial latch. A linear bank cannot reject bursts: a burst's
time-summed damage equals its length for every linear memory that adopts lasting changes
(`CGB.Continual.linear_burst_damage_integral`); rejection needs a nonlinear switch (G23/G24, latch
health H1/H2). Proving both properties for one state transition of the medium is still open.
-/

namespace CGB.Generation

open Real Finset

/-- One pool's trace drops below `c / t` once `t > 2 τ / c`. -/
theorem single_pool_falls_below_power (τ c t : ℝ) (hτ : 0 < τ) (hc : 0 < c) (ht : 2 * τ / c < t) :
    Real.exp (-t / τ) / τ < c / t := by
  have hcpos : 0 < 2 * τ / c := by positivity
  have htpos : 0 < t := lt_trans hcpos ht
  -- exp (t/τ) > (t/τ)^2 / 2
  have hx : 0 < t / τ := div_pos htpos hτ
  have hsq : (t / τ) ^ 2 / 2 < Real.exp (t / τ) := by
    have := Real.quadratic_le_exp_of_nonneg hx.le
    nlinarith [sq_nonneg (t / τ), Real.add_one_le_exp (t / τ)]
  have hexp : Real.exp (-t / τ) = 1 / Real.exp (t / τ) := by
    rw [neg_div, Real.exp_neg, one_div]
  rw [hexp, div_div, div_lt_div_iff₀ (by positivity) htpos]
  -- t < c * (exp(t/τ) * τ)
  have h2 : 2 * τ < c * t := by rw [div_lt_iff₀ hc] at ht; linarith
  have key : t * 2 * τ ^ 2 < c * t * (t * τ) := by
    have : 0 < t * τ := by positivity
    nlinarith
  have hE : (t / τ) ^ 2 / 2 * τ < Real.exp (t / τ) * τ := by nlinarith
  have : (t / τ) ^ 2 / 2 * τ = t ^ 2 / (2 * τ) := by field_simp
  rw [this] at hE
  have hpos2 : 0 < t ^ 2 / (2 * τ) := by positivity
  calc 1 * t = t := one_mul t
    _ < c * (t ^ 2 / (2 * τ)) := by
        rw [mul_div_assoc']
        rw [lt_div_iff₀ (by positivity)]
        nlinarith
    _ ≤ c * (Real.exp (t / τ) * τ) := by nlinarith

/-- The bank's trace is at least any one of its (nonnegative) terms. -/
theorem bank_trace_ge_one_term {m : ℕ} (τ : ℕ → ℝ) (hτ : ∀ k, 0 < τ k) (t : ℝ) (j : ℕ) (hj : j ∈ range (m + 1)) :
    Real.exp (-t / τ j) / τ j ≤ ∑ k ∈ range (m + 1), Real.exp (-t / τ k) / τ k :=
  single_le_sum (f := fun k => Real.exp (-t / τ k) / τ k) (fun k _ => div_nonneg (Real.exp_pos _).le (hτ k).le) hj

/-- A pool with `t ≤ τ ≤ γ t` alone keeps `1 / (e γ t)`. -/
theorem pool_in_band_bound (τ γ t : ℝ) (ht : 0 < t) (_hγ : 0 < γ) (hlo : t ≤ τ) (hhi : τ ≤ γ * t) :
    1 / (Real.exp 1 * (γ * t)) ≤ Real.exp (-t / τ) / τ := by
  have hτ : 0 < τ := lt_of_lt_of_le ht hlo
  have he : Real.exp (-1) ≤ Real.exp (-t / τ) := by
    apply Real.exp_le_exp.mpr
    rw [neg_div, neg_le_neg_iff, div_le_one hτ]
    exact hlo
  have h1 : Real.exp (-1) = 1 / Real.exp 1 := by rw [Real.exp_neg, one_div]
  rw [h1] at he
  calc 1 / (Real.exp 1 * (γ * t)) = (1 / Real.exp 1) / (γ * t) := by rw [div_div]
    _ ≤ (1 / Real.exp 1) / τ := by
        apply div_le_div_of_nonneg_left (by positivity) hτ hhi
    _ ≤ Real.exp (-t / τ) / τ := div_le_div_of_nonneg_right he hτ.le

theorem bank_power_law_lower_bound {m : ℕ} (τ : ℕ → ℝ) (hτ : ∀ k, 0 < τ k) (γ t : ℝ) (ht : 0 < t)
    (hγ : 0 < γ) (j : ℕ) (hj : j ∈ range (m + 1)) (hlo : t ≤ τ j) (hhi : τ j ≤ γ * t) :
    1 / (Real.exp 1 * (γ * t)) ≤ ∑ k ∈ range (m + 1), Real.exp (-t / τ k) / τ k :=
  (pool_in_band_bound (τ j) γ t ht hγ hlo hhi).trans (bank_trace_ge_one_term τ hτ t j hj)

/-- Geometric time constants cover the band: every `t ∈ [τ₀, τ₀ γ^m]` has a level `k ≤ m` with
`t ≤ τ₀ γ^k ≤ γ t`. -/
theorem geometric_bank_covers (τ₀ γ t : ℝ) (m : ℕ) (hτ₀ : 0 < τ₀) (hγ : 1 < γ)
    (hlo : τ₀ ≤ t) (hhi : t ≤ τ₀ * γ ^ m) :
    ∃ k ∈ range (m + 1), t ≤ τ₀ * γ ^ k ∧ τ₀ * γ ^ k ≤ γ * t := by
  classical
  have hex : ∃ k, t ≤ τ₀ * γ ^ k := ⟨m, hhi⟩
  let k := Nat.find hex
  have hk : t ≤ τ₀ * γ ^ k := Nat.find_spec hex
  have hkm : k ≤ m := Nat.find_min' hex hhi
  refine ⟨k, Finset.mem_range.mpr (by omega), hk, ?_⟩
  rcases Nat.eq_zero_or_pos k with h0 | hpos
  · rw [h0, pow_zero, mul_one]
    nlinarith
  · have hprev : ¬ t ≤ τ₀ * γ ^ (k - 1) := Nat.find_min hex (by omega)
    push Not at hprev
    have : τ₀ * γ ^ k = γ * (τ₀ * γ ^ (k - 1)) := by
      rw [← mul_assoc, mul_comm γ τ₀, mul_assoc, ← pow_succ']
      congr 2
      omega
    rw [this]
    have hγ0 : 0 < γ := by linarith
    nlinarith

end CGB.Generation
