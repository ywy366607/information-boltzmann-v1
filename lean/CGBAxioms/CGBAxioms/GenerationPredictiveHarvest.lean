import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

/-!
# Prediction is metabolism under reinvestment (language-stream life, T1)

World contract (declared, a niche like the E. coli rain; not a medium axiom): at each arrival
the world pays every unit of capacity that the body had placed on the arrived byte `|B|` units
(fair odds), and the harvest becomes body capacity again (reinvestment). A body that places the
fraction `c b` of its capacity on byte `b` then multiplies its capacity by `|B| * c x` when `x`
arrives, so its expected log growth per arrival against a stream with law `p` is `growth p c`.

* `growth_eq_log_card_sub_crossEntropy`: growth is `log |B|` minus the body's cross-entropy on
  the stream: the compression rate IS the metabolic rate; no observer converts a score.
* `gibbs_inequality`, `growth_le`: no allocation beats `log |B| - H(p)`;
  `growth_at_law`: placing capacity as the stream's law attains it (Kelly).
* `uniform_growth_zero`: a body that knows nothing (uniform allocation) does not grow.
* `no_allocation_persists`, `persists_of_crossEntropy`: with a maintenance cost `m` per arrival
  (log units), no body persists on a stream with `H(p) > log |B| - m`; a body whose
  cross-entropy is below `log |B| - m` grows.
* `additive_payoff_mode`: WITHOUT reinvestment (additive payoff) the best allocation is all on
  a most likely byte; probability matching needs the multiplicative (reinvested) contract.
-/

namespace CGB.PredictiveHarvest

open Finset

variable {B : Type*} [Fintype B]

/-- Expected log growth per arrival of a body placing capacity fractions `c`. -/
noncomputable def growth (p c : B → ℝ) : ℝ := ∑ b, p b * Real.log (Fintype.card B * c b)

/-- Cross-entropy (nats per arrival) of the body's allocation on the stream. -/
noncomputable def crossEntropy (p c : B → ℝ) : ℝ := -∑ b, p b * Real.log (c b)

/-- Entropy of the stream law (nats per arrival); `Real.log 0 = 0` gives `0 log 0 = 0`. -/
noncomputable def entropy (p : B → ℝ) : ℝ := -∑ b, p b * Real.log (p b)

theorem growth_eq_log_card_sub_crossEntropy [Nonempty B] (p c : B → ℝ) (hp : ∑ b, p b = 1)
    (hc : ∀ b, 0 < c b) :
    growth p c = Real.log (Fintype.card B) - crossEntropy p c := by
  have hB : (0 : ℝ) < Fintype.card B := by exact_mod_cast Fintype.card_pos
  unfold growth crossEntropy
  have h : ∀ b, p b * Real.log (Fintype.card B * c b) =
      p b * Real.log (Fintype.card B) + p b * Real.log (c b) := fun b => by
    rw [Real.log_mul hB.ne' (hc b).ne', mul_add]
  rw [Finset.sum_congr rfl fun b _ => h b, Finset.sum_add_distrib, ← Finset.sum_mul, hp]
  ring

/-- One term of Gibbs' inequality: `x (log y - log x) ≤ y - x` (from `log t ≤ t - 1`). -/
theorem gibbs_term (x y : ℝ) (hx : 0 ≤ x) (hy : 0 < y) :
    x * (Real.log y - Real.log x) ≤ y - x := by
  rcases hx.eq_or_lt with h | h
  · subst h; simpa using hy.le
  · have h1 : Real.log (y / x) ≤ y / x - 1 := Real.log_le_sub_one_of_pos (div_pos hy h)
    rw [Real.log_div hy.ne' h.ne'] at h1
    calc x * (Real.log y - Real.log x) ≤ x * (y / x - 1) := mul_le_mul_of_nonneg_left h1 h.le
      _ = y - x := by field_simp

/-- **Gibbs' inequality**: `Σ p log c ≤ Σ p log p` for a positive allocation summing to one. -/
theorem gibbs_inequality (p c : B → ℝ) (hp0 : ∀ b, 0 ≤ p b) (hp : ∑ b, p b = 1)
    (hc : ∀ b, 0 < c b) (hcs : ∑ b, c b = 1) :
    ∑ b, p b * Real.log (c b) ≤ ∑ b, p b * Real.log (p b) := by
  have h := Finset.sum_le_sum fun b (_ : b ∈ (univ : Finset B)) =>
    gibbs_term (p b) (c b) (hp0 b) (hc b)
  rw [Finset.sum_sub_distrib, hcs, hp, sub_self] at h
  have e : ∑ b, p b * (Real.log (c b) - Real.log (p b)) =
      ∑ b, p b * Real.log (c b) - ∑ b, p b * Real.log (p b) := by
    rw [← Finset.sum_sub_distrib]; exact Finset.sum_congr rfl fun b _ => by ring
  linarith

/-- **Kelly bound**: no allocation grows faster than `log |B| - H(p)`. -/
theorem growth_le [Nonempty B] (p c : B → ℝ) (hp0 : ∀ b, 0 ≤ p b) (hp : ∑ b, p b = 1)
    (hc : ∀ b, 0 < c b) (hcs : ∑ b, c b = 1) :
    growth p c ≤ Real.log (Fintype.card B) - entropy p := by
  rw [growth_eq_log_card_sub_crossEntropy p c hp hc]
  unfold crossEntropy entropy
  linarith [gibbs_inequality p c hp0 hp hc hcs]

/-- Placing capacity as the stream's law attains the bound. -/
theorem growth_at_law [Nonempty B] (p : B → ℝ) (hp : ∑ b, p b = 1) :
    growth p p = Real.log (Fintype.card B) - entropy p := by
  have hB : (0 : ℝ) < Fintype.card B := by exact_mod_cast Fintype.card_pos
  unfold growth entropy
  have h : ∀ b, p b * Real.log (Fintype.card B * p b) =
      p b * Real.log (Fintype.card B) + p b * Real.log (p b) := fun b => by
    rcases eq_or_ne (p b) 0 with h0 | h0
    · simp [h0]
    · rw [Real.log_mul hB.ne' h0, mul_add]
  rw [Finset.sum_congr rfl fun b _ => h b, Finset.sum_add_distrib, ← Finset.sum_mul, hp]
  ring

/-- A body that knows nothing about the stream (uniform allocation) does not grow. -/
theorem uniform_growth_zero [Nonempty B] (p : B → ℝ) :
    growth p (fun _ => 1 / (Fintype.card B : ℝ)) = 0 := by
  have hB : (Fintype.card B : ℝ) ≠ 0 := by exact_mod_cast Fintype.card_ne_zero
  unfold growth
  have h_mul : (Fintype.card B : ℝ) * (1 / (Fintype.card B : ℝ)) = 1 :=
    mul_one_div_cancel hB
  simp only [h_mul, Real.log_one, mul_zero, Finset.sum_const_zero]

/-- **No body persists** on a stream whose entropy exceeds `log |B| - m`: every allocation's
net growth (growth minus a maintenance cost `m` per arrival) is negative. -/
theorem no_allocation_persists [Nonempty B] (p : B → ℝ) (m : ℝ) (hp0 : ∀ b, 0 ≤ p b)
    (hp : ∑ b, p b = 1) (hH : Real.log (Fintype.card B) - m < entropy p)
    (c : B → ℝ) (hc : ∀ b, 0 < c b) (hcs : ∑ b, c b = 1) :
    growth p c - m < 0 := by
  linarith [growth_le p c hp0 hp hc hcs]

/-- A body whose cross-entropy is below `log |B| - m` has positive net growth. -/
theorem persists_of_crossEntropy [Nonempty B] (p c : B → ℝ) (m : ℝ) (hp : ∑ b, p b = 1)
    (hc : ∀ b, 0 < c b) (hce : crossEntropy p c < Real.log (Fintype.card B) - m) :
    0 < growth p c - m := by
  rw [growth_eq_log_card_sub_crossEntropy p c hp hc]
  linarith

/-- Without reinvestment (payoff additive in the capacity placed on the arrived byte) the best
allocation puts everything on a most likely byte: probability matching is a consequence of the
multiplicative contract, not of prediction alone. -/
theorem additive_payoff_mode [DecidableEq B] (p c : B → ℝ) (bs : B) (hmode : ∀ b, p b ≤ p bs)
    (hc0 : ∀ b, 0 ≤ c b) (hcs : ∑ b, c b = 1) :
    ∑ b, p b * c b ≤ ∑ b, p b * (if b = bs then (1 : ℝ) else 0) := by
  have h1 : ∑ b, p b * c b ≤ ∑ b, p bs * c b :=
    Finset.sum_le_sum fun b _ => mul_le_mul_of_nonneg_right (hmode b) (hc0 b)
  rw [← Finset.mul_sum, hcs, mul_one] at h1
  have h2 : ∑ b, p b * (if b = bs then (1 : ℝ) else 0) = p bs := by
    rw [Finset.sum_eq_single bs]
    · simp
    · intro b _ hne
      simp [hne]
    · intro h
      exact False.elim (h (Finset.mem_univ bs))
  rw [h2]
  exact h1

end CGB.PredictiveHarvest
