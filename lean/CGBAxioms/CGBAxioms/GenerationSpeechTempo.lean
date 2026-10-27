import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Analysis.SpecialFunctions.Log.Basic

/-!
# Speech tempo between two finite bodies (generation chain, 2026-10-28, edge N7)

A speaker emits units (bytes, syllables) at rate `r` per tick to a listener. Three finite
resources meet, each already forced elsewhere:

* the listener absorbs at most `C` units per tick (finite write budget per tick: Landauer and the
  write-fee contract), so what arrives above `C` is lost: `absorbed = min r C`;
* a unit is useful only together with what came before it (sequence structure needs memory at
  both ends, G19c), and memory decays geometrically (G20a, `GenerationExport.survive_le_pow`);
  across the gap `1 / r` the earlier part survives with `retention = exp (-1 / (r τ))`;
* the speaker's internal state settles toward its prediction at a geometric rate (G13; the unified
  reference's `(1 - α) ^ A` total-variation bound); with `k` internal reactions per tick it gets
  `k / r` reactions per unit, so a unit is settled to `settled = 1 - (1 - α) ^ (k / r)
  = 1 - exp (-(k L / r))`, `L = -log (1 - α)` (`unsettled_eq_rpow`).

The useful rate is the product. Results:

* `useful_le_rate`, `useful_le_capacity`, `useful_le_inv_rate`: the useful rate is at most `r`,
  at most `C`, and at most `C k L / r`.
* **N7 `tempo_band`**: whatever useful rate `u > 0` some tempo reaches, every tempo below `u` or
  above `C k L / u` does worse. Speaking too slowly and too quickly both lose; the good band is
  bounded by the listener's capacity `C` and the speaker's thinking speed `k L` together.
* `slow_retention_vanishes`: per unit, too slow a tempo leaves nothing of the earlier part
  ("forgot the beginning"), whatever the listener's capacity.
* `fast_settling_vanishes`: per unit, too fast a tempo leaves the speaker unsettled ("said it
  before thinking it through"), whatever the listener's capacity.
* `peaks_wasted`, `uniform_attains` (uniform information density): with a per-slot capacity `C`,
  `n` slots carrying `x i` absorb at most `min (Σ x) (n C)`, and the even spread attains it.
  Peaks above capacity are lost; spreading information evenly is forced, not a style.

Nothing here prices speech: there is no cost term. The losses come from the two bodies' own
finite capacity, memory and settling. A writer with no capacity limit, no forgetting and no
settling (an LLM reading and writing for free) is outside every hypothesis, which is why it does
not feel these limits; the reader still does (`peaks_wasted` holds for the reader's `C`).
-/

namespace CGB.SpeechTempo

open Finset

/-- Units the listener absorbs per tick. -/
noncomputable def absorbed (C r : ℝ) : ℝ := min r C

/-- What survives of the previous unit across the gap `1 / r` (memory time constant `τ`). -/
noncomputable def retention (τ r : ℝ) : ℝ := Real.exp (-(1 / (r * τ)))

/-- How settled a unit is when the speaker has `k / r` reactions of rate `L` for it. -/
noncomputable def settled (k L r : ℝ) : ℝ := 1 - Real.exp (-(k * L / r))

/-- The useful rate. -/
noncomputable def useful (C τ k L r : ℝ) : ℝ := absorbed C r * retention τ r * settled k L r

/-- The settling factor is the unified reference's geometric bound: `(1 - α) ^ (k / r)` with
`L = -log (1 - α)`. -/
theorem unsettled_eq_rpow (α k r : ℝ) (hα1 : α < 1) :
    (1 - α) ^ (k / r) = Real.exp (-(k * (-Real.log (1 - α)) / r)) := by
  rw [Real.rpow_def_of_pos (by linarith)]
  congr 1
  ring

theorem retention_pos (τ r : ℝ) : 0 < retention τ r := Real.exp_pos _

theorem retention_le_one (τ r : ℝ) (hr : 0 < r) (hτ : 0 < τ) : retention τ r ≤ 1 := by
  unfold retention
  rw [Real.exp_le_one_iff]
  have : 0 < 1 / (r * τ) := by positivity
  linarith

theorem settled_nonneg (k L r : ℝ) (h : 0 ≤ k * L / r) : 0 ≤ settled k L r := by
  unfold settled
  have : Real.exp (-(k * L / r)) ≤ 1 := by
    rw [Real.exp_le_one_iff]
    linarith
  linarith

theorem settled_le_one (k L r : ℝ) : settled k L r ≤ 1 := by
  unfold settled
  linarith [Real.exp_pos (-(k * L / r))]

theorem settled_le (k L r : ℝ) : settled k L r ≤ k * L / r := by
  unfold settled
  linarith [Real.add_one_le_exp (-(k * L / r))]

section Bounds

variable (C τ k L r : ℝ) (hC : 0 ≤ C) (hτ : 0 < τ) (hkL : 0 ≤ k * L) (hr : 0 < r)
include hC hτ hkL hr

theorem useful_le_rate : useful C τ k L r ≤ r := by
  have hs0 := settled_nonneg k L r (div_nonneg hkL hr.le)
  have hs1 := settled_le_one k L r
  have hq0 := (retention_pos τ r).le
  have hq1 := retention_le_one τ r hr hτ
  have hm0 : 0 ≤ absorbed C r := le_min hr.le hC
  have hm1 : absorbed C r ≤ r := min_le_left _ _
  unfold useful
  have h1 : absorbed C r * retention τ r ≤ r * 1 := mul_le_mul hm1 hq1 hq0 hr.le
  have h2 := mul_le_mul h1 hs1 hs0 (by linarith)
  linarith

theorem useful_le_capacity : useful C τ k L r ≤ C := by
  have hs0 := settled_nonneg k L r (div_nonneg hkL hr.le)
  have hs1 := settled_le_one k L r
  have hq0 := (retention_pos τ r).le
  have hq1 := retention_le_one τ r hr hτ
  unfold useful
  have h1 : absorbed C r * retention τ r ≤ C * 1 := mul_le_mul (min_le_right _ _) hq1 hq0 hC
  have h2 := mul_le_mul h1 hs1 hs0 (by linarith)
  linarith

theorem useful_le_inv_rate : useful C τ k L r ≤ C * k * L / r := by
  have hs0 := settled_nonneg k L r (div_nonneg hkL hr.le)
  have hq0 := (retention_pos τ r).le
  have hq1 := retention_le_one τ r hr hτ
  unfold useful
  have h1 : absorbed C r * retention τ r ≤ C * 1 := mul_le_mul (min_le_right _ _) hq1 hq0 hC
  have h2 := mul_le_mul h1 (settled_le k L r) hs0 (by linarith)
  calc absorbed C r * retention τ r * settled k L r ≤ C * 1 * (k * L / r) := h2
    _ = C * k * L / r := by ring

/-- **N7**: the good tempo is a bounded band. If some tempo reaches the useful rate `u > 0`, every
tempo slower than `u` or faster than `C k L / u` does worse. -/
theorem tempo_band (u : ℝ) (hu : 0 < u) (h : r < u ∨ C * k * L / u < r) :
    useful C τ k L r < u := by
  rcases h with h | h
  · exact lt_of_le_of_lt (useful_le_rate C τ k L r hC hτ hkL hr) h
  · refine lt_of_le_of_lt (useful_le_inv_rate C τ k L r hC hτ hkL hr) ?_
    rw [div_lt_iff₀ hr]
    have := (div_lt_iff₀ hu).mp h
    linarith

end Bounds

/-- Too slow: per unit, the earlier part is forgotten, whatever the listener's capacity. -/
theorem slow_retention_vanishes (τ ε : ℝ) (hτ : 0 < τ) (hε : 0 < ε) :
    ∃ r₀ > 0, ∀ r, 0 < r → r < r₀ → retention τ r < ε := by
  set M := |Real.log ε| + 1 with hMdef
  have hM : 0 < M := by positivity
  refine ⟨1 / (τ * M), by positivity, fun r hr hr0 => ?_⟩
  have h1 : r * (τ * M) < 1 := by
    have := (lt_div_iff₀ (by positivity : (0 : ℝ) < τ * M)).mp hr0
    linarith
  have h2 : M < 1 / (r * τ) := by
    rw [lt_div_iff₀ (by positivity)]
    nlinarith
  have h3 : -(1 / (r * τ)) < Real.log ε := by
    linarith [neg_abs_le (Real.log ε)]
  exact (Real.exp_lt_exp.mpr h3).trans_eq (Real.exp_log hε)

/-- Too fast: per unit, the speaker is unsettled, whatever the listener's capacity. -/
theorem fast_settling_vanishes (k L ε : ℝ) (hkL : 0 ≤ k * L) (hε : 0 < ε) :
    ∃ r₁, ∀ r, r₁ < r → settled k L r < ε := by
  refine ⟨k * L / ε, fun r hr => ?_⟩
  have hr0 : 0 < r := lt_of_le_of_lt (div_nonneg hkL hε.le) hr
  refine lt_of_le_of_lt (settled_le k L r) ?_
  rw [div_lt_iff₀ hr0]
  have := (div_lt_iff₀ hε).mp hr
  linarith

/-- Uniform information density, upper bound: with a per-slot capacity `C`, the slots absorb at
most the total sent and at most `n C`; anything above capacity in a slot is lost. -/
theorem peaks_wasted {n : ℕ} (x : Fin n → ℝ) (C : ℝ) :
    ∑ i, min (x i) C ≤ min (∑ i, x i) (n * C) := by
  refine le_min (sum_le_sum fun i _ => min_le_left _ _) ?_
  calc ∑ i, min (x i) C ≤ ∑ _i : Fin n, C := sum_le_sum fun i _ => min_le_right _ _
    _ = n * C := by simp

/-- Uniform information density, attainment: the even spread reaches the bound. -/
theorem uniform_attains (n : ℕ) (hn : 0 < n) (I C : ℝ) :
    ∑ _i : Fin n, min (I / n) C = min I (n * C) := by
  have hn0 : (0 : ℝ) < n := by exact_mod_cast hn
  simp only [sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul]
  rw [mul_min_of_nonneg _ _ hn0.le]
  congr 1
  field_simp

end CGB.SpeechTempo
