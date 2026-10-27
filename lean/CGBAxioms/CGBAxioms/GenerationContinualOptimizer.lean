import CGBAxioms.GenerationGradientFilter

/-!
# A continual optimizer: competition, no winner-take-all, a healthy latch (side branch, 2026-10-27)

The optimizer of a stream learner must (i) make units compete so that they differentiate,
(ii) keep one unit from taking everything (G19a, G22c), (iii) hold confirmed structure through
bursts that revert (G24 in optimizer form, `no_linear_gain_separates`), and (iv) keep that hold
releasable and sensing (H1, H2). This file derives the allocation and the latch.

**Allocation.** Units (rows) and input directions (columns of the relation basis) share update
mass `A i j ≥ 0`, and rich-get-richer (positive feedback, G18/G23) favours concentrated
allocations: the payoff `Σ A^2` is convex.

* `polar_conserves_direction_mass`: an orthogonal update (Muon's polar factor, `Oᵀ O = 1`) has
  unit mass along every input direction. Muon is a per-direction conservation law: scarcity
  along each direction, hence competition among rows for it.
* `column_budget_allows_wta`, `row_budget_allows_copies`: with only the per-direction budget the
  payoff maximum `2` is reached by one row taking every direction (winner-take-all); with only a
  per-row budget (the local reservoir of G22) it is reached by every row taking the same
  direction (copies, no differentiation).
* `doubly_budgeted_form`, `doubly_budgeted_max_iff_permutation`: with both budgets the
  allocations are `[[p, 1-p], [1-p, p]]` and the payoff maximum is reached exactly at the
  permutations: every row owns one direction and every direction one row. Competition plus local
  reservoirs is differentiation without winner-take-all.

**Latch.**

* `linear_protect_implies_slow`: a linear gain that limits the damage of a burst of length `c`
  to `ε` adopts at most `1 - (1 - ε)^m` of a lasting change after `m c` steps; a rule that
  defers for `c` steps and then adopts fully has damage `0` and adoption `1` (`deferLatch_*`).
* `deferLatch_holds`, `deferLatch_releases`: the deferral latch holds through any run of
  contrary evidence shorter than its window and releases after a run of the window's length:
  its exit is reachable (H1) and depends on the stimulus (not blind, H2).
* `held_budget_starves`: if consolidated entries keep their share of a conserved budget and
  consolidation is never released, the free budget falls to zero after `B / s` consolidations
  (lock-in of the whole layer). A consolidated entry must give up its claim on the budget.
-/

namespace CGB.Continual

open Finset

/-! ## Allocation -/

/-- Orthonormal columns carry unit mass along every input direction. -/
theorem polar_conserves_direction_mass {m n : ℕ} (O : Matrix (Fin m) (Fin n) ℝ)
    (h : O.transpose * O = 1) (j : Fin n) : ∑ i, O i j ^ 2 = 1 := by
  have := congrFun (congrFun h j) j
  simpa [Matrix.mul_apply, sq, Matrix.one_apply] using this

/-- Payoff of positive feedback: concentrated allocations pay more. -/
def payoff (A : Fin 2 → Fin 2 → ℝ) : ℝ := ∑ i, ∑ j, A i j ^ 2

def ColBudget (A : Fin 2 → Fin 2 → ℝ) : Prop := ∀ j, A 0 j + A 1 j = 1
def RowBudget (A : Fin 2 → Fin 2 → ℝ) : Prop := ∀ i, A i 0 + A i 1 = 1
def Nonneg (A : Fin 2 → Fin 2 → ℝ) : Prop := ∀ i j, 0 ≤ A i j

/-- One row takes every direction. -/
def wta : Fin 2 → Fin 2 → ℝ := fun i _ => if i = 0 then 1 else 0
/-- Every row takes the same direction. -/
def copies : Fin 2 → Fin 2 → ℝ := fun _ j => if j = 0 then 1 else 0

theorem payoff_le_two_of_col {A : Fin 2 → Fin 2 → ℝ} (hn : Nonneg A) (hc : ColBudget A) :
    payoff A ≤ 2 := by
  unfold payoff
  simp only [Fin.sum_univ_two]
  have h0 := hc 0
  have h1 := hc 1
  have := hn 0 0; have := hn 0 1; have := hn 1 0; have := hn 1 1
  nlinarith [mul_nonneg (hn 0 0) (hn 1 0), mul_nonneg (hn 0 1) (hn 1 1)]

theorem payoff_le_two_of_row {A : Fin 2 → Fin 2 → ℝ} (hn : Nonneg A) (hr : RowBudget A) :
    payoff A ≤ 2 := by
  unfold payoff
  simp only [Fin.sum_univ_two]
  have h0 := hr 0
  have h1 := hr 1
  have := hn 0 0; have := hn 0 1; have := hn 1 0; have := hn 1 1
  nlinarith [mul_nonneg (hn 0 0) (hn 0 1), mul_nonneg (hn 1 0) (hn 1 1)]

theorem column_budget_allows_wta :
    Nonneg wta ∧ ColBudget wta ∧ payoff wta = 2 ∧ ¬ RowBudget wta := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · intro i j; unfold wta; split_ifs <;> norm_num
  · intro j; simp [wta]
  · simp [payoff, wta]
  · intro h; have := h 0; simp [wta] at this

theorem row_budget_allows_copies :
    Nonneg copies ∧ RowBudget copies ∧ payoff copies = 2 ∧ ¬ ColBudget copies := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · intro i j; unfold copies; split_ifs <;> norm_num
  · intro i; simp [copies]
  · simp [payoff, copies]
  · intro h; have := h 0; simp [copies] at this

theorem doubly_budgeted_form {A : Fin 2 → Fin 2 → ℝ} (hc : ColBudget A) (hr : RowBudget A) :
    A 1 1 = A 0 0 ∧ A 0 1 = 1 - A 0 0 ∧ A 1 0 = 1 - A 0 0 := by
  have := hc 0; have := hc 1; have := hr 0; have := hr 1
  refine ⟨?_, ?_, ?_⟩ <;> linarith

theorem doubly_budgeted_max_iff_permutation {A : Fin 2 → Fin 2 → ℝ} (_hn : Nonneg A)
    (hc : ColBudget A) (hr : RowBudget A) : payoff A = 2 ↔ (A 0 0 = 1 ∨ A 0 0 = 0) := by
  obtain ⟨h11, h01, h10⟩ := doubly_budgeted_form hc hr
  have hp : payoff A = 2 * A 0 0 ^ 2 + 2 * (1 - A 0 0) ^ 2 := by
    simp only [payoff, Fin.sum_univ_two, h11, h01, h10]; ring
  rw [hp]
  constructor
  · intro h
    have : A 0 0 * (A 0 0 - 1) = 0 := by nlinarith
    rcases mul_eq_zero.mp this with h0 | h0
    · right; exact h0
    · left; linarith
  · rintro (h | h) <;> rw [h] <;> norm_num

/-- At a doubly budgeted maximum the two rows own different directions. -/
theorem doubly_budgeted_max_differentiates {A : Fin 2 → Fin 2 → ℝ} (hn : Nonneg A)
    (hc : ColBudget A) (hr : RowBudget A) (h : payoff A = 2) : A 0 0 ≠ A 1 0 := by
  obtain ⟨_, _, h10⟩ := doubly_budgeted_form hc hr
  rcases (doubly_budgeted_max_iff_permutation hn hc hr).mp h with h0 | h0 <;>
    rw [h10, h0] <;> norm_num

/-! ## Latch -/

theorem linear_protect_implies_slow (K ε : ℝ) (c m : ℕ) (hε : ε ≤ 1)
    (hprot : 1 - (1 - K) ^ c ≤ ε) : 1 - (1 - K) ^ (m * c) ≤ 1 - (1 - ε) ^ m := by
  have h0 : 0 ≤ 1 - ε := by linarith
  have hb : 1 - ε ≤ (1 - K) ^ c := by linarith
  have : (1 - ε) ^ m ≤ ((1 - K) ^ c) ^ m := pow_le_pow_left₀ h0 hb m
  rw [← pow_mul, mul_comm c m] at this
  linarith

/-- Deferral latch: holds the old value until `c` consecutive contrary observations, then
adopts. `run` is the length of the current contrary run. -/
def deferLatch (c : ℕ) (run : ℕ) (v d : ℝ) : ℝ := if c ≤ run then d else v

theorem deferLatch_holds (c run : ℕ) (v d : ℝ) (h : run < c) : deferLatch c run v d = v := by
  unfold deferLatch; rw [if_neg (by omega)]

theorem deferLatch_releases (c run : ℕ) (v d : ℝ) (h : c ≤ run) : deferLatch c run v d = d := by
  unfold deferLatch; rw [if_pos h]

/-- The latch is not blind: the same held state gives different outputs for different
evidence runs (H2 is avoided), and every hold ends after `c` contrary steps (H1). -/
theorem deferLatch_senses (c : ℕ) (v d : ℝ) (hvd : v ≠ d) :
    deferLatch c 0 v d ≠ deferLatch c c v d ∨ c = 0 := by
  rcases Nat.eq_zero_or_pos c with h | h
  · right; exact h
  · left; rw [deferLatch_holds c 0 v d h, deferLatch_releases c c v d le_rfl]; exact hvd

/-- Consolidated entries that keep a share `s` of a conserved budget `B` and are never
released leave `B - n s` free after `n` consolidations; it is gone once `n s ≥ B`. -/
theorem held_budget_starves (B s : ℝ) (n : ℕ) (hn : B ≤ n * s) :
    B - n * s ≤ 0 := by linarith

/-! ## Measuring the own step (found by the v3 mechanism check, 2026-10-27)

The step `a = s + c ξ'` depends on the old batch's noise `ξ'`; the difference of gradients on two
batches is `h a + ξ - ξ'`. Averaged over the noise pairs, the secant numerator is
`h · E[a^2] - c y^2`: biased by the dependence of the step on the noise it was made from
(`secant_own_noise_bias`). The gradient at the old and new parameters on one fresh batch differs
by exactly `h a` (`secant_same_batch_exact`): one extra gradient on the next batch measures the
own step without bias (efference copy, secant, and noise residual at fixed parameters).
-/

theorem secant_own_noise_bias (h s c x y : ℝ) :
    ((h * (s + c * y) + x - y) * (s + c * y) + (h * (s + c * y) - x - y) * (s + c * y) +
      (h * (s - c * y) + x + y) * (s - c * y) + (h * (s - c * y) - x + y) * (s - c * y)) / 4 =
      h * (((s + c * y) ^ 2 + (s - c * y) ^ 2) / 2) - c * y ^ 2 := by
  ring

theorem secant_same_batch_exact (h a ξ : ℝ) : ((h * a + ξ) - ξ) * a = h * a ^ 2 := by
  ring

/-! ## Occurrence must run through every estimate (v4, 2026-10-27)

A row occurs in a batch with probability `p`. Its gradient, curvature and the decrease of a step
differ between batches where it occurs (`1`) and where it does not (`0`).

* `fresh_decrease_mixture`, `conditional_prediction_excess`: the expected decrease of a step on the
  next batch is `p d₁ + (1 - p) d₀`; predicting it from the occurrence-conditional `d₁` alone
  overstates it by `(1 - p)(d₁ - d₀)`, large for rare rows. Scored on fresh data, every step on a
  rare row then looks like a failure and the shared bound collapses. Each row needs its
  occurrence rate and both conditional estimates, each ticking on its own events (W3 for both
  components of the mixture), and the step uses the mixture `p m₁ + (1 - p) m₀`.
* `shared_reference_flips`: judging occurrence against a reference
  shared by the rows (a fraction of the layer's largest row) lets another row's growth erase this
  row's occurrence: one shared potential, the rich-club pattern of G19a. A decision against the
  row's own reference does not depend on any other row by construction (G22c); no theorem is
  needed for that.
* Curvature: a row without curvature data steps at the shared bound; one bound cannot fit two
  validity radii (`no_fixed_bound_two_radii`), so every row needs a curvature, from a prior
  shared inside the clique until its own secant exists.
-/

theorem fresh_decrease_mixture (p d₁ d₀ : ℝ) :
    p * d₁ + (1 - p) * d₀ = d₀ + p * (d₁ - d₀) := by ring

theorem conditional_prediction_excess (p d₁ d₀ : ℝ) :
    d₁ - (p * d₁ + (1 - p) * d₀) = (1 - p) * (d₁ - d₀) := by ring

/-- Shared reference: row `x` occurs when `x ≥ f · max x y`. Growing the other row erases it. -/
theorem shared_reference_flips (f x : ℝ) (hf : 0 < f) (hx : 0 < x) (hfx : f * x ≤ x) :
    x ≥ f * max x x ∧ ¬ (x ≥ f * max x (2 * x / f)) := by
  constructor
  · simpa using hfx
  · intro h
    have hm : 2 * x / f ≤ max x (2 * x / f) := le_max_right _ _
    have : f * (2 * x / f) = 2 * x := by field_simp
    nlinarith [mul_le_mul_of_nonneg_left hm hf.le]

/-! ## v4/v5: what the v3 run and the calibration checks forced (2026-10-27)

* `expected_gradient_is_mixture`: the per-step mean of a row's gradient is the occurrence mixture
  `p g₁ + (1 - p) g₀`, so a filter updated every step estimates exactly what the expected fresh
  decrease needs (`fresh_decrease_mixture`); no occurrence detection is required for the step.
  W3 (`occurrence_clock_beats_time_clock`) stays true but concerns conditional estimates, which
  the step does not use.
* `positive_part_bias`: with no signal and a calibrated variance, `m^2 - P` averages to `0` but
  the per-step plug-in `max(m^2 - P, 0)` averages to `P / 2`: noise-only directions take steps
  whose expected decrease is negative. Clip after averaging.
* `consistent_prediction_excess`: the step acts on the shrunk mean `ĝ = f m` (`0 ≤ f ≤ 1`);
  predicting its decrease with `m` overstates it by `(1 - f) m u ≥ 0` when `u` has the sign of
  `m`. The prediction must use the belief the step uses.
* `softmax_rows_coupled`, `softmax_gradients_sum_zero`: for a softmax output the gradients of the
  class rows sum to zero and the output factor of the curvature has the off-diagonal entry
  `-p (1 - p) ≠ 0` between two classes. An input-side basis then leaves cross-row curvature and the
  per-element secant is not exact (`secant_exact_iff`): the output side needs its own basis, from
  the post-synaptic error covariance `E[δ δᵀ]` (inside the clique). Under a squared loss that
  factor is the identity and the input-side basis alone suffices (`eigenbasis_decouples`).
-/

theorem expected_gradient_is_mixture (p g₁ g₀ : ℝ) :
    p * g₁ + (1 - p) * g₀ = g₀ + p * (g₁ - g₀) := by ring

/-- Zero signal, calibrated variance: `m` is `0, 0, c, -c` with `c^2 = 2P`, so `E[m^2] = P`. The
average of `m^2 - P` is `0`; the average of the per-step plug-in `max(m^2 - P, 0)` is `P / 2`. -/
theorem positive_part_bias (P : ℝ) (hP : 0 < P) :
    ((0 - P) + (0 - P) + (2 * P - P) + (2 * P - P)) / 4 = 0 ∧
      (max (0 - P) 0 + max (0 - P) 0 + max (2 * P - P) 0 + max (2 * P - P) 0) / 4 = P / 2 := by
  refine ⟨by ring, ?_⟩
  rw [max_eq_right (by linarith), max_eq_left (by linarith)]
  ring

theorem consistent_prediction_excess (m u f h : ℝ) (hf : f ≤ 1) (hmu : 0 ≤ m * u) :
    (m * u - h * u ^ 2 / 2) - (f * m * u - h * u ^ 2 / 2) = (1 - f) * (m * u) ∧
      0 ≤ (1 - f) * (m * u) := by
  refine ⟨by ring, mul_nonneg (by linarith) hmu⟩

/-- Two-class softmax: the class-row gradients `(p - y)` and `((1 - p) - (1 - y))` sum to zero. -/
theorem softmax_gradients_sum_zero (p y : ℝ) : (p - y) + ((1 - p) - (1 - y)) = 0 := by ring

/-- The output factor of the softmax curvature, `diag p - p pᵀ` for two classes, has the
off-diagonal entry `-p (1 - p)`, nonzero for `0 < p < 1`. -/
theorem softmax_rows_coupled (p : ℝ) (h0 : 0 < p) (h1 : p < 1) : -(p * (1 - p)) ≠ 0 := by
  have : 0 < p * (1 - p) := mul_pos h0 (by linarith)
  linarith

/-! ## v5: sparse rows, events and rates (2026-10-27)

The v4 calibration check located the remaining softmax bias: head rows are predicted within 4%,
tail rows ten times too high. A tail row's gradient is a spike every few hundred steps.

* `ema_forgets_rare_spikes`: an average with factor `β` holds `β^k` of a spike after `k` quiet
  steps; for the registered `β = 0.95` and spikes 200 steps apart this is below the spike's share
  `1/200` of the long-run power.
  A short-window noise estimate is far too small just before the next spike, the gain is near 1,
  and the estimate jumps to the spike: the expected gradient of a rare row is overstated.
* `partition_is_unbiased`: split each sample's contribution into an event part and a background
  part by any rule that looks only at that sample; the two expected parts add to the expected
  gradient whatever the rule. The rule changes efficiency, never the target, so it is a
  registered choice and not a correction.
* `event_part_is_rate_times_size`: the expected event part per step is the event rate times the
  mean event contribution. The rate is a count (it ticks every step); the size ticks only on
  events (W3, `occurrence_clock_beats_time_clock`, now applied to the object it is about).
  Events come from the per-sample post-synaptic error δ (a label that occurs, a unit that fires),
  the post-synaptic factor that the three-factor rule already provides.
-/

/-- The registered window (factor 0.95) with spikes 200 steps apart: the held share 0.95^200 is
below the spike's share 1/200 of the long-run power. -/
theorem ema_forgets_rare_spikes : ((19 : ℝ) / 20) ^ 200 < 1 / 200 := by norm_num

theorem partition_is_unbiased {ι : Type*} (s : Finset ι) (w g : ι → ℝ) (ev : ι → Bool) :
    ∑ i ∈ s, w i * g i =
      ∑ i ∈ s, w i * (if ev i then g i else 0) + ∑ i ∈ s, w i * (if ev i then 0 else g i) := by
  rw [← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun i _ => ?_
  cases ev i <;> simp

theorem event_part_is_rate_times_size (n B μ : ℝ) (hB : B ≠ 0) :
    (n * μ) / B = (n / B) * μ := by
  field_simp

/-! ## A beneficial latch, not only a healthy one (from the v4 run, 2026-10-27)

H1 and H2 make a latch healthy: it can always be left and it keeps sensing. The v4 run showed a
healthy latch can still cost: a hard latch (gain zero) froze a drifting backbone (task C: 7.96 vs
1.73 without latch), and a fixed window held a 16-step burst as long as a 500-step backbone. In a
world of slow drift plus rare revertible jumps (W7) a latch is *beneficial* when, besides H1 and
H2 and giving up its budget (`held_budget_starves`), it

* **B1 passes drift**: a frozen estimate under a random walk has an error that grows without
  bound (`frozen_hold_loses_on_drift`), so small innovations must still be written;
* **B2 bounds influence**: under a mixture of small noise and rare large jumps the Bayes update
  is `y · w(y)` with `w(y) = 1 / (1 + c exp(a y^2))`; it is below `1 / (c a y)`, so the influence
  of a single large innovation goes to zero (`mixture_update_bounded`), while a linear gain's
  grows without bound; at `y = 0` the weight is `1 / (1 + c)`, near full for a small jump prior;
* **B3 deepens with age**: if the prior odds of a lasting change fall with the age of the held
  value (a decreasing hazard, as for the heavy-tailed durations of language), the number of
  consecutive contrary observations needed to release never decreases with age
  (`release_threshold_monotone_in_age`); with a constant hazard it is constant, so a fixed
  window is right only in a memoryless world. The rule of succession `1 / (T + 2)` (a uniform
  prior on an unknown constant hazard) gives odds `1 / (T + 1)`, decreasing
  (`succession_odds_decreasing`). The depth of the hold is the continuous shell again, now indexed
  by the age of the evidence.
-/

theorem frozen_hold_loses_on_drift (e₀ q M : ℝ) (hq : 0 < q) : ∃ n : ℕ, M < e₀ + n * q := by
  obtain ⟨n, hn⟩ := exists_nat_gt ((M - e₀) / q)
  refine ⟨n, ?_⟩
  rw [div_lt_iff₀ hq] at hn
  linarith

theorem mixture_update_bounded (c a y : ℝ) (hc : 0 < c) (ha : 0 < a) (hy : 0 < y) :
    y / (1 + c * Real.exp (a * y ^ 2)) < 1 / (c * a * y) := by
  have he : a * y ^ 2 < Real.exp (a * y ^ 2) := by
    have := Real.add_one_lt_exp (show a * y ^ 2 ≠ 0 by positivity)
    linarith
  have hd : c * a * y ^ 2 < 1 + c * Real.exp (a * y ^ 2) := by nlinarith
  have hpos : 0 < c * a * y := by positivity
  rw [div_lt_div_iff₀ (by positivity) hpos]
  nlinarith

theorem mixture_weight_at_zero (c a : ℝ) : 1 / (1 + c * Real.exp (a * 0 ^ 2)) = 1 / (1 + c) := by
  simp

theorem release_threshold_monotone_in_age (o : ℕ → ℝ) (hanti : Antitone o) (L : ℝ) (k T₁ T₂ : ℕ)
    (hL : 0 ≤ L) (hT : T₁ ≤ T₂) (hrel : 1 ≤ o T₂ * L ^ k) : 1 ≤ o T₁ * L ^ k := by
  have := hanti hT
  have hp : 0 ≤ L ^ k := pow_nonneg hL k
  nlinarith

theorem succession_odds_decreasing : Antitone (fun T : ℕ => (1 : ℝ) / ((T : ℝ) + 1)) := by
  intro a b hab
  have : (a : ℝ) ≤ b := by exact_mod_cast hab
  apply one_div_le_one_div_of_le (by positivity)
  linarith

/-! ## A linear cascade cannot reject bursts (2026-10-27)

A linear time-invariant memory (one pool, or a Benna-Fusi chain of pools with growing time
constants, whose mean dynamics are linear) answers a lasting step input with its step response
`a n`. A burst of length `L` is a step up at `0` and a step down at `L`, so its response is
`a n - a (n - L)` (zero before the step down). If the memory adopts lasting changes fully, `a n = 1`
from some `N` on, then the burst's damage summed over time is exactly `L`, whatever the filter
(`linear_burst_damage_integral`): a linear cascade can delay and spread the damage of a burst,
never reduce it. Depth by age (B3) can come from a cascade; rejection of bursts (B2) needs a
nonlinear switch (G23/G24), as in the deferral latch (`deferLatch_holds`: zero damage for bursts
shorter than its window).
-/

/-- Response of a linear memory to a burst of length `L`, from its step response `a`. -/
def burstResponse (a : ℕ → ℝ) (L n : ℕ) : ℝ := a n - (if L ≤ n then a (n - L) else 0)

theorem linear_burst_damage_integral (a : ℕ → ℝ) (N L : ℕ) (hfull : ∀ n, N ≤ n → a n = 1) :
    ∑ n ∈ Finset.range (N + L), burstResponse a L n = L := by
  unfold burstResponse
  rw [Finset.sum_sub_distrib]
  have hshift : ∑ n ∈ Finset.range (N + L), (if L ≤ n then a (n - L) else 0) =
      ∑ n ∈ Finset.range N, a n := by
    rw [add_comm N L, Finset.sum_range_add]
    have h0 : ∑ n ∈ Finset.range L, (if L ≤ n then a (n - L) else 0) = 0 :=
      Finset.sum_eq_zero fun n hn => by
        rw [Finset.mem_range] at hn
        rw [if_neg (by omega)]
    rw [h0, zero_add]
    refine Finset.sum_congr rfl fun n _ => ?_
    rw [if_pos (by omega)]
    congr 1
    omega
  rw [hshift, Finset.sum_range_add, add_sub_cancel_left]
  rw [Finset.sum_congr rfl fun i _ => hfull (N + i) (by omega)]
  simp

/-! ## v6: what the hold must take from the clique (2026-10-27)

CLO5 on a stationary task kept re-accepting single outliers as changes: the rule of succession
`1/(T+2)` (a uniform prior on the change rate: "every second step may change" at age 0) let a
single jump-like observation through whenever its value was young, the acceptance reset the age,
and the accepted value was given the jump's prior variance.

* `beta_hazard_decreasing`, `beta_hazard_succession`: with a Beta(α, β) prior on the change rate
  the hazard after `T` quiet observations is `α / (α + β + T)`: still decreasing (B3 holds) and
  equal to the rule of succession at `α = β = 1`. The clique pools thousands of element histories,
  so `α, β` are learned (empirical Bayes); in a world without lasting changes `α` is small and a
  single outlier is never accepted.
* `accepted_run_variance`: after a run of `k` observations with noise variance `s` the new value
  is known to within the variance of their mean, `s / k`; the jump-size prior no longer applies.
-/

theorem beta_hazard_decreasing (α β : ℝ) (hα : 0 < α) (hβ : 0 ≤ β) :
    Antitone (fun T : ℕ => α / (α + β + T)) := by
  intro a b hab
  have : (a : ℝ) ≤ b := by exact_mod_cast hab
  apply div_le_div_of_nonneg_left hα.le (by positivity)
  linarith

theorem beta_hazard_succession (T : ℕ) : (1 : ℝ) / (1 + 1 + T) = 1 / ((T : ℝ) + 2) := by
  ring_nf

theorem accepted_run_variance (s : ℝ) (k : ℕ) (hk : 0 < k) :
    (∑ _i ∈ Finset.range k, s) / (k : ℝ) ^ 2 = s / k := by
  rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul]
  have : (k : ℝ) ≠ 0 := by exact_mod_cast hk.ne'
  field_simp

/-! ## The step acts on the posterior mean without a second shrink (2026-10-27)

`optimal_step_newton_times_signal` gives the shrink `S / (S + N)` for one raw observation with no
other knowledge: the shrink is the prior "the gradient is small" applied once. The filter's `m` is
already a posterior mean with variance `P`. Under that posterior, the expected squared error of the
action `f m` is `P + (1 - f)^2 m^2` (`posterior_shrink_error`), smallest only at `f = 1`
(`posterior_mean_is_bayes_action`): a second shrink counts the prior again at every step, and as
the gradient shrinks near the optimum the steps stall (task B, CLO5: the signal fraction's median
is 0 from step 50 on). The prior belongs in the filter's start (`m₀ = 0`, `P₀` from the clique),
once.
-/

/-- Posterior of the gradient as the two points `m ± √P` (mean `m`, variance `P`): the mean squared
error of the action `f m` is `P + (1 - f)^2 m^2`. -/
theorem posterior_shrink_error (m P f : ℝ) (hP : 0 ≤ P) :
    ((m + Real.sqrt P - f * m) ^ 2 + (m - Real.sqrt P - f * m) ^ 2) / 2 = P + (1 - f) ^ 2 * m ^ 2 := by
  have hs := Real.sq_sqrt hP
  nlinarith [hs]

theorem posterior_mean_is_bayes_action (m P f : ℝ) (hm : m ≠ 0) (hf : f ≠ 1) :
    P + (1 - 1) ^ 2 * m ^ 2 < P + (1 - f) ^ 2 * m ^ 2 := by
  have h1 : (1 - f) ≠ 0 := sub_ne_zero.mpr (Ne.symm hf)
  have : 0 < (1 - f) ^ 2 * m ^ 2 := by positivity
  simpa using this

end CGB.Continual
