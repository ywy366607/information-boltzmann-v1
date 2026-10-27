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

end CGB.Continual
