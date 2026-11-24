import Mathlib.Basic.Real.Basic
import Mathlib.Data.Fintype.Fin
import Mathlib.Tactic.FinCases
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring
import Mathlib.Analysis.Normed.Group.Basic

/-!
# The executed budgeted AdamW step (2026-11-15)

This is the arithmetic contract of `budgeted_adamw.py`, on arbitrary real tensor
entries. Fin 2 denotes the four coarse cells inside EACH physical tensor tile,
not an artificial learning task. The same allocation is broadcast over all
entries of a cell. Adam's proposal d is an input, including its real m/v state;
no descent, learned functional diversity, or brain equivalence is assumed.

The coherence observer uses matching EMA weights for gradient energy and m.
Its stationary independent-noise interpretation is an external statistical
premise, not an identity about correlated language gradients.
-/
namespace CGB.BudgetedPlasticity

/-- For an inherited moment, ||m|| <= old_survivor_bound + ||recent||.
    The implementation stores the old RMS, not the old vector. -/
theorem inherited_moment_lower_bound (observed oldBound recent : ℝ)
    (hr : 0 ≤ recent) (ht : observed ≤ oldBound + recent) :
    max 0 (observed - oldBound) ≤ recent := by
  exact max_le hr (by linarith)

/-- The reverse triangle inequality supplies the companion upper bound.
    Recent = observed - surviving old moment; no old vector is stored. -/
theorem inherited_moment_upper_bound {E : Type*} [SeminormedAddCommGroup E]
    (observed survivingOld recent : E) (oldBound : ℝ)
    (hdecomposition : observed = survivingOld + recent)
    (hold : ‖survivingOld‖ ≤ oldBound) :
    ‖recent‖ ≤ ‖observed‖ + oldBound := by
  have hrecent : recent = observed - survivingOld := by
    rw [hdecomposition]
    abel
  rw [hrecent]
  have htriangle := norm_sub_le observed survivingOld
  linarith

noncomputable def cool (signal energy noiseWeight : ℝ) : ℝ :=
  max 0 (min 1 ((signal - noiseWeight * energy) / ((1 - noiseWeight) * energy)))

theorem cool_bounds (s e c : ℝ) : 0 ≤ cool s e c ∧ cool s e c ≤ 1 := by
  constructor
  · exact le_max_left _ _
  · exact max_le (by norm_num) (min_le_left _ _)

/-- Actual Torch/Triton cooling is monotone in the signal-power estimate. -/
theorem cool_mono_signal (s upper e c : ℝ) (he : 0 < e) (hc : c < 1)
    (hs : s ≤ upper) : cool s e c ≤ cool upper e c := by
  have hd : 0 < (1 - c) * e := mul_pos (by linarith) he
  unfold cool
  apply max_le_max (le_refl 0)
  apply min_le_min (le_refl 1)
  apply (div_le_div_iff_of_pos_right hd).2
  linarith

/-- With uncertain inherited direction, using the upper bound never cools
    more than an observer that had the unavailable exact recent signal. -/
theorem upper_bound_cooling_is_nonpremature (recent upper e c : ℝ)
    (hr : 0 ≤ recent) (hu : recent ≤ upper) (he : 0 < e) (hc : c < 1) :
    cool (recent * recent) e c ≤ cool (upper * upper) e c := by
  apply cool_mono_signal _ _ _ _ he hc
  nlinarith

theorem noise_dominated_holds (s e c : ℝ) (he : 0 < e) (hc : c < 1)
    (hs : s ≤ c * e) : cool s e c = 0 := by
  have hd : 0 < (1 - c) * e := mul_pos (by linarith) he
  have hx : (s - c * e) / ((1 - c) * e) ≤ 0 :=
    div_nonpos_of_nonpos_of_nonneg (by linarith) hd.le
  unfold cool
  exact max_eq_left (le_trans (min_le_right _ _) hx)

theorem coherent_evidence_reopens (e c : ℝ) (he : 0 < e) (hc : c < 1) :
    cool e e c = 1 := by
  have hd : (1 - c) * e ≠ 0 := ne_of_gt (mul_pos (by linarith) he)
  have hx : (e - c * e) / ((1 - c) * e) = 1 := by
    rw [show e - c * e = (1 - c) * e by ring, div_self hd]
  simp [cool, hx]

def phaseNext (β old cue : ℝ) : ℝ := β * old + (1 - β) * cue

theorem phase_bounds (β old cue : ℝ) (hb : 0 ≤ β) (hb1 : β ≤ 1)
    (ho : -1 ≤ old ∧ old ≤ 1) (hc : -1 ≤ cue ∧ cue ≤ 1) :
    -1 ≤ phaseNext β old cue ∧ phaseNext β old cue ≤ 1 := by
  have h1 := mul_le_mul_of_nonneg_left ho.1 hb
  have h2 := mul_le_mul_of_nonneg_left ho.2 hb
  have h3 := mul_le_mul_of_nonneg_left hc.1 (show 0 ≤ 1 - β by linarith)
  have h4 := mul_le_mul_of_nonneg_left hc.2 (show 0 ≤ 1 - β by linarith)
  unfold phaseNext
  constructor <;> nlinarith

def allocation (phase : ℝ) (i j : Fin 2) : ℝ :=
  if i = j then 1 + phase else 1 - phase

theorem allocation_bounds (p : ℝ) (hp : -1 ≤ p ∧ p ≤ 1) (i j : Fin 2) :
    0 ≤ allocation p i j ∧ allocation p i j ≤ 2 := by
  unfold allocation
  split_ifs <;> constructor <;> linarith

theorem row_budget (p : ℝ) (i : Fin 2) :
    allocation p i 0 + allocation p i 1 = 2 := by
  fin_cases i <;> norm_num [allocation]

theorem column_budget (p : ℝ) (j : Fin 2) :
    allocation p 0 j + allocation p 1 j = 2 := by
  fin_cases j <;> norm_num [allocation]

theorem spent_and_idle (p k : ℝ) (i : Fin 2) :
    k * allocation p i 0 + k * allocation p i 1 + 2 * (1 - k) = 2 := by
  calc
    _ = k * (allocation p i 0 + allocation p i 1) + 2 * (1 - k) := by ring
    _ = 2 := by rw [row_budget]; ring

theorem idle_nonnegative (k : ℝ) (hk : k ≤ 1) : 0 ≤ 2 * (1 - k) := by linarith

/-- This is the actual parameter operation, including local AdamW decay. -/
def nextParameter (θ d η decay gate phase : ℝ) (i j : Fin 2) : ℝ :=
  θ - η * gate * (allocation phase i j * d + decay * θ)

theorem closed_gate_preserves_weight (θ d η decay phase : ℝ) (i j : Fin 2) :
    nextParameter θ d η decay 0 phase i j = θ := by simp [nextParameter]

theorem neutral_recovers_adamw (θ d η decay : ℝ) (i j : Fin 2) :
    nextParameter θ d η decay 1 0 i j = θ * (1 - η * decay) - η * d := by
  simp only [nextParameter, allocation]
  split_ifs <;> ring

theorem executed_displacement (θ d η decay k p : ℝ) (i j : Fin 2) :
    θ - nextParameter θ d η decay k p i j =
      η * k * (allocation p i j * d + decay * θ) := by
  unfold nextParameter
  ring

theorem proposal_is_not_reversed (p k d : ℝ) (hp : -1 ≤ p ∧ p ≤ 1)
    (hk : 0 ≤ k) (i j : Fin 2) : 0 ≤ d * (k * allocation p i j * d) := by
  have h := mul_nonneg hk (allocation_bounds p hp i j).1
  rw [show d * (k * allocation p i j * d) =
    (k * allocation p i j) * d ^ 2 by ring]
  exact mul_nonneg h (sq_nonneg d)

end CGB.BudgetedPlasticity
