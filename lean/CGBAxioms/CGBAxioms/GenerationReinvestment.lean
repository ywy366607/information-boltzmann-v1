import Mathlib.Algebra.BigOperators.Field
import Mathlib.Analysis.SpecialFunctions.Log.Deriv
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

/-!
# Reinvestment as one balanced exchange (language organism, 2026-11-03)

User decisions of 2026-11-03: occupancy is reinvested multiplicatively, and metabolism is gross
balanced exchange. `GenerationPredictiveHarvest` states the world contract for a whole body. Here
the capacity is a finite conserved population of shares `a` over addresses; each address bets `w i`
on the arrived byte. The body total is fixed, so only the shares are physical state; the
cumulative wealth of the contract is a bookkeeping number, never stored.

* `reinvest_total`, `gross_balanced`: reinvestment keeps the total, and what the winners gain is
  exactly what the losers give up: a gross balanced exchange with net zero.
* `grows_iff`, `no_exchange_iff`: an address grows exactly when it beat the prediction, and nothing
  is exchanged exactly when every live address agreed with the prediction.
* `mean_multiplier_one`, `reinvest_scale`: the mean multiplier is one for every byte and the
  absolute level of the bets does not matter. This is adaptation of the response to the
  prediction (compare `centred_threshold_resolves`, `integral_feedback_adapts`), with no reference
  state added.
* `wealth_eq_mixture`, `wealth_ge_address`: the product of predictions along a stream equals the
  Bayes mixture of the addresses, so it is at least any one address weighted by its first share.
* `log_predicted_deriv`: the derivative of the arrived byte's log prediction with respect to one
  bet is the share divided by the prediction, which equals the new share divided by the old bet.

None of this is a statement about any body: the bets `w` are fixed here, no transport is included,
and how a table realises the step is a separate obligation.
-/

namespace CGB.Reinvestment

open Finset

variable {A : Type*} [Fintype A]

/-- What the shares `a` predict for the arrived byte, whose bets are `w`. -/
noncomputable def predicted (a w : A → ℝ) : ℝ := ∑ i, a i * w i

/-- One reinvestment step on the arrived byte. -/
noncomputable def reinvest (a w : A → ℝ) (i : A) : ℝ := a i * w i / predicted a w

theorem reinvest_total (a w : A → ℝ) (h : predicted a w ≠ 0) : ∑ i, reinvest a w i = 1 := by
  unfold reinvest
  rw [← Finset.sum_div]
  exact div_self h

/-- Two distributions of the same total: the summed gains equal the summed losses. -/
theorem gains_eq_losses (a b : A → ℝ) (h : ∑ i, a i = ∑ i, b i) :
    ∑ i, max (b i - a i) 0 = ∑ i, max (a i - b i) 0 := by
  have hi : ∀ i, max (b i - a i) 0 - max (a i - b i) 0 = b i - a i := fun i => by
    rcases le_total (a i) (b i) with hle | hle
    · rw [max_eq_left (by linarith), max_eq_right (by linarith)]; ring
    · rw [max_eq_right (by linarith), max_eq_left (by linarith)]; ring
  have hs : ∑ i, (max (b i - a i) 0 - max (a i - b i) 0) = 0 := by
    rw [Finset.sum_congr rfl fun i _ => hi i, Finset.sum_sub_distrib, h]; ring
  rw [Finset.sum_sub_distrib] at hs
  linarith

/-- **Gross balanced exchange**: reinvestment imports into the winners exactly what it exports from
the losers. -/
theorem gross_balanced (a w : A → ℝ) (ha : ∑ i, a i = 1) (h : predicted a w ≠ 0) :
    ∑ i, max (reinvest a w i - a i) 0 = ∑ i, max (a i - reinvest a w i) 0 :=
  gains_eq_losses a (reinvest a w) (by rw [ha, reinvest_total a w h])

/-- An address with a positive share grows exactly when its bet beats the prediction. -/
theorem grows_iff (a w : A → ℝ) (i : A) (hi : 0 < a i) (h : 0 < predicted a w) :
    a i < reinvest a w i ↔ predicted a w < w i := by
  unfold reinvest
  rw [lt_div_iff₀ h]
  exact mul_lt_mul_iff_right₀ hi

/-- **Exact prediction exchanges nothing**, and only then: the shares stay put exactly when every
live address bet the prediction. -/
theorem no_exchange_iff (a w : A → ℝ) (h : predicted a w ≠ 0) :
    (∀ i, reinvest a w i = a i) ↔ ∀ i, a i ≠ 0 → w i = predicted a w := by
  constructor
  · intro hs i hi
    have := hs i
    unfold reinvest at this
    rw [div_eq_iff h] at this
    exact mul_left_cancel₀ hi this
  · intro hs i
    unfold reinvest
    rw [div_eq_iff h]
    by_cases hi : a i = 0
    · simp [hi]
    · rw [hs i hi]

/-- **Adaptation of the total**: the share-weighted multiplier is one for every byte. -/
theorem mean_multiplier_one (a w : A → ℝ) (h : predicted a w ≠ 0) :
    ∑ i, a i * (w i / predicted a w) = 1 := by
  have := reinvest_total a w h
  simpa [reinvest, mul_div_assoc] using this

/-- **Adaptation of the level**: multiplying every bet by the same positive number changes nothing.
Only the bet relative to the prediction acts. -/
theorem reinvest_scale (a w : A → ℝ) (c : ℝ) (hc : c ≠ 0) (i : A) :
    reinvest a (fun j => c * w j) i = reinvest a w i := by
  unfold reinvest predicted
  have : ∑ j, a j * (c * w j) = c * ∑ j, a j * w j := by
    rw [Finset.mul_sum]; exact Finset.sum_congr rfl fun j _ => by ring
  rw [this, mul_left_comm (a i) c, mul_div_mul_left _ _ hc]

theorem predicted_pos [Nonempty A] (a w : A → ℝ) (ha : ∀ i, 0 < a i) (hw : ∀ i, 0 < w i) :
    0 < predicted a w :=
  Finset.sum_pos (fun i _ => mul_pos (ha i) (hw i)) Finset.univ_nonempty

theorem reinvest_pos [Nonempty A] (a w : A → ℝ) (ha : ∀ i, 0 < a i) (hw : ∀ i, 0 < w i) (i : A) :
    0 < reinvest a w i :=
  div_pos (mul_pos (ha i) (hw i)) (predicted_pos a w ha hw)

variable {B : Type*}

/-- Cumulative wealth of the contract along a stream, relative to no bets: a bookkeeping number. -/
noncomputable def wealth (w : B → A → ℝ) : (A → ℝ) → List B → ℝ
  | _, [] => 1
  | a, y :: ys => predicted a (w y) * wealth w (reinvest a (w y)) ys

/-- **Kelly equals Bayes**: the product of the predictions equals the mixture of the addresses'
own products of bets. -/
theorem wealth_eq_mixture [Nonempty A] (w : B → A → ℝ) (hw : ∀ y i, 0 < w y i) (ys : List B)
    (a : A → ℝ) (ha : ∀ i, 0 < a i) (hs : ∑ i, a i = 1) :
    wealth w a ys = ∑ i, a i * (ys.map fun y => w y i).prod := by
  induction ys generalizing a with
  | nil => simp [wealth, hs]
  | cons y ys ih =>
    have hp := predicted_pos a (w y) ha (hw y)
    simp only [wealth, List.map_cons, List.prod_cons]
    rw [ih _ (reinvest_pos a (w y) ha (hw y)) (reinvest_total a (w y) hp.ne'), Finset.mul_sum]
    refine Finset.sum_congr rfl fun i _ => ?_
    unfold reinvest
    field_simp

/-- The wealth is at least what one address would have made from its first share. -/
theorem wealth_ge_address [Nonempty A] (w : B → A → ℝ) (hw : ∀ y i, 0 < w y i) (ys : List B)
    (a : A → ℝ) (ha : ∀ i, 0 < a i) (hs : ∑ i, a i = 1) (i : A) :
    a i * (ys.map fun y => w y i).prod ≤ wealth w a ys := by
  rw [wealth_eq_mixture w hw ys a ha hs]
  have hprod : ∀ j, 0 < (ys.map fun y => w y j).prod := fun j =>
    List.prod_pos (by simp only [List.mem_map]; rintro _ ⟨y, _, rfl⟩; exact hw y j)
  exact Finset.single_le_sum (f := fun j => a j * (ys.map fun y => w y j).prod)
    (fun j _ => (mul_pos (ha j) (hprod j)).le) (Finset.mem_univ i)

/-- Changing one bet changes the prediction by the share times the change. -/
theorem predicted_update [DecidableEq A] (a w : A → ℝ) (i : A) (θ : ℝ) :
    predicted a (Function.update w i (w i + θ)) = predicted a w + a i * θ := by
  unfold predicted
  have h : ∀ j, a j * Function.update w i (w i + θ) j =
      a j * w j + if j = i then a i * θ else 0 := fun j => by
    by_cases hj : j = i
    · subst hj; simp; ring
    · simp [hj]
  rw [Finset.sum_congr rfl fun j _ => h j, Finset.sum_add_distrib]
  simp

/-- **Credit from the same event**: the derivative of the arrived byte's log prediction with respect
to one bet is the share over the prediction. -/
theorem log_predicted_deriv [DecidableEq A] (a w : A → ℝ) (i : A) (h : predicted a w ≠ 0) :
    HasDerivAt (fun θ => Real.log (predicted a (Function.update w i (w i + θ))))
      (a i / predicted a w) 0 := by
  have hf : HasDerivAt (fun θ : ℝ => predicted a w + a i * θ) (a i) 0 := by
    simpa using ((hasDerivAt_id (0 : ℝ)).const_mul (a i)).const_add (predicted a w)
  have hlog := hf.log (by simpa using h)
  simp only [mul_zero, add_zero] at hlog
  exact hlog.congr_of_eventuallyEq (Filter.Eventually.of_forall fun θ => by
    simp only [predicted_update])

/-- The same credit written with the new share: the update needs only the share after the arrival
and the bet itself. -/
theorem credit_eq_new_share (a w : A → ℝ) (i : A) (hw : w i ≠ 0) :
    a i / predicted a w = reinvest a w i / w i := by
  unfold reinvest
  field_simp

end CGB.Reinvestment
