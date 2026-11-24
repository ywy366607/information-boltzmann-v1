import Mathlib.Basic.Real.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Ring

/-!
Actual categorical selection accounting: the runtime logs all 3 fire and 4
draft propensities. Q is the expected real consequence of changing a draw,
with current beliefs and the subsequent body/world/learning transition held
as the same program. It can include a finite discounted continuation.

These are fixed-current-state interventional identities, not derivatives or
a claim about retraining an actor under a different policy. A baseline must
be determined before the current draw; Python uses only earlier real gains.
The supported finite action type corresponds to positive logged probabilities.
-/
namespace CGB.LoggedChoiceEffect
open scoped BigOperators

theorem weighted_cell (p q r b : ℝ) (hp : p ≠ 0) :
    p * (r-b) * (1-q/p) = (p-q)*(r-b) := by
  field_simp [hp]

theorem centered_choice {A : Type*} [Fintype A]
    (p q r : A → ℝ) (b : ℝ) (hp : ∀ a, p a ≠ 0)
    (sp : ∑ a, p a = 1) (sq : ∑ a, q a = 1) :
    (∑ a, p a * (r a-b) * (1-q a/p a)) = ∑ a, (p a-q a)*r a := by
  have hzero : ∑ a, (p a-q a) = 0 := by rw [Finset.sum_sub_distrib,sp,sq]; ring
  calc
    _ = ∑ a, (p a-q a)*(r a-b) := by
      apply Finset.sum_congr rfl
      intro a _
      exact weighted_cell _ _ _ _ (hp a)
    _ = (∑ a, (p a-q a)*r a) - ∑ a, (p a-q a)*b := by
      simp_rw [mul_sub,Finset.sum_sub_distrib]
    _ = _ := by rw [← Finset.sum_mul,hzero,zero_mul,sub_zero]

theorem two_stage_cell (p q s t r : ℝ) (hp : p ≠ 0) (hs : s ≠ 0) :
    p*s*r*(1-q/p)*(1-t/s) = (p-q)*(s-t)*r := by
  field_simp [hp,hs]

theorem factorial_baseline_zero {J A : Type*} [Fintype J] [Fintype A]
    (p q : J → ℝ) (s t : J → A → ℝ) (b : ℝ)
    (ss : ∀ j, ∑ a, s j a = 1) (st : ∀ j, ∑ a, t j a = 1) :
    (∑ j, ∑ a, (p j-q j)*(s j a-t j a)*b) = 0 := by
  apply Finset.sum_eq_zero
  intro j _
  have hz : ∑ a, (s j a-t j a) = 0 := by
    rw [Finset.sum_sub_distrib,ss j,st j]; ring
  calc
    _ = ∑ a, ((p j-q j)*b)*(s j a-t j a) := by
      apply Finset.sum_congr rfl; intro a _; ring
    _ = _ := by rw [← Finset.mul_sum,hz,mul_zero]

theorem centered_factorial {J A : Type*} [Fintype J] [Fintype A]
    (p q : J → ℝ) (s t r : J → A → ℝ) (b : ℝ)
    (hp : ∀ j, p j ≠ 0) (hs : ∀ j a, s j a ≠ 0)
    (ss : ∀ j, ∑ a, s j a = 1) (st : ∀ j, ∑ a, t j a = 1) :
    (∑ j, ∑ a, p j*s j a*(r j a-b)*(1-q j/p j)*(1-t j a/s j a)) =
      ∑ j, ∑ a, (p j-q j)*(s j a-t j a)*r j a := by
  calc
    _ = ∑ j, ∑ a, (p j-q j)*(s j a-t j a)*(r j a-b) := by
      apply Finset.sum_congr rfl; intro j _
      apply Finset.sum_congr rfl; intro a _
      exact two_stage_cell _ _ _ _ _ (hp j) (hs j a)
    _ = (∑ j, ∑ a, (p j-q j)*(s j a-t j a)*r j a) -
          ∑ j, ∑ a, (p j-q j)*(s j a-t j a)*b := by
      simp_rw [mul_sub,Finset.sum_sub_distrib]
    _ = _ := by rw [factorial_baseline_zero p q s t b ss st,sub_zero]

end CGB.LoggedChoiceEffect
