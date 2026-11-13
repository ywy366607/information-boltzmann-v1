import CGBAxioms.GenerationActiveNavigation
import CGBAxioms.LanguageLifeKernel
import Mathlib.Tactic

/-! Local admission: retained re-encoding, convex correction, and same-state
credit obstruction. This does not assume or prove predictive/behavioral benefit;
those are the preregistered real-language tests of the same running loop. -/

example (x y w B : ℝ) (hw0 : 0 ≤ w) (hw1 : w ≤ 1)
    (hx : |x| ≤ B) (hy : |y| ≤ B) : |(1-w)*x+w*y| ≤ B := by
  calc
    |(1-w)*x+w*y| ≤ |(1-w)*x|+|w*y| := abs_add_le _ _
    _ = (1-w)*|x|+w*|y| := by
      rw [abs_mul, abs_mul, abs_of_nonneg (sub_nonneg.mpr hw1), abs_of_nonneg hw0]
    _ ≤ (1-w)*B+w*B := add_le_add
      (mul_le_mul_of_nonneg_left hx (sub_nonneg.mpr hw1))
      (mul_le_mul_of_nonneg_left hy hw0)
    _ = B := by ring

example {E L : ℝ → ℝ} {θ e l : ℝ}
    (hE : HasDerivAt E e θ) (hL : HasDerivAt L l (E θ)) :
    HasDerivAt (fun t => L (E t)) (l*e) θ := by
  simpa using CGB.Generation.ActiveNavigation.reactivation_exact_held (a := 1) 0 hE
    (by simpa using hL)

example {S A C : Type} (advance : S → A → S) (rule : S → C) (s : S)
    (a b : A) (x y : C) (hs : advance s a = advance s b) (hc : x ≠ y) :
    ¬ (rule (advance s a) = x ∧ rule (advance s b) = y) :=
  CGB.LanguageLifeKernel.delayed_action_credit_requires_state_separation
    advance rule s a b x y hs hc
