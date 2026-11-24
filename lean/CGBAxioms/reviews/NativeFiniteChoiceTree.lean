import Mathlib.Tactic

/-! Body binding: navigation_generative_cycle question selection, then
navigation_joint_cycle sampled draft selection, then the original actor.
pQ is binding_choice(features)+wood_choice_logits. pD is the original
joint_selection.distribution conditional on the recorded generated drafts.
pA is the original habit+centered precision offset at that leaf. q is the
same finite_relation_use_v5 model expectation, at the actual bound Q and P.
The audit enumerates these categorical choices in complete private bodies.
-/
namespace CGB.NativeFiniteChoiceTree
noncomputable section
variable {Q D A : Type*} [Fintype Q] [Fintype D] [Fintype A]

def conditional (pA q : Q → D → A → ℝ) (r : Q) (d : D) :=
  ∑ a, pA r d a * q r d a

def value (pQ : Q → ℝ) (pD : Q → D → ℝ) (pA q : Q → D → A → ℝ) :=
  ∑ r, pQ r * ∑ d, pD r d * conditional pA q r d

theorem finite_tree_flatten (pQ : Q → ℝ) (pD : Q → D → ℝ)
    (pA q : Q → D → A → ℝ) :
    value pQ pD pA q = ∑ r, ∑ d, ∑ a, (pQ r * pD r d * pA r d a) * q r d a := by
  simp only [value, conditional, Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro r _
  apply Finset.sum_congr rfl
  intro d _
  apply Finset.sum_congr rfl
  intro a _
  ring

theorem joint_probability_normalized (pQ : Q → ℝ) (pD : Q → D → ℝ)
    (pA : Q → D → A → ℝ) (hQ : ∑ r, pQ r = 1)
    (hD : ∀ r, ∑ d, pD r d = 1) (hA : ∀ r d, ∑ a, pA r d a = 1) :
    ∑ r, ∑ d, ∑ a, pQ r * pD r d * pA r d a = 1 := by
  simp_rw [← Finset.mul_sum, hA, mul_one, ← Finset.mul_sum, hD, mul_one]
  exact hQ

theorem reference_gap (reference : ℝ) (pQ : Q → ℝ) (pD : Q → D → ℝ)
    (pA q : Q → D → A → ℝ) (hQ : ∑ r, pQ r = 1) :
    reference - value pQ pD pA q =
      ∑ r, pQ r * (reference - ∑ d, pD r d * conditional pA q r d) := by
  simp only [mul_sub, Finset.sum_sub_distrib, ← Finset.sum_mul, hQ, one_mul, value]

end
end CGB.NativeFiniteChoiceTree
