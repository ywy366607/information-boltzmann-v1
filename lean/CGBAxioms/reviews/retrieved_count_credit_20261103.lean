module

public import Mathlib.Algebra.BigOperators.Field
public import Mathlib.Analysis.Calculus.Deriv.Mul
public import Mathlib.Analysis.Calculus.Deriv.Inv
public import Mathlib.Analysis.Calculus.Deriv.Add
import Mathlib.Tactic.Ring

public section

/-! A retrieved association receives later evidence through its actual mixture read.
These are event-local derivatives, NOT derivatives through old writes or all GDN layers.
The existing count controls read precision, assimilation and eviction. Its optional
feedback exchange preserves the total of the currently eligible counts only;
ordinary boundary writes remain open updates. -/
namespace RetrievedCountCreditReview

/-- Changing one participant's mixture weight changes the normalized read by
its value relative to the other participants; the denominator must be nonzero. -/
theorem normalized_read_deriv (a b v c : ℝ) (h : a + c ≠ 0) :
    HasDerivAt (fun x => (b + x * v) / (a + x))
      ((v * a - b) / (a + c)^2) c := by
  convert (((hasDerivAt_id c).mul_const v).const_add b).fun_div
    ((hasDerivAt_id c).const_add a) h using 1 <;> dsimp <;> ring

/-- The existing count-to-trust map has this local derivative. It is not a
derivative of the old sequence of writes that produced the count. -/
theorem count_trust_deriv (c : ℝ) (h : c + 1 ≠ 0) :
    HasDerivAt (fun x : ℝ => x / (x + 1)) (1 / (c + 1)^2) c := by
  convert (hasDerivAt_id c).fun_div ((hasDerivAt_id c).add_const 1) h using 1 <;>
    dsimp <;> ring

/-- Any common fractional move toward normalized reinvestment preserves the
eligible pool's total. Shortening the common move at a numerical ceiling also
preserves this identity; ordinary boundary writes are outside this statement.
`GenerationReinvestment.reinvest` and `predicted` are unfolded in the statement
to avoid importing unrelated logarithmic-wealth proofs into this runtime check. -/
theorem partial_exchange_total {A : Type*} [Fintype A]
    (c w : A → ℝ) (r : ℝ) (h : (∑ i, c i * w i) ≠ 0) :
    (∑ i, ((1-r)*c i + r*(∑ j, c j)*(c i*w i/(∑ j, c j*w j)))) =
      ∑ i, c i := by
  simp only [Finset.sum_add_distrib, ← Finset.mul_sum]
  rw [← Finset.sum_div, div_self h]
  ring

end RetrievedCountCreditReview

#print axioms RetrievedCountCreditReview.normalized_read_deriv
#print axioms RetrievedCountCreditReview.count_trust_deriv
#print axioms RetrievedCountCreditReview.partial_exchange_total
