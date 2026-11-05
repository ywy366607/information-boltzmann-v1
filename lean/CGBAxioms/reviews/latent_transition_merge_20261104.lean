import CGBAxioms.InformationEconomics
import Mathlib.Tactic

/-! Admission for finite merge, before runtime testing. This is scalar/component
algebra, not an assertion that cosine similarity implies language equivalence.
The exact two-history condition is `causal_carrier_contract_iff` in the import:
readout and ALL encoded successor fibres must agree. Here a convex merge has
an explicitly bounded local value displacement. A long-run bound additionally
requires accounting for repeated merges and decoder/transition sensitivity.
-/

namespace LatentTransitionMergeReview

example (x y w : ℝ) (hw0 : 0 ≤ w) (hw1 : w ≤ 1) :
    |((1 - w) * x + w * y) - x| ≤ |y - x| := by
  have hid : ((1 - w) * x + w * y) - x = w * (y - x) := by ring
  rw [hid, abs_mul, abs_of_nonneg hw0]
  nlinarith [abs_nonneg (y - x)]

example (x y w L : ℝ) (hw0 : 0 ≤ w) (hw1 : w ≤ 1) (hL : 0 ≤ L)
    (f : ℝ → ℝ) (hlip : ∀ a b, |f a - f b| ≤ L * |a - b|) :
    |f ((1 - w) * x + w * y) - f x| ≤ L * |y - x| := by
  have hid : ((1 - w) * x + w * y) - x = w * (y - x) := by ring
  calc
    |f ((1 - w) * x + w * y) - f x| ≤ L * |((1 - w) * x + w * y) - x| := hlip _ _
    _ = L * (w * |y - x|) := by rw [hid, abs_mul, abs_of_nonneg hw0]
    _ ≤ L * |y - x| := by
      have h := mul_le_mul_of_nonneg_right hw1 (abs_nonneg (y - x))
      simpa using mul_le_mul_of_nonneg_left h hL

example (x w : ℝ) : (1 - w) * x + w * x = x := by ring

end LatentTransitionMergeReview
