import Mathlib.Tactic

namespace CGB.NativeChoiceCreditTransport

/-- Actual binding_choice has one bias-free linear row, 3*128+4 coordinates.
    Binary log-softmax score derivative uses the two recorded coefficients.
    They are retained separately to match the FP32 implementation. -/
def score (c0 c1 : ℝ) (f0 f1 : Fin 388 → ℝ) (j : Fin 388) : ℝ :=
  c0 * f0 j + c1 * f1 j

theorem binary_score (s p : ℝ) (f0 f1 : Fin 388 → ℝ) (j : Fin 388) :
    score (p-s) (s-p) f0 f1 j = (s-p)*(f1 j-f0 j) := by
  simp only [score]; ring

/-- Each of the first 384 coordinates is a content coordinate. The actual
    held-preview intervention equates these, retaining different role tags. -/
theorem held_content (s p : ℝ) (f0 f1 : Fin 388 → ℝ)
    (same : ∀ j, j.val < 384 → f1 j = f0 j)
    (j : Fin 388) (content : j.val < 384) :
    score (p-s) (s-p) f0 f1 j = 0 := by
  rw [binary_score, same j content]; ring

/-- With both recorded content vectors fixed, changing only the prior changes
    the probability coefficient along the same feature difference. -/
theorem prior_path (s p q : ℝ) (f0 f1 : Fin 388 → ℝ) (j : Fin 388) :
    score (q-s) (s-q) f0 f1 j - score (p-s) (s-p) f0 f1 j =
    (p-q)*(f1 j-f0 j) := by
  simp only [score]; ring

/-- autonomous_cycle.poll accumulates scores; _advance applies the later
    real advantage to this sum before clipping and optimizer transformation. -/
theorem delayed_factor {n : ℕ} (a : ℝ) (e : Fin n → Fin 388 → ℝ)
    (j : Fin 388) :
    -a * (∑ i, e i j) = ∑ i, -a * e i j := by
  exact Finset.mul_sum ..

end CGB.NativeChoiceCreditTransport
