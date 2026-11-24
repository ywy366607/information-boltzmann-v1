import Mathlib.Tactic

/-! Bindings: native_amortized_decision.predict has G=D K D and A=G+ridge I.
native_value_residual_filter.coefficients reuses its Cholesky factor.
Each eigenmode below has eigenvalue sigma and residual coordinate r.
The final observable is the original response/successor contraction followed
by the actual actor; c is its signed contrast coefficient in that mode.
-/
namespace CGB.NativePersistentResidualFilter
noncomputable section

theorem one_step_remaining (sigma ridge r : ℝ) (h : sigma+ridge ≠ 0) :
    r-sigma*(r/(sigma+ridge)) = ridge*r/(sigma+ridge) := by
  field_simp; ring

theorem two_step_remaining (sigma ridge r : ℝ) (h : sigma+ridge ≠ 0) :
    r-sigma*(r/(sigma+ridge)+ridge*r/(sigma+ridge)^2) =
      (ridge/(sigma+ridge))^2*r := by
  field_simp; ring

theorem positive_mode_contraction (sigma ridge : ℝ)
    (hs : 0 < sigma) (hr : 0 < ridge) :
    0 < ridge/(sigma+ridge) ∧ ridge/(sigma+ridge) < 1 := by
  constructor
  · positivity
  · exact (div_lt_one (by positivity)).mpr (by linarith)

theorem second_fit_reduces_mode (q : ℝ) (h0 : 0 < q) (h1 : q < 1) :
    q^2 < q := by nlinarith

theorem observed_residual_transport {I : Type*} [Fintype I]
    (c r q : I → ℝ) :
    (∑ i, c i*(q i*r i)) - (∑ i, c i*(q i^2*r i)) =
      ∑ i, c i*(q i*(1-q i)*r i) := by
  rw [← Finset.sum_sub_distrib]
  apply Finset.sum_congr rfl
  intro i _
  ring

/-- Opposite signed observable coefficients can cancel. The actual root
contraction is therefore carried through, alongside modal training errors. -/
theorem contraction_cancellation (u v : ℝ) :
    u+v=0 ↔ v = -u := by constructor <;> intro h <;> linarith

end
end CGB.NativePersistentResidualFilter
