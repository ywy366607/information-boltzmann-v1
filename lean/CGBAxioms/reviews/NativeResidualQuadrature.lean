import Mathlib.Tactic

/-! Body binding: native_residual_quadrature uses the SAME 4096 posterior
points of joint_successor, and a partition of their stable sample indices.
`b` is navigation_internal_rollout.continuation, trained by actual finite-U
outcomes. `v` is prospective_decision.value at each native boundary.
For stratum g, w_g = card(g)/4096, p_gi = 1/card(g). The readout is
mean_all(b) + sum_g w_g * (v(sample_g)-b(sample_g)).
The finite average below is the expectation over a uniform local draw,
conditional on the body, points, proxy and partition fixed BEFORE that draw.
No world label or observed successor is supplied to this computation.
-/
namespace CGB.NativeResidualQuadrature
noncomputable section

theorem residual_average_exact {G I : Type*} [Fintype G] [Fintype I]
    (w : G → ℝ) (p b v : G → I → ℝ) :
    (∑ g, w g * ∑ i, p g i * b g i) +
      (∑ g, w g * ∑ i, p g i * (v g i - b g i)) =
      ∑ g, w g * ∑ i, p g i * v g i := by
  simp_rw [mul_sub, Finset.sum_sub_distrib, mul_sub]
  rw [Finset.sum_sub_distrib]
  ring

theorem constant_residual_exact {G : Type*} [Fintype G]
    (w baseline residual : G → ℝ) (c : ℝ)
    (normalized : ∑ g, w g = 1) (flat : ∀ g, residual g = c) :
    (∑ g, w g * baseline g) + ∑ g, w g * residual g =
      (∑ g, w g * baseline g) + c := by
  simp_rw [flat]
  rw [← Finset.sum_mul, normalized, one_mul]

theorem residual_error_bound {G : Type*} [Fintype G]
    (w r expected bound : G → ℝ)
    (positive : ∀ g, 0 ≤ w g)
    (bounded : ∀ g, |r g - expected g| ≤ bound g) :
    |∑ g, w g * (r g - expected g)| ≤ ∑ g, w g * bound g := by
  calc
    _ ≤ ∑ g, |w g * (r g - expected g)| := Finset.abs_sum_le_sum_abs _ _
    _ = ∑ g, w g * |r g - expected g| := by
      apply Finset.sum_congr rfl
      intro g _
      rw [abs_mul, abs_of_nonneg (positive g)]
    _ ≤ _ := Finset.sum_le_sum fun g _ => mul_le_mul_of_nonneg_left (bounded g) (positive g)

theorem split_proxy_variance (w a b : ℝ) :
    w * (a - (w*a+(1-w)*b))^2 +
      (1-w) * (b - (w*a+(1-w)*b))^2 =
      w*(1-w)*(a-b)^2 := by ring

theorem split_proxy_gain_nonnegative (w a b : ℝ) (lower : 0 ≤ w) (upper : w ≤ 1) :
    0 ≤ w*(1-w)*(a-b)^2 := by
  exact mul_nonneg (mul_nonneg lower (sub_nonneg.mpr upper)) (sq_nonneg _)

end
end CGB.NativeResidualQuadrature
