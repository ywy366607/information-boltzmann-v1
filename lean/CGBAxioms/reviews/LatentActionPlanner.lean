import Mathlib.Tactic

/-! Bound to latent_action_planner.value: p is original schema5 mass,
q is the positive content-conditioned mixture proposal; mu is the frozen
batched action planner and v the unchanged native decision endpoint.
Native and arrived world labels commit only after the full actual choice.
-/
namespace CGB.LatentActionPlanner
noncomputable section

/- Production decision_budget reads the learned candidate expectation and
   commits at most two native teachers at the root action boundary. This
   conditional error statement is distinct from the residual reference below.
   Numerical admission binds p/mu; observed teacher residuals measure coverage.
-/
theorem candidate_readout_bound {I : Type*} [Fintype I]
    (p mu v : I → ℝ) (epsilon : ℝ) (normal : ∑ i, p i = 1)
    (nonnegative : ∀ i, 0 ≤ p i) (error : ∀ i, |v i-mu i| ≤ epsilon) :
    |(∑ i, p i*v i)-(∑ i, p i*mu i)| ≤ epsilon := by
  have difference : (∑ i, p i*v i)-(∑ i, p i*mu i) = ∑ i, p i*(v i-mu i) := by
    simp_rw [mul_sub]
    rw [Finset.sum_sub_distrib]
  rw [difference]
  calc
    |∑ i, p i*(v i-mu i)| ≤ ∑ i, |p i*(v i-mu i)| := Finset.abs_sum_le_sum_abs _ _
    _ = ∑ i, p i*|v i-mu i| := by
      apply Finset.sum_congr rfl
      intro i _
      rw [abs_mul, abs_of_nonneg (nonnegative i)]
    _ ≤ ∑ i, p i*epsilon := Finset.sum_le_sum fun i _ => mul_le_mul_of_nonneg_left (error i) (nonnegative i)
    _ = epsilon := by rw [← Finset.sum_mul, normal, one_mul]

theorem candidate_ranking_bound (vi vj mui muj epsilon : ℝ)
    (ei : |vi-mui| ≤ epsilon) (ej : |vj-muj| ≤ epsilon) (rank : muj ≤ mui) :
    vj-vi ≤ 2*epsilon := by
  have hi := (abs_le.mp ei).1
  have hj := (abs_le.mp ej).2
  linarith

theorem importance_correction {I : Type*} [Fintype I]
    (p q mu v : I → ℝ) (normal : ∑ i, q i = 1)
    (support : ∀ i, q i ≠ 0) :
    (∑ i, q i * ((∑ j, p j * mu j) + p i / q i * (v i-mu i))) =
      ∑ i, p i*v i := by
  have cancel : ∀ i, q i * (p i / q i * (v i-mu i)) = p i*(v i-mu i) := by
    intro i
    field_simp [support i]
  simp_rw [mul_add, cancel]
  rw [Finset.sum_add_distrib, ← Finset.sum_mul, normal, one_mul]
  simp_rw [mul_sub]
  rw [Finset.sum_sub_distrib]
  ring

theorem mixture_mass (p learned : ℝ) (hl : 0 ≤ learned) :
    p ≤ 2*((p+learned)/2) := by linarith

theorem mixture_support (p learned : ℝ) (hp : 0 < p) (hl : 0 ≤ learned) :
    0 < (p+learned)/2 := by linarith

/- The largest-mass exact contribution is outside the stochastic tail.
   Production normalizes q on the remaining original indices. -/
theorem exact_plus_tail {I : Type*} [Fintype I]
    (exact : ℝ) (p q mu v : I → ℝ) (normal : ∑ i, q i = 1)
    (support : ∀ i, q i ≠ 0) :
    exact + (∑ i, q i * ((∑ j, p j*mu j)+p i/q i*(v i-mu i))) =
    exact + ∑ i, p i*v i := by
  rw [importance_correction p q mu v normal support]

theorem full_correction {I : Type*} [Fintype I] (p mu v : I → ℝ) :
    (∑ i, p i*mu i)+(∑ i, p i*(v i-mu i)) = ∑ i, p i*v i := by
  simp_rw [mul_sub]
  rw [Finset.sum_sub_distrib]
  ring

end
end CGB.LatentActionPlanner
