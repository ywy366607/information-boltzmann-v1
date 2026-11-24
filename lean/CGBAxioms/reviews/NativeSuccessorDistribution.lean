import Mathlib.Tactic

/-! Body binding: successor_distribution.quadrature supplies fixed antithetic
Student-t marginal quantiles, normalized weights, and encoded successor points.
G is the actual prospective_decision.value composition: raw lift, relation
address, typed question, native word execution, actor, finite-U score.
The formulas preserve action contrasts and identify exactly what a mean-only
handoff discards. They do not assert a numerical integration error bound.
The finite posterior scored by delayed learning is overlap_consensus, the same
function that execute_programs uses. Numeric binding is tested in
tests/test_successor_distribution.py against the actual production functions.
-/
namespace CGB.NativeSuccessorDistribution
noncomputable section

def expected {K : Type*} [Fintype K] (w v : K → ℝ) := ∑ k, w k * v k

theorem centered_native_read {K : Type*} [Fintype K] (w v : K → ℝ)
    (normalized : ∑ k, w k = 1) (anchor : ℝ) :
    expected w v = anchor + ∑ k, w k * (v k-anchor) := by
  unfold expected
  simp only [mul_sub, Finset.sum_sub_distrib, ← Finset.sum_mul, normalized, one_mul]
  ring

theorem paired_latent_center (mu sigma direction : ℝ) :
    ((mu+sigma*direction)+(mu-sigma*direction))/2 = mu := by ring

theorem zero_scale_recovers_native_read {K : Type*} [Fintype K]
    (w : K → ℝ) (normalized : ∑ k, w k = 1) (G : ℝ → ℝ) (mu : ℝ) :
    expected w (fun _ => G mu) = G mu := by
  simp [expected, ← Finset.sum_mul, normalized]

theorem action_contrast_uses_same_mass {K : Type*} [Fintype K] (w vi vj : K → ℝ) :
    expected w vi - expected w vj = expected w (fun k => vi k-vj k) := by
  simp [expected, mul_sub, Finset.sum_sub_distrib]

theorem mean_handoff_discards_native_curvature (G : ℝ → ℝ) (mu delta : ℝ) :
    (G (mu+delta)+G (mu-delta))/2-G mu =
      (G (mu+delta)-2*G mu+G (mu-delta))/2 := by ring

theorem branch_rename_invariant {K : Type*} [Fintype K] (w v : K → ℝ) (e : K ≃ K) :
    expected (w ∘ e) (v ∘ e) = expected w v := by
  unfold expected
  exact e.sum_comp (fun k => w k*v k)

end
end CGB.NativeSuccessorDistribution
