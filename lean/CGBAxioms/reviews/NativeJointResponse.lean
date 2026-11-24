import Mathlib.Tactic

/-! Production binding (schema6): joint_response.distribution reconstructs
N_ij=noise_i δ_ij+(B_ij-B_ii δ_ij)/df from the SAME finite residual scatter.
Both the outer imagine and the two finite-U expected readers call
response_hypotheses. Their predictive scale is (1+leverage)N; their PAYMENT
score remains the original coordinate Student-t. Batched_context_scores
pads each own-fact-excluded design with zero feature/residual rows. The
identities below verify the shared marginal, unchanged ridge normal equations
and interchange of the existing response/decision weighted finite evaluation.
Actual LU, positive matrix, values and gradients are bound in
tests/test_joint_response.py, rather than postulated by this file.
-/
namespace CGB.NativeJointResponse
noncomputable section

def noiseEntry {O : Type*} [DecidableEq O] (noise : O → ℝ)
    (scatter : O → O → ℝ) (df : ℝ) (i j : O) : ℝ :=
  (if i=j then noise i else 0) +
    (scatter i j - if i=j then scatter i i else 0)/df

theorem joint_response_keeps_predictive_marginals {O : Type*} [DecidableEq O]
    (noise : O → ℝ) (scatter : O → O → ℝ) (df leverage : ℝ) (i : O) :
    (1+leverage)*noiseEntry noise scatter df i i = (1+leverage)*noise i := by
  simp [noiseEntry]

theorem padding_keeps_ridge_normal_matrix {I P F : Type*}
    [Fintype I] [Fintype P] (x : I → F → ℝ) (a b : F) (ridge : ℝ) :
    ridge + (∑ r : I ⊕ P, (Sum.elim x (fun _ _ => 0) r a) *
      (Sum.elim x (fun _ _ => 0) r b)) =
    ridge + ∑ r : I, x r a * x r b := by
  simp [Fintype.sum_sum_type]

theorem padding_keeps_ridge_right_hand_side {I P F O : Type*}
    [Fintype I] [Fintype P] (x : I → F → ℝ) (y : I → O → ℝ) (a : F) (b : O) :
    (∑ r : I ⊕ P, (Sum.elim x (fun _ _ => 0) r a) *
      (Sum.elim y (fun _ _ => 0) r b)) = ∑ r : I, x r a * y r b := by
  simp [Fintype.sum_sum_type]

theorem joint_trajectory_weighted_read {H K : Type*} [Fintype H] [Fintype K]
    (p : H → ℝ) (w v : H → K → ℝ) :
    (∑ h, p h * ∑ k, w h k * v h k) =
    ∑ h, ∑ k, (p h * w h k) * v h k := by
  simp_rw [Finset.mul_sum, mul_assoc]

end
end CGB.NativeJointResponse
