import Mathlib.Tactic

/-! Production mapping: navigation_momentum_wood._packed_momentum_read.
For a fixed source, RMSNorm's denominator is identical across all equal-width
layers with the same epsilon. W is q_proj, gamma is layer.norm.weight, and x
is the post-feature_write source. Fold gamma into W, concatenate layers, then
apply the unchanged per-head normalization and current P read. This lemma is
the coordinate equality before head normalization; PyTorch admission checks
FP32 rounding and parameter-version invalidation at 32 layers / 8 heads.
-/
namespace CGB.PackedMomentumQuery

theorem fold_rms_query {I : Type*} [Fintype I]
    (W gamma x : I → ℝ) (den : ℝ) :
    (∑ k, W k * (gamma k * x k / den)) =
      ∑ k, (W k * gamma k) * (x k / den) := by
  apply Finset.sum_congr rfl
  intro k _
  ring

-- Packing the batch/layer axes changes only the storage index, not any term
-- in the original per-head P^T q contraction. q is the normalized query.
theorem packed_read_coordinate {I : Type*} [Fintype I]
    (q : ℕ → ℕ → I → ℝ) (P : ℕ → I → ℝ) (batch layer : ℕ) :
    (∑ k, q batch layer k * P layer k) =
      (fun b l => ∑ k, q b l k * P l k) batch layer := rfl

end CGB.PackedMomentumQuery
