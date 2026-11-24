import Mathlib.Tactic

/-!
Bindings: native_amortized_decision.features, latent_action_planner.features/
predict and autonomous_cycle.poll/take_boundary. This file adds the anchored
feature inversion, an explicit repeated-feature KRR confidence/error example,
and finite-prefix first-release mass AND tangent accounting. Existing actor
contrast invariance and causal-tag transport theorems are inherited separately.

The normalizer uses a fixed nonzero scalar already present in the packet.
`repeatNovelty` is the normalized Schur residual for n identical kernel rows;
its reduction from the actual Cholesky implementation is checked numerically.
`stay` is the conditional internal-choice probability on each continuing path;
`dstay` is its directional derivative, with the actual selector features held
fixed as in production detached head-score eligibility. Tail mass and its
tangent remain outstanding at a finite observation/checkpoint prefix.
-/
namespace CGB.NativeTargetClosureAndStopping
noncomputable section

/-- The existing fixed anchor recovers scale after positive normalization. -/
theorem anchor_recovers_coordinate (v anchor scale : ℝ)
    (ha : anchor ≠ 0) (hs : scale ≠ 0) :
    anchor * (v / scale) / (anchor / scale) = v := by
  field_simp

/-- Injectivity of the normalized augmented vector on a fixed anchor slice. -/
theorem anchored_view_injective {I : Type*} (v w : I → ℝ)
    (a : I) (s t : ℝ) (hs : s ≠ 0) (ht : t ≠ 0)
    (ha : v a ≠ 0) (anchor : w a = v a)
    (same : ∀ i, v i / s = w i / t) : v = w := by
  funext i
  have hv := anchor_recovers_coordinate (v i) (v a) s ha hs
  have hw := anchor_recovers_coordinate (w i) (v a) t ha ht
  rw [same i, same a, anchor] at hv
  exact hv.symm.trans hw

def repeatNovelty (n k ridge : ℝ) : ℝ := ridge / (ridge + n*k)

/-- Repeated-feature novelty is the actual diagonal-normalized Schur residual. -/
theorem repeated_feature_schur (n k ridge : ℝ) (hk : k ≠ 0)
    (hd : ridge+n*k ≠ 0) :
    (k - n*k^2/(ridge+n*k))/k = repeatNovelty n k ridge := by
  unfold repeatNovelty
  have h : (k - n * k ^ 2 / (ridge + n * k)) = (ridge * k) / (ridge + n * k) := by
    rw [sub_div' hd]
    congr 1
    ring
  rw [h]
  rw [div_div, mul_comm (ridge + n * k) k, ← div_div, mul_div_cancel_right₀ _ hk]

/-- More copies reduce novelty even when the associated labels disagree. -/
theorem repeated_novelty_strictly_decreases (n m k ridge : ℝ)
    (hn : 0 ≤ n) (hm : n < m) (hk : 0 < k) (hr : 0 < ridge) :
    repeatNovelty m k ridge < repeatNovelty n k ridge := by
  unfold repeatNovelty
  apply div_lt_div_of_pos_left hr
  · positivity
  · nlinarith

/-- Same represented condition with opposite residual targets retains this
mean-square floor for EVERY single prediction, including the ridge optimum. -/
theorem same_feature_error_floor (prediction delta : ℝ) :
    delta^2 ≤ ((prediction-delta)^2+(prediction+delta)^2)/2 := by
  nlinarith [sq_nonneg prediction]

/-- A finite repetition count yielding arbitrarily prescribed low novelty;
the target-error floor above is independent of this count. -/
theorem novelty_below_threshold (n k ridge eps : ℝ)
    (hn : 0 ≤ n) (hk : 0 < k) (hr : 0 < ridge) (_he : 0 < eps)
    (budget : ridge < eps*(ridge+n*k)) :
    repeatNovelty n k ridge < eps := by
  unfold repeatNovelty
  exact (div_lt_iff₀ (by positivity : 0 < ridge+n*k)).mpr budget

def survival (stay : ℕ → ℝ) : ℕ → ℝ
  | 0 => 1
  | n+1 => survival stay n * stay n

def release (stay : ℕ → ℝ) (n : ℕ) : ℝ :=
  survival stay n * (1-stay n)

def survivalTangent (stay dstay : ℕ → ℝ) : ℕ → ℝ
  | 0 => 0
  | n+1 => survivalTangent stay dstay n * stay n + survival stay n * dstay n

def releaseTangent (stay dstay : ℕ → ℝ) (n : ℕ) : ℝ :=
  survivalTangent stay dstay n * (1-stay n) - survival stay n * dstay n

/-- Exact finite-prefix probability accounting, retaining the unfinished tail. -/
theorem prefix_release_and_tail (stay : ℕ → ℝ) (N : ℕ) :
    (∑ n ∈ Finset.range N, release stay n) + survival stay N = 1 := by
  induction N with
  | zero => simp [survival]
  | succ N ih =>
    rw [Finset.sum_range_succ]
    change (∑ n ∈ Finset.range N, release stay n) +
      survival stay N*(1-stay N) + survival stay N*stay N = 1
    nlinarith

/-- Exact directional probability accounting. Dropping the unfinished tail
also drops its score credit, even when all completed releases were retained. -/
theorem prefix_tangent_and_tail (stay dstay : ℕ → ℝ) (N : ℕ) :
    (∑ n ∈ Finset.range N, releaseTangent stay dstay n) +
      survivalTangent stay dstay N = 0 := by
  induction N with
  | zero => simp [survivalTangent]
  | succ N ih =>
    rw [Finset.sum_range_succ]
    change (∑ n ∈ Finset.range N, releaseTangent stay dstay n) +
      (survivalTangent stay dstay N*(1-stay N)-survival stay N*dstay N) +
      (survivalTangent stay dstay N*stay N+survival stay N*dstay N) = 0
    nlinarith

end
end CGB.NativeTargetClosureAndStopping
