import CGBAxioms.GenerationInformationProjection

/-! Variational abduction, same-disturbance intervention, and target information. -/

namespace CGB.CounterfactualActiveInference

open Finset
open CGB.InformationEnergy (kl)
open CGB.InformationProjection (project kl_nonneg_supported kl_project_le)

variable {U : Type*} [Fintype U]

/-- `joint u` is the prior times the likelihood of the observed outcome at the factual action. -/
noncomputable def posterior (joint : U → ℝ) (u : U) : ℝ :=
  joint u / (∑ v, joint v)

/-- The finite VFE on trials supported by the factual joint. -/
noncomputable def vfe (q joint : U → ℝ) : ℝ :=
  ∑ u, q u * (Real.log (q u) - Real.log (joint u))

theorem vfe_decomposition (q joint : U → ℝ) (hq : ∀ u, 0 ≤ q u)
    (hq1 : ∑ u, q u = 1) (hs : ∀ u, 0 < q u → 0 < joint u)
    (hz : 0 < ∑ u, joint u) :
    vfe q joint = kl q (posterior joint) - Real.log (∑ u, joint u) := by
  have ht : ∀ u, q u * (Real.log (q u) - Real.log (joint u)) =
      q u * (Real.log (q u) - Real.log (posterior joint u)) -
        q u * Real.log (∑ v, joint v) := by
    intro u
    by_cases h : q u = 0
    · simp [h]
    · rw [posterior, Real.log_div (hs u (lt_of_le_of_ne (hq u) (Ne.symm h))).ne' hz.ne']
      ring
  unfold vfe kl
  simp_rw [ht]
  rw [sum_sub_distrib, ← sum_mul, hq1, one_mul]

theorem posterior_minimizes_vfe (q joint : U → ℝ) (hq : ∀ u, 0 ≤ q u)
    (hq1 : ∑ u, q u = 1) (hj : ∀ u, 0 ≤ joint u)
    (hs : ∀ u, 0 < q u → 0 < joint u) (hz : 0 < ∑ u, joint u) :
    vfe (posterior joint) joint ≤ vfe q joint := by
  have hp : ∀ u, 0 ≤ posterior joint u := fun u => div_nonneg (hj u) hz.le
  have hp1 : ∑ u, posterior joint u = 1 := by
    simp_rw [posterior, div_eq_mul_inv]
    rw [← sum_mul, mul_inv_cancel₀ hz.ne']
  have hps : ∀ u, 0 < posterior joint u → 0 < joint u := by
    intro u h
    exact (div_pos_iff.mp h).resolve_right (fun h' => not_lt_of_ge hz.le h'.2) |>.1
  have hk := kl_nonneg_supported q (posterior joint) hq hp
    (fun u hu => div_pos (hs u hu) hz) hq1 hp1
  rw [vfe_decomposition _ _ hp hp1 hps hz, vfe_decomposition _ _ hq hq1 hs hz]
  have hself : kl (posterior joint) (posterior joint) = 0 := by simp [kl]
  linarith

/-- On the posterior support, the action contrast is the factual predictor's
contrast. This covers additive disturbances coordinatewise. -/
theorem posterior_residual (joint factual alternative : U → ℝ)
    (observed μ₀ μ₁ : ℝ) (hz : 0 < ∑ u, joint u)
    (hobs : ∀ u, joint u ≠ 0 → factual u = observed)
    (hcontrast : ∀ u, joint u ≠ 0 → alternative u - factual u = μ₁ - μ₀) :
    (∑ u, posterior joint u * alternative u) = observed + μ₁ - μ₀ := by
  have ht : ∀ u, posterior joint u * alternative u =
      posterior joint u * (observed + μ₁ - μ₀) := by
    intro u
    by_cases h : joint u = 0
    · simp [posterior, h]
    · congr 1
      have ho := hobs u h
      have hc := hcontrast u h
      linarith
  simp_rw [ht]
  rw [← sum_mul]
  have hp1 : ∑ u, posterior joint u = 1 := by
    simp_rw [posterior, div_eq_mul_inv]
    rw [← sum_mul, mul_inv_cancel₀ hz.ne']
  rw [hp1, one_mul]

/-- The paper's update is the interventional readout of a VFE minimizer
when the contrast is constant on the factual posterior support. -/
theorem variational_abduction_then_intervention
    (joint factual alternative : U → ℝ) (observed μ₀ μ₁ : ℝ)
    (hj : ∀ u, 0 ≤ joint u) (hz : 0 < ∑ u, joint u)
    (hobs : ∀ u, joint u ≠ 0 → factual u = observed)
    (hc : ∀ u, joint u ≠ 0 → alternative u - factual u = μ₁ - μ₀) :
    (∀ q : U → ℝ, (∀ u, 0 ≤ q u) → (∑ u, q u = 1) →
      (∀ u, 0 < q u → 0 < joint u) → vfe (posterior joint) joint ≤ vfe q joint) ∧
    (∑ u, posterior joint u * alternative u) = observed + μ₁ - μ₀ := by
  exact ⟨fun q hq hq1 hs => posterior_minimizes_vfe q joint hq hq1 hj hs hz,
    posterior_residual joint factual alternative observed μ₀ μ₁ hz hobs hc⟩

/-- Away from exact transfer, the missing quantity is the action contrast
averaged over the abducted posterior, not the unconditional contrast. -/
theorem posterior_contrast_correction (q factual alternative : U → ℝ)
    (observed μ₀ μ₁ : ℝ) (hq1 : ∑ u, q u = 1)
    (hobs : ∀ u, q u ≠ 0 → factual u = observed) :
    (∑ u, q u * alternative u) - (observed + μ₁ - μ₀) =
      ∑ u, q u * ((alternative u - factual u) - (μ₁ - μ₀)) := by
  have ht : ∀ u, q u * ((alternative u - factual u) - (μ₁ - μ₀)) =
      q u * alternative u - q u * (observed + μ₁ - μ₀) := by
    intro u
    by_cases h : q u = 0
    · simp [h]
    · rw [hobs u h]; ring
  simp_rw [ht]
  rw [sum_sub_distrib, ← sum_mul, hq1, one_mul]

/-- Pointwise epistemic value about the alternative outcome is bounded by
epistemic value about the entire disturbance. For each probe, averaging
over its possible observed outcomes preserves the bound. -/
theorem counterfactual_epistemic_le_disturbance {Y : Type*} [Fintype Y]
    (alternative : U → Y) (q prior : U → ℝ) (hq : ∀ u, 0 ≤ q u)
    (hp : ∀ u, 0 ≤ prior u) (hs : ∀ u, 0 < q u → 0 < prior u) :
    kl (project alternative q) (project alternative prior) ≤ kl q prior :=
  kl_project_le alternative q prior hq hp hs

/-- Squared risk when transporting a fraction `k` of the factual residual. -/
noncomputable def transferRisk (w factual alternative : U → ℝ) (k : ℝ) : ℝ :=
  ∑ u, w u * (alternative u - k * factual u)^2

/-- Exact excess risk relative to the optimal scalar transfer gain. -/
theorem transfer_excess_risk (w factual alternative : U → ℝ) (k : ℝ)
    (A : ℝ) (hA : A = ∑ u, w u * (factual u)^2) (hpos : 0 < A)
    (B : ℝ) (hB : B = ∑ u, w u * factual u * alternative u) :
    transferRisk w factual alternative k - transferRisk w factual alternative (B / A) =
      A * (k - B / A)^2 := by
  have expand (t : ℝ) : transferRisk w factual alternative t =
      transferRisk w factual alternative 0 + t^2 * A - 2 * t * B := by
    unfold transferRisk
    rw [hA, hB, mul_sum, mul_sum, ← sum_add_distrib, ← sum_sub_distrib]
    exact sum_congr rfl fun u _ => by ring
  rw [expand k, expand (B / A)]
  field_simp [hpos.ne']
  ring

/-- The calibrated gain dominates both zero reuse and literal residual reuse. -/
theorem calibrated_transfer_le (w factual alternative : U → ℝ) (k : ℝ)
    (A : ℝ) (hA : A = ∑ u, w u * (factual u)^2) (hpos : 0 < A)
    (B : ℝ) (hB : B = ∑ u, w u * factual u * alternative u) :
    transferRisk w factual alternative (B / A) ≤ transferRisk w factual alternative k := by
  have h := transfer_excess_risk w factual alternative k A hA hpos B hB
  linarith [mul_nonneg hpos.le (sq_nonneg (k - B / A))]

/-- An estimated gain beats a comparator exactly when it is closer to the
true gain in squared distance. No assumption that estimation is correct. -/
theorem estimated_gain_improves_iff (w factual alternative : U → ℝ) (k comparator : ℝ)
    (A : ℝ) (hA : A = ∑ u, w u * (factual u)^2) (hpos : 0 < A)
    (B : ℝ) (hB : B = ∑ u, w u * factual u * alternative u) :
    transferRisk w factual alternative k < transferRisk w factual alternative comparator ↔
      (k - B / A)^2 < (comparator - B / A)^2 := by
  have hk := transfer_excess_risk w factual alternative k A hA hpos B hB
  have hc := transfer_excess_risk w factual alternative comparator A hA hpos B hB
  constructor
  · intro h
    by_contra! hn
    nlinarith [mul_nonneg hpos.le (sub_nonneg.mpr hn)]
  · intro h
    nlinarith [mul_pos hpos (sub_pos.mpr h)]

end CGB.CounterfactualActiveInference
