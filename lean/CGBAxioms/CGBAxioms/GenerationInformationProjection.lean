import CGBAxioms.GenerationInformationEnergy
import CGBAxioms.Life

/-!
# Information under finite deterministic observation

Data processing for `Medium.condMutualInfo`, including sparse laws and arbitrary
nonnegative total mass. Only the third variable is observed; the conditioning
variable and the tag are unchanged. No stationarity or frozen-body premise is used.
-/

namespace CGB.InformationProjection

open Finset

/-- Gibbs' pointwise bound on compatible, possibly zero, supports. -/
theorem gibbs_term_supported (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y)
    (hs : 0 < x → 0 < y) :
    x * (Real.log y - Real.log x) ≤ y - x := by
  by_cases h : x = 0
  · simpa [h] using hy
  · exact PredictiveHarvest.gibbs_term x y hx (hs (lt_of_le_of_ne hx (Ne.symm h)))

/-- Finite log-sum, with zero entries allowed and no normalization premise. -/
theorem log_sum_le {ι : Type*} [Fintype ι] (x y : ι → ℝ)
    (hx : ∀ i, 0 ≤ x i) (hy : ∀ i, 0 ≤ y i)
    (hs : ∀ i, 0 < x i → 0 < y i) :
    (∑ i, x i) * (Real.log (∑ i, x i) - Real.log (∑ i, y i)) ≤
      ∑ i, x i * (Real.log (x i) - Real.log (y i)) := by
  classical
  let X := ∑ i, x i
  let Y := ∑ i, y i
  have hX0 : 0 ≤ X := sum_nonneg fun i _ => hx i
  by_cases hX : X = 0
  · have hz : ∀ i, x i = 0 := by
      intro i
      have hi : x i ≤ X := single_le_sum (fun j _ => hx j) (mem_univ i)
      exact le_antisymm (by simpa [hX] using hi) (hx i)
    simp [hz]
  · have hXp : 0 < X := lt_of_le_of_ne hX0 (Ne.symm hX)
    have hYp : 0 < Y := by
      obtain ⟨i, _, hi⟩ := (sum_pos_iff_of_nonneg (fun i _ => hx i)).mp hXp
      exact lt_of_lt_of_le (hs i hi) (single_le_sum (fun j _ => hy j) (mem_univ i))
    have h := sum_le_sum fun i (_ : i ∈ (univ : Finset ι)) =>
      gibbs_term_supported (x i) (y i * X / Y) (hx i)
        (div_nonneg (mul_nonneg (hy i) hXp.le) hYp.le)
        (fun hi => div_pos (mul_pos (hs i hi) hXp) hYp)
    have he : ∀ i, x i * (Real.log (y i * X / Y) - Real.log (x i)) =
        -(x i * (Real.log (x i) - Real.log (y i))) +
          x i * (Real.log X - Real.log Y) := by
      intro i
      by_cases hi : x i = 0
      · simp [hi]
      · rw [Real.log_div (mul_pos (hs i (lt_of_le_of_ne (hx i) (Ne.symm hi))) hXp).ne'
            hYp.ne', Real.log_mul (hs i (lt_of_le_of_ne (hx i) (Ne.symm hi))).ne' hXp.ne']
        ring
    have hmass : ∑ i, y i * X / Y = X := by
      simp_rw [div_eq_mul_inv]
      rw [← sum_mul, ← sum_mul]
      change Y * X * Y⁻¹ = X
      field_simp
    simp_rw [he] at h
    rw [sum_add_distrib, sum_neg_distrib, ← sum_mul, sum_sub_distrib, hmass] at h
    change X * (Real.log X - Real.log Y) ≤ _
    change -(∑ i, x i * (Real.log (x i) - Real.log (y i))) +
      X * (Real.log X - Real.log Y) ≤ X - X at h
    linarith

/-- The existing relative entropy is nonnegative for normalized compatible supports.
The strictly positive case reuses `InformationEnergy.kl_nonneg` unchanged. -/
theorem kl_nonneg_supported {ι : Type*} [Fintype ι] (x y : ι → ℝ)
    (hx : ∀ i, 0 ≤ x i) (hy : ∀ i, 0 ≤ y i)
    (hs : ∀ i, 0 < x i → 0 < y i)
    (hX : ∑ i, x i = 1) (hY : ∑ i, y i = 1) :
    0 ≤ InformationEnergy.kl x y := by
  by_cases hp : ∀ i, 0 < y i
  · exact InformationEnergy.kl_nonneg x y hx hX hp hY
  · simpa [InformationEnergy.kl, hX, hY] using log_sum_le x y hx hy hs

/-- A support gap makes relative entropy strictly positive for compatible
nonnegative weights of equal mass. No normalization is needed. -/
theorem kl_pos_of_support_gap {ι : Type*} [Fintype ι] (x y : ι → ℝ)
    (hx : ∀ i, 0 ≤ x i) (hy : ∀ i, 0 ≤ y i)
    (hs : ∀ i, 0 < x i → 0 < y i)
    (hm : ∑ i, x i = ∑ i, y i)
    (hgap : ∃ i, x i = 0 ∧ 0 < y i) :
    0 < InformationEnergy.kl x y := by
  obtain ⟨i, hxi, hyi⟩ := hgap
  have hlt : (∑ i, x i * (Real.log (y i) - Real.log (x i))) <
      ∑ i, (y i - x i) :=
    sum_lt_sum (fun j _ => gibbs_term_supported (x j) (y j) (hx j) (hy j) (hs j))
      ⟨i, mem_univ i, by simpa [hxi] using hyi⟩
  rw [sum_sub_distrib, hm, sub_self] at hlt
  have he : (∑ i, x i * (Real.log (y i) - Real.log (x i))) =
      -InformationEnergy.kl x y := by
    unfold InformationEnergy.kl
    rw [← sum_neg_distrib]
    exact sum_congr rfl fun j _ => by ring
  rw [he] at hlt
  linarith

/-- Push a finite weight function through a deterministic observation. -/
noncomputable def project {γ δ : Type*} [Fintype γ] (f : γ → δ)
    (p : γ → ℝ) (d : δ) : ℝ := by
  classical
  exact ∑ c, if f c = d then p c else 0

theorem project_nonneg {γ δ : Type*} [Fintype γ] (f : γ → δ)
    (p : γ → ℝ) (hp : ∀ c, 0 ≤ p c) (d : δ) : 0 ≤ project f p d := by
  classical
  exact sum_nonneg fun c _ => by split_ifs <;> simp_all

/-- Observation preserves total mass, including zero total mass. -/
theorem sum_project {γ δ : Type*} [Fintype γ] [Fintype δ]
    (f : γ → δ) (p : γ → ℝ) : ∑ d, project f p d = ∑ c, p c := by
  classical
  unfold project
  rw [sum_comm]
  simp

/-- Relative entropy contracts under finite deterministic observation, without
normalization, provided the reference is positive wherever the source is positive. -/
theorem kl_project_le {γ δ : Type*} [Fintype γ] [Fintype δ]
    (f : γ → δ) (x y : γ → ℝ) (hx : ∀ c, 0 ≤ x c)
    (hy : ∀ c, 0 ≤ y c) (hs : ∀ c, 0 < x c → 0 < y c) :
    InformationEnergy.kl (project f x) (project f y) ≤ InformationEnergy.kl x y := by
  classical
  have h := sum_le_sum fun d (_ : d ∈ (univ : Finset δ)) =>
    log_sum_le (fun c => if f c = d then x c else 0)
      (fun c => if f c = d then y c else 0)
      (fun c => by split_ifs <;> simp_all)
      (fun c => by split_ifs <;> simp_all)
      (fun c => by split_ifs <;> simp_all)
  change InformationEnergy.kl (project f x) (project f y) ≤ _ at h
  have he : ∀ d c,
      (if f c = d then x c else 0) *
          (Real.log (if f c = d then x c else 0) -
            Real.log (if f c = d then y c else 0)) =
        if f c = d then x c * (Real.log (x c) - Real.log (y c)) else 0 := by
    intro d c
    split_ifs <;> simp
  simp_rw [he] at h
  rw [sum_comm] at h
  simpa [InformationEnergy.kl] using h

/-- Conditional-product reference for one (not necessarily normalized) joint slice. -/
noncomputable def reference {β γ : Type*} [Fintype β] [Fintype γ]
    (p : β → γ → ℝ) (b : β) (c : γ) : ℝ :=
  (∑ c', p b c') * (∑ b', p b' c) / (∑ b', ∑ c', p b' c')

private theorem marginals_pos {β γ : Type*} [Fintype β] [Fintype γ]
    (p : β → γ → ℝ) (hp : ∀ b c, 0 ≤ p b c) (b : β) (c : γ)
    (h : 0 < p b c) :
    0 < (∑ c', p b c') ∧ 0 < (∑ b', p b' c) ∧
      0 < (∑ b', ∑ c', p b' c') := by
  have hr : 0 < ∑ c', p b c' :=
    lt_of_lt_of_le h (single_le_sum (fun c' _ => hp b c') (mem_univ c))
  have hc : 0 < ∑ b', p b' c :=
    lt_of_lt_of_le h (single_le_sum (fun b' _ => hp b' c) (mem_univ b))
  exact ⟨hr, hc, lt_of_lt_of_le hr
    (single_le_sum (fun b' _ => sum_nonneg fun c' _ => hp b' c') (mem_univ b))⟩

theorem reference_nonneg {β γ : Type*} [Fintype β] [Fintype γ]
    (p : β → γ → ℝ) (hp : ∀ b c, 0 ≤ p b c) (b : β) (c : γ) :
    0 ≤ reference p b c := by
  unfold reference
  exact div_nonneg (mul_nonneg (sum_nonneg fun c' _ => hp b c')
    (sum_nonneg fun b' _ => hp b' c))
    (sum_nonneg fun b' _ => sum_nonneg fun c' _ => hp b' c')

theorem reference_pos_of_pos {β γ : Type*} [Fintype β] [Fintype γ]
    (p : β → γ → ℝ) (hp : ∀ b c, 0 ≤ p b c) (b : β) (c : γ)
    (h : 0 < p b c) : 0 < reference p b c := by
  obtain ⟨hr, hc, ht⟩ := marginals_pos p hp b c h
  exact div_pos (mul_pos hr hc) ht

/-- The conditional-product reference has exactly the source's total mass,
including when that mass is zero. -/
theorem reference_sum {β γ : Type*} [Fintype β] [Fintype γ]
    (p : β → γ → ℝ) :
    (∑ b, ∑ c, reference p b c) = ∑ b, ∑ c, p b c := by
  unfold reference
  simp only [div_eq_mul_inv, ← sum_mul, ← mul_sum]
  have hc : (∑ c, ∑ b, p b c) = ∑ b, ∑ c, p b c := sum_comm
  rw [hc]
  by_cases ht : (∑ b, ∑ c, p b c) = 0
  · simp [ht]
  · field_simp

/-- CMI is a sum of relative entropies against the conditional-product reference.
This identity handles empty slices and zero entries directly. -/
theorem condMutualInfo_eq_kl {α β γ : Type*}
    [Fintype α] [Fintype β] [Fintype γ] (p : α → β → γ → ℝ)
    (hp : ∀ a b c, 0 ≤ p a b c) :
    Medium.condMutualInfo p =
      ∑ a, ∑ b, InformationEnergy.kl (p a b) (reference (p a) b) := by
  classical
  unfold Medium.condMutualInfo InformationEnergy.kl
  refine sum_congr rfl fun a _ => sum_congr rfl fun b _ => sum_congr rfl fun c _ => ?_
  by_cases h : p a b c = 0
  · simp [h]
  · rw [ite_eq_right h]
    obtain ⟨hr, hc, ht⟩ := marginals_pos (p a) (hp a) b c
      (lt_of_le_of_ne (hp a b c) (Ne.symm h))
    unfold reference
    rw [Real.log_div (mul_ne_zero h ht.ne') (mul_pos hr hc).ne',
      Real.log_mul h ht.ne', Real.log_div (mul_pos hr hc).ne' ht.ne',
      Real.log_mul hr.ne' hc.ne']
    ring

/-- Nonnegative, unnormalized finite conditional mutual information. -/
theorem condMutualInfo_nonneg {α β γ : Type*}
    [Fintype α] [Fintype β] [Fintype γ] (p : α → β → γ → ℝ)
    (hp : ∀ a b c, 0 ≤ p a b c) : 0 ≤ Medium.condMutualInfo p := by
  rw [condMutualInfo_eq_kl p hp]
  refine sum_nonneg fun a _ => ?_
  have h := log_sum_le
    (fun bc : β × γ => p a bc.1 bc.2)
    (fun bc : β × γ => reference (p a) bc.1 bc.2)
    (fun bc => hp a bc.1 bc.2)
    (fun bc => reference_nonneg (p a) (hp a) bc.1 bc.2)
    (fun bc => reference_pos_of_pos (p a) (hp a) bc.1 bc.2)
  simp only [Fintype.sum_prod_type] at h
  rw [reference_sum, sub_self, mul_zero] at h
  simpa [InformationEnergy.kl] using h

/-- A missing entry inside a positive marginal rectangle forces positive CMI.
This applies to sparse, unnormalized physical laws without expanding their logs. -/
theorem condMutualInfo_pos_of_missing_rectangle {α β γ : Type*}
    [Fintype α] [Fintype β] [Fintype γ] (p : α → β → γ → ℝ)
    (hp : ∀ a b c, 0 ≤ p a b c)
    (hgap : ∃ a b c, p a b c = 0 ∧
      0 < (∑ c', p a b c') ∧ 0 < (∑ b', p a b' c)) :
    0 < Medium.condMutualInfo p := by
  let x : α × β × γ → ℝ := fun abc => p abc.1 abc.2.1 abc.2.2
  let y : α × β × γ → ℝ := fun abc => reference (p abc.1) abc.2.1 abc.2.2
  have he : InformationEnergy.kl x y = Medium.condMutualInfo p := by
    rw [condMutualInfo_eq_kl p hp]
    simp only [InformationEnergy.kl, x, y, Fintype.sum_prod_type]
  rw [← he]
  apply kl_pos_of_support_gap x y
    (fun abc => hp abc.1 abc.2.1 abc.2.2)
    (fun abc => reference_nonneg (p abc.1) (hp abc.1) abc.2.1 abc.2.2)
    (fun abc => reference_pos_of_pos (p abc.1) (hp abc.1) abc.2.1 abc.2.2)
  · simp only [x, y, Fintype.sum_prod_type]
    exact (sum_congr rfl fun a _ => reference_sum (p a)).symm
  · obtain ⟨a, b, c, hz, hr, hc⟩ := hgap
    refine ⟨(a, b, c), hz, ?_⟩
    have ht : 0 < ∑ b', ∑ c', p a b' c' :=
      lt_of_lt_of_le hr (single_le_sum
        (fun b' _ => sum_nonneg fun c' _ => hp a b' c') (mem_univ b))
    exact div_pos (mul_pos hr hc) ht

private theorem project_sum {β γ δ : Type*} [Fintype β] [Fintype γ]
    (f : γ → δ) (p : β → γ → ℝ) (d : δ) :
    (∑ b, project f (p b) d) = project f (fun c => ∑ b, p b c) d := by
  classical
  unfold project
  rw [sum_comm]
  exact sum_congr rfl fun c _ => by split_ifs <;> simp

/-- The conditional-product reference commutes with observing the third variable. -/
theorem reference_project {β γ δ : Type*} [Fintype β] [Fintype γ] [Fintype δ]
    (f : γ → δ) (p : β → γ → ℝ) (b : β) (d : δ) :
    reference (fun b => project f (p b)) b d = project f (reference p b) d := by
  classical
  unfold reference
  simp_rw [sum_project, project_sum]
  unfold project
  have he : ∀ c,
      (if f c = d then (∑ c', p b c') * (∑ b', p b' c) /
          (∑ b', ∑ c', p b' c') else 0) =
        (∑ c', p b c') * (if f c = d then ∑ b', p b' c else 0) /
          (∑ b', ∑ c', p b' c') := by
    intro c
    split_ifs <;> simp
  simp_rw [he, div_eq_mul_inv]
  rw [← sum_mul, ← mul_sum]

/-- **Finite deterministic CMI data processing**, including arbitrary sparse supports
and unnormalized nonnegative weights. No strict-positivity or stationarity premise. -/
theorem condMutualInfo_project_le {α β γ δ : Type*}
    [Fintype α] [Fintype β] [Fintype γ] [Fintype δ]
    (p : α → β → γ → ℝ) (f : γ → δ) (hp : ∀ a b c, 0 ≤ p a b c) :
    Medium.condMutualInfo (fun a b => project f (p a b)) ≤ Medium.condMutualInfo p := by
  rw [condMutualInfo_eq_kl _ (fun a b d => project_nonneg f (p a b) (hp a b) d),
    condMutualInfo_eq_kl p hp]
  refine sum_le_sum fun a _ => sum_le_sum fun b _ => ?_
  have he : reference (fun b => project f (p a b)) b =
      project f (reference (p a) b) := funext fun d => reference_project f (p a) b d
  rw [he]
  exact kl_project_le f (p a b) (reference (p a) b) (hp a b)
    (reference_nonneg (p a) (hp a) b) (reference_pos_of_pos (p a) (hp a) b)

end CGB.InformationProjection
