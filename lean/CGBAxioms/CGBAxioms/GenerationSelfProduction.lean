import CGBAxioms.LanguageLifeAdmission
import CGBAxioms.RenewableLearningControls
import CGBAxioms.RenewableLearningClosure
import CGBAxioms.GenerationAssociationCapacity
import CGBAxioms.GenerationColumnMemory
import CGBAxioms.GraceRecoveryClosure
import CGBAxioms.GraceRecoveryAccounting
import CGBAxioms.SensoryMaintenanceLimits
import CGBAxioms.SensoryMaintenanceClosure

/-!
# Conditional existence of an open self-producing niche

This module is deliberately world-independent.  It does not assert that a
language world supplies the contract; it proves that once a state carrier,
action-conditioned boundary response and viable invariant region are supplied,
one rule can generate an unbounded trajectory and maintain the specified
region. A mathematical consequence function can use the supplied response;
this does not establish that an organism has learned a world model.
-/

namespace CGB.SelfProduction

open CGB.LanguageLifeAdmission
open CGB.ActiveInference
open CGB.CausalLearning
open CGB.InformationEconomics
open CGB.LanguageLifeKernel

/-- On a finite alphabet with at least two symbols, a positive finite
two-phase score step strictly improves the probability of the observed
symbol on the same frozen feature. It does not assert future improvement. -/
theorem two_phase_finite_score_step_improves {B : Type*} [Fintype B] [DecidableEq B]
    (z : B → ℝ) (target : B) (η : ℝ) (hη : 0 < η)
    (hother : ∃ b, b ≠ target) :
    CGB.Medium.softmax z target < CGB.Medium.softmax
      (fun b => z b + η * ((if b = target then 1 else 0) - CGB.Medium.softmax z b)) target := by
  let _ : Nonempty B := ⟨target⟩
  let p := CGB.Medium.softmax z
  let d := fun b => η * ((if b = target then 1 else 0) - p b)
  have hden : 0 < ∑ b, Real.exp (z b) :=
    Finset.sum_pos (fun _ _ => Real.exp_pos _) Finset.univ_nonempty
  have hp (b : B) : 0 < p b := div_pos (Real.exp_pos _) hden
  have hpt : p target ≤ 1 := by
    have hsum := CGB.Medium.softmax_sum z
    have hle : p target ≤ ∑ b, p b :=
      Finset.single_le_sum (fun b _ => (hp b).le) (Finset.mem_univ target)
    linarith
  have hd (b : B) : d b ≤ d target := by
    by_cases he : b = target
    · subst b; exact le_rfl
    · dsimp [d]; simp only [he, ite_false, ite_true];
      have := hp b
      nlinarith
  have hdstrict (b : B) (he : b ≠ target) : d b < d target := by
    dsimp [d]; simp only [he, ite_false, ite_true]
    have := hp b
    nlinarith
  have hsum : (∑ b, Real.exp (z b) * Real.exp (d b)) <
      ∑ b, Real.exp (z b) * Real.exp (d target) := by
    apply Finset.sum_lt_sum
    · intro b _
      exact mul_le_mul_of_nonneg_left (Real.exp_le_exp.mpr (hd b)) (Real.exp_pos _).le
    · obtain ⟨b, hb⟩ := hother
      exact ⟨b, Finset.mem_univ b,
        mul_lt_mul_of_pos_left (Real.exp_lt_exp.mpr (hdstrict b hb)) (Real.exp_pos _)⟩
  have hmean : (∑ b, Real.exp (z b) * Real.exp (d b)) /
      (∑ b, Real.exp (z b)) < Real.exp (d target) := by
    apply (div_lt_iff₀ hden).2
    simpa only [← Finset.sum_mul, mul_comm] using hsum
  have hiff := CGB.finite_reweighting_improves_iff {target}
    (fun b => Real.exp (z b)) (fun b => Real.exp (d b))
    (fun _ => Real.exp_pos _) (fun _ => Real.exp_pos _) (Finset.singleton_nonempty target)
  have hresult := hiff.mpr (by simpa using hmean)
  simpa [CGB.Medium.softmax, d, p, Real.exp_add] using hresult

/-- If every byte logit changes by at most δ, the observed-byte negative
log likelihood changes by at most 2δ. This is a per-readout bound, not a
claim that the approximate state follows the reference on later ticks. -/
theorem softmax_nll_logit_error_bound {B : Type*} [Fintype B] [Nonempty B]
    (z z' : B → ℝ) (target : B) (δ : ℝ)
    (h : ∀ b, |z' b - z b| ≤ δ) :
    |(-Real.log (CGB.Medium.softmax z' target)) -
        (-Real.log (CGB.Medium.softmax z target))| ≤ 2 * δ := by
  have hpos (x : B → ℝ) : 0 < ∑ b, Real.exp (x b) :=
    Finset.sum_pos (fun _ _ => Real.exp_pos _) Finset.univ_nonempty
  have hlogsum (x y : B → ℝ) (hxy : ∀ b, y b ≤ x b + δ) :
      Real.log (∑ b, Real.exp (y b)) ≤ Real.log (∑ b, Real.exp (x b)) + δ := by
    have hs : (∑ b, Real.exp (y b)) ≤ Real.exp δ * ∑ b, Real.exp (x b) := by
      rw [Finset.mul_sum]
      apply Finset.sum_le_sum
      intro b _
      calc
        Real.exp (y b) ≤ Real.exp (x b + δ) := Real.exp_le_exp.mpr (hxy b)
        _ = Real.exp δ * Real.exp (x b) := by rw [Real.exp_add]; ring
    have hl := Real.log_le_log (hpos y) hs
    rw [Real.log_mul (ne_of_gt (Real.exp_pos δ)) (ne_of_gt (hpos x)), Real.log_exp] at hl
    linarith
  have hforward : ∀ b, z' b ≤ z b + δ := by
    intro b
    have hb := (abs_le.mp (h b)).2
    linarith
  have hreverse : ∀ b, z b ≤ z' b + δ := by
    intro b
    have hb := (abs_le.mp (h b)).1
    linarith
  have h1 := hlogsum z z' hforward
  have h2 := hlogsum z' z hreverse
  have ht := abs_le.mp (h target)
  have hnll (x : B → ℝ) : -Real.log (CGB.Medium.softmax x target) =
      Real.log (∑ b, Real.exp (x b)) - x target := by
    rw [CGB.Medium.softmax, Real.log_div (ne_of_gt (Real.exp_pos _)) (ne_of_gt (hpos x)), Real.log_exp]
    ring
  rw [hnll z', hnll z]
  apply abs_le.mpr
  constructor <;> linarith

/-- The associative interference certificate propagates through an affine
language head into a per-byte NLL bound. The reference logits use the value
associated with the same probe. No future distribution or training result is
assumed. An old-state residual, if present, must be included separately. -/
theorem association_affine_language_nll_bound {dv dk n nb : ℕ}
    (k q : Fin n → Fin dk → ℝ) (v : Fin n → Fin dv → ℝ)
    (head : Fin nb → Fin dv → ℝ) (bias : Fin nb → ℝ)
    (hv : ∀ i o, |v i o| ≤ 1) (i : Fin n) (target : Fin nb)
    (c : ℝ) (hc : ∀ b, (∑ o, |head b o|) ≤ c) :
    let err := ∑ j, |(∑ t, k j t * q i t) - (if j = i then 1 else 0)|
    let actual := fun b => bias b + ∑ o, head b o *
      CGB.Generation.associationRead (CGB.Generation.associationState k v) (q i) o
    let ideal := fun b => bias b + ∑ o, head b o * v i o
    |(-Real.log (CGB.Medium.softmax actual target)) -
      (-Real.log (CGB.Medium.softmax ideal target))| ≤ 2 * (err * c) := by
  classical
  let _ : Nonempty (Fin nb) := ⟨target⟩
  dsimp only
  let err := ∑ j, |(∑ t, k j t * q i t) - (if j = i then 1 else 0)|
  have herr : 0 ≤ err := Finset.sum_nonneg (fun _ _ => abs_nonneg _)
  apply softmax_nll_logit_error_bound
  intro b
  have hdif : (bias b + ∑ o, head b o *
      CGB.Generation.associationRead (CGB.Generation.associationState k v) (q i) o) -
      (bias b + ∑ o, head b o * v i o) =
      ∑ o, head b o * (CGB.Generation.associationRead
        (CGB.Generation.associationState k v) (q i) o - v i o) := by
    simp only [mul_sub, Finset.sum_sub_distrib]
    ring
  rw [hdif]
  calc
    |∑ o, head b o * (CGB.Generation.associationRead
        (CGB.Generation.associationState k v) (q i) o - v i o)| ≤
      ∑ o, |head b o * (CGB.Generation.associationRead
        (CGB.Generation.associationState k v) (q i) o - v i o)| :=
      Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ o, |head b o| * err := by
      apply Finset.sum_le_sum
      intro o _
      rw [abs_mul]
      exact mul_le_mul_of_nonneg_left
        (CGB.Generation.association_readout_interference_bound k q v hv i o)
        (abs_nonneg _)
    _ = err * (∑ o, |head b o|) := by
      rw [← Finset.sum_mul]
      ring
    _ ≤ err * c := mul_le_mul_of_nonneg_left (hc b) herr

/-- For a fixed future query, externally retrieved views share the burden of
distinguishing histories with the finite body. If every listed history needs
a different answer, the number of histories cannot exceed the product of
body states and possible returned views. No assumption stores the external
world inside the body; retrieval itself must actually supply the view. -/
theorem delayed_readout_capacity_with_retrieval
    {History State View Answer : Type*}
    [Fintype History] [Fintype State] [Fintype View]
    (internal : History → State) (returned : History → View)
    (required : History → Answer) (read : State → View → Answer)
    (hrequired : Function.Injective required)
    (hcorrect : ∀ h, read (internal h) (returned h) = required h) :
    Fintype.card History ≤ Fintype.card State * Fintype.card View := by
  have hinj : Function.Injective (fun h => (internal h, returned h)) := by
    intro h h' same
    apply hrequired
    rw [← hcorrect h, ← hcorrect h']
    exact congrArg (fun p : State × View => read p.1 p.2) same
  simpa using Fintype.card_le_of_injective (fun h => (internal h, returned h)) hinj

/-- Assemble the existing row delta and associative readout contracts on the
same value-by-key matrix. Unit-key write error contracts by `1 - β`; any
query changes by its overlap with the written key times the innovation.
This is the mean-state reference update, not the Bayesian assimilation path. -/
theorem association_delta_read_contract
    {dv dk : ℕ} (state : Fin dv → Fin dk → ℝ)
    (key query : Fin dk → ℝ) (value : Fin dv → ℝ) (β : ℝ)
    (hkey : CGB.Generation.dot key key = 1) :
    (∀ o, CGB.Generation.associationRead
        (fun i => CGB.Generation.tiedWrite key β (value i) (state i)) key o - value o =
      (1 - β) * (CGB.Generation.associationRead state key o - value o)) ∧
    (∀ o, CGB.Generation.associationRead
        (fun i => CGB.Generation.tiedWrite key β (value i) (state i)) query o -
      CGB.Generation.associationRead state query o =
      β * (value o - CGB.Generation.associationRead state key o) *
        CGB.Generation.dot query key) := by
  constructor
  · intro o
    simpa [CGB.Generation.associationRead, CGB.Generation.dot, mul_comm] using
      CGB.Generation.tied_write_error_shrinks key (state o) hkey β (value o)
  · intro o
    simpa [CGB.Generation.associationRead, CGB.Generation.dot, mul_comm] using
      CGB.Generation.tied_write_cross_read key query (state o) β (value o)

/-- For an exactly one-hot key, no global decay, and precision-derived gain,
the delta write agrees with slotwise precision assimilation: only the keyed
slot changes. This does not assert equality for mixed keys or nonzero decay. -/
theorem one_hot_delta_is_precision_assimilation
    {n : ℕ} (slot : Fin n) (row : Fin n → ℝ) (p w value : ℝ)
    (hp : 0 < p) (hw : 0 ≤ w) :
    CGB.Generation.tiedWrite (fun j => if j = slot then 1 else 0)
      (w / (p + w)) value row =
      fun j => if j = slot then (p * row j + w * value) / (p + w) else row j := by
  funext j
  by_cases hj : j = slot
  · subst j
    simp [CGB.Generation.tiedWrite, CGB.Generation.dot]
    exact (CGB.Generation.precision_assimilation_is_innovation p w (row slot) value hp hw).symm
  · simp [CGB.Generation.tiedWrite, hj]

/-- Two slots with the same current mean but different positive precisions
produce different next means on the same nonmatching observation. Hence a
deterministic transition reading only that mean cannot reproduce both. -/
theorem precision_distinguishes_equal_mean_histories
    (p₁ p₂ w mean observed : ℝ)
    (hp₁ : 0 < p₁) (hp₂ : 0 < p₂) (hw : 0 < w)
    (hpne : p₁ ≠ p₂) (hne : observed ≠ mean) :
    let next₁ := (p₁ * mean + w * observed) / (p₁ + w)
    let next₂ := (p₂ * mean + w * observed) / (p₂ + w)
    next₁ ≠ next₂ ∧
      ¬ ∃ transition : ℝ → ℝ → ℝ,
        transition mean observed = next₁ ∧ transition mean observed = next₂ := by
  dsimp only
  have hd₁ : p₁ + w ≠ 0 := ne_of_gt (by positivity)
  have hd₂ : p₂ + w ≠ 0 := ne_of_gt (by positivity)
  have hdiff : (p₁ * mean + w * observed) / (p₁ + w) ≠
      (p₂ * mean + w * observed) / (p₂ + w) := by
    intro he
    field_simp at he
    have hprod : w * (p₁ - p₂) * (observed - mean) = 0 := by nlinarith [he]
    exact (mul_ne_zero (mul_ne_zero (ne_of_gt hw) (sub_ne_zero.mpr hpne))
      (sub_ne_zero.mpr hne)) hprod
  exact ⟨hdiff, fun ⟨_, h₁, h₂⟩ => hdiff (h₁.symm.trans h₂)⟩

/-- A normalized mixed key gives a concrete obstruction to general equality:
the delta write couples the first slot to the second, while independent
precision assimilation with zero target leaves its zero mean unchanged. -/
theorem mixed_key_delta_differs_from_slot_assimilation :
    let key : Fin 2 → ℝ := ![3 / 5, 4 / 5]
    let row : Fin 2 → ℝ := ![0, 1]
    CGB.Generation.dot key key = 1 ∧
    CGB.Generation.tiedWrite key (1 / 2) 0 row 0 = -(6 / 25 : ℝ) ∧
    ((1 : ℝ) * row 0 + (1 / 2 : ℝ) * 0) / (1 + 1 / 2) = 0 := by
  norm_num [CGB.Generation.dot, CGB.Generation.tiedWrite, Fin.sum_univ_two]

/-- Strictly positive soft addresses necessarily overlap. Consequently a
nonzero delta write changes every strictly positive raw query in any output
coordinate with nonzero innovation. This is exact-real arithmetic; floating
point softmax can underflow and dual or signed queries are not covered. -/
theorem positive_soft_query_changes_under_delta
    {dk : ℕ} [NeZero dk] (key query row : Fin dk → ℝ) (β value : ℝ)
    (hk : ∀ i, 0 < key i) (hq : ∀ i, 0 < query i)
    (hb : β ≠ 0) (he : value ≠ CGB.Generation.dot key row) :
    0 < CGB.Generation.dot query key ∧
      CGB.Generation.dot query (CGB.Generation.tiedWrite key β value row) ≠
        CGB.Generation.dot query row := by
  have hpos : 0 < CGB.Generation.dot query key := by
    unfold CGB.Generation.dot
    apply Finset.sum_pos
    · intro i _
      exact mul_pos (hq i) (hk i)
    · exact ⟨⟨0, NeZero.pos dk⟩, Finset.mem_univ _⟩
  refine ⟨hpos, ?_⟩
  intro same
  have h := CGB.Generation.tied_write_cross_read key query row β value
  rw [same, sub_self] at h
  exact (mul_ne_zero (mul_ne_zero hb (sub_ne_zero.mpr he)) (ne_of_gt hpos)) h.symm

/-- Query change under the existing natural-parameter assimilation: `weight`
is the per-slot sum of observation weights and `weighted` the corresponding
value-weighted sum. The identity includes zero observation weight and allows
slot contributions to cancel; no separate resource state is introduced. -/
theorem association_assimilation_query_change
    {dv dk : ℕ} (state weighted : Fin dv → Fin dk → ℝ)
    (precision weight query : Fin dk → ℝ)
    (hp : ∀ j, 0 < precision j) (hw : ∀ j, 0 ≤ weight j) (o : Fin dv) :
    CGB.Generation.associationRead
      (fun i j => (precision j * state i j + weighted i j) /
        (precision j + weight j)) query o -
    CGB.Generation.associationRead state query o =
      ∑ j, query j * ((weighted o j - weight j * state o j) /
        (precision j + weight j)) := by
  simp only [CGB.Generation.associationRead, ← Finset.sum_sub_distrib]
  apply Finset.sum_congr rfl
  intro j _
  have hden : precision j + weight j ≠ 0 := ne_of_gt (by linarith [hp j, hw j])
  field_simp; ring

/-- On a fixed query and target, the existing assimilation improves squared
prediction error exactly when its query change points sufficiently towards the
target without overshooting. The target is an externally specified future
value, not assumed to be known during the update. -/
theorem association_assimilation_improves_iff
    {dv dk : ℕ} (state weighted : Fin dv → Fin dk → ℝ)
    (precision weight query : Fin dk → ℝ)
    (hp : ∀ j, 0 < precision j) (hw : ∀ j, 0 ≤ weight j)
    (o : Fin dv) (target : ℝ) :
    let old := CGB.Generation.associationRead state query o
    let delta := ∑ j, query j * ((weighted o j - weight j * state o j) /
      (precision j + weight j))
    let updated := CGB.Generation.associationRead
      (fun i j => (precision j * state i j + weighted i j) /
        (precision j + weight j)) query o
    (updated - target) ^ 2 < (old - target) ^ 2 ↔
      delta * (2 * (old - target) + delta) < 0 := by
  dsimp only
  have h := association_assimilation_query_change state weighted precision weight query hp hw o
  have hnew : CGB.Generation.associationRead
      (fun i j => (precision j * state i j + weighted i j) /
        (precision j + weight j)) query o =
      CGB.Generation.associationRead state query o +
        ∑ j, query j * ((weighted o j - weight j * state o j) /
          (precision j + weight j)) := by linarith
  rw [hnew]
  constructor <;> intro hgain <;> nlinarith

/-- Under slotwise assimilation and an arbitrary affine-head update, a
target's softmax probability rises exactly when its multiplicative change
exceeds the old distribution's average. The change includes head, query and
state changes; no learned head or update rule is assumed. -/
theorem association_assimilation_softmax_improves_iff
    {dv dk nb : ℕ} (state weighted : Fin dv → Fin dk → ℝ)
    (precision weight query query' : Fin dk → ℝ)
    (head head' : Fin nb → Fin dv → ℝ) (bias bias' : Fin nb → ℝ)
    (hp : ∀ j, 0 < precision j) (hw : ∀ j, 0 ≤ weight j)
    (target : Fin nb) :
    let old := fun b => bias b + ∑ i, head b i * CGB.Generation.associationRead state query i
    let updated := fun b => bias' b + ∑ i, head' b i * CGB.Generation.associationRead
      (fun i j => (precision j * state i j + weighted i j) / (precision j + weight j)) query' i
    let delta := fun b => (bias' b - bias b) +
      ∑ i, ((head' b i - head b i) * CGB.Generation.associationRead state query i +
        head' b i * ((∑ j, state i j * (query' j - query j)) +
          ∑ j, query' j * ((weighted i j - weight j * state i j) / (precision j + weight j))))
    CGB.Medium.softmax old target < CGB.Medium.softmax updated target ↔
      (∑ i, Real.exp (old i) * Real.exp (delta i)) /
        (∑ i, Real.exp (old i)) < Real.exp (delta target) := by
  dsimp only
  let old := fun b => bias b + ∑ i, head b i * CGB.Generation.associationRead state query i
  let updated := fun b => bias' b + ∑ i, head' b i * CGB.Generation.associationRead
    (fun i j => (precision j * state i j + weighted i j) / (precision j + weight j)) query' i
  let delta := fun b => (bias' b - bias b) +
    ∑ i, ((head' b i - head b i) * CGB.Generation.associationRead state query i +
      head' b i * ((∑ j, state i j * (query' j - query j)) +
        ∑ j, query' j * ((weighted i j - weight j * state i j) / (precision j + weight j))))
  have hquery : ∀ i, CGB.Generation.associationRead state query' i -
      CGB.Generation.associationRead state query i = ∑ j, state i j * (query' j - query j) := by
    intro i
    simp only [CGB.Generation.associationRead, ← Finset.sum_sub_distrib]
    apply Finset.sum_congr rfl
    intro j _
    ring
  have hread : ∀ i, CGB.Generation.associationRead
      (fun i j => (precision j * state i j + weighted i j) / (precision j + weight j)) query' i =
      CGB.Generation.associationRead state query i +
        ((∑ j, state i j * (query' j - query j)) +
          ∑ j, query' j * ((weighted i j - weight j * state i j) / (precision j + weight j))) := by
    intro i
    have h := association_assimilation_query_change state weighted precision weight query' hp hw i
    have hq := hquery i
    linarith
  have hchange : ∀ b, updated b = old b + delta b := by
    intro b
    dsimp [updated, old, delta]
    simp_rw [hread, mul_add, sub_mul, Finset.sum_add_distrib, Finset.sum_sub_distrib]
    ring
  have h := CGB.finite_reweighting_improves_iff {target}
    (fun i => Real.exp (old i)) (fun i => Real.exp (delta i))
    (fun i => Real.exp_pos _) (fun i => Real.exp_pos _)
    (Finset.singleton_nonempty target)
  simp only [Finset.sum_singleton] at h
  have hcancel : Real.exp (old target) * Real.exp (delta target) /
      Real.exp (old target) = Real.exp (delta target) := by
    field_simp
  rw [hcancel] at h
  change CGB.Medium.softmax old target < CGB.Medium.softmax updated target ↔
    (∑ i, Real.exp (old i) * Real.exp (delta i)) /
      (∑ i, Real.exp (old i)) < Real.exp (delta target)
  simpa only [CGB.Medium.softmax, hchange, Real.exp_add] using h

/-- With no intervening prior step, repeated identical slot evidence is
accumulated by the existing natural/precision update. Its resulting mean
retains exactly `p₀ / (p₀ + n*w)` of the initial error. This describes one
slot in exact arithmetic, not a learned language-quality guarantee. -/
theorem repeated_assimilation_retention
    (precision natural : ℕ → ℝ) (p₀ m₀ w target : ℝ)
    (hp : 0 < p₀) (hw : 0 ≤ w)
    (hprec0 : precision 0 = p₀) (hnat0 : natural 0 = p₀ * m₀)
    (hprec : ∀ n, precision (n + 1) = precision n + w)
    (hnat : ∀ n, natural (n + 1) = natural n + w * target) :
    ∀ n, precision n = p₀ + (n : ℝ) * w ∧
      natural n / precision n - target =
        (p₀ / (p₀ + (n : ℝ) * w)) * (m₀ - target) ∧
      ((n : ℝ) * w ≤ p₀ → (1 : ℝ) / 2 ≤ p₀ / (p₀ + (n : ℝ) * w)) := by
  have hclosed : ∀ n, precision n = p₀ + (n : ℝ) * w ∧
      natural n = p₀ * m₀ + (n : ℝ) * w * target := by
    intro n
    induction n with
    | zero => simp [hprec0, hnat0]
    | succ n ih =>
      rw [hprec, hnat, ih.1, ih.2]
      push_cast
      constructor <;> ring
  intro n
  obtain ⟨hnp, hnn⟩ := hclosed n
  refine ⟨hnp, ?_, ?_⟩
  · rw [hnp, hnn]
    have hd : p₀ + (n : ℝ) * w ≠ 0 := by positivity
    field_simp
    ring
  · intro hbudget
    have hd : 0 < p₀ + (n : ℝ) * w := by positivity
    apply (le_div_iff₀ hd).2
    linarith

/-- The one-slot retention law composes into the actual value-by-key
associative readout. Each slot may have its own precision and evidence
weight. The resulting terms can cancel, so the equality does not claim that
an individual association survives merely because a query error is small. -/
theorem repeated_assimilation_association_readout
    {dv dk : ℕ} (precision : ℕ → Fin dk → ℝ)
    (natural : ℕ → Fin dv → Fin dk → ℝ)
    (p₀ w query : Fin dk → ℝ) (m₀ target : Fin dv → Fin dk → ℝ)
    (hp : ∀ j, 0 < p₀ j) (hw : ∀ j, 0 ≤ w j)
    (hprec0 : ∀ j, precision 0 j = p₀ j)
    (hnat0 : ∀ i j, natural 0 i j = p₀ j * m₀ i j)
    (hprec : ∀ n j, precision (n + 1) j = precision n j + w j)
    (hnat : ∀ n i j, natural (n + 1) i j = natural n i j + w j * target i j)
    (n : ℕ) (i : Fin dv) :
    CGB.Generation.associationRead (fun o j => natural n o j / precision n j) query i -
        CGB.Generation.associationRead target query i =
      ∑ j, query j * (p₀ j / (p₀ j + (n : ℝ) * w j)) * (m₀ i j - target i j) := by
  simp only [CGB.Generation.associationRead, ← Finset.sum_sub_distrib]
  apply Finset.sum_congr rfl
  intro j _
  have h := (repeated_assimilation_retention (fun n => precision n j)
    (fun n => natural n i j) (p₀ j) (m₀ i j) (w j) (target i j)
    (hp j) (hw j) (hprec0 j) (hnat0 i j)
    (fun n => hprec n j) (fun n => hnat n i j) n).2.1
  have hm := congrArg (fun z : ℝ => z * query j) h
  nlinarith [hm]

/-- A family of future queries remains accurate when its target readout is
accurate and the summed retained slot deviations fit the remaining tolerance.
This gives an explicit no-cancellation sufficient certificate for mixed slots. -/
theorem repeated_assimilation_probe_family
    {dv dk m : ℕ} (precision : ℕ → Fin dk → ℝ)
    (natural : ℕ → Fin dv → Fin dk → ℝ)
    (p₀ w : Fin dk → ℝ) (m₀ target : Fin dv → Fin dk → ℝ)
    (keys : Fin m → Fin dk → ℝ) (answers : Fin m → Fin dv → ℝ)
    (ε₀ ε : ℝ) (n : ℕ)
    (hp : ∀ j, 0 < p₀ j) (hw : ∀ j, 0 ≤ w j)
    (hprec0 : ∀ j, precision 0 j = p₀ j)
    (hnat0 : ∀ i j, natural 0 i j = p₀ j * m₀ i j)
    (hprec : ∀ t j, precision (t + 1) j = precision t j + w j)
    (hnat : ∀ t i j, natural (t + 1) i j = natural t i j + w j * target i j)
    (hε : 0 ≤ ε) (htarget : CGB.Generation.probeFamilyAccurate target keys answers ε₀)
    (hbudget : ∀ r i, (∑ j, |keys r j * (p₀ j / (p₀ j + (n : ℝ) * w j)) *
      (m₀ i j - target i j)|) + ε₀ ≤ ε) :
    CGB.Generation.probeFamilyAccurate
      (fun i j => natural n i j / precision n j) keys answers ε := by
  refine ⟨hε, ?_⟩
  intro r i
  have hid := repeated_assimilation_association_readout precision natural p₀ w (keys r)
    m₀ target hp hw hprec0 hnat0 hprec hnat n i
  have htri := abs_add_le
    (CGB.Generation.associationRead (fun o j => natural n o j / precision n j) (keys r) i -
      CGB.Generation.associationRead target (keys r) i)
    (CGB.Generation.associationRead target (keys r) i - answers r i)
  have hsum := Finset.abs_sum_le_sum_abs (s := Finset.univ)
    (f := fun j : Fin dk => keys r j * (p₀ j / (p₀ j + (n : ℝ) * w j)) *
      (m₀ i j - target i j))
  rw [hid] at htri
  have heq : (∑ j, keys r j * (p₀ j / (p₀ j + (n : ℝ) * w j)) *
      (m₀ i j - target i j)) +
      (CGB.Generation.associationRead target (keys r) i - answers r i) =
      CGB.Generation.associationRead (fun o j => natural n o j / precision n j) (keys r) i -
        answers r i := by rw [← hid]; ring
  rw [heq] at htri
  have ht := htarget.2 r i
  have hb := hbudget r i
  linarith

/-- In a declared joint distribution, an oracle using the actual conditional
byte law at each body state has no higher expected log loss than any fixed
strictly positive normalized predictor. The weights are state probabilities;
no additional resource state is introduced. This does not establish that the
body can learn the conditional law, or that prediction improves viability. -/
theorem state_conditioned_logloss_le_fixed
    {State Byte : Type*} [Fintype State] [Fintype Byte]
    (weight : State → ℝ) (law : State → Byte → ℝ) (fixed : Byte → ℝ)
    (hweight : ∀ s, 0 ≤ weight s)
    (hlaw0 : ∀ s b, 0 ≤ law s b) (hlaw : ∀ s, ∑ b, law s b = 1)
    (hfixed0 : ∀ b, 0 < fixed b) (hfixed : ∑ b, fixed b = 1) :
    (∑ s, weight s * PredictiveHarvest.entropy (law s)) ≤
      ∑ s, weight s * PredictiveHarvest.crossEntropy (law s) fixed := by
  refine Finset.sum_le_sum fun s _ => mul_le_mul_of_nonneg_left ?_ (hweight s)
  unfold PredictiveHarvest.entropy PredictiveHarvest.crossEntropy
  have hg := PredictiveHarvest.gibbs_inequality (law s) fixed (hlaw0 s)
    (hlaw s) hfixed0 hfixed
  linarith

/-- If two equally likely histories share the same accessible state but
require opposite next-byte predictions, any positive binary predictor pays
at least `log 2` nats of expected loss. A predictor with the distinguishing
information can assign probability one to the correct byte and has zero loss.
The equal probabilities are a world condition, not a universal drift law. -/
theorem merged_histories_binary_logloss_floor
    {History State : Type*} (encode : History → State) (predict : State → ℝ)
    (h₀ h₁ : History) (hmerge : encode h₀ = encode h₁)
    (hp : 0 < predict (encode h₀)) (hp1 : predict (encode h₀) < 1) :
    -(Real.log (1 - predict (encode h₀)) + Real.log (predict (encode h₁))) / 2 ≥
      Real.log 2 := by
  rw [← hmerge]
  let p := predict (encode h₀)
  have hp' : 0 < p := hp
  have hq : 0 < 1 - p := by linarith [hp1]
  have hproduct : 0 < (1 - p) * p := mul_pos hq hp'
  have hupper : (1 - p) * p ≤ (1 / 2 : ℝ) ^ 2 := by
    nlinarith [sq_nonneg (p - 1 / 2)]
  have hlog := Real.log_le_log hproduct hupper
  rw [Real.log_mul (ne_of_gt hq) (ne_of_gt hp'), Real.log_pow] at hlog
  have hhalf : Real.log (1 / 2 : ℝ) = -Real.log 2 := by
    rw [show (1 / 2 : ℝ) = (2 : ℝ)⁻¹ by norm_num, Real.log_inv]
  rw [hhalf] at hlog
  norm_num at hlog
  change -(Real.log (1 - p) + Real.log p) / 2 ≥ Real.log 2
  linarith

/-- For a state-dependent predictor, expected log loss is the conditional
entropy plus the state-weighted KL mismatch. This uses a specified joint law,
not a new resource account, and does not assert the predictor is learned. -/
theorem state_conditioned_logloss_decomposition
    {State Byte : Type*} [Fintype State] [Fintype Byte]
    (weight : State → ℝ) (law predictor : State → Byte → ℝ)
    (hweight : ∀ s, 0 ≤ weight s)
    (hlaw0 : ∀ s b, 0 ≤ law s b) (hlaw : ∀ s, ∑ b, law s b = 1)
    (hpredictor0 : ∀ s b, 0 < predictor s b)
    (hpredictor : ∀ s, ∑ b, predictor s b = 1) :
    (∑ s, weight s * PredictiveHarvest.crossEntropy (law s) (predictor s)) =
      (∑ s, weight s * PredictiveHarvest.entropy (law s)) +
        (∑ s, weight s * InformationEnergy.kl (law s) (predictor s)) ∧
    0 ≤ ∑ s, weight s * InformationEnergy.kl (law s) (predictor s) := by
  have hpoint (s : State) : PredictiveHarvest.crossEntropy (law s) (predictor s) =
      PredictiveHarvest.entropy (law s) + InformationEnergy.kl (law s) (predictor s) := by
    unfold PredictiveHarvest.crossEntropy PredictiveHarvest.entropy InformationEnergy.kl
    have hterm : ∀ b, law s b * (Real.log (law s b) - Real.log (predictor s b)) =
        law s b * Real.log (law s b) - law s b * Real.log (predictor s b) := by
      intro b
      ring
    simp_rw [hterm, Finset.sum_sub_distrib]
    ring
  constructor
  · simp_rw [hpoint, mul_add, Finset.sum_add_distrib]
  · exact Finset.sum_nonneg fun s _ => mul_nonneg (hweight s)
      (InformationEnergy.kl_nonneg (law s) (predictor s) (hlaw0 s)
        (hlaw s) (hpredictor0 s) (hpredictor s))

/-- The existing renewable witness actually reopens an excluded candidate;
no single old `advance` step from the same memory can implement that change. -/
theorem renewable_reopening_not_single_old_advance
    (a : CGB.RenewableLearning.Action) (o : CGB.LearningMaintenance.Outcome) :
    CGB.RenewableLearning.memory
      (CGB.RenewableLearning.next true true .miss .aReady) ≠
    CausalLearning.advance CGB.RenewableLearning.possible
      (CGB.RenewableLearning.memory .aReady) () a o := by
  fin_cases a <;> fin_cases o <;> decide

/-- The same existing renewable process has arbitrarily long supported
prefixes with more than any requested number of paid writes. This is a
finite-prefix support statement, not an almost-sure infinite-path claim. -/
theorem renewable_witness_unbounded_supported_writes :
    ∀ B : ℕ, ∃ T : ℕ,
      0 < CGB.RenewableLearningControls.weight
        (CGB.RenewableLearning.process true true false)
        CGB.RenewableLearningControls.path T ∧
      B < ∑ t ∈ Finset.range T,
        CGB.RenewableLearning.writeCost true
          (CGB.RenewableLearning.sense (CGB.RenewableLearningControls.path t).2 ())
          ((CGB.RenewableLearningControls.path t).1 ()) := by
  intro B
  refine ⟨3 + 10 * (B + 1), ?_⟩
  have h := CGB.RenewableLearningControls.arbitrary_relearning (B + 1)
  refine ⟨h.1, ?_⟩
  rw [h.2.1]
  omega

/-- One stochastic body-world process has operational maintenance and
arbitrarily many paid writes on supported finite prefixes; blocking reopening
in the same world and initial conditions lowers the registered survival score
by at least one half at tick six. The intervention also changes the declared
outgoing accounting, and the prefix claim is not almost-sure infinite writing. -/
theorem renewable_maintenance_and_supported_relearning :
    CGB.Exchange.Persists CGB.RenewableLearningClosure.body
      CGB.RenewableLearning.world CGB.RenewableLearningClosure.ec
      CGB.RenewableLearningClosure.material 0 ∧
    CGB.Exchange.SelfMaintains CGB.RenewableLearningClosure.body
      CGB.RenewableLearning.world CGB.RenewableLearningClosure.ec
      CGB.RenewableLearningClosure.material ∧
    (∀ B : ℕ, ∃ T : ℕ,
      0 < CGB.RenewableLearningControls.weight
        (CGB.RenewableLearning.process true true false)
        CGB.RenewableLearningControls.path T ∧
      B < ∑ t ∈ Finset.range T,
        CGB.RenewableLearning.writeCost true
          (CGB.RenewableLearning.sense (CGB.RenewableLearningControls.path t).2 ())
          ((CGB.RenewableLearningControls.path t).1 ())) ∧
    CGB.Exchange.survival (CGB.RenewableLearning.plant true true)
      CGB.RenewableLearning.world (CGB.RenewableLearning.he true true)
      (CGB.RenewableLearning.ledger true true) false
      (fun _ => CGB.RenewableLearning.world.sense) 6 = 1 ∧
    CGB.Exchange.survival (CGB.RenewableLearning.plant false true)
      CGB.RenewableLearning.world (CGB.RenewableLearning.he false true)
      (CGB.RenewableLearning.ledger false true) false
      (fun _ => CGB.RenewableLearning.world.sense) 6 ≤ 1 / 2 := by
  exact ⟨CGB.RenewableLearningClosure.maintenance.1,
    CGB.RenewableLearningClosure.maintenance.2,
    renewable_witness_unbounded_supported_writes,
    CGB.RenewableLearningLife.natural_survival 6,
    CGB.RenewableLearningControls.paired_failure false true (by simp)⟩

/-- The already-certified stochastic process also has a concrete world-side
action contrast. On live bodies its world law depends on the body only through
the selected action. Two body states select different actions and give
different next-world probabilities. This is not a fixed-body action
intervention or a proof about a language world. -/
theorem renewable_maintenance_world_action_contrast :
    CGB.Exchange.SelfMaintains CGB.RenewableLearningClosure.body
      CGB.RenewableLearning.world CGB.RenewableLearningClosure.ec
      CGB.RenewableLearningClosure.material ∧
    (∀ (w v : CGB.RenewableLearning.Env) (q q' : CGB.RenewableLearning.Cell),
      q ≠ .vacuum → q' ≠ .vacuum →
      CGB.RenewableLearning.policy q = CGB.RenewableLearning.policy q' →
      CGB.RenewableLearning.moves w (fun _ => q) v =
        CGB.RenewableLearning.moves w (fun _ => q') v) ∧
    CGB.RenewableLearning.policy .aReady = 0 ∧
    CGB.RenewableLearning.policy .bReady = 1 ∧
    CGB.RenewableLearning.moves .feedA (fun _ => .aReady) .missA = 0 ∧
    CGB.RenewableLearning.moves .feedA (fun _ => .bReady) .missA = 1 := by
  have ha : CGB.RenewableLearning.policy .aReady = 0 := by decide
  have hb : CGB.RenewableLearning.policy .bReady = 1 := by decide
  refine ⟨CGB.RenewableLearningClosure.maintenance.2, ?_, ha, hb, ?_, ?_⟩
  · intro w v q q' hq hq' hpolicy
    cases w <;> simp [CGB.RenewableLearning.moves, hq, hq', hpolicy]
  · simp [CGB.RenewableLearning.moves, ha]
  · simp [CGB.RenewableLearning.moves, hb]

/-- A live representative for each of the old world's three action values.
It is used only to expose the already factorized world transition. -/
def renewableActionRepresentative (a : CGB.RenewableLearning.Action) :
    CGB.RenewableLearning.Cell :=
  if a = 0 then .aReady else if a = 1 then .bReady else .search

/-- The existing world law viewed as a kernel of an explicitly supplied
action. The factorization below makes the choice of live representative
irrelevant. -/
noncomputable def renewableActionWorld
    (w : CGB.RenewableLearning.Env) (a : CGB.RenewableLearning.Action)
    (v : CGB.RenewableLearning.Env) : ℝ :=
  CGB.RenewableLearning.moves w (fun _ => renewableActionRepresentative a) v

/-- This action kernel is normalized and agrees with the existing world for
every live body. Explicit actions induce different next-world laws at a fixed
world state. A do-action interpretation assumes this world mechanism remains
invariant under intervention; multi-step survival is a separate obligation. -/
theorem renewable_action_world_contract :
    (∀ a, renewableActionRepresentative a ≠ .vacuum ∧
      CGB.RenewableLearning.policy (renewableActionRepresentative a) = a) ∧
    (∀ w a, (∀ v, 0 ≤ renewableActionWorld w a v) ∧
      ∑ v, renewableActionWorld w a v = 1) ∧
    (∀ w v q, q ≠ .vacuum →
      CGB.RenewableLearning.moves w (fun _ => q) v =
        renewableActionWorld w (CGB.RenewableLearning.policy q) v) ∧
    renewableActionWorld .feedA 0 .missA = 0 ∧
    renewableActionWorld .feedA 1 .missA = 1 := by
  have hr : ∀ a, renewableActionRepresentative a ≠ .vacuum ∧
      CGB.RenewableLearning.policy (renewableActionRepresentative a) = a := by
    intro a
    fin_cases a <;> decide
  refine ⟨hr, ?_, ?_, ?_, ?_⟩
  · intro w a
    exact ⟨fun v => CGB.RenewableLearning.world.moves_nonneg w
      (fun _ => renewableActionRepresentative a) v,
      CGB.RenewableLearning.world.moves_sum w (fun _ => renewableActionRepresentative a)⟩
  · intro w v q hq
    exact renewable_maintenance_world_action_contrast.2.1 w v q
      (renewableActionRepresentative (CGB.RenewableLearning.policy q)) hq
      (hr _).1 (hr _).2.symm
  · simpa [renewableActionWorld, renewableActionRepresentative] using
      renewable_maintenance_world_action_contrast.2.2.2.2.1
  · simpa [renewableActionWorld, renewableActionRepresentative] using
      renewable_maintenance_world_action_contrast.2.2.2.2.2

/-- The existing stochastic body/world process with its action port exposed.
The body event still uses the current world stimulus and the old event rule;
only the world kernel receives the explicit intervention. Vacuum retains its
original world transition. This is a mathematical intervention interface. -/
noncomputable def renewableActionMachine :
    CGB.Interaction.Machine CGB.RenewableLearning.S
      CGB.RenewableLearning.Action (Option Unit) where
  step a z e z' :=
    (if z.1 () = .vacuum then CGB.RenewableLearning.moves z.2 z.1 z'.2
      else renewableActionWorld z.2 a z'.2) *
    (CGB.RenewableLearning.plant true true).eventStep CGB.RenewableLearning.world.clock
      (CGB.RenewableLearning.world.sense z.2) z.1 e z'.1
  nonneg a z e z' := by
    apply mul_nonneg
    · split_ifs
      · exact CGB.RenewableLearning.world.moves_nonneg _ _ _
      · exact (renewable_action_world_contract.2.1 z.2 a).1 z'.2
    · exact (CGB.RenewableLearning.plant true true).eventStep_nonneg
        (CGB.RenewableLearning.he true true) _ _ _ _
  sum_one a z := by
    have hw : (∑ v, if z.1 () = .vacuum then
        CGB.RenewableLearning.moves z.2 z.1 v else renewableActionWorld z.2 a v) = 1 := by
      split_ifs
      · exact CGB.RenewableLearning.world.moves_sum _ _
      · exact (renewable_action_world_contract.2.1 z.2 a).2
    simp only [Fintype.sum_prod_type, ← Finset.sum_mul, hw, one_mul]
    exact (CGB.RenewableLearning.plant true true).eventStep_sum
      CGB.RenewableLearning.world.clock (CGB.RenewableLearning.world.sense z.2) z.1

/-- Supplying the old policy reproduces every joint event and successor-state
probability of the original uncut process, including the vacuum case. -/
theorem renewable_action_machine_reproduces
    (t : ℕ) (z z' : CGB.RenewableLearning.S) (e : Option Unit) :
    renewableActionMachine.step (CGB.RenewableLearning.policy (z.1 ())) z e z' =
      (CGB.RenewableLearning.process true true false).step t z e z' := by
  change (if z.1 () = .vacuum then CGB.RenewableLearning.moves z.2 z.1 z'.2
      else renewableActionWorld z.2 (CGB.RenewableLearning.policy (z.1 ())) z'.2) *
    (CGB.RenewableLearning.plant true true).eventStep CGB.RenewableLearning.world.clock
      (CGB.RenewableLearning.world.sense z.2) z.1 e z'.1 =
    CGB.RenewableLearning.moves z.2 z.1 z'.2 *
      (CGB.RenewableLearning.plant true true).eventStep CGB.RenewableLearning.world.clock
        (CGB.RenewableLearning.world.sense z.2) z.1 e z'.1
  by_cases hz : z.1 () = .vacuum
  · simp [hz]
  · simp only [hz, ↓reduceIte]
    have hx : (fun _ : Unit => z.1 ()) = z.1 := by funext u; cases u; rfl
    have h := renewable_action_world_contract.2.2.1 z.2 z'.2 (z.1 ()) hz
    rw [hx] at h
    rw [← h]

/-- Restore the existing policy to the exposed action port. -/
noncomputable def renewableActionAutonomous :
    CGB.Interaction.Machine CGB.RenewableLearning.S ℕ (Option Unit) where
  step _ z := renewableActionMachine.step (CGB.RenewableLearning.policy (z.1 ())) z
  nonneg _ z := renewableActionMachine.nonneg (CGB.RenewableLearning.policy (z.1 ())) z
  sum_one _ z := renewableActionMachine.sum_one (CGB.RenewableLearning.policy (z.1 ())) z

/-- Closing the explicit action port with the existing policy recovers the
whole original machine, not just a one-step marginal or a sample trace. -/
theorem renewable_action_autonomous_eq :
    renewableActionAutonomous = CGB.RenewableLearning.process true true false := by
  unfold renewableActionAutonomous CGB.RenewableLearning.process CGB.MaterialSource.machine
  congr 1
  funext t z e z'
  exact renewable_action_machine_reproduces t z z' e

/-- Under the declared action-port intervention, holding action 0 sends the
existing initial state to the trap and then to vacuum in two certain steps. -/
theorem renewable_action_zero_failure_steps :
    renewableActionMachine.step 0 CGB.RenewableLearning.initial (some ())
      ((fun _ => .search), .trap) = 1 ∧
    renewableActionMachine.step 0 ((fun _ => .search), .trap) (some ())
      ((fun _ => .vacuum), .trap) = 1 := by
  have hzero : CGB.RenewableLearning.policy .aReady = 0 := by decide
  constructor <;>
    norm_num [renewableActionMachine, renewableActionWorld, renewableActionRepresentative,
      CGB.RenewableLearning.initial, CGB.RenewableLearning.moves,
      CGB.RenewableLearning.plant, CGB.RenewableLearning.world,
      CGB.RenewableLearning.sense, CGB.RenewableLearning.next,
      CGB.Medium.eventStep, hzero, funext_iff]

/-- Reuse the same intervenable machine with a fixed action at each tick. -/
noncomputable def renewableActionAlways (a : CGB.RenewableLearning.Action) :
    CGB.Interaction.Machine CGB.RenewableLearning.S ℕ (Option Unit) where
  step _ := renewableActionMachine.step a
  nonneg _ := renewableActionMachine.nonneg a
  sum_one _ := renewableActionMachine.sum_one a

/-- In the specified intervention model, constant action zero reaches vacuum
with probability one after two ticks and remains there. This is a distinct
intervention from the registered marginal-ghost cut, not a general survival
or language-world theorem. -/
theorem renewable_action_zero_eventual_vacuum :
    ∀ n, 2 ≤ n → CGB.Energetics.law (renewableActionAlways 0)
      CGB.RenewableLearning.initial n ((fun _ => .vacuum), .trap) = 1 := by
  let P := renewableActionAlways 0
  let init := CGB.RenewableLearning.initial
  let middle : CGB.RenewableLearning.S := ((fun _ => .search), .trap)
  let dead : CGB.RenewableLearning.S := ((fun _ => .vacuum), .trap)
  have hstep (t : ℕ) (x y : CGB.RenewableLearning.S) (e : Option Unit) :
      CGB.Energetics.law P init t x * P.step t x e y ≤
        CGB.Energetics.law P init (t + 1) y := by
    have hn := (CGB.Energetics.law_probability P init t).1
    calc
      _ ≤ CGB.Energetics.law P init t x * (∑ e, P.step t x e y) := by
        apply mul_le_mul_of_nonneg_left _ (hn x)
        exact Finset.single_le_sum (fun e _ => P.nonneg t x e y) (Finset.mem_univ _)
      _ ≤ _ := by
        apply Finset.single_le_sum _ (Finset.mem_univ x)
        intro z _
        exact mul_nonneg (hn z) (Finset.sum_nonneg (fun e _ => P.nonneg t z e y))
  have hupper (n : ℕ) (z : CGB.RenewableLearning.S) :
      CGB.Energetics.law P init n z ≤ 1 := by
    calc
      _ ≤ ∑ z, CGB.Energetics.law P init n z :=
        Finset.single_le_sum (fun z _ => (CGB.Energetics.law_probability P init n).1 z)
          (Finset.mem_univ z)
      _ = 1 := (CGB.Energetics.law_probability P init n).2
  have hstart : CGB.Energetics.law P init 0 init = 1 := by
    simp [CGB.Energetics.law, CGB.EventChannels.Kernel.stateLaw]
  have hfirst : P.step 0 init (some ()) middle = 1 := renewable_action_zero_failure_steps.1
  have hsecond : P.step 1 middle (some ()) dead = 1 := renewable_action_zero_failure_steps.2
  have hstay (t : ℕ) : P.step t dead none dead = 1 := by
    norm_num [P, renewableActionAlways, renewableActionMachine,
      CGB.RenewableLearning.moves, CGB.RenewableLearning.plant,
      CGB.RenewableLearning.world, CGB.RenewableLearning.sense,
      CGB.Medium.eventStep, dead, funext_iff]
  have hfirstlaw : 1 ≤ CGB.Energetics.law P init 1 middle := by
    simpa [hstart, hfirst] using hstep 0 init middle (some ())
  have hsecondlaw : 1 ≤ CGB.Energetics.law P init 2 dead := by
    have h := hstep 1 middle dead (some ())
    rw [hsecond, mul_one] at h
    exact hfirstlaw.trans h
  have hforever (k : ℕ) : 1 ≤ CGB.Energetics.law P init (k + 2) dead := by
    induction k with
    | zero => exact hsecondlaw
    | succ k ih =>
      have h := hstep (k + 2) dead dead none
      rw [hstay, mul_one] at h
      simpa only [show (k + 1) + 2 = (k + 2) + 1 by omega] using ih.trans h
  intro n hn
  obtain ⟨k, rfl⟩ := Nat.exists_eq_add_of_le hn
  have h := hforever k
  have hu := hupper (2 + k) dead
  have he : CGB.Energetics.law P init (2 + k) dead = 1 := by
    have : 1 ≤ CGB.Energetics.law P init (2 + k) dead := by simpa [Nat.add_comm] using h
    exact le_antisymm hu this
  exact he

/-- In the same certified world, aligned ready bodies have two equally likely
next-world states while their next body state is the same. The current sensory
labels of those world states coincide, yet a subsequent action 0 distinguishes
them by the world's next `missB` probability. This is a conditional one-step
world-law statement, not a claim of infinite entropy or learned prediction. -/
theorem renewable_world_hidden_branch_and_delayed_consequence :
    (∀ q : CGB.RenewableLearning.Cell, q = .aReady ∨ q = .bReady →
      let w := if q = .aReady then CGB.RenewableLearning.Env.feedA else .feedB
      renewableActionMachine.step (CGB.RenewableLearning.policy q)
        ((fun _ => q), w) (some ()) ((fun _ => q), .feedA) = 1 / 2 ∧
      renewableActionMachine.step (CGB.RenewableLearning.policy q)
        ((fun _ => q), w) (some ()) ((fun _ => q), .feedB) = 1 / 2) ∧
    CGB.RenewableLearning.sense .feedA () = CGB.RenewableLearning.sense .feedB () ∧
    renewableActionWorld .feedA 0 .missB = 0 ∧
    renewableActionWorld .feedB 0 .missB = 1 := by
  have ha : CGB.RenewableLearning.policy .aReady = 0 := by decide
  have hb : CGB.RenewableLearning.policy .bReady = 1 := by decide
  constructor
  · intro q hq
    rcases hq with rfl | rfl <;>
      norm_num [renewableActionMachine, renewableActionWorld, renewableActionRepresentative,
        CGB.RenewableLearning.moves, CGB.RenewableLearning.plant,
        CGB.RenewableLearning.world, CGB.RenewableLearning.sense,
        CGB.RenewableLearning.next, CGB.Medium.eventStep, ha, hb, funext_iff]
  · norm_num [renewableActionWorld, renewableActionRepresentative,
      CGB.RenewableLearning.moves, CGB.RenewableLearning.sense, ha, hb]

/-- The aligned `aReady/feedA` state used above has a positive-probability
three-step prefix from the actual initial state in the original process. -/
theorem renewable_world_aligned_state_supported :
    ∃ path : ℕ → CGB.RenewableLearning.S,
      path 0 = CGB.RenewableLearning.initial ∧
      path 3 = ((fun _ => .aReady), .feedA) ∧
      0 < CGB.RenewableLearningControls.weight
        (CGB.RenewableLearning.process true true false) path 3 := by
  let path : ℕ → CGB.RenewableLearning.S := fun n =>
    if n = 3 then ((fun _ => .aReady), .feedA)
    else CGB.RenewableLearningControls.path n
  have hs : CGB.RenewableLearning.policy .search = 2 := by decide
  have ha : CGB.RenewableLearning.policy .aSpent = 0 := by decide
  refine ⟨path, ?_, ?_, ?_⟩
  · rfl
  · simp [path]
  · norm_num [CGB.RenewableLearningControls.weight, Finset.prod_range_succ,
      CGB.RenewableLearningControls.tag, path, CGB.RenewableLearningControls.path,
      CGB.RenewableLearning.process, CGB.MaterialSource.machine,
      CGB.Medium.jointEventKernel, CGB.Medium.eventWorldStep, CGB.Medium.eventStep,
      CGB.RenewableLearning.plant, CGB.RenewableLearning.world,
      CGB.RenewableLearning.initial, CGB.RenewableLearning.moves,
      CGB.RenewableLearning.sense, CGB.RenewableLearning.next, hs, ha, funext_iff]

/-- A positive-probability hidden world change in the certified process can
flow through mismatch, reopening, probing and a new write on the same body.
At the resulting state the learned action and an action override induce distinct
next-world outcomes. This is a supported branch, not a language capability. -/
theorem renewable_hidden_branch_reopens_and_rewrites (t : ℕ) :
    let P := CGB.RenewableLearning.process true true false
    let s0 : CGB.RenewableLearning.S := ((fun _ => .aReady), .feedA)
    let s1 : CGB.RenewableLearning.S := ((fun _ => .aReady), .feedB)
    let s2 : CGB.RenewableLearning.S := ((fun _ => .aReady), .missB)
    let s3 : CGB.RenewableLearning.S := ((fun _ => .search), .probeB)
    let s4 : CGB.RenewableLearning.S := ((fun _ => .search), .answerB)
    let s5 : CGB.RenewableLearning.S := ((fun _ => .bSpent), .feedB)
    let s6 : CGB.RenewableLearning.S := ((fun _ => .bReady), .feedB)
    P.step t s0 (some ()) s1 = 1 / 2 ∧
    P.step (t + 1) s1 (some ()) s2 = 1 ∧
    P.step (t + 2) s2 (some ()) s3 = 1 ∧
    P.step (t + 3) s3 (some ()) s4 = 1 ∧
    P.step (t + 4) s4 (some ()) s5 = 1 ∧
    P.step (t + 5) s5 (some ()) s6 = 1 / 2 ∧
    CGB.RenewableLearning.writeCost true .miss .aReady = 1 ∧
    CGB.RenewableLearning.writeCost true .answerB .search = 1 ∧
    CGB.RenewableLearning.topup true .feed .bSpent = 2 ∧
    CGB.RenewableLearning.policy .bReady = 1 ∧
    renewableActionMachine.step 1 s6 (some ()) ((fun _ => .bReady), .feedA) = 1 / 2 ∧
    renewableActionMachine.step 1 s6 (some ()) ((fun _ => .bReady), .feedB) = 1 / 2 ∧
    renewableActionMachine.step 0 s6 (some ()) ((fun _ => .bReady), .missB) = 1 := by
  have hs : CGB.RenewableLearning.policy .search = 2 := by decide
  have ha : CGB.RenewableLearning.policy .aReady = 0 := by decide
  have hb : CGB.RenewableLearning.policy .bSpent = 1 := by decide
  have hc : CGB.RenewableLearning.policy .bReady = 1 := by decide
  dsimp only
  norm_num [renewableActionMachine, renewableActionWorld, renewableActionRepresentative,
    CGB.RenewableLearning.process, CGB.MaterialSource.machine,
    CGB.Medium.jointEventKernel, CGB.Medium.eventWorldStep, CGB.Medium.eventStep,
    CGB.RenewableLearning.plant, CGB.RenewableLearning.world,
    CGB.RenewableLearning.moves, CGB.RenewableLearning.sense,
    CGB.RenewableLearning.next, CGB.RenewableLearning.writeCost,
    CGB.RenewableLearning.topup, hs, ha, hb, hc, funext_iff]

/-- The recorded normal and reopen-disabled paths agree through tick four.
On the same mismatch, the normal body reopens while the disabled body holds
the old candidate; the following world move distinguishes the resulting
actions. Both paths and the paired survival bound are existing certificates. -/
theorem renewable_reopen_control_branch :
    (∀ n, n ≤ 4 → CGB.RenewableLearningControls.path n =
      CGB.RenewableLearningControls.failingPath true n) ∧
    CGB.RenewableLearning.next true true .miss .aReady = .search ∧
    CGB.RenewableLearning.next false true .miss .aReady = .aHeld ∧
    CGB.RenewableLearning.policy .search = 2 ∧
    CGB.RenewableLearning.policy .aHeld = 0 ∧
    CGB.RenewableLearning.moves .probeB (fun _ => .search) .answerB = 1 ∧
    CGB.RenewableLearning.moves .probeB (fun _ => .aHeld) .trap = 1 := by
  have hs : CGB.RenewableLearning.policy .search = 2 := by decide
  have ha : CGB.RenewableLearning.policy .aHeld = 0 := by decide
  refine ⟨?_, ?_, ?_, hs, ha, ?_, ?_⟩
  · intro n hn
    have he : n = 0 ∨ n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 := by omega
    rcases he with rfl | rfl | rfl | rfl | rfl <;>
      norm_num [CGB.RenewableLearningControls.path, CGB.RenewableLearningControls.failingPath,
        CGB.RenewableLearningControls.cycle, CGB.RenewableLearningControls.index]
  · decide
  · decide
  · simp [CGB.RenewableLearning.moves, hs]
  · simp [CGB.RenewableLearning.moves, ha]

/-- A proof-only eleven-tick block in the existing stochastic process. Both
branches start at the same aligned state; the true branch visits the
existing B-to-A reopening cycle, while the false branch stays aligned. -/
def renewableObservableBlock (bit : Bool) (i : Fin 11) : CGB.RenewableLearning.S :=
  if bit then
    match i.val with
    | 0 => ((fun _ => .aReady), .feedA)
    | 1 => ((fun _ => .aReady), .feedB)
    | 2 => ((fun _ => .aReady), .missB)
    | 3 => ((fun _ => .search), .probeB)
    | 4 => ((fun _ => .search), .answerB)
    | 5 => ((fun _ => .bSpent), .feedB)
    | 6 => ((fun _ => .bReady), .feedA)
    | 7 => ((fun _ => .bReady), .missA)
    | 8 => ((fun _ => .search), .probeA)
    | 9 => ((fun _ => .search), .answerA)
    | _ => ((fun _ => .aSpent), .feedA)
  else ((fun _ => .aReady), .feedA)

/-- Every step of either block, including the transition to either next block,
has positive probability in the unchanged autonomous process. The world sensor
at offset two distinguishes the two blocks. These are finite-path support
facts, not an almost-sure statement or a language-performance experiment. -/
theorem renewable_observable_block_contract :
    (∀ bit next : Bool, ∀ i : Fin 11, ∀ t,
      0 < (CGB.RenewableLearning.process true true false).step t
        (renewableObservableBlock bit i) (some ())
        (if h : i.val + 1 < 11 then
          renewableObservableBlock bit ⟨i.val + 1, h⟩
        else renewableObservableBlock next ⟨0, by decide⟩)) ∧
    (∀ bit : Bool, CGB.RenewableLearning.sense (renewableObservableBlock bit ⟨2, by decide⟩).2 () =
      if bit then .miss else .feed) := by
  have hs : CGB.RenewableLearning.policy .search = 2 := by decide
  have ha : CGB.RenewableLearning.policy .aReady = 0 := by decide
  have hb : CGB.RenewableLearning.policy .bReady = 1 := by decide
  have hc : CGB.RenewableLearning.policy .bSpent = 1 := by decide
  have hd : CGB.RenewableLearning.policy .aSpent = 0 := by decide
  constructor
  · intro bit next i t
    cases bit <;> cases next <;> fin_cases i <;>
      norm_num [renewableObservableBlock, CGB.RenewableLearning.process,
        CGB.MaterialSource.machine, CGB.Medium.jointEventKernel, CGB.Medium.eventWorldStep,
        CGB.Medium.eventStep, CGB.RenewableLearning.plant, CGB.RenewableLearning.world,
        CGB.RenewableLearning.moves, CGB.RenewableLearning.sense,
        CGB.RenewableLearning.next, hs, ha, hb, hc, hd, funext_iff]
  · intro bit
    cases bit <;> decide

/-- A family of proof paths assembled solely from the existing world/body
transitions. The bit sequence selects which supported branch is described;
it is not an input or a stored counter in the organism. -/
def renewableObservablePath (bits : ℕ → Bool) (n : ℕ) : CGB.RenewableLearning.S :=
  renewableObservableBlock (bits (n / 11)) ⟨n % 11, Nat.mod_lt _ (by decide)⟩

/-- Every finite prefix of every selected branch sequence has positive
probability from the aligned state, and the existing sensor at offset two of
each block recovers its selected bit. This supplies arbitrarily many distinct
observable finite prefixes without assuming an infinite-path measure. -/
theorem renewable_observable_path_contract (bits : ℕ → Bool) :
    renewableObservablePath bits 0 = ((fun _ => .aReady), .feedA) ∧
    (∀ n, 0 < (CGB.RenewableLearning.process true true false).step n
      (renewableObservablePath bits n) (some ()) (renewableObservablePath bits (n + 1))) ∧
    (∀ k, CGB.RenewableLearning.sense (renewableObservablePath bits (11 * k + 2)).2 () =
      if bits k then .miss else .feed) := by
  constructor
  · cases h : bits 0 <;> simp [renewableObservablePath, renewableObservableBlock, h]
  constructor
  · intro n
    have h := renewable_observable_block_contract.1 (bits (n / 11))
      (bits (n / 11 + 1)) ⟨n % 11, Nat.mod_lt _ (by decide)⟩ n
    change 0 < (CGB.RenewableLearning.process true true false).step n
      (renewableObservablePath bits n) (some ()) _ at h
    have hn : renewableObservablePath bits (n + 1) =
        (if h : n % 11 + 1 < 11 then
          renewableObservableBlock (bits (n / 11)) ⟨n % 11 + 1, h⟩
        else renewableObservableBlock (bits (n / 11 + 1)) ⟨0, by decide⟩) := by
      by_cases hsmall : n % 11 + 1 < 11
      · have hdiv : (n + 1) / 11 = n / 11 := by omega
        have hmod : (n + 1) % 11 = n % 11 + 1 := by omega
        simp [renewableObservablePath, hsmall, hdiv, hmod]
      · have hdiv : (n + 1) / 11 = n / 11 + 1 := by omega
        have hmod : (n + 1) % 11 = 0 := by omega
        simp [renewableObservablePath, hsmall, hdiv, hmod]
    rw [hn]
    exact h
  · intro k
    have hdiv : (11 * k + 2) / 11 = k := by omega
    have hmod : (11 * k + 2) % 11 = 2 := by omega
    simpa [renewableObservablePath, hdiv, hmod] using renewable_observable_block_contract.2 (bits k)

/-- Attach the previously verified three-step initial prefix to the family of
observable paths. The sequence selects a proof path, not an input port. -/
noncomputable def renewableObservableFromBirth (bits : ℕ → Bool) : ℕ → CGB.RenewableLearning.S
  | 0 => CGB.RenewableLearning.initial
  | 1 => ((fun _ => .search), .answerA)
  | 2 => ((fun _ => .aSpent), .feedA)
  | n + 3 => renewableObservablePath bits n

/-- From the same original initial state, every finite prefix of every chosen
bit sequence has positive support; sensory symbols distinguish the blocks,
and each true block includes two actual paid writes. Individual infinite
paths may have probability zero; no entropy-rate or full life claim is made. -/
theorem renewable_observable_from_birth (bits : ℕ → Bool) :
    renewableObservableFromBirth bits 0 = CGB.RenewableLearning.initial ∧
    (∀ N, 0 < CGB.RenewableLearningControls.weight
      (CGB.RenewableLearning.process true true false) (renewableObservableFromBirth bits) N) ∧
    (∀ k, CGB.RenewableLearning.sense
      (renewableObservableFromBirth bits (3 + 11 * k + 2)).2 () =
        if bits k then .miss else .feed) ∧
    (∀ k, bits k = true →
      CGB.RenewableLearning.writeCost true
        (CGB.RenewableLearning.sense (renewableObservableFromBirth bits (3 + 11 * k + 2)).2 ())
        ((renewableObservableFromBirth bits (3 + 11 * k + 2)).1 ()) = 1 ∧
      CGB.RenewableLearning.writeCost true
        (CGB.RenewableLearning.sense (renewableObservableFromBirth bits (3 + 11 * k + 4)).2 ())
        ((renewableObservableFromBirth bits (3 + 11 * k + 4)).1 ()) = 1) := by
  have hs : CGB.RenewableLearning.policy .search = 2 := by decide
  have ha : CGB.RenewableLearning.policy .aSpent = 0 := by decide
  have hstep (n : ℕ) : 0 < (CGB.RenewableLearning.process true true false).step n
      (renewableObservableFromBirth bits n) (some ()) (renewableObservableFromBirth bits (n + 1)) := by
    rcases n with _ | n
    · norm_num [renewableObservableFromBirth, CGB.RenewableLearning.process,
        CGB.MaterialSource.machine, CGB.Medium.jointEventKernel, CGB.Medium.eventWorldStep,
        CGB.Medium.eventStep, CGB.RenewableLearning.initial, CGB.RenewableLearning.plant,
        CGB.RenewableLearning.world, CGB.RenewableLearning.moves,
        CGB.RenewableLearning.sense, CGB.RenewableLearning.next, hs, funext_iff]
    · rcases n with _ | n
      · norm_num [renewableObservableFromBirth, CGB.RenewableLearning.process,
          CGB.MaterialSource.machine, CGB.Medium.jointEventKernel, CGB.Medium.eventWorldStep,
          CGB.Medium.eventStep, CGB.RenewableLearning.plant, CGB.RenewableLearning.world,
          CGB.RenewableLearning.moves, CGB.RenewableLearning.sense,
          CGB.RenewableLearning.next, hs, funext_iff]
      · rcases n with _ | n
        · have hstart := (renewable_observable_path_contract bits).1
          simp only [renewableObservableFromBirth, Nat.reduceAdd, hstart]
          norm_num [CGB.RenewableLearning.process, CGB.MaterialSource.machine,
            CGB.Medium.jointEventKernel, CGB.Medium.eventWorldStep, CGB.Medium.eventStep,
            CGB.RenewableLearning.plant, CGB.RenewableLearning.world,
            CGB.RenewableLearning.moves, CGB.RenewableLearning.sense,
            CGB.RenewableLearning.next, ha, funext_iff]
        · have h := (renewable_observable_path_contract bits).2.1 n
          change 0 < (CGB.RenewableLearning.process true true false).step (n + 3)
            (renewableObservablePath bits n) (some ()) (renewableObservablePath bits (n + 1)) at h
          simpa [renewableObservableFromBirth, show n + 3 + 1 = (n + 1) + 3 by omega] using h
  have hlive (n : ℕ) : (renewableObservableFromBirth bits n).1 () ≠ .vacuum := by
    rcases n with _ | _ | _ | n
    · simp [renewableObservableFromBirth, CGB.RenewableLearning.initial]
    · simp [renewableObservableFromBirth]
    · simp [renewableObservableFromBirth]
    · change (renewableObservableBlock (bits (n / 11))
        ⟨n % 11, Nat.mod_lt _ (by decide)⟩).1 () ≠ .vacuum
      generalize bits (n / 11) = b
      generalize (⟨n % 11, Nat.mod_lt _ (by decide)⟩ : Fin 11) = i
      cases b <;> fin_cases i <;> decide
  refine ⟨rfl, ?_, ?_, ?_⟩
  · intro N
    unfold CGB.RenewableLearningControls.weight
    apply Finset.prod_pos
    intro n _
    simpa only [CGB.RenewableLearningControls.tag, hlive n, ↓reduceIte] using hstep n
  · intro k
    simpa [renewableObservableFromBirth, show 3 + 11 * k + 2 = (11 * k + 2) + 3 by omega] using
      (renewable_observable_path_contract bits).2.2 k
  · intro k hbit
    have hdiv2 : (11 * k + 2) / 11 = k := by omega
    have hmod2 : (11 * k + 2) % 11 = 2 := by omega
    have hdiv4 : (11 * k + 4) / 11 = k := by omega
    have hmod4 : (11 * k + 4) % 11 = 4 := by omega
    constructor
    · rw [show 3 + 11 * k + 2 = (11 * k + 2) + 3 by omega]
      simp [renewableObservableFromBirth, renewableObservablePath, hdiv2, hmod2,
        renewableObservableBlock, hbit, CGB.RenewableLearning.sense,
        CGB.RenewableLearning.writeCost]
    · rw [show 3 + 11 * k + 4 = (11 * k + 4) + 3 by omega]
      simp [renewableObservableFromBirth, renewableObservablePath, hdiv4, hmod4,
        renewableObservableBlock, hbit, CGB.RenewableLearning.sense,
        CGB.RenewableLearning.writeCost]

/-- A minimal open self-production contract.  `response` is the boundary
response to an action; `step` is the organism's own internal update. -/
structure Contract (S A I : Type*) where
  viable : S → Prop
  initial : S
  action : S → A
  response : S → A → I
  step : S → I → S
  initial_viable : viable initial
  closed : ∀ s, viable s → viable (step s (response s (action s)))

/-- The trajectory generated by one contract, with no external reset or
time-dependent rule. -/
def trajectory {S A I : Type*} (c : Contract S A I) : ℕ → S
  | 0 => c.initial
  | n + 1 => c.step (trajectory c n) (c.response (trajectory c n) (c.action (trajectory c n)))

/-- The same rule generates a valid state at every finite time. -/
theorem trajectory_viable {S A I : Type*} (c : Contract S A I) (n : ℕ) :
    c.viable (trajectory c n) := by
  induction n with
  | zero => exact c.initial_viable
  | succ n ih =>
      exact c.closed (trajectory c n) ih

/-- Re-run the same transition and boundary response under a specified action
sequence. This is a mathematical intervention, not a second organism state. -/
def controlledTrajectory {S A I : Type*} (c : Contract S A I) (actions : ℕ → A) : ℕ → S
  | 0 => c.initial
  | n + 1 => c.step (controlledTrajectory c actions n)
      (c.response (controlledTrajectory c actions n) (actions n))

/-- Replaying the actual actions in the same world reproduces the generated
trajectory. -/
theorem actual_actions_reproduce_trajectory {S A I : Type*} (c : Contract S A I) :
    ∀ n, controlledTrajectory c (fun t => c.action (trajectory c t)) n = trajectory c n := by
  intro n
  induction n with
  | zero => rfl
  | succ n ih =>
    simp only [controlledTrajectory, trajectory, ih]

/-- If each passive use spends from a finite opportunity potential, the
total passive intake from that point cannot exceed the starting potential.
The potential is a function of the existing state, not an extra state. -/
theorem passive_opportunity_budget {S A I : Type*}
    (c : Contract S A I) (actions : ℕ → A) (passive : A)
    (opportunity : S → ℕ) (intake : S → I → ℕ)
    (hspend : ∀ s, opportunity (c.step s (c.response s passive)) +
      intake s (c.response s passive) ≤ opportunity s)
    (start : ℕ) (hpassive : ∀ n, start ≤ n → actions n = passive) (T : ℕ) :
    opportunity (controlledTrajectory c actions (start + T)) +
      (∑ n ∈ Finset.range T,
        let s := controlledTrajectory c actions (start + n)
        intake s (c.response s (actions (start + n)))) ≤
      opportunity (controlledTrajectory c actions start) := by
  induction T with
  | zero => simp
  | succ T ih =>
    rw [Finset.sum_range_succ]
    have ha : actions (start + T) = passive := hpassive _ (by omega)
    have hs := hspend (controlledTrajectory c actions (start + T))
    have hnext : controlledTrajectory c actions (start + (T + 1)) =
        c.step (controlledTrajectory c actions (start + T))
          (c.response (controlledTrajectory c actions (start + T)) passive) := by
      rw [show start + (T + 1) = (start + T) + 1 by omega, controlledTrajectory, ha]
    rw [hnext, ha]
    dsimp only at ih ⊢
    omega

/-- A finite passive opportunity potential supplies the universal
counterfactual bound required by `counterfactual_passive_bound_forces_activity`. -/
theorem passive_opportunity_bounds_eventual_interventions {S A I : Type*}
    (c : Contract S A I) (passive : A) (opportunity : S → ℕ) (intake : S → I → ℕ)
    (hspend : ∀ s, opportunity (c.step s (c.response s passive)) +
      intake s (c.response s passive) ≤ opportunity s) :
    ∀ actions : ℕ → A, eventuallyPassive actions passive →
      ∃ B, ∀ T, cumulative (fun n =>
        let s := controlledTrajectory c actions n
        intake s (c.response s (actions n))) T ≤ B := by
  intro actions hpassive
  obtain ⟨start, hstart⟩ := hpassive
  let signal : ℕ → ℕ := fun n =>
    let s := controlledTrajectory c actions n
    intake s (c.response s (actions n))
  refine ⟨cumulative signal start + opportunity (controlledTrajectory c actions start), ?_⟩
  intro T
  change cumulative signal T ≤ _
  by_cases ht : T ≤ start
  · have hle : cumulative signal T ≤ cumulative signal start :=
      Finset.sum_le_sum_of_subset (Finset.range_mono ht)
    omega
  · have hle : start ≤ T := by omega
    have hbudget := passive_opportunity_budget c actions passive opportunity intake
      hspend start hstart (T - start)
    have hs : cumulative signal T = cumulative signal start +
        ∑ n ∈ Finset.range (T - start), signal (start + n) := by
      unfold cumulative
      conv_lhs => rw [show T = start + (T - start) by omega]
      exact Finset.sum_range_add signal start (T - start)
    change opportunity (controlledTrajectory c actions (start + (T - start))) +
      (∑ n ∈ Finset.range (T - start), signal (start + n)) ≤ _ at hbudget
    rw [hs]
    omega

/-- If every eventually passive intervention in the same world has bounded
intake, an actual trajectory with unbounded intake cannot become permanently
passive. The universal premise also covers the always-passive intervention. -/
theorem counterfactual_passive_bound_forces_activity {S A I : Type*}
    (c : Contract S A I) (passive : A) (intake : S → I → ℕ)
    (hworld : ∀ actions : ℕ → A, eventuallyPassive actions passive →
      ∃ B, ∀ T, cumulative (fun n =>
        let s := controlledTrajectory c actions n
        intake s (c.response s (actions n))) T ≤ B)
    (hactual : unbounded (fun n =>
      let s := trajectory c n
      intake s (c.response s (c.action s)))) :
    ¬ eventuallyPassive (fun n => c.action (trajectory c n)) passive := by
  intro hpassive
  obtain ⟨B, hB⟩ := hworld (fun n => c.action (trajectory c n)) hpassive
  obtain ⟨T, hT⟩ := hactual B
  have heq : (fun n =>
      let s := controlledTrajectory c (fun t => c.action (trajectory c t)) n
      intake s (c.response s (c.action (trajectory c n)))) =
      (fun n => let s := trajectory c n; intake s (c.response s (c.action s))) := by
    funext n
    rw [actual_actions_reproduce_trajectory]
  rw [heq] at hB
  exact (Nat.not_lt_of_ge (hB T)) hT

/-- Finite passive opportunity and unbounded actual intake force recurrent
non-passive action in the same deterministic world. -/
theorem passive_opportunity_forces_activity {S A I : Type*}
    (c : Contract S A I) (passive : A) (opportunity : S → ℕ) (intake : S → I → ℕ)
    (hspend : ∀ s, opportunity (c.step s (c.response s passive)) +
      intake s (c.response s passive) ≤ opportunity s)
    (hactual : unbounded (fun n =>
      let s := trajectory c n
      intake s (c.response s (c.action s)))) :
    ¬ eventuallyPassive (fun n => c.action (trajectory c n)) passive := by
  exact counterfactual_passive_bound_forces_activity c passive intake
    (passive_opportunity_bounds_eventual_interventions c passive opportunity intake hspend)
    hactual

/-- **Unbounded self-production.**  The contract generates one trajectory
indexed by all natural causal ticks, and every prefix remains in its viable
niche.  This is the deterministic infinite-duration statement. -/
theorem exists_infinite_self_producing_trajectory {S A I : Type*}
    (c : Contract S A I) :
    ∃ x : ℕ → S,
      x 0 = c.initial ∧
      (∀ n, x (n + 1) = c.step (x n) (c.response (x n) (c.action (x n)))) ∧
      (∀ n, c.viable (x n)) := by
  refine ⟨trajectory c, rfl, ?_, ?_⟩
  · intro n
    rfl
  · exact trajectory_viable c

/-- The action-conditioned boundary rule itself is an exact self-consequence
model when that rule is available to the organism's persistent state. -/
theorem action_consequence_model_exists {S A I : Type*} (c : Contract S A I) :
    ∃ predict : S → A → S,
      ∀ s a, predict s a = c.step s (c.response s a) := by
  exact ⟨fun s a => c.step s (c.response s a), fun _ _ => rfl⟩

/-- If two actions have different boundary consequences at a viable state, the
exact self-consequence model preserves that distinction. -/
theorem action_distinction_preserved {S A I : Type*} (c : Contract S A I)
    {s : S} (_hs : c.viable s) {a a' : A}
    (h : c.step s (c.response s a) ≠ c.step s (c.response s a')) :
    ∃ predict : S → A → S,
      predict s a ≠ predict s a' ∧
      (∀ t b, predict t b = c.step t (c.response t b)) := by
  refine ⟨fun t b => c.step t (c.response t b), h, ?_⟩
  intro t b
  rfl

/-! ## Language-life bridge

The base admission contract already supplies one persistent memory state and
its byte-conditioned transition.  The bridge below adds only a boundary
response, a viable seed, and a faithful-candidate condition.  Candidate
retention plus the encoded transition then derive the viable-niche closure;
no second memory or learning rule is introduced.
-/

/-! ## Minimal generators

`LanguageCandidate` is the assembled interface.  The generator below removes
its two supplied read/transition functions: their existence is derived from
history-factorization conditions already proved in `LanguageLifeKernel`. -/

structure MinimalLanguageGenerator
    (Hist Model Ctx Act Obs Byte Answer : Type*)
    [Fintype Model] [DecidableEq Model]
    [Fintype Byte] [Nonempty Byte] where
  reserve : ℕ
  possible : Support Model Ctx Act Obs
  encode : Hist → Memory Model reserve
  required : Hist → Answer
  advanceHistory : Hist → Byte → Hist
  fallback : Answer
  readable : ∀ h h', encode h = encode h' → required h = required h'
  transition_compatible : ∀ h h' b, encode h = encode h' →
    encode (advanceHistory h b) = encode (advanceHistory h' b)
  context : Hist → Byte → Ctx
  action : Hist → Byte → Act
  outcome : Hist → Byte → Obs
  transition_correct : ∀ h b,
    CausalLearning.advance possible (encode h)
      (context h b) (action h b) (outcome h b) =
      encode (advanceHistory h b)
  stream : Hist → Byte → ℝ
  allocation : Memory Model reserve → Byte → ℝ
  stream_nonneg : ∀ h b, 0 ≤ stream h b
  stream_norm : ∀ h, ∑ b, stream h b = 1
  allocation_pos : ∀ s b, 0 < allocation s b
  allocation_norm : ∀ s, ∑ b, allocation s b = 1
  trajectory : ℕ → Hist
  signal : ℕ → ℕ
  policy : Memory Model reserve → Act
  passive : Act
  response : Act → Byte → ℝ
  predict : Byte → ℝ
  viable_signal : unbounded signal
  passive_bound : passiveBounded (fun t => policy (encode (trajectory t))) signal passive
  action_sensitive : actionSensitive response

noncomputable def materializeLanguageCandidate
    {Hist Model Ctx Act Obs Byte Answer : Type*}
    [Fintype Model] [DecidableEq Model]
    [Fintype Byte] [Nonempty Byte]
    (G : MinimalLanguageGenerator Hist Model Ctx Act Obs Byte Answer) :
    LanguageCandidate Hist Model Ctx Act Obs Byte Answer := by
  let readWitness : ∃ read : Memory Model G.reserve → Answer,
      ∀ h, read (G.encode h) = G.required h :=
    (history_readout_exists_iff G.encode G.required G.fallback).mpr G.readable
  let nextWitness : ∃ next : Memory Model G.reserve → Byte → Memory Model G.reserve,
      ∀ h b, next (G.encode h) b = G.encode (G.advanceHistory h b) :=
    (history_transition_exists_iff G.encode G.advanceHistory
      (CausalLearning.initialMemory Model G.reserve)).mpr
      (fun h h' same b => G.transition_compatible h h' b same)
  let read := Classical.choose readWitness
  let next := Classical.choose nextWitness
  refine
    { reserve := G.reserve
      possible := G.possible
      encode := G.encode
      required := G.required
      advanceHistory := G.advanceHistory
      read := read
      next := next
      read_correct := ?_
      next_correct := ?_
      context := G.context
      action := G.action
      outcome := G.outcome
      write_refines := ?_
      stream := G.stream
      allocation := G.allocation
      stream_nonneg := G.stream_nonneg
      stream_norm := G.stream_norm
      allocation_pos := G.allocation_pos
      allocation_norm := G.allocation_norm
      trajectory := G.trajectory
      signal := G.signal
      policy := G.policy
      passive := G.passive
      response := G.response
      predict := G.predict
      viable_signal := G.viable_signal
      passive_bound := G.passive_bound
      action_sensitive := G.action_sensitive }
  · exact Classical.choose_spec readWitness
  · exact Classical.choose_spec nextWitness
  · intro h b
    calc
      next (G.encode h) b = G.encode (G.advanceHistory h b) :=
        Classical.choose_spec nextWitness h b
      _ = CausalLearning.advance G.possible (G.encode h)
          (G.context h b) (G.action h b) (G.outcome h b) :=
        (G.transition_correct h b).symm

structure LanguageSelfProductionCandidate
    (Hist Model Ctx Act Obs Byte Answer : Type*)
    [Fintype Model] [DecidableEq Model]
    [Fintype Byte] [Nonempty Byte] where
  base : LanguageCandidate Hist Model Ctx Act Obs Byte Answer
  seed : Hist
  seed_nonempty : (base.encode seed).candidates.Nonempty
  world_response : Memory Model base.reserve → Act → Byte
  faithful : ∀ h b, ∃ x, x ∈ (base.encode h).candidates ∧
    base.possible x (base.context h b) (base.action h b) (base.outcome h b) = true

structure MinimalLanguageSelfProductionGenerator
    (Hist Model Ctx Act Obs Byte Answer : Type*)
    [Fintype Model] [DecidableEq Model]
    [Fintype Byte] [Nonempty Byte] where
  base : MinimalLanguageGenerator Hist Model Ctx Act Obs Byte Answer
  seed : Hist
  seed_nonempty : (base.encode seed).candidates.Nonempty
  world_response : Memory Model base.reserve → Act → Byte
  faithful : ∀ h b, ∃ x, x ∈ (base.encode h).candidates ∧
    base.possible x (base.context h b) (base.action h b) (base.outcome h b) = true

noncomputable def materializeLanguageSelfProductionGenerator
    {Hist Model Ctx Act Obs Byte Answer : Type*}
    [Fintype Model] [DecidableEq Model]
    [Fintype Byte] [Nonempty Byte]
    (G : MinimalLanguageSelfProductionGenerator Hist Model Ctx Act Obs Byte Answer) :
    LanguageSelfProductionCandidate Hist Model Ctx Act Obs Byte Answer :=
  { base := materializeLanguageCandidate G.base
    seed := G.seed
    seed_nonempty := by simpa [materializeLanguageCandidate] using G.seed_nonempty
    world_response := G.world_response
    faithful := by
      intro h b
      simpa [materializeLanguageCandidate] using G.faithful h b }


def language_viable
    {Hist Model Ctx Act Obs Byte Answer : Type*}
    [Fintype Model] [DecidableEq Model]
    [Fintype Byte] [Nonempty Byte]
    (L : LanguageSelfProductionCandidate Hist Model Ctx Act Obs Byte Answer)
    (s : Memory Model L.base.reserve) : Prop :=
  ∃ h, s = L.base.encode h ∧ (L.base.encode h).candidates.Nonempty

theorem language_viable_next
    {Hist Model Ctx Act Obs Byte Answer : Type*}
    [Fintype Model] [DecidableEq Model]
    [Fintype Byte] [Nonempty Byte]
    (L : LanguageSelfProductionCandidate Hist Model Ctx Act Obs Byte Answer)
    {s : Memory Model L.base.reserve} (hs : language_viable L s) (b : Byte) :
    language_viable L (L.base.next s b) := by
  rcases hs with ⟨h, rfl, _⟩
  obtain ⟨x, hx, htrue⟩ := L.faithful h b
  have hret := CausalLearning.retain_consistent L.base.possible
    (L.base.encode h) (L.base.context h b) (L.base.action h b)
    (L.base.outcome h b) x hx htrue
  have hnext := L.base.next_correct h b
  refine ⟨L.base.advanceHistory h b, hnext, ?_⟩
  have hmem : x ∈ (L.base.encode (L.base.advanceHistory h b)).candidates := by
    rw [← hnext, L.base.write_refines h b]
    exact hret
  exact ⟨x, hmem⟩

/-! The finite candidate state gives an intrinsic information-dissipation
quantity.  It counts future-model distinctions removed by one write; it is not
an extra physical cost parameter. -/

def candidate_erase_count {H C A O : Type*} [Fintype H] [DecidableEq H]
    (possible : Support H C A O) {B : Nat} (m : Memory H B)
    (c : C) (a : A) (o : O) : Nat :=
  m.candidates.card - (advance possible m c a o).candidates.card

theorem candidate_erase_balance {H C A O : Type*} [Fintype H] [DecidableEq H]
    (possible : Support H C A O) {B : Nat} (m : Memory H B)
    (c : C) (a : A) (o : O) :
    (advance possible m c a o).candidates.card +
        candidate_erase_count possible m c a o = m.candidates.card := by
  have hw := write_contract possible m c a o
  unfold candidate_erase_count
  omega

theorem candidate_erase_ge_fee {H C A O : Type*} [Fintype H] [DecidableEq H]
    (possible : Support H C A O) {B : Nat} (m : Memory H B)
    (c : C) (a : A) (o : O) :
    fee possible m c a o ≤ candidate_erase_count possible m c a o := by
  have hw := write_contract possible m c a o
  unfold candidate_erase_count
  omega

theorem effective_write_erases_distinction {H C A O : Type*}
    [Fintype H] [DecidableEq H]
    (possible : Support H C A O) {B : Nat} (m : Memory H B)
    (c : C) (a : A) (o : O)
    (hwrite : fee possible m c a o = 1) :
    0 < candidate_erase_count possible m c a o := by
  have hge := candidate_erase_ge_fee possible m c a o
  omega

theorem language_state_capacity
    {Hist Model Ctx Act Obs Byte Answer : Type*}
    [Fintype Model] [DecidableEq Model]
    [Fintype Byte] [Nonempty Byte]
    (L : LanguageSelfProductionCandidate Hist Model Ctx Act Obs Byte Answer) :
    Fintype.card (Memory Model L.base.reserve) =
      2 ^ Fintype.card Model * (L.base.reserve + 1) := by
  exact CausalLearning.memory_cardinality L.base.reserve

def language_contract
    {Hist Model Ctx Act Obs Byte Answer : Type*}
    [Fintype Model] [DecidableEq Model]
    [Fintype Byte] [Nonempty Byte]
    (L : LanguageSelfProductionCandidate Hist Model Ctx Act Obs Byte Answer) :
    Contract (Memory Model L.base.reserve) Act Byte where
  viable := language_viable L
  initial := L.base.encode L.seed
  action := L.base.policy
  response := L.world_response
  step := L.base.next
  initial_viable := ⟨L.seed, rfl, L.seed_nonempty⟩
  closed := by
    intro s hs
    rcases hs with ⟨h, rfl, _⟩
    exact language_viable_next L ⟨h, rfl, by assumption⟩ _

/-- A history generated by the same policy and boundary response as the
self-production contract; it is a mathematical history, not stored runtime
state. -/
def language_generated_history
    {Hist Model Ctx Act Obs Byte Answer : Type*}
    [Fintype Model] [DecidableEq Model] [Fintype Byte] [Nonempty Byte]
    (L : LanguageSelfProductionCandidate Hist Model Ctx Act Obs Byte Answer) : ℕ → Hist
  | 0 => L.seed
  | n + 1 =>
    let h := language_generated_history L n
    let s := L.base.encode h
    L.base.advanceHistory h (L.world_response s (L.base.policy s))

/-- The generated history and the existing state recursion agree without a
separately postulated trajectory-alignment premise. -/
theorem generated_history_encodes_self_production
    {Hist Model Ctx Act Obs Byte Answer : Type*}
    [Fintype Model] [DecidableEq Model] [Fintype Byte] [Nonempty Byte]
    (L : LanguageSelfProductionCandidate Hist Model Ctx Act Obs Byte Answer) :
    ∀ n, L.base.encode (language_generated_history L n) =
      trajectory (language_contract L) n := by
  intro n
  induction n with
  | zero => rfl
  | succ n ih =>
    change L.base.encode (L.base.advanceHistory (language_generated_history L n)
      (L.world_response (L.base.encode (language_generated_history L n))
        (L.base.policy (L.base.encode (language_generated_history L n))))) = _
    rw [← L.base.next_correct, ih]
    rfl

/-- On the generated trajectory, a boundary-derived nonnegative integer
signal supports the existing active conclusion when the explicit unbounded
intake and passive-depletion conditions hold. No extra stored state is used. -/
theorem generated_boundary_signal_forces_active
    {Hist Model Ctx Act Obs Byte Answer : Type*}
    [Fintype Model] [DecidableEq Model] [Fintype Byte] [Nonempty Byte]
    (L : LanguageSelfProductionCandidate Hist Model Ctx Act Obs Byte Answer)
    (intake : Memory Model L.base.reserve → Byte → ℕ)
    (hunbounded : unbounded (fun n =>
      let s := trajectory (language_contract L) n
      intake s (L.world_response s (L.base.policy s))))
    (hpassive : passiveBounded
      (fun n => L.base.policy (trajectory (language_contract L) n))
      (fun n => let s := trajectory (language_contract L) n
        intake s (L.world_response s (L.base.policy s))) L.base.passive) :
    (∀ n, language_viable L (trajectory (language_contract L) n)) ∧
      ¬ eventuallyPassive (fun n => L.base.policy (trajectory (language_contract L) n))
        L.base.passive := by
  exact ⟨trajectory_viable (language_contract L),
    viable_not_eventually_passive _ _ _ hunbounded hpassive⟩

/-- The admission history follows the self-production trajectory when its
encoded initial state and every action-conditioned transition agree.  Only
encoded state equality is required; histories themselves need not coincide. -/
theorem language_admission_trajectory_aligned
    {Hist Model Ctx Act Obs Byte Answer : Type*}
    [Fintype Model] [DecidableEq Model]
    [Fintype Byte] [Nonempty Byte]
    (L : LanguageSelfProductionCandidate Hist Model Ctx Act Obs Byte Answer)
    (hseed : L.base.encode (L.base.trajectory 0) = L.base.encode L.seed)
    (hstep : ∀ n, L.base.encode (L.base.trajectory (n + 1)) =
      L.base.next (L.base.encode (L.base.trajectory n))
        (L.world_response (L.base.encode (L.base.trajectory n))
          (L.base.policy (L.base.encode (L.base.trajectory n))))) :
    ∀ n, L.base.encode (L.base.trajectory n) = trajectory (language_contract L) n := by
  intro n
  induction n with
  | zero => exact hseed
  | succ n ih =>
    rw [hstep, ih]
    rfl

/-- With the alignment conditions, the existing active-inference conclusion
applies to the same viable state trajectory produced by the language contract. -/
theorem language_self_production_active_if_aligned
    {Hist Model Ctx Act Obs Byte Answer : Type*}
    [Fintype Model] [DecidableEq Model]
    [Fintype Byte] [Nonempty Byte]
    (L : LanguageSelfProductionCandidate Hist Model Ctx Act Obs Byte Answer)
    (hseed : L.base.encode (L.base.trajectory 0) = L.base.encode L.seed)
    (hstep : ∀ n, L.base.encode (L.base.trajectory (n + 1)) =
      L.base.next (L.base.encode (L.base.trajectory n))
        (L.world_response (L.base.encode (L.base.trajectory n))
          (L.base.policy (L.base.encode (L.base.trajectory n))))) :
    (∀ n, language_viable L (trajectory (language_contract L) n)) ∧
    ¬ eventuallyPassive (fun n => L.base.policy (trajectory (language_contract L) n))
      L.base.passive := by
  constructor
  · exact trajectory_viable (language_contract L)
  · have halign := language_admission_trajectory_aligned L hseed hstep
    have hactive := viable_not_eventually_passive
      (fun n => L.base.policy (L.base.encode (L.base.trajectory n)))
      L.base.signal L.base.passive L.base.viable_signal L.base.passive_bound
    intro hpassive
    apply hactive
    rcases hpassive with ⟨t, ht⟩
    refine ⟨t, ?_⟩
    intro n hn
    change L.base.policy (L.base.encode (L.base.trajectory n)) = L.base.passive
    rw [halign n]
    exact ht n hn

/-- The active and self-production halves use one trajectory once the two
explicit encoded alignment conditions hold. This is the compact bridge for
the joint language-life admission: the base's admission contract, the
self-producing viable trajectory, and non-eventual passivity are all stated
on the same encoded state sequence. -/
theorem language_joint_admission_self_production_active_if_aligned
    {Hist Model Ctx Act Obs Byte Answer : Type*}
    [Fintype Model] [DecidableEq Model]
    [Fintype Byte] [Nonempty Byte]
    (L : LanguageSelfProductionCandidate Hist Model Ctx Act Obs Byte Answer)
    (hseed : L.base.encode (L.base.trajectory 0) = L.base.encode L.seed)
    (hstep : ∀ n, L.base.encode (L.base.trajectory (n + 1)) =
      L.base.next (L.base.encode (L.base.trajectory n))
        (L.world_response (L.base.encode (L.base.trajectory n))
          (L.base.policy (L.base.encode (L.base.trajectory n))))) :
    LanguageLifeAdmission.joint_admission_spec L.base ∧
    (∀ n, language_viable L (trajectory (language_contract L) n)) ∧
    ¬ eventuallyPassive (fun n => L.base.policy (trajectory (language_contract L) n))
      L.base.passive ∧
    (∀ n, L.base.encode (L.base.trajectory n) = trajectory (language_contract L) n) := by
  have ha := language_self_production_active_if_aligned L hseed hstep
  refine ⟨LanguageLifeAdmission.joint_admission L.base, ha.1, ha.2, ?_⟩
  exact language_admission_trajectory_aligned L hseed hstep

/-- Probe loss counts distinct, previously recoverable probes belonging to
candidates removed by one update. -/
def probeLoss {H P S : Type*} [DecidableEq H] [DecidableEq P]
    (probe : H → P) (recoverable : S → P → Bool)
    (before after : S) (old new : Finset H) : Nat :=
  (old.filter (fun h => h ∉ new ∧
    recoverable before (probe h) = true ∧
    recoverable after (probe h) = false)).image probe |>.card

/-- Under explicit injective-probe and recoverability conditions, probe loss is
exactly the candidate distinction erased by the update. -/
theorem probeLoss_eq_candidate_erase {H P S : Type*}
    [Fintype H] [DecidableEq H] [DecidableEq P]
    (probe : H → P) (recoverable : S → P → Bool)
    (before after : S) (old new : Finset H)
    (probe_injective : Function.Injective probe)
    (hkeep : ∀ h ∈ old, recoverable before (probe h) = true)
    (herase : ∀ h ∈ old, h ∉ new → recoverable after (probe h) = false) :
    probeLoss probe recoverable before after old new = old.card - (old ∩ new).card := by
  unfold probeLoss
  have hf : old.filter (fun h => h ∉ new ∧
      recoverable before (probe h) = true ∧
      recoverable after (probe h) = false) = old \ new := by
    ext h
    by_cases ho : h ∈ old <;> by_cases hn : h ∈ new <;> simp [ho, hn]
    · exact ⟨hkeep h ho, herase h ho (by simp [hn])⟩
  rw [Finset.card_image_of_injective _ probe_injective, hf]
  rw [Finset.card_sdiff]
  simp [Finset.inter_comm]

/-- In the historical monotone candidate controller, dissipation is the number
of future-model distinctions erased by this write.  This is a useful
information-loss observable, not a food, energy, or auxiliary reserve field;
the controller itself is finite and therefore cannot be the renewable GDN
implementation. -/
def language_dissipation
    {Hist Model Ctx Act Obs Byte Answer : Type*}
    [Fintype Model] [DecidableEq Model]
    [Fintype Byte] [Nonempty Byte]
    (L : LanguageSelfProductionCandidate Hist Model Ctx Act Obs Byte Answer)
    (s : Memory Model L.base.reserve) (b : Byte) : Nat :=
  s.candidates.card - (L.base.next s b).candidates.card

/-- Every externally controlled trajectory remains in the existing language
viability region, since the language contract admits every boundary byte. -/
theorem language_controlled_trajectory_viable
    {Hist Model Ctx Act Obs Byte Answer : Type*}
    [Fintype Model] [DecidableEq Model] [Fintype Byte] [Nonempty Byte]
    (L : LanguageSelfProductionCandidate Hist Model Ctx Act Obs Byte Answer)
    (actions : ℕ → Act) :
    ∀ n, language_viable L (controlledTrajectory (language_contract L) actions n) := by
  intro n
  induction n with
  | zero => exact (language_contract L).initial_viable
  | succ n ih =>
    change language_viable L (L.base.next
      (controlledTrajectory (language_contract L) actions n)
      (L.world_response (controlledTrajectory (language_contract L) actions n) (actions n)))
    exact language_viable_next L ih _

/-- The candidate-based language contract eventually stops changing its
entire memory, for every action sequence. This is a limitation of this
particular one-way, finite-write contract, not of finite-state systems. -/
theorem language_controlled_memory_eventually_stops
    {Hist Model Ctx Act Obs Byte Answer : Type*}
    [Fintype Model] [DecidableEq Model] [Fintype Byte] [Nonempty Byte]
    (L : LanguageSelfProductionCandidate Hist Model Ctx Act Obs Byte Answer)
    (actions : ℕ → Act) :
    ∃ t, ∀ n, t ≤ n →
      controlledTrajectory (language_contract L) actions (n + 1) =
        controlledTrajectory (language_contract L) actions n := by
  classical
  let x := controlledTrajectory (language_contract L) actions
  let r : ℕ → ℕ := fun n => (x n).reserve.val
  have hstep : ∀ n, ∃ h,
      x n = L.base.encode h ∧
      x (n + 1) = CausalLearning.advance L.base.possible (x n)
        (L.base.context h (L.world_response (x n) (actions n)))
        (L.base.action h (L.world_response (x n) (actions n)))
        (L.base.outcome h (L.world_response (x n) (actions n))) := by
    intro n
    obtain ⟨h, hh, _⟩ := language_controlled_trajectory_viable L actions n
    change x n = L.base.encode h at hh
    refine ⟨h, hh, ?_⟩
    change L.base.next (x n) (L.world_response (x n) (actions n)) = _
    rw [hh, L.base.write_refines]
  have hmono : Antitone r := antitone_nat_of_succ_le (by
    intro n
    obtain ⟨h, hh, hn⟩ := hstep n
    have hb := (write_contract L.base.possible (x n)
      (L.base.context h (L.world_response (x n) (actions n)))
      (L.base.action h (L.world_response (x n) (actions n)))
      (L.base.outcome h (L.world_response (x n) (actions n)))).2.1
    change (x (n + 1)).reserve.val ≤ (x n).reserve.val
    rw [hn]
    omega)
  have hexists : ∃ m, ∃ n, r n = m := ⟨r 0, 0, rfl⟩
  obtain ⟨t, ht⟩ := Nat.find_spec hexists
  refine ⟨t, ?_⟩
  intro n htn
  have hmin : r t ≤ r (n + 1) := by
    rw [ht]
    exact Nat.find_min' hexists ⟨n + 1, rfl⟩
  have hupper : r n ≤ r t := hmono htn
  obtain ⟨h, hh, hn⟩ := hstep n
  have hb := (write_contract L.base.possible (x n)
    (L.base.context h (L.world_response (x n) (actions n)))
    (L.base.action h (L.world_response (x n) (actions n)))
    (L.base.outcome h (L.world_response (x n) (actions n)))).2.1
  have hfee : fee L.base.possible (x n)
    (L.base.context h (L.world_response (x n) (actions n)))
    (L.base.action h (L.world_response (x n) (actions n)))
    (L.base.outcome h (L.world_response (x n) (actions n))) = 0 := by
    change (x t).reserve.val ≤ (x (n + 1)).reserve.val at hmin
    change (x n).reserve.val ≤ (x t).reserve.val at hupper
    rw [hn] at hmin
    omega
  change x (n + 1) = x n
  rw [hn]
  unfold fee at hfee
  unfold CausalLearning.advance
  split_ifs with hw
  · simp [hw] at hfee
  · rfl

/-- The autonomous self-production trajectory inherits the same eventual
memory freeze; infinite viability does not by itself imply lifelong learning. -/
theorem language_self_production_memory_eventually_stops
    {Hist Model Ctx Act Obs Byte Answer : Type*}
    [Fintype Model] [DecidableEq Model] [Fintype Byte] [Nonempty Byte]
    (L : LanguageSelfProductionCandidate Hist Model Ctx Act Obs Byte Answer) :
    ∃ t, ∀ n, t ≤ n →
      trajectory (language_contract L) (n + 1) = trajectory (language_contract L) n := by
  obtain ⟨t, ht⟩ := language_controlled_memory_eventually_stops L
    (fun n => L.base.policy (trajectory (language_contract L) n))
  refine ⟨t, ?_⟩
  intro n hn
  have he := ht n hn
  have hr := actual_actions_reproduce_trajectory (language_contract L)
  change ∀ n, controlledTrajectory (language_contract L)
    (fun n => L.base.policy (trajectory (language_contract L) n)) n =
    trajectory (language_contract L) n at hr
  rw [hr (n + 1), hr n] at he
  exact he

/-- In the original candidate contract, eventual memory freeze also freezes
the policy action and body-determined boundary byte. Hence that particular
autonomous witness cannot represent an indefinitely changing boundary stream.
This does not apply to the separate open-world contract with external state. -/
theorem language_candidate_eventually_constant_boundary
    {Hist Model Ctx Act Obs Byte Answer : Type*}
    [Fintype Model] [DecidableEq Model] [Fintype Byte] [Nonempty Byte]
    (L : LanguageSelfProductionCandidate Hist Model Ctx Act Obs Byte Answer) :
    ∃ t, ∀ n, t ≤ n →
      trajectory (language_contract L) n = trajectory (language_contract L) t ∧
      L.base.policy (trajectory (language_contract L) n) =
        L.base.policy (trajectory (language_contract L) t) ∧
      L.world_response (trajectory (language_contract L) n)
          (L.base.policy (trajectory (language_contract L) n)) =
        L.world_response (trajectory (language_contract L) t)
          (L.base.policy (trajectory (language_contract L) t)) := by
  obtain ⟨t, ht⟩ := language_self_production_memory_eventually_stops L
  refine ⟨t, ?_⟩
  intro n hn
  have hstate : trajectory (language_contract L) n = trajectory (language_contract L) t := by
    obtain ⟨k, rfl⟩ := Nat.exists_eq_add_of_le hn
    induction k with
    | zero => simp
    | succ k ih =>
      rw [show t + (k + 1) = (t + k) + 1 by omega, ht (t + k) (by omega),
        ih (by omega)]
  simp [hstate]

/-- Candidate elimination telescopes along every intervention, not merely
along the organism's chosen policy. -/
theorem language_controlled_erasure_balance
    {Hist Model Ctx Act Obs Byte Answer : Type*}
    [Fintype Model] [DecidableEq Model] [Fintype Byte] [Nonempty Byte]
    (L : LanguageSelfProductionCandidate Hist Model Ctx Act Obs Byte Answer)
    (actions : ℕ → Act) (T : ℕ) :
    (controlledTrajectory (language_contract L) actions T).candidates.card +
      (∑ n ∈ Finset.range T,
        let s := controlledTrajectory (language_contract L) actions n
        language_dissipation L s (L.world_response s (actions n))) =
      (L.base.encode L.seed).candidates.card := by
  induction T with
  | zero => simp [controlledTrajectory, language_contract]
  | succ T ih =>
    let s := controlledTrajectory (language_contract L) actions T
    let b := L.world_response s (actions T)
    have hv : language_viable L s := language_controlled_trajectory_viable L actions T
    obtain ⟨h, hs, _⟩ := hv
    have hsub : (L.base.next s b).candidates ⊆ s.candidates := by
      rw [hs, L.base.write_refines h b]
      exact (write_contract L.base.possible (L.base.encode h)
        (L.base.context h b) (L.base.action h b) (L.base.outcome h b)).1
    have hcard := Finset.card_le_card hsub
    change s.candidates.card +
      (∑ n ∈ Finset.range T,
        let s := controlledTrajectory (language_contract L) actions n
        language_dissipation L s (L.world_response s (actions n))) = _ at ih
    rw [Finset.sum_range_succ]
    change (L.base.next s b).candidates.card +
      ((∑ n ∈ Finset.range T,
        let s := controlledTrajectory (language_contract L) actions n
        language_dissipation L s (L.world_response s (actions n))) +
        language_dissipation L s b) = _
    simp only [language_dissipation] at ih ⊢
    omega

/-- Consequently, permanently eliminating a finite candidate family cannot
produce the unbounded signal for any policy, even an active one. -/
theorem language_controlled_erasure_not_unbounded
    {Hist Model Ctx Act Obs Byte Answer : Type*}
    [Fintype Model] [DecidableEq Model] [Fintype Byte] [Nonempty Byte]
    (L : LanguageSelfProductionCandidate Hist Model Ctx Act Obs Byte Answer)
    (actions : ℕ → Act) :
    ¬ unbounded (fun n =>
      let s := controlledTrajectory (language_contract L) actions n
      language_dissipation L s (L.world_response s (actions n))) := by
  apply not_unbounded_of_bounded
  refine ⟨(L.base.encode L.seed).candidates.card, ?_⟩
  intro T
  have h := language_controlled_erasure_balance L actions T
  have hbound : (∑ n ∈ Finset.range T,
      let s := controlledTrajectory (language_contract L) actions n
      language_dissipation L s (L.world_response s (actions n))) ≤
        (L.base.encode L.seed).candidates.card := by omega
  simpa only [cumulative] using hbound

/-- A concrete bridge from language candidates to a finite probe state.  The
bridge carries no new learning rule; it records exactly what an implementation
must expose before candidate dissipation can be measured as probe loss. -/
structure LanguageProbeBridge
    {Hist Model Ctx Act Obs Byte Answer : Type*}
    [Fintype Model] [DecidableEq Model]
    [Fintype Byte] [Nonempty Byte]
    (L : LanguageSelfProductionCandidate Hist Model Ctx Act Obs Byte Answer)
    (P S : Type*) [DecidableEq P] where
  probe : Model → P
  probe_injective : Function.Injective probe
  state_of : Memory Model L.base.reserve → S
  recoverable : S → P → Bool

/-- Once the bridge's three recoverability conditions hold, its measured probe
loss equals the language candidate dissipation for that byte. -/
theorem language_probe_loss_eq_dissipation
    {Hist Model Ctx Act Obs Byte Answer P S : Type*}
    [Fintype Model] [DecidableEq Model]
    [Fintype Byte] [Nonempty Byte] [DecidableEq P]
    (L : LanguageSelfProductionCandidate Hist Model Ctx Act Obs Byte Answer)
    (B : LanguageProbeBridge L P S) (h : Hist) (b : Byte)
    (hkeep : ∀ x ∈ (L.base.encode h).candidates,
      B.recoverable (B.state_of (L.base.encode h)) (B.probe x) = true)
    (herase : ∀ x ∈ (L.base.encode h).candidates,
      x ∉ (L.base.next (L.base.encode h) b).candidates →
      B.recoverable (B.state_of (L.base.next (L.base.encode h) b)) (B.probe x) = false) :
    probeLoss B.probe B.recoverable
      (B.state_of (L.base.encode h))
      (B.state_of (L.base.next (L.base.encode h) b))
      (L.base.encode h).candidates
      (L.base.next (L.base.encode h) b).candidates =
      language_dissipation L (L.base.encode h) b := by
  have hsub : (L.base.next (L.base.encode h) b).candidates ⊆
      (L.base.encode h).candidates := by
    rw [L.base.write_refines h b]
    exact (write_contract L.base.possible (L.base.encode h)
      (L.base.context h b) (L.base.action h b) (L.base.outcome h b)).1
  have hloss := probeLoss_eq_candidate_erase
    B.probe B.recoverable
    (B.state_of (L.base.encode h))
    (B.state_of (L.base.next (L.base.encode h) b))
    (L.base.encode h).candidates
    (L.base.next (L.base.encode h) b).candidates
    B.probe_injective hkeep herase
  rw [hloss]
  unfold language_dissipation
  rw [Finset.inter_eq_right.mpr hsub]


/-- Reference bridge: candidate identity is a lossless probe and candidate memory
itself is the state.  This proves the bridge contract is consistent; a GDN
implementation must replace this identity map and re-prove its conditions. -/
noncomputable def candidateReferenceProbeBridge
    {Hist Model Ctx Act Obs Byte Answer : Type*}
    [Fintype Model] [DecidableEq Model]
    [Fintype Byte] [Nonempty Byte]
    (L : LanguageSelfProductionCandidate Hist Model Ctx Act Obs Byte Answer) :
    LanguageProbeBridge L Model (Memory Model L.base.reserve) where
  probe := id
  probe_injective := Function.injective_id
  state_of := id
  recoverable := fun s x => decide (x ∈ s.candidates)

theorem candidate_reference_probe_loss_eq_dissipation
    {Hist Model Ctx Act Obs Byte Answer : Type*}
    [Fintype Model] [DecidableEq Model]
    [Fintype Byte] [Nonempty Byte]
    (L : LanguageSelfProductionCandidate Hist Model Ctx Act Obs Byte Answer)
    (h : Hist) (b : Byte) :
    probeLoss (candidateReferenceProbeBridge L).probe
      (candidateReferenceProbeBridge L).recoverable
      (L.base.encode h) (L.base.next (L.base.encode h) b)
      (L.base.encode h).candidates
      (L.base.next (L.base.encode h) b).candidates =
      language_dissipation L (L.base.encode h) b := by
  apply language_probe_loss_eq_dissipation
  · intro x hx
    simp [candidateReferenceProbeBridge, hx]
  · intro x hx hnot
    simp [candidateReferenceProbeBridge, hnot]

theorem language_dissipation_encoded_eq_candidate_erase
    {Hist Model Ctx Act Obs Byte Answer : Type*}
    [Fintype Model] [DecidableEq Model]
    [Fintype Byte] [Nonempty Byte]
    (L : LanguageSelfProductionCandidate Hist Model Ctx Act Obs Byte Answer)
    (h : Hist) (b : Byte) :
    language_dissipation L (L.base.encode h) b =
      candidate_erase_count L.base.possible (L.base.encode h)
        (L.base.context h b) (L.base.action h b) (L.base.outcome h b) := by
  unfold language_dissipation candidate_erase_count
  rw [L.base.write_refines h b]

/-- Along an encoded history, cumulative candidate erasure is exactly the
initial candidate count minus the current count. This exposes the finite
elimination budget already implicit in the admission contract. -/
theorem language_cumulative_erasure_balance
    {Hist Model Ctx Act Obs Byte Answer : Type*}
    [Fintype Model] [DecidableEq Model]
    [Fintype Byte] [Nonempty Byte]
    (L : LanguageSelfProductionCandidate Hist Model Ctx Act Obs Byte Answer)
    (h : ℕ → Hist) (b : ℕ → Byte)
    (hstep : ∀ n, h (n + 1) = L.base.advanceHistory (h n) (b n))
    (T : ℕ) :
    (L.base.encode (h T)).candidates.card +
      ∑ n ∈ Finset.range T, language_dissipation L (L.base.encode (h n)) (b n) =
      (L.base.encode (h 0)).candidates.card := by
  induction T with
  | zero => simp
  | succ T ih =>
    have hb := candidate_erase_balance L.base.possible (L.base.encode (h T))
      (L.base.context (h T) (b T)) (L.base.action (h T) (b T))
      (L.base.outcome (h T) (b T))
    have hn : L.base.next (L.base.encode (h T)) (b T) =
        L.base.encode (h (T + 1)) := by
      rw [hstep T]
      exact L.base.next_correct (h T) (b T)
    rw [← L.base.write_refines (h T) (b T), hn,
      ← language_dissipation_encoded_eq_candidate_erase L (h T) (b T)] at hb
    rw [Finset.sum_range_succ]
    omega

/-- The total number of eliminated distinctions cannot exceed the finite
initial candidate set, regardless of the length of the stream. -/
theorem language_cumulative_erasure_bounded
    {Hist Model Ctx Act Obs Byte Answer : Type*}
    [Fintype Model] [DecidableEq Model]
    [Fintype Byte] [Nonempty Byte]
    (L : LanguageSelfProductionCandidate Hist Model Ctx Act Obs Byte Answer)
    (h : ℕ → Hist) (b : ℕ → Byte)
    (hstep : ∀ n, h (n + 1) = L.base.advanceHistory (h n) (b n))
    (T : ℕ) :
    (∑ n ∈ Finset.range T, language_dissipation L (L.base.encode (h n)) (b n)) ≤
      (L.base.encode (h 0)).candidates.card := by
  have hb := language_cumulative_erasure_balance L h b hstep T
  omega

/-- Finite one-way candidate elimination cannot itself supply the unbounded
signal assumed by the active-inference contract. -/
theorem candidate_erasure_not_unbounded_signal
    {Hist Model Ctx Act Obs Byte Answer : Type*}
    [Fintype Model] [DecidableEq Model]
    [Fintype Byte] [Nonempty Byte]
    (L : LanguageSelfProductionCandidate Hist Model Ctx Act Obs Byte Answer)
    (h : ℕ → Hist) (b : ℕ → Byte)
    (hstep : ∀ n, h (n + 1) = L.base.advanceHistory (h n) (b n)) :
    ¬ unbounded (fun n => language_dissipation L (L.base.encode (h n)) (b n)) := by
  apply not_unbounded_of_bounded
  refine ⟨(L.base.encode (h 0)).candidates.card, ?_⟩
  intro T
  exact language_cumulative_erasure_bounded L h b hstep T

/-- The existing four-way language admission and the open self-production
contract can be satisfied by the same persistent memory state. -/
theorem language_self_production_bridge
    {Hist Model Ctx Act Obs Byte Answer : Type*}
    [Fintype Model] [DecidableEq Model]
    [Fintype Byte] [Nonempty Byte]
    (L : LanguageSelfProductionCandidate Hist Model Ctx Act Obs Byte Answer)
    (_hJoint : LanguageLifeAdmission.joint_admission_spec L.base) :
    ∃ x : ℕ → Memory Model L.base.reserve,
        x 0 = L.base.encode L.seed ∧
        (∀ n, x (n + 1) = L.base.next (x n)
          (L.world_response (x n) (L.base.policy (x n)))) ∧
        (∀ n, language_viable L (x n)) := by
  exact exists_infinite_self_producing_trajectory (language_contract L)

/-- The named joint-admission proposition and the infinite self-producing
trajectory now form one explicit language-life existence statement. -/
theorem language_joint_admission_and_self_production
    {Hist Model Ctx Act Obs Byte Answer : Type*}
    [Fintype Model] [DecidableEq Model]
    [Fintype Byte] [Nonempty Byte]
    (L : LanguageSelfProductionCandidate Hist Model Ctx Act Obs Byte Answer) :
    LanguageLifeAdmission.joint_admission_spec L.base ∧
      ∃ x : ℕ → Memory Model L.base.reserve,
        x 0 = L.base.encode L.seed ∧
        (∀ n, x (n + 1) = L.base.next (x n)
          (L.world_response (x n) (L.base.policy (x n)))) ∧
        (∀ n, language_viable L (x n)) := by
  refine ⟨LanguageLifeAdmission.joint_admission L.base, ?_⟩
  exact language_self_production_bridge L
    (LanguageLifeAdmission.joint_admission L.base)

/-- In the joined language contract, the self-consequence predictor is a
single function of the persistent memory, action, and boundary response. -/
theorem language_self_consequence_model_exists
    {Hist Model Ctx Act Obs Byte Answer : Type*}
    [Fintype Model] [DecidableEq Model]
    [Fintype Byte] [Nonempty Byte]
    (L : LanguageSelfProductionCandidate Hist Model Ctx Act Obs Byte Answer) :
    ∃ predict : Memory Model L.base.reserve → Act → Memory Model L.base.reserve,
      ∀ s a, predict s a = L.base.next s (L.world_response s a) := by
  exact action_consequence_model_exists (language_contract L)

/-- The candidate specification assembled from the existing contract. Its
name is historical: it does not cover all original life requirements. Every
conjunct is a field of the admitted base or a theorem above. -/
def language_infinite_stream_life_spec
    {Hist Model Ctx Act Obs Byte Answer : Type*}
    [Fintype Model] [DecidableEq Model]
    [Fintype Byte] [Nonempty Byte]
    (L : LanguageSelfProductionCandidate Hist Model Ctx Act Obs Byte Answer) : Prop :=
  LanguageLifeAdmission.joint_admission_spec L.base ∧
    (∀ h h', L.base.encode h = L.base.encode h' →
      L.base.required h = L.base.required h') ∧
    (∀ h h' b, L.base.encode h = L.base.encode h' →
      L.base.encode (L.base.advanceHistory h b) =
        L.base.encode (L.base.advanceHistory h' b)) ∧
    (Fintype.card (Memory Model L.base.reserve) =
      2 ^ Fintype.card Model * (L.base.reserve + 1)) ∧
    (∀ h b,
      language_dissipation L (L.base.encode h) b =
        candidate_erase_count L.base.possible (L.base.encode h)
          (L.base.context h b) (L.base.action h b) (L.base.outcome h b)) ∧
    (∃ x : ℕ → Memory Model L.base.reserve, ∃ bytes : ℕ → Byte,
      x 0 = L.base.encode L.seed ∧
      (∀ n, bytes n = L.world_response (x n) (L.base.policy (x n))) ∧
      (∀ n, x (n + 1) = L.base.next (x n) (bytes n)) ∧
      (∀ n, language_viable L (x n))) ∧
    (∃ predict : Memory Model L.base.reserve → Act → Memory Model L.base.reserve,
      ∀ s a, predict s a = L.base.next s (L.world_response s a))

/-- Joint candidate-contract theorem. The finite state is the compressed
carrier; the two fibre conditions make it a causal
sufficient statistic; the next conjuncts expose capacity, dissipation,
    infinite viability, and a mathematical action-consequence function using
    the supplied world response. It does not establish a learned self-model. -/
theorem language_infinite_stream_life_complete
    {Hist Model Ctx Act Obs Byte Answer : Type*}
    [Fintype Model] [DecidableEq Model]
    [Fintype Byte] [Nonempty Byte]
    (L : LanguageSelfProductionCandidate Hist Model Ctx Act Obs Byte Answer) :
    language_infinite_stream_life_spec L := by
  have hj := language_joint_admission_and_self_production L
  rcases hj with ⟨hjoint, ⟨x, hx0, hxstep, hxviable⟩⟩
  have hread : ∀ h h', L.base.encode h = L.base.encode h' →
      L.base.required h = L.base.required h' := by
    intro h h' same
    rw [← L.base.read_correct h, ← L.base.read_correct h', same]
  have htransition : ∀ h h' b, L.base.encode h = L.base.encode h' →
      L.base.encode (L.base.advanceHistory h b) =
        L.base.encode (L.base.advanceHistory h' b) := by
    intro h h' b same
    rw [← L.base.next_correct h b, ← L.base.next_correct h' b, same]
  refine ⟨hjoint, hread, htransition, language_state_capacity L, ?_, ?_,
    language_self_consequence_model_exists L⟩
  · intro h b
    exact language_dissipation_encoded_eq_candidate_erase L h b
  · refine ⟨x, (fun n => L.world_response (x n) (L.base.policy (x n))),
      hx0, ?_, ?_, hxviable⟩
    · intro n
      rfl
    · intro n
      exact hxstep n

/-- If the same language candidate's allocation beats the declared
reinvestment cost on every encoded history, its predictive stream contract
has positive expected net log growth on every such history. This is the
smallest formal bridge from language prediction to the declared information
economy; it does not identify that growth with physical energy or trajectory
level survival. -/
theorem language_candidate_predictive_growth_positive
    {Hist Model Ctx Act Obs Byte Answer : Type*}
    [Fintype Model] [DecidableEq Model]
    [Fintype Byte] [Nonempty Byte]
    (L : LanguageSelfProductionCandidate Hist Model Ctx Act Obs Byte Answer)
    (m : ℝ)
    (hce : ∀ h, PredictiveHarvest.crossEntropy (L.base.stream h)
      (L.base.allocation (L.base.encode h)) <
      Real.log (Fintype.card Byte) - m) :
    ∀ h, 0 < PredictiveHarvest.growth (L.base.stream h)
      (L.base.allocation (L.base.encode h)) - m := by
  intro h
  exact PredictiveHarvest.persists_of_crossEntropy
    (L.base.stream h) (L.base.allocation (L.base.encode h)) m
    (L.base.stream_norm h) (L.base.allocation_pos (L.base.encode h)) (hce h)

/-- Conditional joint existence theorem for the current language candidate.
Under the two encoded trajectory-alignment conditions, a cross-entropy
threshold, and a block-level learning minorization recurrence, one and the
same candidate has joint admission, an infinite viable self-production
trajectory, non-eventual passivity, positive expected predictive growth, and
the stated finite-block learning-failure bound. Every condition is explicit;
none is silently promoted to an axiom. -/
theorem language_conditional_joint_existence
    {Hist Model Ctx Act Obs Byte Answer : Type*}
    [Fintype Model] [DecidableEq Model]
    [Fintype Byte] [Nonempty Byte]
    (L : LanguageSelfProductionCandidate Hist Model Ctx Act Obs Byte Answer)
    (hseed : L.base.encode (L.base.trajectory 0) = L.base.encode L.seed)
    (hstep : ∀ n, L.base.encode (L.base.trajectory (n + 1)) =
      L.base.next (L.base.encode (L.base.trajectory n))
        (L.world_response (L.base.encode (L.base.trajectory n))
          (L.base.policy (L.base.encode (L.base.trajectory n)))))
    (m : ℝ)
    (hce : ∀ h, PredictiveHarvest.crossEntropy (L.base.stream h)
      (L.base.allocation (L.base.encode h)) <
      Real.log (Fintype.card Byte) - m)
    (p : ℝ) (hp1 : p ≤ 1) (q : ℕ → ℝ) (hq0 : q 0 ≤ 1)
    (hrec : ∀ n, q (n + 1) ≤ (1 - p) * q n) :
    LanguageLifeAdmission.joint_admission_spec L.base ∧
    (∀ n, language_viable L (trajectory (language_contract L) n)) ∧
    ¬ eventuallyPassive (fun n => L.base.policy (trajectory (language_contract L) n))
      L.base.passive ∧
    (∀ h, 0 < PredictiveHarvest.growth (L.base.stream h)
      (L.base.allocation (L.base.encode h)) - m) ∧
    (∀ n, q n ≤ (1 - p) ^ n) := by
  have hj := language_joint_admission_self_production_active_if_aligned L hseed hstep
  refine ⟨hj.1, hj.2.1, hj.2.2.1, ?_, ?_⟩
  · exact language_candidate_predictive_growth_positive L m hce
  · intro n
    induction n with
    | zero => simpa using hq0
    | succ n ih =>
      calc
        q (n + 1) ≤ (1 - p) * q n := hrec n
        _ ≤ (1 - p) * ((1 - p) ^ n) :=
          mul_le_mul_of_nonneg_left ih (sub_nonneg.mpr hp1)
        _ = (1 - p) ^ (n + 1) := by rw [pow_succ]; ring

/-- Minimal-generator refinement of the conditional joint existence theorem.
The generator supplies the finite causal carrier; the only remaining
conditions are the same two encoded alignment equations, predictive payment
threshold and learning minorization. -/
theorem minimal_generator_conditional_joint_existence
    {Hist Model Ctx Act Obs Byte Answer : Type*}
    [Fintype Model] [DecidableEq Model]
    [Fintype Byte] [Nonempty Byte]
    (G : MinimalLanguageSelfProductionGenerator Hist Model Ctx Act Obs Byte Answer)
    (hseed : (materializeLanguageSelfProductionGenerator G).base.encode
      ((materializeLanguageSelfProductionGenerator G).base.trajectory 0) =
      (materializeLanguageSelfProductionGenerator G).base.encode
        (materializeLanguageSelfProductionGenerator G).seed)
    (hstep : ∀ n, (materializeLanguageSelfProductionGenerator G).base.encode
      ((materializeLanguageSelfProductionGenerator G).base.trajectory (n + 1)) =
      (materializeLanguageSelfProductionGenerator G).base.next
        ((materializeLanguageSelfProductionGenerator G).base.encode
          ((materializeLanguageSelfProductionGenerator G).base.trajectory n))
        ((materializeLanguageSelfProductionGenerator G).world_response
          ((materializeLanguageSelfProductionGenerator G).base.encode
            ((materializeLanguageSelfProductionGenerator G).base.trajectory n))
          ((materializeLanguageSelfProductionGenerator G).base.policy
            ((materializeLanguageSelfProductionGenerator G).base.encode
              ((materializeLanguageSelfProductionGenerator G).base.trajectory n)))))
    (m : ℝ)
    (hce : ∀ h, PredictiveHarvest.crossEntropy
      ((materializeLanguageSelfProductionGenerator G).base.stream h)
      ((materializeLanguageSelfProductionGenerator G).base.allocation
        ((materializeLanguageSelfProductionGenerator G).base.encode h)) <
      Real.log (Fintype.card Byte) - m)
    (p : ℝ) (hp1 : p ≤ 1) (q : ℕ → ℝ) (hq0 : q 0 ≤ 1)
    (hrec : ∀ n, q (n + 1) ≤ (1 - p) * q n) :
    LanguageLifeAdmission.joint_admission_spec
        (materializeLanguageSelfProductionGenerator G).base ∧
    (∀ n, language_viable (materializeLanguageSelfProductionGenerator G)
      (trajectory (language_contract (materializeLanguageSelfProductionGenerator G)) n)) ∧
    ¬ eventuallyPassive
      (fun n => (materializeLanguageSelfProductionGenerator G).base.policy
        (trajectory (language_contract (materializeLanguageSelfProductionGenerator G)) n))
      (materializeLanguageSelfProductionGenerator G).base.passive ∧
    (∀ h, 0 < PredictiveHarvest.growth
      ((materializeLanguageSelfProductionGenerator G).base.stream h)
      ((materializeLanguageSelfProductionGenerator G).base.allocation
        ((materializeLanguageSelfProductionGenerator G).base.encode h)) - m) ∧
    (∀ n, q n ≤ (1 - p) ^ n) := by
  exact language_conditional_joint_existence
    (materializeLanguageSelfProductionGenerator G) hseed hstep m hce p hp1 q hq0 hrec

/-- The minimal generator is sufficient: `read` and `next` are materialized by
history factorization, after which the complete joint theorem applies unchanged. -/
theorem minimal_generator_infinite_stream_life_complete
    {Hist Model Ctx Act Obs Byte Answer : Type*}
    [Fintype Model] [DecidableEq Model]
    [Fintype Byte] [Nonempty Byte]
    (G : MinimalLanguageSelfProductionGenerator Hist Model Ctx Act Obs Byte Answer) :
    language_infinite_stream_life_spec
      (materializeLanguageSelfProductionGenerator G) := by
  exact language_infinite_stream_life_complete
    (materializeLanguageSelfProductionGenerator G)

/-- Given an admissible minimal generator, there is a candidate satisfying
the candidate specification. This assumes the generator and does not establish
existence of an organism satisfying all original life requirements. -/
theorem exists_language_infinite_stream_life_of_minimal_generator
    {Hist Model Ctx Act Obs Byte Answer : Type*}
    [Fintype Model] [DecidableEq Model]
    [Fintype Byte] [Nonempty Byte]
    (G : MinimalLanguageSelfProductionGenerator Hist Model Ctx Act Obs Byte Answer) :
    ∃ L : LanguageSelfProductionCandidate Hist Model Ctx Act Obs Byte Answer,
      language_infinite_stream_life_spec L := by
  exact ⟨materializeLanguageSelfProductionGenerator G,
    minimal_generator_infinite_stream_life_complete G⟩

/-! ## Reusable general-state language carrier

The previous candidate-only generator demands a monotonically shrinking
candidate set. The carrier below keeps history factorization and world closure
for general state types, including finite ones, leaving transitions to the
history rule. It does not postulate a separate resource or a learning rule. -/

structure RenewableLanguageGenerator (Hist State Byte Act Answer : Type*) where
  encode : Hist → State
  required : Hist → Answer
  advanceHistory : Hist → Byte → Hist
  fallback : Answer
  seed : Hist
  viable : State → Prop
  seed_viable : viable (encode seed)
  readable : ∀ h h', encode h = encode h' → required h = required h'
  transition_compatible : ∀ h h' b, encode h = encode h' →
    encode (advanceHistory h b) = encode (advanceHistory h' b)
  policy : State → Act
  world_response : State → Act → Byte
  world_closed : ∀ h, viable (encode h) →
    viable (encode (advanceHistory h (world_response (encode h) (policy (encode h)))))

noncomputable def renewableLanguageRead
    {Hist State Byte Act Answer : Type*}
    (G : RenewableLanguageGenerator Hist State Byte Act Answer) : State → Answer :=
  Classical.choose ((history_readout_exists_iff G.encode G.required G.fallback).2 G.readable)

noncomputable def renewableLanguageNext
    {Hist State Byte Act Answer : Type*}
    (G : RenewableLanguageGenerator Hist State Byte Act Answer) : State → Byte → State :=
  Classical.choose ((history_transition_exists_iff G.encode G.advanceHistory (G.encode G.seed)).2
    (fun h h' same b => G.transition_compatible h h' b same))

theorem renewable_language_carrier_correct
    {Hist State Byte Act Answer : Type*}
    (G : RenewableLanguageGenerator Hist State Byte Act Answer) :
    (∀ h, renewableLanguageRead G (G.encode h) = G.required h) ∧
    (∀ h b, renewableLanguageNext G (G.encode h) b = G.encode (G.advanceHistory h b)) := by
  constructor
  · exact Classical.choose_spec ((history_readout_exists_iff G.encode G.required G.fallback).2 G.readable)
  · exact Classical.choose_spec ((history_transition_exists_iff G.encode G.advanceHistory
      (G.encode G.seed)).2 (fun h h' same b => G.transition_compatible h h' b same))

/-- The viability predicate uses the same encoded state; `next` is derived
from history factorization rather than supplied as an independent carrier. -/
noncomputable def renewableLanguageContract
    {Hist State Byte Act Answer : Type*}
    (G : RenewableLanguageGenerator Hist State Byte Act Answer) : Contract State Act Byte where
  viable := fun s => ∃ h, s = G.encode h ∧ G.viable s
  initial := G.encode G.seed
  action := G.policy
  response := G.world_response
  step := renewableLanguageNext G
  initial_viable := ⟨G.seed, rfl, G.seed_viable⟩
  closed := by
    intro s hs
    obtain ⟨h, rfl, hv⟩ := hs
    rw [(renewable_language_carrier_correct G).2]
    exact ⟨G.advanceHistory h _, rfl, G.world_closed h hv⟩

/-- The same state carrier has its required readout, byte transition, and an
unbounded viable trajectory. Continued adaptation, useful prediction and
active action remain separate world/transition conditions. -/
theorem renewable_language_carrier_trajectory
    {Hist State Byte Act Answer : Type*}
    (G : RenewableLanguageGenerator Hist State Byte Act Answer) :
    (∀ h, renewableLanguageRead G (G.encode h) = G.required h) ∧
    (∀ h b, renewableLanguageNext G (G.encode h) b = G.encode (G.advanceHistory h b)) ∧
    (∀ n, (renewableLanguageContract G).viable (trajectory (renewableLanguageContract G) n)) := by
  exact ⟨(renewable_language_carrier_correct G).1,
    (renewable_language_carrier_correct G).2, trajectory_viable (renewableLanguageContract G)⟩

/-! ### Renewable language life gate

`RenewableLanguageGenerator` is the minimal state carrier: it supplies only
history factorization, an input transition, a viable seed, policy, and a
world response.  The wrapper below adds the information-economy and learning
conditions without prescribing a neural mechanism.  In particular, the
learning condition below compares both allocations against the same pre-update
stream law, so changing the law itself cannot masquerade as improvement.
-/

def renewablePredictiveUpdateEvent
    {Hist State Byte Act Answer : Type*}
    [Fintype Byte] [Nonempty Byte]
    (carrier : RenewableLanguageGenerator Hist State Byte Act Answer)
    (stream allocation : State → Byte → ℝ)
    (s : State) (b : Byte) : Prop :=
  let s' := renewableLanguageNext carrier s b
  s' ≠ s ∧
    PredictiveHarvest.crossEntropy (stream s) (allocation s') <
      PredictiveHarvest.crossEntropy (stream s) (allocation s)

/-- The precision-derived gain used by the active GDN path is a valid binary
predictive update whenever incoming precision is positive.  This is the
smallest analytic bridge from natural-parameter assimilation to the renewable
life gate: no fixed learning-rate parameter is introduced. -/
theorem precision_binary_update_is_predictive_improvement
    (r : ℝ) (y : Bool) (p w : ℝ)
    (hr0 : 0 < r) (hr1 : r < 1)
    (hp : 0 < p) (hw : 0 < w) :
    let β := w / (p + w)
    let r' := r + β * ((if y then 1 else 0) - r)
    (0 ≤ r' ∧ r' ≤ 1 ∧ r' ≠ r) ∧
      PredictiveHarvest.crossEntropy (CGB.Generation.binaryPointLaw y)
          (CGB.Generation.binaryAllocation r') <
        PredictiveHarvest.crossEntropy (CGB.Generation.binaryPointLaw y)
          (CGB.Generation.binaryAllocation r) := by
  have hg := CGB.Generation.precision_assimilation_gain_bounds p w hp hw.le
  have hβ0 : 0 < w / (p + w) := by positivity
  simpa using CGB.Generation.binary_delta_crossEntropy_decreases r y
    (w / (p + w)) hr0 hr1 hβ0 hg.2.le

/-- Fitting an arrived bit does not by itself improve the underlying stream:
if the old binary predictor already equals the stream law, its expected log
loss cannot improve after the precision-derived update. The same update still
improves loss on the arrived bit. This compares both predictors on one law. -/
theorem precision_binary_sample_gain_not_future_guarantee
    (r : ℝ) (y : Bool) (p w : ℝ)
    (hr0 : 0 < r) (hr1 : r < 1) (hp : 0 < p) (hw : 0 < w) :
    let β := w / (p + w)
    let r' := r + β * ((if y then 1 else 0) - r)
    PredictiveHarvest.crossEntropy (CGB.Generation.binaryPointLaw y)
        (CGB.Generation.binaryAllocation r') <
      PredictiveHarvest.crossEntropy (CGB.Generation.binaryPointLaw y)
        (CGB.Generation.binaryAllocation r) ∧
    PredictiveHarvest.crossEntropy (CGB.Generation.binaryAllocation r)
        (CGB.Generation.binaryAllocation r) ≤
      PredictiveHarvest.crossEntropy (CGB.Generation.binaryAllocation r)
        (CGB.Generation.binaryAllocation r') := by
  dsimp only
  let β := w / (p + w)
  let r' := r + β * ((if y then 1 else 0) - r)
  have hb0 : 0 < β := by dsimp [β]; positivity
  have hb1 : β < 1 := (CGB.Generation.precision_assimilation_gain_bounds p w hp hw.le).2
  have hr'0 : 0 < r' := by cases y <;> simp [r'] <;> nlinarith
  have hr'1 : r' < 1 := by cases y <;> simp [r'] <;> nlinarith
  constructor
  · exact (precision_binary_update_is_predictive_improvement r y p w hr0 hr1 hp hw).2
  · have h := PredictiveHarvest.gibbs_inequality
        (CGB.Generation.binaryAllocation r) (CGB.Generation.binaryAllocation r')
        (by intro b; cases b <;> simp [CGB.Generation.binaryAllocation] <;> linarith)
        (by simp [CGB.Generation.binaryAllocation])
        (by intro b; cases b <;> simp [CGB.Generation.binaryAllocation] <;> linarith)
        (by simp [CGB.Generation.binaryAllocation])
    unfold PredictiveHarvest.crossEntropy
    linarith

/-- Categorical precision-weighted assimilation improves the next symbol's
log loss exactly when that symbol repeats the observed symbol; a different
symbol becomes strictly less likely. This is a one-step realised outcome,
not a claim about expected loss for a general language distribution. -/
theorem categorical_precision_next_loss_direction
    {Byte : Type*} [DecidableEq Byte] (c : Byte → ℝ) (observed next : Byte)
    (p w : ℝ) (hp : 0 < p) (hw : 0 < w)
    (hc0 : 0 < c next) (hc1 : c next < 1) :
    let q := c next + (w / (p + w)) * ((if next = observed then 1 else 0) - c next)
    0 < q ∧
      ((-Real.log q < -Real.log (c next)) ↔ next = observed) ∧
      ((-Real.log (c next) < -Real.log q) ↔ next ≠ observed) := by
  dsimp only
  let β := w / (p + w)
  have hb0 : 0 < β := by dsimp [β]; positivity
  have hb1 : β < 1 := (CGB.Generation.precision_assimilation_gain_bounds p w hp hw.le).2
  by_cases he : next = observed
  · have hpos : 0 < c next + β * (1 - c next) := by nlinarith
    have hincrease : c next < c next + β * (1 - c next) := by nlinarith
    have hlog := Real.strictMonoOn_log hc0 hpos hincrease
    have hless : -Real.log (c next + β * (1 - c next)) < -Real.log (c next) := by linarith
    have hnot : ¬(-Real.log (c next) < -Real.log (c next + β * (1 - c next))) := by linarith
    simpa [he, β] using And.intro hpos (And.intro hless hnot)
  · have hpos : 0 < c next + β * (0 - c next) := by nlinarith
    have hdecrease : c next + β * (0 - c next) < c next := by nlinarith
    have hlog := Real.strictMonoOn_log hpos hc0 hdecrease
    have hless : -Real.log (c next) < -Real.log (c next + β * (0 - c next)) := by linarith
    have hnot : ¬(-Real.log (c next + β * (0 - c next)) < -Real.log (c next)) := by linarith
    simpa [he, β] using And.intro hpos (And.intro hnot hless)

structure RenewableLanguageLifeGenerator
    (Hist State Byte Act Answer : Type*)
    [Fintype Byte] [Nonempty Byte] where
  carrier : RenewableLanguageGenerator Hist State Byte Act Answer
  stream : State → Byte → ℝ
  allocation : State → Byte → ℝ
  stream_nonneg : ∀ s b, 0 ≤ stream s b
  stream_norm : ∀ s, ∑ b, stream s b = 1
  allocation_pos : ∀ s b, 0 < allocation s b
  allocation_norm : ∀ s, ∑ b, allocation s b = 1
  signal : ℕ → ℕ
  passive : Act
  response : Act → Byte → ℝ
  predict : Byte → ℝ
  viable_signal : unbounded signal
  passive_bound : passiveBounded
    (fun n => carrier.policy (trajectory (renewableLanguageContract carrier) n))
    signal passive
  action_sensitive : actionSensitive response
  recurrent_predictive_learning : ∀ N, ∃ n, N ≤ n ∧
    renewablePredictiveUpdateEvent carrier stream allocation
      (trajectory (renewableLanguageContract carrier) n)
      (carrier.world_response
        (trajectory (renewableLanguageContract carrier) n)
        (carrier.policy (trajectory (renewableLanguageContract carrier) n)))

/-- The renewable carrier, plus explicit predictive payment and recurrent
learning conditions, yields one general-state language life gate. Recurrent
learning is stated as a change of the same state under the same world
response, so the result rules out eventual freezing without adding a clock or
a second reserve. -/
theorem renewable_language_life_gate
    {Hist State Byte Act Answer : Type*}
    [Fintype Byte] [Nonempty Byte]
    (G : RenewableLanguageLifeGenerator Hist State Byte Act Answer)
    (m : ℝ)
    (hce : ∀ s, PredictiveHarvest.crossEntropy (G.stream s) (G.allocation s) <
      Real.log (Fintype.card Byte) - m) :
    (∀ h, renewableLanguageRead G.carrier (G.carrier.encode h) = G.carrier.required h) ∧
    (∀ h b, renewableLanguageNext G.carrier (G.carrier.encode h) b =
      G.carrier.encode (G.carrier.advanceHistory h b)) ∧
    (∀ n, (renewableLanguageContract G.carrier).viable
      (trajectory (renewableLanguageContract G.carrier) n)) ∧
    ¬ eventuallyPassive
      (fun n => G.carrier.policy (trajectory (renewableLanguageContract G.carrier) n))
      G.passive ∧
    ¬ (∀ a b, G.predict b = G.response a b) ∧
    (∀ s, 0 < PredictiveHarvest.growth (G.stream s) (G.allocation s) - m) ∧
    (∀ N, ∃ n, N ≤ n ∧
      renewablePredictiveUpdateEvent G.carrier G.stream G.allocation
        (trajectory (renewableLanguageContract G.carrier) n)
        (G.carrier.world_response
          (trajectory (renewableLanguageContract G.carrier) n)
          (G.carrier.policy (trajectory (renewableLanguageContract G.carrier) n)))) ∧
    ¬ (∃ N, ∀ n, N ≤ n →
      trajectory (renewableLanguageContract G.carrier) (n + 1) =
        trajectory (renewableLanguageContract G.carrier) n) := by
  let C := renewableLanguageContract G.carrier
  have hcarrier := renewable_language_carrier_correct G.carrier
  have htraj : ∀ n, C.viable (trajectory C n) := by
    intro n
    exact trajectory_viable C n
  have hactive := active_closure
    (fun n => G.carrier.policy (trajectory C n)) G.signal G.passive
    G.response G.predict G.viable_signal G.passive_bound G.action_sensitive
  have hgrowth : ∀ s, 0 < PredictiveHarvest.growth (G.stream s)
      (G.allocation s) - m := by
    intro s
    exact PredictiveHarvest.persists_of_crossEntropy
      (G.stream s) (G.allocation s) m (G.stream_norm s)
      (G.allocation_pos s) (hce s)
  have hlearn : ∀ N, ∃ n, N ≤ n ∧
      renewablePredictiveUpdateEvent G.carrier G.stream G.allocation
        (trajectory C n)
        (G.carrier.world_response (trajectory C n)
          (G.carrier.policy (trajectory C n))) := by
    intro N
    simpa [C] using G.recurrent_predictive_learning N
  have hnofreeze : ¬ (∃ N, ∀ n, N ≤ n →
      trajectory C (n + 1) = trajectory C n) := by
    rintro ⟨N, hfreeze⟩
    obtain ⟨n, hn, he⟩ := hlearn N
    let s := trajectory C n
    let b := G.carrier.world_response s (G.carrier.policy s)
    have hc : renewableLanguageNext G.carrier s b ≠ s := by
      simpa [renewablePredictiveUpdateEvent, s, b] using he.1
    have hs : trajectory C (n + 1) = renewableLanguageNext G.carrier s b := by
      rfl
    have hf := hfreeze n hn
    apply hc
    rw [← hs, hf]
  refine ⟨hcarrier.1, ?_⟩
  refine ⟨hcarrier.2, ?_⟩
  refine ⟨htraj, ?_⟩
  refine ⟨hactive.1, ?_⟩
  refine ⟨hactive.2, ?_⟩
  refine ⟨hgrowth, ?_⟩
  exact ⟨hlearn, hnofreeze⟩

/-- Existential form of the renewable gate.  The generated trajectory is the
same one used by the carrier, active signal, and recurrent-learning clauses;
it is not a separately chosen witness. -/
theorem exists_renewable_language_life_of_generator
    {Hist State Byte Act Answer : Type*}
    [Fintype Byte] [Nonempty Byte]
    (G : RenewableLanguageLifeGenerator Hist State Byte Act Answer)
    (m : ℝ)
    (hce : ∀ s, PredictiveHarvest.crossEntropy (G.stream s) (G.allocation s) <
      Real.log (Fintype.card Byte) - m) :
    ∃ x : ℕ → State,
      x 0 = G.carrier.encode G.carrier.seed ∧
      (∀ n, x (n + 1) = renewableLanguageNext G.carrier (x n)
        (G.carrier.world_response (x n) (G.carrier.policy (x n)))) ∧
      (∀ n, (renewableLanguageContract G.carrier).viable (x n)) ∧
      ¬ eventuallyPassive
        (fun n => G.carrier.policy (x n)) G.passive ∧
      ¬ (∀ a b, G.predict b = G.response a b) ∧
      (∀ s, 0 < PredictiveHarvest.growth (G.stream s) (G.allocation s) - m) ∧
      (∀ N, ∃ n, N ≤ n ∧ renewablePredictiveUpdateEvent
        G.carrier G.stream G.allocation (x n)
        (G.carrier.world_response (x n) (G.carrier.policy (x n)))) ∧
      ¬ (∃ N, ∀ n, N ≤ n → x (n + 1) = x n) := by
  have hg := renewable_language_life_gate G m hce
  rcases hg with ⟨hread, hnext, hviable, hpassive, hblind,
    hgrowth, hlearn, hnofreeze⟩
  let C := renewableLanguageContract G.carrier
  refine ⟨trajectory C, rfl, ?_, ?_, ?_, ?_, hgrowth, ?_, ?_⟩
  · intro n
    rfl
  · exact hviable
  · exact hpassive
  · exact hblind
  · exact hlearn
  · exact hnofreeze

/-! ## World state is not organism memory

The organism carrier is separate from the external world, which may have its
own independently evolving state, including an unbounded stream cursor. -/

structure RenewableOpenWorld
    {Hist State Byte Act Answer : Type*}
    (G : RenewableLanguageGenerator Hist State Byte Act Answer) (World : Type*) where
  initial : World
  boundary : World → State → Act → Byte
  evolve : World → State → Act → Byte → World
  closed : ∀ h w, G.viable (G.encode h) →
    G.viable (G.encode (G.advanceHistory h (boundary w (G.encode h) (G.policy (G.encode h)))))

/-- One joint trajectory, with action carried into the transition so the
same world can also be replayed under action interventions. -/
noncomputable def renewableOpenContract
    {Hist State Byte Act Answer World : Type*}
    (G : RenewableLanguageGenerator Hist State Byte Act Answer)
    (W : RenewableOpenWorld G World) : Contract (State × World) Act (Act × Byte) where
  viable := fun p => ∃ h, p.1 = G.encode h ∧ G.viable p.1
  initial := (G.encode G.seed, W.initial)
  action := fun p => G.policy p.1
  response := fun p a => (a, W.boundary p.2 p.1 a)
  step := fun p input =>
    (renewableLanguageNext G p.1 input.2, W.evolve p.2 p.1 input.1 input.2)
  initial_viable := ⟨G.seed, rfl, G.seed_viable⟩
  closed := by
    intro p hp
    obtain ⟨h, hh, hv⟩ := hp
    rw [hh] at hv
    change ∃ h', renewableLanguageNext G p.1 (W.boundary p.2 p.1 (G.policy p.1)) = G.encode h' ∧
      G.viable (renewableLanguageNext G p.1 (W.boundary p.2 p.1 (G.policy p.1)))
    rw [hh, (renewable_language_carrier_correct G).2]
    exact ⟨G.advanceHistory h _, rfl, W.closed h p.2 hv⟩

/-- The proof-side history follows the actual open-world trajectory. It is not
stored by the organism. -/
noncomputable def renewableOpenHistory
    {Hist State Byte Act Answer World : Type*}
    (G : RenewableLanguageGenerator Hist State Byte Act Answer)
    (W : RenewableOpenWorld G World) : ℕ → Hist
  | 0 => G.seed
  | n + 1 =>
    let z := trajectory (renewableOpenContract G W) n
    G.advanceHistory (renewableOpenHistory G W n) (W.boundary z.2 z.1 (G.policy z.1))

/-- One open-world trajectory remains viable and realizes the readout for
the history of its own delivered inputs. The external world and proof-side
history need not be finite or stored inside the organism. -/
theorem renewable_open_world_trajectory
    {Hist State Byte Act Answer World : Type*}
    (G : RenewableLanguageGenerator Hist State Byte Act Answer)
    (W : RenewableOpenWorld G World) :
    ∃ x : ℕ → State × World,
      x 0 = (G.encode G.seed, W.initial) ∧
      (∀ n, x (n + 1) = (renewableOpenContract G W).step (x n)
        ((renewableOpenContract G W).response (x n) ((renewableOpenContract G W).action (x n)))) ∧
      (∀ n, (renewableOpenContract G W).viable (x n)) ∧
      ∃ h : ℕ → Hist,
        h 0 = G.seed ∧
        (∀ n, h (n + 1) = G.advanceHistory (h n)
          (W.boundary (x n).2 (x n).1 (G.policy (x n).1))) ∧
        (∀ n, G.encode (h n) = (x n).1 ∧
          renewableLanguageRead G (x n).1 = G.required (h n)) := by
  let C := renewableOpenContract G W
  let h := renewableOpenHistory G W
  have halign : ∀ n, G.encode (h n) = (trajectory C n).1 := by
    intro n
    induction n with
    | zero => rfl
    | succ n ih =>
      change G.encode (G.advanceHistory (h n)
        (W.boundary (trajectory C n).2 (trajectory C n).1 (G.policy (trajectory C n).1))) =
        renewableLanguageNext G (trajectory C n).1
          (W.boundary (trajectory C n).2 (trajectory C n).1 (G.policy (trajectory C n).1))
      rw [← ih, (renewable_language_carrier_correct G).2]
  refine ⟨trajectory C, rfl, ?_, trajectory_viable C, h, rfl, ?_, ?_⟩
  · intro n; rfl
  · intro n; rfl
  · intro n
    refine ⟨halign n, ?_⟩
    rw [← halign n]
    exact (renewable_language_carrier_correct G).1 (h n)

/-- Conditional joint gate on the *open* body-world trajectory. The actual
boundary selects each learning event; the world may have its own unbounded
state. Signal, passive limitation, predictive payment and recurrent useful
updates are assumptions on this trajectory, not consequences of a local
score step. The world-closure condition is the existing strong `W.closed`. -/
theorem renewable_open_language_joint_gate
    {Hist State Byte Act Answer World : Type*}
    [Fintype Byte] [Nonempty Byte]
    (G : RenewableLanguageGenerator Hist State Byte Act Answer)
    (W : RenewableOpenWorld G World)
    (stream allocation : State → Byte → ℝ)
    (_hstream_nonneg : ∀ s b, 0 ≤ stream s b)
    (hstream : ∀ s, ∑ b, stream s b = 1)
    (halloc : ∀ s b, 0 < allocation s b)
    (_halloc_norm : ∀ s, ∑ b, allocation s b = 1)
    (m : ℝ)
    (hce : ∀ n, PredictiveHarvest.crossEntropy
      (stream (trajectory (renewableOpenContract G W) n).1)
      (allocation (trajectory (renewableOpenContract G W) n).1) <
      Real.log (Fintype.card Byte) - m)
    (signal : ℕ → ℕ) (passive : Act)
    (hunbounded : unbounded signal)
    (hpassive : passiveBounded
      (fun n => G.policy (trajectory (renewableOpenContract G W) n).1) signal passive)
    (haction : ∃ n a a',
      W.boundary (trajectory (renewableOpenContract G W) n).2
        (trajectory (renewableOpenContract G W) n).1 a ≠
      W.boundary (trajectory (renewableOpenContract G W) n).2
        (trajectory (renewableOpenContract G W) n).1 a')
    (hlearn : ∀ N, ∃ n, N ≤ n ∧
      renewablePredictiveUpdateEvent G stream allocation
        (trajectory (renewableOpenContract G W) n).1
        (W.boundary (trajectory (renewableOpenContract G W) n).2
          (trajectory (renewableOpenContract G W) n).1
          (G.policy (trajectory (renewableOpenContract G W) n).1))) :
    ∃ x : ℕ → State × World,
      x 0 = (G.encode G.seed, W.initial) ∧
      (∀ n, x (n + 1) = (renewableOpenContract G W).step (x n)
        ((renewableOpenContract G W).response (x n) ((renewableOpenContract G W).action (x n)))) ∧
      (∀ n, (renewableOpenContract G W).viable (x n)) ∧
      (∃ h : ℕ → Hist, h 0 = G.seed ∧
        (∀ n, h (n + 1) = G.advanceHistory (h n)
          (W.boundary (x n).2 (x n).1 (G.policy (x n).1))) ∧
        (∀ n, G.encode (h n) = (x n).1 ∧
          renewableLanguageRead G (x n).1 = G.required (h n))) ∧
      ¬ eventuallyPassive (fun n => G.policy (x n).1) passive ∧
      ¬ (∃ predict : World → State → Byte, ∀ n a,
        predict (x n).2 (x n).1 = W.boundary (x n).2 (x n).1 a) ∧
      (∀ n, 0 < PredictiveHarvest.growth (stream (x n).1) (allocation (x n).1) - m) ∧
      (∀ N, ∃ n, N ≤ n ∧ renewablePredictiveUpdateEvent G stream allocation (x n).1
        (W.boundary (x n).2 (x n).1 (G.policy (x n).1))) ∧
      ¬ (∃ N, ∀ n, N ≤ n → (x (n + 1)).1 = (x n).1) := by
  let C := renewableOpenContract G W
  obtain ⟨x, hx0, hxstep, hxviable, h, hh0, hhstep, hread⟩ :=
    renewable_open_world_trajectory G W
  have hx : x = trajectory C := by
    funext n
    induction n with
    | zero => exact hx0
    | succ n ih =>
      rw [hxstep n, ih]
      rfl
  subst x
  refine ⟨trajectory C, hx0, hxstep, hxviable, ⟨h, hh0, hhstep, hread⟩, ?_, ?_, ?_, hlearn, ?_⟩
  · exact viable_not_eventually_passive _ signal passive hunbounded hpassive
  · rintro ⟨predict, hpred⟩
    obtain ⟨n, a, a', hne⟩ := haction
    apply hne
    exact (hpred n a).symm.trans (hpred n a')
  · intro n
    exact PredictiveHarvest.persists_of_crossEntropy _ _ m (hstream _) (halloc _) (hce n)
  · rintro ⟨N, hfreeze⟩
    obtain ⟨n, hn, hevent⟩ := hlearn N
    have hchange := hevent.1
    apply hchange
    have hsame := hfreeze n hn
    change renewableLanguageNext G (trajectory C n).1
      (W.boundary (trajectory C n).2 (trajectory C n).1
        (G.policy (trajectory C n).1)) = (trajectory C n).1 at hsame
    exact hsame

/-- The categorical innovation condition applied to the actual boundary and
the next actual boundary on one viable open-world trajectory. The next symbol
is used only to assess the result; it is not an input to the earlier update. -/
theorem open_world_categorical_next_loss_direction
    {Hist State Byte Act Answer World : Type*} [DecidableEq Byte]
    (G : RenewableLanguageGenerator Hist State Byte Act Answer)
    (W : RenewableOpenWorld G World) (allocation : State → Byte → ℝ)
    (n : ℕ) (p w : ℝ) (hp : 0 < p) (hw : 0 < w)
    (hprob : ∀ b, 0 < allocation (trajectory (renewableOpenContract G W) n).1 b ∧
      allocation (trajectory (renewableOpenContract G W) n).1 b < 1)
    (hupdate : ∀ b, allocation (trajectory (renewableOpenContract G W) (n + 1)).1 b =
      allocation (trajectory (renewableOpenContract G W) n).1 b + (w / (p + w)) *
        ((if b = W.boundary (trajectory (renewableOpenContract G W) n).2
          (trajectory (renewableOpenContract G W) n).1
          (G.policy (trajectory (renewableOpenContract G W) n).1) then 1 else 0) -
          allocation (trajectory (renewableOpenContract G W) n).1 b)) :
    let C := renewableOpenContract G W
    let s := (trajectory C n).1
    let s' := (trajectory C (n + 1)).1
    let observed := W.boundary (trajectory C n).2 s (G.policy s)
    let next := W.boundary (trajectory C (n + 1)).2 s' (G.policy s')
    C.viable (trajectory C (n + 1)) ∧
      ((-Real.log (allocation s' next) < -Real.log (allocation s next)) ↔ next = observed) ∧
      ((-Real.log (allocation s next) < -Real.log (allocation s' next)) ↔ next ≠ observed) := by
  dsimp only
  let C := renewableOpenContract G W
  let s := (trajectory C n).1
  let s' := (trajectory C (n + 1)).1
  let observed := W.boundary (trajectory C n).2 s (G.policy s)
  let next := W.boundary (trajectory C (n + 1)).2 s' (G.policy s')
  have h := categorical_precision_next_loss_direction (allocation s) observed next p w hp hw
    (hprob next).1 (hprob next).2
  dsimp only at h
  refine ⟨trajectory_viable C (n + 1), ?_⟩
  rw [hupdate next]
  exact h.2

/-- An actual open-world update improves the next prediction when the next
delivered bit repeats the current one and the same body's probability obeys
the existing precision-derived innovation update. Both scores target the
actual next bit; the condition is not a claim that arbitrary streams repeat. -/
theorem open_world_precision_update_improves_next_prediction
    {Hist State Act Answer World : Type*}
    (G : RenewableLanguageGenerator Hist State Bool Act Answer)
    (W : RenewableOpenWorld G World) (prob : State → ℝ)
    (n : ℕ) (p w : ℝ) (hp : 0 < p) (hw : 0 < w)
    (hprob0 : 0 < prob (trajectory (renewableOpenContract G W) n).1)
    (hprob1 : prob (trajectory (renewableOpenContract G W) n).1 < 1)
    (hupdate :
      prob (trajectory (renewableOpenContract G W) (n + 1)).1 =
        prob (trajectory (renewableOpenContract G W) n).1 + (w / (p + w)) *
          ((if W.boundary (trajectory (renewableOpenContract G W) n).2
            (trajectory (renewableOpenContract G W) n).1
            (G.policy (trajectory (renewableOpenContract G W) n).1) then 1 else 0) -
            prob (trajectory (renewableOpenContract G W) n).1))
    (hnext : W.boundary (trajectory (renewableOpenContract G W) (n + 1)).2
        (trajectory (renewableOpenContract G W) (n + 1)).1
        (G.policy (trajectory (renewableOpenContract G W) (n + 1)).1) =
      W.boundary (trajectory (renewableOpenContract G W) n).2
        (trajectory (renewableOpenContract G W) n).1
        (G.policy (trajectory (renewableOpenContract G W) n).1)) :
    (renewableOpenContract G W).viable (trajectory (renewableOpenContract G W) (n + 1)) ∧
    (trajectory (renewableOpenContract G W) (n + 1)).1 ≠
      (trajectory (renewableOpenContract G W) n).1 ∧
    PredictiveHarvest.crossEntropy
        (CGB.Generation.binaryPointLaw
          (W.boundary (trajectory (renewableOpenContract G W) (n + 1)).2
            (trajectory (renewableOpenContract G W) (n + 1)).1
            (G.policy (trajectory (renewableOpenContract G W) (n + 1)).1)))
        (CGB.Generation.binaryAllocation
          (prob (trajectory (renewableOpenContract G W) (n + 1)).1)) <
      PredictiveHarvest.crossEntropy
        (CGB.Generation.binaryPointLaw
          (W.boundary (trajectory (renewableOpenContract G W) (n + 1)).2
            (trajectory (renewableOpenContract G W) (n + 1)).1
            (G.policy (trajectory (renewableOpenContract G W) (n + 1)).1)))
        (CGB.Generation.binaryAllocation
          (prob (trajectory (renewableOpenContract G W) n).1)) := by
  let C := renewableOpenContract G W
  let s := (trajectory C n).1
  let s' := (trajectory C (n + 1)).1
  let b := W.boundary (trajectory C n).2 s (G.policy s)
  have h := precision_binary_update_is_predictive_improvement (prob s) b p w hprob0 hprob1 hp hw
  dsimp only at h
  have hpchange : prob s' ≠ prob s := by rw [hupdate]; exact h.1.2.2
  refine ⟨trajectory_viable C (n + 1), ?_, ?_⟩
  · intro heq
    exact hpchange (congrArg prob heq)
  · rw [hnext, hupdate]
    exact h.2

/-- Intervene on delivered input only: the body receives the ordinary `quiet`
symbol while the world keeps evolving under the actual action and the world
response it produced. The clock index is external and no extra body state or
separate missing-observation channel is introduced. -/
noncomputable def maskedOpenRun
    {Hist State Byte Act Answer World : Type*}
    (G : RenewableLanguageGenerator Hist State Byte Act Answer)
    (W : RenewableOpenWorld G World) (quiet : Byte) : ℕ → State × World → State × World
  | 0, p => p
  | n + 1, p =>
      let z := maskedOpenRun G W quiet n p
      let a := G.policy z.1
      let emitted := W.boundary z.2 z.1 a
      (renewableLanguageNext G z.1 quiet, W.evolve z.2 z.1 a emitted)

/-- Under the same fixed delivered symbol, identical starting bodies have
identical body states and actions for any two evolving external worlds.
If those worlds instead require opposite equally likely next-bit predictions,
the common body-only prediction has at least one bit of average log loss.
The separation of future targets is an explicit world condition. -/
theorem masked_open_world_information_limit
    {Hist State Byte Act Answer World : Type*}
    (G : RenewableLanguageGenerator Hist State Byte Act Answer)
    (W : RenewableOpenWorld G World) (quiet : Byte) (s : State) (w₀ w₁ : World)
    (target : State × World → Bool) (predict : State → ℝ) (n : ℕ)
    (hzero : target (maskedOpenRun G W quiet n (s, w₀)) = false)
    (hone : target (maskedOpenRun G W quiet n (s, w₁)) = true)
    (hp : 0 < predict (maskedOpenRun G W quiet n (s, w₀)).1)
    (hp1 : predict (maskedOpenRun G W quiet n (s, w₀)).1 < 1) :
    (maskedOpenRun G W quiet n (s, w₀)).1 = (maskedOpenRun G W quiet n (s, w₁)).1 ∧
    G.policy (maskedOpenRun G W quiet n (s, w₀)).1 =
      G.policy (maskedOpenRun G W quiet n (s, w₁)).1 ∧
    target (maskedOpenRun G W quiet n (s, w₀)) = false ∧
    target (maskedOpenRun G W quiet n (s, w₁)) = true ∧
    -(Real.log (1 - predict (maskedOpenRun G W quiet n (s, w₀)).1) +
      Real.log (predict (maskedOpenRun G W quiet n (s, w₁)).1)) / 2 ≥ Real.log 2 := by
  have heq_all : ∀ t, (maskedOpenRun G W quiet t (s, w₀)).1 =
      (maskedOpenRun G W quiet t (s, w₁)).1 := by
    intro t
    induction t with
    | zero => rfl
    | succ t ih => simp only [maskedOpenRun]; rw [ih]
  have heq := heq_all n
  exact ⟨heq, congrArg G.policy heq, hzero, hone,
    merged_histories_binary_logloss_floor
      (fun w => (maskedOpenRun G W quiet n (s, w)).1) predict w₀ w₁ heq hp hp1⟩

/-- The present `RenewableOpenWorld.closed` quantifies over every hidden
world state at a viable encoded body. It therefore excludes a pair of worlds
that require two different actions for even one-step body viability. This
pinpoints a limitation of that contract, not an impossibility of life in such
worlds. -/
theorem open_world_closure_excludes_conflicting_safe_actions
    {Hist State Byte Act Answer World : Type*}
    (G : RenewableLanguageGenerator Hist State Byte Act Answer)
    (W : RenewableOpenWorld G World) (h : Hist) (w₀ w₁ : World) (a₀ a₁ : Act)
    (hv : G.viable (G.encode h)) (hne : a₀ ≠ a₁)
    (h₀ : ∀ a, G.viable (G.encode (G.advanceHistory h
      (W.boundary w₀ (G.encode h) a))) → a = a₀)
    (h₁ : ∀ a, G.viable (G.encode (G.advanceHistory h
      (W.boundary w₁ (G.encode h) a))) → a = a₁) : False := by
  have ha := h₀ _ (W.closed h w₀ hv)
  have hb := h₁ _ (W.closed h w₁ hv)
  exact hne (ha.symm.trans hb)

/-- For a joint body-world viability predicate, two worlds hidden behind the
same body can require conflicting safe actions. In that case the same
body-only policy cannot keep both successor pairs viable. The predicate is
an assessment of the existing joint state, not an additional body account. -/
theorem indistinguishable_joint_worlds_cannot_both_stay_viable
    {State World Act Byte : Type*}
    (boundary : World → State → Act → Byte)
    (bodyStep : State → Byte → State)
    (worldStep : World → State → Act → Byte → World)
    (viable : State × World → Prop) (policy : State → Act)
    (s : State) (w₀ w₁ : World) (a₀ a₁ : Act) (hne : a₀ ≠ a₁)
    (h₀ : ∀ a, viable (bodyStep s (boundary w₀ s a),
        worldStep w₀ s a (boundary w₀ s a)) → a = a₀)
    (h₁ : ∀ a, viable (bodyStep s (boundary w₁ s a),
        worldStep w₁ s a (boundary w₁ s a)) → a = a₁) :
    ¬ (viable (bodyStep s (boundary w₀ s (policy s)),
          worldStep w₀ s (policy s) (boundary w₀ s (policy s))) ∧
       viable (bodyStep s (boundary w₁ s (policy s)),
          worldStep w₁ s (policy s) (boundary w₁ s (policy s)))) := by
  rintro ⟨hv₀, hv₁⟩
  exact hne ((h₀ _ hv₀).symm.trans (h₁ _ hv₁))

/-- A body-only policy preserving a joint viable region exists exactly when,
for each body state, there is one action safe for every compatible world in
that region. Given a viable initial pair, the selected policy generates one
joint trajectory that stays viable at every finite tick. The predicate and
one-step safe actions are conditions to verify, not a proof that a language
world or a complete living organism supplies them. Classical choice here is
an existence proof, not a claimed executable or efficient policy. -/
theorem joint_viability_policy_generation
    {State World Act : Type*}
    (step : State → World → Act → State × World)
    (viable : State × World → Prop) (seed : State × World) (hseed : viable seed) :
    (∀ s, ∃ a, ∀ w, viable (s, w) → viable (step s w a)) ↔
    (∃ policy : State → Act,
      (∀ s w, viable (s, w) → viable (step s w (policy s))) ∧
      ∃ run : ℕ → State × World,
        run 0 = seed ∧
        (∀ n, run (n + 1) = step (run n).1 (run n).2 (policy (run n).1)) ∧
        (∀ n, viable (run n))) := by
  constructor
  · intro hs
    let policy : State → Act := fun s => Classical.choose (hs s)
    have hsafe : ∀ s w, viable (s, w) → viable (step s w (policy s)) := by
      intro s w hv
      exact Classical.choose_spec (hs s) w hv
    let c : Contract (State × World) Act Act := {
      viable := viable
      initial := seed
      action := fun z => policy z.1
      response := fun _ a => a
      step := fun z a => step z.1 z.2 a
      initial_viable := hseed
      closed := by intro z hz; exact hsafe z.1 z.2 hz }
    refine ⟨policy, hsafe, trajectory c, rfl, ?_, ?_⟩
    · intro n; rfl
    · intro n; exact trajectory_viable c n
  · rintro ⟨policy, hsafe, _⟩ s
    exact ⟨policy s, hsafe s⟩

private theorem closed_region_law_zero
    {S E : Type*} [Fintype S] [DecidableEq S] [Fintype E]
    (Q : CGB.Interaction.Machine S ℕ E) (region : S → Prop) (seed : S)
    (hseed : region seed)
    (hsafe : ∀ t x e y, region x → 0 < Q.step t x e y → region y) :
    ∀ n z, ¬ region z → CGB.Energetics.law Q seed n z = 0 := by
  classical
  intro n
  induction n with
  | zero =>
    intro z hz
    have hn : z ≠ seed := by intro he; subst z; exact hz hseed
    simp [CGB.Energetics.law, CGB.EventChannels.Kernel.stateLaw, hn]
  | succ n ih =>
    intro z hz
    change (∑ y, CGB.Energetics.law Q seed n y * ∑ e, Q.step n y e z) = 0
    apply Finset.sum_eq_zero
    intro y _
    by_cases hy : region y
    · have he (e : E) : Q.step n y e z = 0 := by
        by_cases hp : 0 < Q.step n y e z
        · exact False.elim (hz (hsafe n y e z hy hp))
        · exact le_antisymm (le_of_not_gt hp) (Q.nonneg n y e z)
      simp [he]
    · rw [ih y hy, zero_mul]

/-- Stochastic version: a body-only policy keeps every positive-probability
successor in the joint viable region iff each body has an action safe for all
compatible viable worlds and all supported events. The generated closed-loop
law has zero mass outside the region at every finite tick. This is not an
infinite-path measure or a proof that the safe-action premise is populated by
a real language world. -/
theorem stochastic_joint_viability_policy_generation
    {State World Act Event : Type*} [Fintype State] [DecidableEq State]
    [Fintype World] [DecidableEq World] [Fintype Event]
    (P : CGB.Interaction.Machine (State × World) Act Event)
    (viable : State × World → Prop) (seed : State × World) (hseed : viable seed) :
    (∀ s, ∃ a, ∀ w e z', viable (s, w) → 0 < P.step a (s, w) e z' → viable z') ↔
    (∃ policy : State → Act,
      (∀ s w e z', viable (s, w) → 0 < P.step (policy s) (s, w) e z' → viable z') ∧
      ∃ process : CGB.Interaction.Machine (State × World) ℕ Event,
        (∀ t z e z', process.step t z e z' = P.step (policy z.1) z e z') ∧
        (∀ n z, ¬ viable z → CGB.Energetics.law process seed n z = 0)) := by
  classical
  constructor
  · intro hs
    let policy : State → Act := fun s => Classical.choose (hs s)
    have hsafe : ∀ s w e z', viable (s, w) →
        0 < P.step (policy s) (s, w) e z' → viable z' := by
      intro s w e z' hv hp
      exact Classical.choose_spec (hs s) w e z' hv hp
    let Q : CGB.Interaction.Machine (State × World) ℕ Event := {
      step := fun _ z => P.step (policy z.1) z
      nonneg := fun _ z => P.nonneg (policy z.1) z
      sum_one := fun _ z => P.sum_one (policy z.1) z }
    refine ⟨policy, hsafe, Q, (by intros; rfl), ?_⟩
    exact closed_region_law_zero Q viable seed hseed
      (fun t z e z' hv hp => hsafe z.1 z.2 e z' (by simpa using hv) hp)
  · rintro ⟨policy, hsafe, _⟩ s
    exact ⟨policy s, hsafe s⟩

/-- Generate one stochastic policy whose same closed loop both stays within
the joint region and reaches the declared event within the finite horizon.
The decreasing rank is a proof certificate, not stored body state. The mark
must be connected separately to an actual useful event in the world. -/
theorem stochastic_joint_recovery_policy_generation
    {State World Act Event : Type*} [Fintype State] [DecidableEq State]
    [Fintype World] [DecidableEq World] [Fintype Event]
    (P : CGB.Interaction.Machine (State × World) Act Event)
    (viable : State × World → Prop) (seed : State × World) (hseed : viable seed)
    (rank : State × World → ℕ) (horizon : ℕ)
    (mark : ℕ → State × World → Event → State × World → Bool)
    (hbound : ∀ z, viable z → rank z < horizon)
    (hlocal : ∀ s, ∃ a, ∀ t w e z', viable (s, w) →
      0 < P.step a (s, w) e z' →
      viable z' ∧ (mark t (s, w) e z' = false → rank z' < rank (s, w))) :
    ∃ policy : State → Act,
      ∃ process : CGB.Interaction.Machine (State × World) ℕ Event,
        (∀ t z e z', process.step t z e z' = P.step (policy z.1) z e z') ∧
        (∀ n z, ¬ viable z → CGB.Energetics.law process seed n z = 0) ∧
        (∀ t z, viable z → CGB.EventChannels.Kernel.firstHit
          (CGB.Energetics.events process) mark t horizon z = 1) := by
  classical
  let policy : State → Act := fun s => Classical.choose (hlocal s)
  let Q : CGB.Interaction.Machine (State × World) ℕ Event := {
    step := fun _ z => P.step (policy z.1) z
    nonneg := fun _ z => P.nonneg (policy z.1) z
    sum_one := fun _ z => P.sum_one (policy z.1) z }
  have hs (t : ℕ) (z : State × World) (e : Event) (z' : State × World)
      (hv : viable z) (hp : 0 < Q.step t z e z') :
      viable z' ∧ (mark t z e z' = false → rank z' < rank z) := by
    exact Classical.choose_spec (hlocal z.1) t z.2 e z' (by simpa using hv) hp
  let cert : CGB.RecoveryCriteria.Ranked (CGB.Energetics.events Q) mark horizon := {
    region := viable
    rank := rank
    bounded := hbound
    closed := fun t z e z' hv hp => (hs t z e z' hv hp).1
    progresses := fun t z e z' hv hp hm => (hs t z e z' hv hp).2 hm }
  refine ⟨policy, Q, (by intros; rfl), ?_, ?_⟩
  · exact closed_region_law_zero Q viable seed hseed
      (fun t z e z' hv hp => (hs t z e z' hv hp).1)
  · intro t z hv
    exact CGB.RecoveryCriteria.ranked_firstHit cert t horizon z hv (hbound z hv)

/-- The existing ranked recovery certificate forces a marked event in every
`horizon`-tick window along every infinite sequence of supported transitions.
The sequence is a mathematical path, not an asserted infinite-path measure;
the mark can denote any actually declared event. -/
theorem ranked_supported_path_recurrent
    {S E : Type*} [Fintype S] [Fintype E]
    (K : ℕ → CGB.EventChannels.Kernel S E)
    (mark : ℕ → S → E → S → Bool) (horizon : ℕ)
    (C : CGB.RecoveryCriteria.Ranked K mark horizon)
    (path : ℕ → S) (event : ℕ → E) (hseed : C.region (path 0))
    (hsupport : ∀ t, 0 < (K t).step (path t) (event t) (path (t + 1))) :
    ∀ start, ∃ t, start ≤ t ∧ t < start + horizon ∧
      mark t (path t) (event t) (path (t + 1)) = true := by
  have hregion : ∀ t, C.region (path t) := by
    intro t
    induction t with
    | zero => exact hseed
    | succ t ih => exact C.closed t (path t) (event t) (path (t + 1)) ih (hsupport t)
  have descent (start k : ℕ)
      (hmiss : ∀ j, j < k → mark (start + j) (path (start + j))
        (event (start + j)) (path (start + j + 1)) = false) :
      C.rank (path (start + k)) + k ≤ C.rank (path start) := by
    induction k with
    | zero => simp
    | succ k ih =>
      have hm : mark (start + k) (path (start + k)) (event (start + k))
          (path (start + k + 1)) = false := hmiss k (by omega)
      have hd := C.progresses (start + k) (path (start + k)) (event (start + k))
        (path (start + k + 1)) (hregion (start + k)) (hsupport (start + k)) hm
      have hi := ih (fun j hj => hmiss j (by omega))
      have he : start + (k + 1) = start + k + 1 := by omega
      rw [he]
      omega
  intro start
  by_contra hn
  have hmiss : ∀ j, j < horizon → mark (start + j) (path (start + j))
      (event (start + j)) (path (start + j + 1)) = false := by
    intro j hj
    cases hm : mark (start + j) (path (start + j)) (event (start + j))
        (path (start + j + 1)) with
    | false => rfl
    | true =>
      exfalso
      exact hn ⟨start + j, by omega, by omega, hm⟩
  have hlower := descent start horizon hmiss
  have hupper := C.bounded (path start) (hregion start)
  omega

/-- A normalized finite event kernel has an infinite supported path from any
initial state; a ranked certificate makes its marked events recurrent. This
constructs a path, not a probability measure assigning it positive mass. -/
theorem ranked_exists_recurrently_marked_path
    {S E : Type*} [Fintype S] [Fintype E]
    (K : ℕ → CGB.EventChannels.Kernel S E)
    (mark : ℕ → S → E → S → Bool) (horizon : ℕ)
    (C : CGB.RecoveryCriteria.Ranked K mark horizon) (seed : S) (hseed : C.region seed) :
    ∃ path : ℕ → S, ∃ event : ℕ → E,
      path 0 = seed ∧
      (∀ t, 0 < (K t).step (path t) (event t) (path (t + 1))) ∧
      (∀ start, ∃ t, start ≤ t ∧ t < start + horizon ∧
        mark t (path t) (event t) (path (t + 1)) = true) := by
  classical
  have hnext (t : ℕ) (z : S) : ∃ p : E × S, 0 < (K t).step z p.1 p.2 := by
    by_contra hn
    have hz (e : E) (z' : S) : (K t).step z e z' = 0 := by
      have hp : ¬ 0 < (K t).step z e z' := by
        intro h; exact hn ⟨(e, z'), h⟩
      exact le_antisymm (le_of_not_gt hp) ((K t).nonneg z e z')
    have hsum := (K t).sum_one z
    simp [hz] at hsum
  let chooseNext := fun t z => Classical.choose (hnext t z)
  let path : ℕ → S := fun t => Nat.rec seed (fun n z => (chooseNext n z).2) t
  let event : ℕ → E := fun t => (chooseNext t (path t)).1
  have hsupport : ∀ t, 0 < (K t).step (path t) (event t) (path (t + 1)) := by
    intro t
    exact Classical.choose_spec (hnext t (path t))
  exact ⟨path, event, rfl, hsupport,
    ranked_supported_path_recurrent K mark horizon C path event hseed hsupport⟩

/-- One step of the existing open contract under an explicit action. -/
noncomputable def openActionStep
    {Hist State Byte Act Answer World : Type*}
    (G : RenewableLanguageGenerator Hist State Byte Act Answer)
    (W : RenewableOpenWorld G World) (z : State × World) (a : Act) : State × World :=
  let b := W.boundary z.2 z.1 a
  (renewableLanguageNext G z.1 b, W.evolve z.2 z.1 a b)

/-- On the same masked open-world runs, combine the prediction loss with a
world-dependent viability requirement. This does not assert that real
language worlds meet the two opposite-target or safe-action conditions. -/
theorem masked_world_prediction_and_viability_limit
    {Hist State Byte Act Answer World : Type*}
    (G : RenewableLanguageGenerator Hist State Byte Act Answer)
    (W : RenewableOpenWorld G World) (quiet : Byte) (s : State) (w₀ w₁ : World)
    (target : State × World → Bool) (predict : State → ℝ) (n : ℕ)
    (viable : State × World → Prop) (a₀ a₁ : Act) (hne : a₀ ≠ a₁)
    (hzero : target (maskedOpenRun G W quiet n (s, w₀)) = false)
    (hone : target (maskedOpenRun G W quiet n (s, w₁)) = true)
    (hp : 0 < predict (maskedOpenRun G W quiet n (s, w₀)).1)
    (hp1 : predict (maskedOpenRun G W quiet n (s, w₀)).1 < 1)
    (hattain₀ : viable (openActionStep G W (maskedOpenRun G W quiet n (s, w₀)) a₀))
    (hattain₁ : viable (openActionStep G W (maskedOpenRun G W quiet n (s, w₁)) a₁))
    (h₀ : ∀ a, viable (openActionStep G W (maskedOpenRun G W quiet n (s, w₀)) a) → a = a₀)
    (h₁ : ∀ a, viable (openActionStep G W (maskedOpenRun G W quiet n (s, w₁)) a) → a = a₁) :
    viable (openActionStep G W (maskedOpenRun G W quiet n (s, w₀)) a₀) ∧
    viable (openActionStep G W (maskedOpenRun G W quiet n (s, w₁)) a₁) ∧
    (¬ (viable (openActionStep G W (maskedOpenRun G W quiet n (s, w₀))
          (G.policy (maskedOpenRun G W quiet n (s, w₀)).1)) ∧
        viable (openActionStep G W (maskedOpenRun G W quiet n (s, w₁))
          (G.policy (maskedOpenRun G W quiet n (s, w₁)).1)))) ∧
    -(Real.log (1 - predict (maskedOpenRun G W quiet n (s, w₀)).1) +
      Real.log (predict (maskedOpenRun G W quiet n (s, w₁)).1)) / 2 ≥ Real.log 2 := by
  have hi := masked_open_world_information_limit G W quiet s w₀ w₁ target predict n
    hzero hone hp hp1
  refine ⟨hattain₀, hattain₁, ?_, hi.2.2.2.2⟩
  rintro ⟨hv₀, hv₁⟩
  have ha := h₀ _ hv₀
  have hb := h₁ _ hv₁
  rw [← hi.2.1] at hb
  exact hne (ha.symm.trans hb)

/-- If hidden world states can give different boundary inputs at the same
organism state and action, no state-only predictor is correct for both. -/
theorem open_world_input_not_body_determined
    {Hist State Byte Act Answer World : Type*}
    (G : RenewableLanguageGenerator Hist State Byte Act Answer)
    (W : RenewableOpenWorld G World) (s : State) (a : Act)
    (w₁ w₂ : World) (hdifferent : W.boundary w₁ s a ≠ W.boundary w₂ s a)
    (predict : State → Act → Byte) :
    ¬ (predict s a = W.boundary w₁ s a ∧ predict s a = W.boundary w₂ s a) := by
  rintro ⟨h₁, h₂⟩
  exact hdifferent (h₁.symm.trans h₂)

/-- Exact body-only prediction of the next body state for every world state
is possible precisely when hidden world variation has no effect on that
next state. This is an expressibility result, not a learned self-model. -/
theorem open_world_body_prediction_iff
    {Hist State Byte Act Answer World : Type*}
    (G : RenewableLanguageGenerator Hist State Byte Act Answer)
    (W : RenewableOpenWorld G World) :
    (∃ predict : State → Act → State, ∀ s a w,
      predict s a = renewableLanguageNext G s (W.boundary w s a)) ↔
    (∀ s a w₁ w₂,
      renewableLanguageNext G s (W.boundary w₁ s a) =
        renewableLanguageNext G s (W.boundary w₂ s a)) := by
  constructor
  · rintro ⟨predict, hp⟩ s a w₁ w₂
    exact (hp s a w₁).symm.trans (hp s a w₂)
  · intro h
    exact ⟨fun s a => renewableLanguageNext G s (W.boundary W.initial s a),
      fun s a w => h s a W.initial w⟩

/-- Restrict exact consequence prediction to joint states reachable from the
initial condition under action interventions. A body-state readout exists iff
histories merged into the same body state have identical next-body consequences
for every proposed action. This is representability, not identification from
observations or proof that the predictor is learned. -/
theorem reachable_consequence_readout_iff
    {Hist State Byte Act Answer World : Type*}
    (G : RenewableLanguageGenerator Hist State Byte Act Answer)
    (W : RenewableOpenWorld G World) :
    let c := renewableOpenContract G W
    let joint := fun p : (ℕ → Act) × ℕ => controlledTrajectory c p.1 p.2
    let body := fun p => (joint p).1
    let consequence := fun p a =>
      renewableLanguageNext G (body p) (W.boundary (joint p).2 (body p) a)
    (∃ predict : State → Act → State, ∀ p a, predict (body p) a = consequence p a) ↔
      (∀ p q, body p = body q → ∀ a, consequence p a = consequence q a) := by
  dsimp only
  simpa only [funext_iff] using
    (history_readout_exists_iff
      (fun p : (ℕ → Act) × ℕ => (controlledTrajectory (renewableOpenContract G W) p.1 p.2).1)
      (fun p a => renewableLanguageNext G
        (controlledTrajectory (renewableOpenContract G W) p.1 p.2).1
        (W.boundary (controlledTrajectory (renewableOpenContract G W) p.1 p.2).2
          (controlledTrajectory (renewableOpenContract G W) p.1 p.2).1 a))
      (fun _ => G.encode G.seed))

/-- Interface witness using the existing renewable cell rule. -/
def existingRenewableEncode (h : List CGB.RenewableLearning.Signal) : CGB.RenewableLearning.Cell :=
  h.foldl (fun q b => CGB.RenewableLearning.next true true b q) .search

def existingRenewableResponse (q : CGB.RenewableLearning.Cell)
    (a : CGB.RenewableLearning.Action) : CGB.RenewableLearning.Signal :=
  match q with
  | .search => if a = 2 then .answerA else .calm
  | .aSpent => .feed
  | .aReady => if a = 0 then .miss else .calm
  | _ => .calm

noncomputable def existingRenewableCarrier :
    RenewableLanguageGenerator (List CGB.RenewableLearning.Signal)
      CGB.RenewableLearning.Cell CGB.RenewableLearning.Signal
      CGB.RenewableLearning.Action CGB.RenewableLearning.Cell where
  encode := existingRenewableEncode
  required := existingRenewableEncode
  advanceHistory := fun h b => h ++ [b]
  fallback := .search
  seed := []
  viable := fun q => q ≠ .vacuum
  seed_viable := by decide
  readable := fun _ _ h => h
  transition_compatible := by
    intro h h' b hh
    simp only [existingRenewableEncode, List.foldl_append, List.foldl_cons, List.foldl_nil]
    change CGB.RenewableLearning.next true true b (existingRenewableEncode h) =
      CGB.RenewableLearning.next true true b (existingRenewableEncode h')
    rw [hh]
  policy := CGB.RenewableLearning.policy
  world_response := existingRenewableResponse
  world_closed := by
    intro h hv
    change existingRenewableEncode h ≠ .vacuum at hv
    simp only [existingRenewableEncode, List.foldl_append, List.foldl_cons, List.foldl_nil]
    change CGB.RenewableLearning.next true true
      (existingRenewableResponse (existingRenewableEncode h)
        (CGB.RenewableLearning.policy (existingRenewableEncode h)))
      (existingRenewableEncode h) ≠ .vacuum
    generalize existingRenewableEncode h = q at hv ⊢
    cases q <;> first | contradiction | decide

/-- A minimal open-world witness: the external stream chooses between the
two existing answers on a probe. Its natural-number cursor belongs to the
world, not the eight-state organism. -/
def renewableStreamBoundary (bits : ℕ → Bool) (w : ℕ)
    (q : CGB.RenewableLearning.Cell) (a : CGB.RenewableLearning.Action) :
    CGB.RenewableLearning.Signal :=
  match q with
  | .search => if a = 2 then (if bits w then .answerA else .answerB) else .calm
  | .aSpent | .bSpent => .feed
  | .aReady | .bReady => if a = CGB.RenewableLearning.policy q then .miss else .calm
  | _ => .calm

noncomputable def existingRenewableOpenWorld (bits : ℕ → Bool) :
    RenewableOpenWorld existingRenewableCarrier ℕ where
  initial := 0
  boundary := renewableStreamBoundary bits
  evolve := fun w _ _ _ => w + 1
  closed := by
    intro h w hv
    change existingRenewableEncode h ≠ .vacuum at hv
    simp only [existingRenewableCarrier, existingRenewableEncode, List.foldl_append,
      List.foldl_cons, List.foldl_nil]
    change CGB.RenewableLearning.next true true
      (renewableStreamBoundary bits w (existingRenewableEncode h)
        (CGB.RenewableLearning.policy (existingRenewableEncode h)))
      (existingRenewableEncode h) ≠ .vacuum
    generalize existingRenewableEncode h = q at hv ⊢
    cases hbit : bits w <;> cases q <;> first | contradiction | (simp [renewableStreamBoundary, hbit]; decide)

/-- In the existing open-stream witness, every action intervention leaves the
external cursor on exactly the same trajectory. Actions affect the boundary
response but not the evolution of this world's own state. -/
theorem renewable_open_world_is_exogenous (bits : ℕ → Bool)
    (actions : ℕ → CGB.RenewableLearning.Action) :
    ∀ n, (controlledTrajectory (renewableOpenContract existingRenewableCarrier
      (existingRenewableOpenWorld bits)) actions n).2 = n := by
  let c := renewableOpenContract existingRenewableCarrier (existingRenewableOpenWorld bits)
  intro n
  induction n with
  | zero => rfl
  | succ n ih =>
    change (c.step (controlledTrajectory c actions n)
      (c.response (controlledTrajectory c actions n) (actions n))).2 = n + 1
    change (controlledTrajectory c actions n).2 + 1 = n + 1
    rw [ih]

/-- The external cursor advances each tick while the same finite organism
stays in its stated non-vacuum region, for any external bit sequence. -/
theorem existing_renewable_open_stream_witness (bits : ℕ → Bool) :
    ∃ x : ℕ → CGB.RenewableLearning.Cell × ℕ,
      x 0 = (.search, 0) ∧
      (∀ n, x (n + 1) = (renewableOpenContract existingRenewableCarrier
        (existingRenewableOpenWorld bits)).step (x n)
        ((renewableOpenContract existingRenewableCarrier (existingRenewableOpenWorld bits)).response
          (x n) ((renewableOpenContract existingRenewableCarrier (existingRenewableOpenWorld bits)).action (x n)))) ∧
      (∀ n, (x n).1 ≠ .vacuum) ∧ (∀ n, (x n).2 = n) := by
  let C := renewableOpenContract existingRenewableCarrier (existingRenewableOpenWorld bits)
  refine ⟨trajectory C, rfl, ?_, ?_, ?_⟩
  · intro n
    rfl
  · intro n
    have hv := trajectory_viable C n
    rcases hv with ⟨h, he, hnonvacuum⟩
    exact hnonvacuum
  · intro n
    induction n with
    | zero => rfl
    | succ n ih =>
      change (trajectory C n).2 + 1 = n + 1
      rw [ih]

/-- On the initial probe, the existing finite carrier records which external
answer arrived. The value is supplied by the world, not the body state. -/
theorem renewable_open_first_input_recorded (bits : ℕ → Bool) :
    (trajectory (renewableOpenContract existingRenewableCarrier
      (existingRenewableOpenWorld bits)) 1).1 =
      if bits 0 then .aSpent else .bSpent := by
  have hp : CGB.RenewableLearning.policy .search = 2 := by decide
  have ha := (renewable_language_carrier_correct existingRenewableCarrier).2 [] .answerA
  have hb := (renewable_language_carrier_correct existingRenewableCarrier).2 [] .answerB
  simp [existingRenewableCarrier, existingRenewableEncode, CGB.RenewableLearning.next] at ha hb
  change renewableLanguageNext existingRenewableCarrier .search .answerA = .aSpent at ha
  change renewableLanguageNext existingRenewableCarrier .search .answerB = .bSpent at hb
  change renewableLanguageNext existingRenewableCarrier .search
    (renewableStreamBoundary bits 0 .search (CGB.RenewableLearning.policy .search)) = _
  rw [hp]
  cases hbit : bits 0 <;> simp [renewableStreamBoundary, hbit, ha, hb]

/-- Two worlds differing at the first observed bit start with the same body
but lead to distinguishable body states after one boundary input. -/
theorem renewable_open_distinct_inputs_reach_distinct_states
    (bits₁ bits₂ : ℕ → Bool) (hdifferent : bits₁ 0 ≠ bits₂ 0) :
    (trajectory (renewableOpenContract existingRenewableCarrier (existingRenewableOpenWorld bits₁)) 0).1 =
      (trajectory (renewableOpenContract existingRenewableCarrier (existingRenewableOpenWorld bits₂)) 0).1 ∧
    (trajectory (renewableOpenContract existingRenewableCarrier (existingRenewableOpenWorld bits₁)) 1).1 ≠
      (trajectory (renewableOpenContract existingRenewableCarrier (existingRenewableOpenWorld bits₂)) 1).1 := by
  constructor
  · rfl
  · rw [renewable_open_first_input_recorded, renewable_open_first_input_recorded]
    cases h₁ : bits₁ 0 <;> cases h₂ : bits₂ 0 <;> simp_all

/-- In this witness world, replacing the initial probe by the passive action
changes the next body state: the passive step retains `search`, while the
actual probe records the external answer. -/
theorem renewable_open_first_action_affects_body (bits : ℕ → Bool) :
    (controlledTrajectory (renewableOpenContract existingRenewableCarrier
      (existingRenewableOpenWorld bits)) (fun _ => 0) 1).1 = .search ∧
    (trajectory (renewableOpenContract existingRenewableCarrier
      (existingRenewableOpenWorld bits)) 1).1 ≠ .search := by
  constructor
  · have hc := (renewable_language_carrier_correct existingRenewableCarrier).2 [] .calm
    simp [existingRenewableCarrier, existingRenewableEncode, CGB.RenewableLearning.next] at hc
    change renewableLanguageNext existingRenewableCarrier .search .calm = .search at hc
    change renewableLanguageNext existingRenewableCarrier .search
      (renewableStreamBoundary bits 0 .search 0) = .search
    simpa [renewableStreamBoundary] using hc
  · rw [renewable_open_first_input_recorded]
    cases bits 0 <;> decide

/-- In the open-stream witness an always-passive intervention stays in the
non-vacuum `search` state forever. The witness therefore does not establish
that activity is necessary for its non-vacuum viability predicate. -/
theorem renewable_open_always_passive_stays_search (bits : ℕ → Bool) :
    ∀ n, controlledTrajectory (renewableOpenContract existingRenewableCarrier
      (existingRenewableOpenWorld bits)) (fun _ => 0) n = (.search, n) := by
  let C := renewableOpenContract existingRenewableCarrier (existingRenewableOpenWorld bits)
  have hc := (renewable_language_carrier_correct existingRenewableCarrier).2 [] .calm
  simp [existingRenewableCarrier, existingRenewableEncode, CGB.RenewableLearning.next] at hc
  change renewableLanguageNext existingRenewableCarrier .search .calm = .search at hc
  intro n
  induction n with
  | zero => rfl
  | succ n ih =>
    change C.step (controlledTrajectory C (fun _ => 0) n)
      (C.response (controlledTrajectory C (fun _ => 0) n) 0) = (.search, n + 1)
    rw [ih]
    change (renewableLanguageNext existingRenewableCarrier .search
      (renewableStreamBoundary bits n .search 0), n + 1) = (.search, n + 1)
    simp [renewableStreamBoundary, hc]

/-- The open witness returns to the probing state every three ticks for any
external bit stream, and each probe changes the finite body state. -/
theorem renewable_open_stream_recurrently_writes (bits : ℕ → Bool) :
    ∀ k,
      (trajectory (renewableOpenContract existingRenewableCarrier
        (existingRenewableOpenWorld bits)) (3 * k)).1 = .search ∧
      (trajectory (renewableOpenContract existingRenewableCarrier
        (existingRenewableOpenWorld bits)) (3 * k + 1)).1 ≠ .search := by
  let C := renewableOpenContract existingRenewableCarrier (existingRenewableOpenWorld bits)
  have hp : CGB.RenewableLearning.policy .search = 2 := by decide
  have ha0 := (renewable_language_carrier_correct existingRenewableCarrier).2 [] .answerA
  have hb0 := (renewable_language_carrier_correct existingRenewableCarrier).2 [] .answerB
  have ha1 := (renewable_language_carrier_correct existingRenewableCarrier).2 [.answerA] .feed
  have hb1 := (renewable_language_carrier_correct existingRenewableCarrier).2 [.answerB] .feed
  have ha2 := (renewable_language_carrier_correct existingRenewableCarrier).2 [.answerA, .feed] .miss
  have hb2 := (renewable_language_carrier_correct existingRenewableCarrier).2 [.answerB, .feed] .miss
  simp [existingRenewableCarrier, existingRenewableEncode, CGB.RenewableLearning.next] at ha0 hb0 ha1 hb1 ha2 hb2
  change renewableLanguageNext existingRenewableCarrier .search .answerA = .aSpent at ha0
  change renewableLanguageNext existingRenewableCarrier .search .answerB = .bSpent at hb0
  change renewableLanguageNext existingRenewableCarrier .aSpent .feed = .aReady at ha1
  change renewableLanguageNext existingRenewableCarrier .bSpent .feed = .bReady at hb1
  change renewableLanguageNext existingRenewableCarrier .aReady .miss = .search at ha2
  change renewableLanguageNext existingRenewableCarrier .bReady .miss = .search at hb2
  have hstep (n : ℕ) : (trajectory C (n + 1)).1 =
      renewableLanguageNext existingRenewableCarrier (trajectory C n).1
        (renewableStreamBoundary bits (trajectory C n).2 (trajectory C n).1
          (CGB.RenewableLearning.policy (trajectory C n).1)) := by rfl
  have hcycle (n : ℕ) (hs : (trajectory C n).1 = .search) :
      (trajectory C (n + 1)).1 ≠ .search ∧ (trajectory C (n + 3)).1 = .search := by
    cases hbit : bits (trajectory C n).2 with
    | false =>
      have h1 : (trajectory C (n + 1)).1 = .bSpent := by
        rw [hstep, hs, hp]
        simp [renewableStreamBoundary, hbit, hb0]
      have h2 : (trajectory C (n + 2)).1 = .bReady := by
        rw [show n + 2 = (n + 1) + 1 by omega, hstep, h1]
        simp [renewableStreamBoundary, hb1]
      constructor
      · rw [h1]; decide
      · rw [show n + 3 = (n + 2) + 1 by omega, hstep, h2]
        simp [renewableStreamBoundary, hb2]
    | true =>
      have h1 : (trajectory C (n + 1)).1 = .aSpent := by
        rw [hstep, hs, hp]
        simp [renewableStreamBoundary, hbit, ha0]
      have h2 : (trajectory C (n + 2)).1 = .aReady := by
        rw [show n + 2 = (n + 1) + 1 by omega, hstep, h1]
        simp [renewableStreamBoundary, ha1]
      constructor
      · rw [h1]; decide
      · rw [show n + 3 = (n + 2) + 1 by omega, hstep, h2]
        simp [renewableStreamBoundary, ha2]
  intro k
  induction k with
  | zero =>
    have hs : (trajectory C 0).1 = .search := rfl
    simpa using And.intro hs (hcycle 0 hs).1
  | succ k ih =>
    have hs : (trajectory C (3 * (k + 1))).1 = .search := by
      simpa only [show 3 * (k + 1) = 3 * k + 3 by omega] using (hcycle (3 * k) ih.1).2
    exact ⟨hs, (hcycle (3 * (k + 1)) hs).1⟩

/-- Each probe stores the current external bit for two successive body
states, then reopens the body on the third tick. This is bounded retention,
not a claim of lasting or useful language memory. -/
theorem renewable_open_probe_records_then_reopens (bits : ℕ → Bool) (k : ℕ) :
    (trajectory (renewableOpenContract existingRenewableCarrier
      (existingRenewableOpenWorld bits)) (3 * k + 1)).1 =
        (if bits (3 * k) then .aSpent else .bSpent) ∧
    (trajectory (renewableOpenContract existingRenewableCarrier
      (existingRenewableOpenWorld bits)) (3 * k + 2)).1 =
        (if bits (3 * k) then .aReady else .bReady) ∧
    (trajectory (renewableOpenContract existingRenewableCarrier
      (existingRenewableOpenWorld bits)) (3 * k + 3)).1 = .search := by
  let C := renewableOpenContract existingRenewableCarrier (existingRenewableOpenWorld bits)
  have hs : (trajectory C (3 * k)).1 = .search :=
    (renewable_open_stream_recurrently_writes bits k).1
  have hcursor (n : ℕ) : (trajectory C n).2 = n := by
    induction n with
    | zero => rfl
    | succ n ih =>
      change (trajectory C n).2 + 1 = n + 1
      rw [ih]
  have hp : CGB.RenewableLearning.policy .search = 2 := by decide
  have ha0 := (renewable_language_carrier_correct existingRenewableCarrier).2 [] .answerA
  have hb0 := (renewable_language_carrier_correct existingRenewableCarrier).2 [] .answerB
  have ha1 := (renewable_language_carrier_correct existingRenewableCarrier).2 [.answerA] .feed
  have hb1 := (renewable_language_carrier_correct existingRenewableCarrier).2 [.answerB] .feed
  simp [existingRenewableCarrier, existingRenewableEncode, CGB.RenewableLearning.next] at ha0 hb0 ha1 hb1
  change renewableLanguageNext existingRenewableCarrier .search .answerA = .aSpent at ha0
  change renewableLanguageNext existingRenewableCarrier .search .answerB = .bSpent at hb0
  change renewableLanguageNext existingRenewableCarrier .aSpent .feed = .aReady at ha1
  change renewableLanguageNext existingRenewableCarrier .bSpent .feed = .bReady at hb1
  have hstep (n : ℕ) : (trajectory C (n + 1)).1 =
      renewableLanguageNext existingRenewableCarrier (trajectory C n).1
        (renewableStreamBoundary bits (trajectory C n).2 (trajectory C n).1
          (CGB.RenewableLearning.policy (trajectory C n).1)) := by rfl
  have h1 : (trajectory C (3 * k + 1)).1 =
      (if bits (3 * k) then .aSpent else .bSpent) := by
    rw [hstep, hs, hp, hcursor]
    cases hb : bits (3 * k) <;> simp [renewableStreamBoundary, hb, ha0, hb0]
  have h2 : (trajectory C (3 * k + 2)).1 =
      (if bits (3 * k) then .aReady else .bReady) := by
    rw [show 3 * k + 2 = (3 * k + 1) + 1 by omega, hstep, h1]
    cases hb : bits (3 * k) <;> simp [renewableStreamBoundary, ha1, hb1]
  have h3 := (renewable_open_stream_recurrently_writes bits (k + 1)).1
  have h3' : (trajectory C (3 * k + 3)).1 = .search := by
    simpa only [show 3 * (k + 1) = 3 * k + 3 by omega] using h3
  exact ⟨h1, h2, h3'⟩

/-- In the witness world the existing certified policy reads the recorded
distinction for two ticks; the always-passive intervention remains at the
undifferentiated fallback action. This is a task-specific functional contrast. -/
theorem renewable_open_probe_enables_conditional_action (bits : ℕ → Bool) (k : ℕ) :
    CGB.RenewableLearning.policy
      (trajectory (renewableOpenContract existingRenewableCarrier
        (existingRenewableOpenWorld bits)) (3 * k + 1)).1 =
      (if bits (3 * k) then 0 else 1) ∧
    CGB.RenewableLearning.policy
      (trajectory (renewableOpenContract existingRenewableCarrier
        (existingRenewableOpenWorld bits)) (3 * k + 2)).1 =
      (if bits (3 * k) then 0 else 1) ∧
    CGB.RenewableLearning.policy
      (controlledTrajectory (renewableOpenContract existingRenewableCarrier
        (existingRenewableOpenWorld bits)) (fun _ => 0) (3 * k + 2)).1 = 2 := by
  have h := renewable_open_probe_records_then_reopens bits k
  have hpassive := renewable_open_always_passive_stays_search bits (3 * k + 2)
  constructor
  · rw [h.1]
    cases bits (3 * k) <;> decide
  constructor
  · rw [h.2.1]
    cases bits (3 * k) <;> decide
  · rw [hpassive]
    change CGB.RenewableLearning.policy .search = 2
    decide

/-- Recurrent writing is not enough for delayed credit: if two streams differ
at one probe but agree at the next, the body first distinguishes them and
then merges them. No body-only readout can recover the earlier bit after that
merge. A world observation or accessible external record can change the
information available to a later rule and is not ruled out here. -/
theorem renewable_open_rewriting_loses_old_distinction
    (bits₁ bits₂ : ℕ → Bool) (k : ℕ)
    (hdifferent : bits₁ (3 * k) ≠ bits₂ (3 * k))
    (hsame : bits₁ (3 * (k + 1)) = bits₂ (3 * (k + 1))) :
    let body₁ := fun n => (trajectory (renewableOpenContract existingRenewableCarrier
      (existingRenewableOpenWorld bits₁)) n).1
    let body₂ := fun n => (trajectory (renewableOpenContract existingRenewableCarrier
      (existingRenewableOpenWorld bits₂)) n).1
    body₁ (3 * k + 1) ≠ body₂ (3 * k + 1) ∧
    body₁ (3 * (k + 1) + 1) = body₂ (3 * (k + 1) + 1) ∧
    ¬ ∃ read : CGB.RenewableLearning.Cell → Bool,
      read (body₁ (3 * (k + 1) + 1)) = bits₁ (3 * k) ∧
      read (body₂ (3 * (k + 1) + 1)) = bits₂ (3 * k) := by
  dsimp only
  have h₁ := (renewable_open_probe_records_then_reopens bits₁ k).1
  have h₂ := (renewable_open_probe_records_then_reopens bits₂ k).1
  have hnext₁ := (renewable_open_probe_records_then_reopens bits₁ (k + 1)).1
  have hnext₂ := (renewable_open_probe_records_then_reopens bits₂ (k + 1)).1
  have hmerge : (trajectory (renewableOpenContract existingRenewableCarrier
      (existingRenewableOpenWorld bits₁)) (3 * (k + 1) + 1)).1 =
      (trajectory (renewableOpenContract existingRenewableCarrier
        (existingRenewableOpenWorld bits₂)) (3 * (k + 1) + 1)).1 := by
    rw [hnext₁, hnext₂, hsame]
  refine ⟨?_, hmerge, ?_⟩
  · rw [h₁, h₂]
    cases hbit₁ : bits₁ (3 * k) <;> cases hbit₂ : bits₂ (3 * k) <;> simp_all
  · rintro ⟨read, hread₁, hread₂⟩
    apply hdifferent
    exact hread₁.symm.trans ((congrArg read hmerge).trans hread₂)

/-- The derived transition admits the old witness's writing, renewal and
reopening steps on one finite carrier. -/
theorem existing_renewable_carrier_reuses_old_rule :
    renewableLanguageNext existingRenewableCarrier .search .answerA = .aSpent ∧
    renewableLanguageNext existingRenewableCarrier .aSpent .feed = .aReady ∧
    renewableLanguageNext existingRenewableCarrier .aReady .miss = .search := by
  have h0 := (renewable_language_carrier_correct existingRenewableCarrier).2 [] .answerA
  have h1 := (renewable_language_carrier_correct existingRenewableCarrier).2 [.answerA] .feed
  have h2 := (renewable_language_carrier_correct existingRenewableCarrier).2 [.answerA, .feed] .miss
  simp [existingRenewableCarrier, existingRenewableEncode, CGB.RenewableLearning.next] at h0 h1 h2
  exact ⟨h0, h1, h2⟩

/-- In the specified deterministic witness world the same finite carrier
repeats the writing, refill and reopening cycle at every natural time scale.
This is a mathematical interface witness, not a language capability claim. -/
theorem existing_renewable_autonomous_cycle :
    ∀ k, trajectory (renewableLanguageContract existingRenewableCarrier) (3 * k) = .search ∧
      trajectory (renewableLanguageContract existingRenewableCarrier) (3 * k + 1) = .aSpent ∧
      trajectory (renewableLanguageContract existingRenewableCarrier) (3 * k + 2) = .aReady := by
  let C := renewableLanguageContract existingRenewableCarrier
  have hp0 : CGB.RenewableLearning.policy .search = 2 := by decide
  have hp2 : CGB.RenewableLearning.policy .aReady = 0 := by decide
  have hw := existing_renewable_carrier_reuses_old_rule
  have hstep (t : ℕ) : trajectory C (t + 1) =
      renewableLanguageNext existingRenewableCarrier (trajectory C t)
        (existingRenewableResponse (trajectory C t) (CGB.RenewableLearning.policy (trajectory C t))) := by
    rfl
  have hcycle (t : ℕ) (hs : trajectory C t = .search) :
      trajectory C (t + 1) = .aSpent ∧ trajectory C (t + 2) = .aReady ∧
      trajectory C (t + 3) = .search := by
    have h1 : trajectory C (t + 1) = .aSpent := by
      rw [hstep, hs]
      simpa [existingRenewableResponse, hp0] using hw.1
    have h2 : trajectory C (t + 2) = .aReady := by
      rw [show t + 2 = (t + 1) + 1 by omega, hstep, h1]
      simpa [existingRenewableResponse] using hw.2.1
    refine ⟨h1, h2, ?_⟩
    rw [show t + 3 = (t + 2) + 1 by omega, hstep, h2]
    simpa [existingRenewableResponse, hp2] using hw.2.2
  intro k
  induction k with
  | zero =>
    have hzero : trajectory C 0 = .search := rfl
    have h := hcycle 0 hzero
    simpa using And.intro hzero (And.intro h.1 h.2.1)
  | succ k ih =>
    have hold := hcycle (3 * k) ih.1
    have hs : trajectory C (3 * (k + 1)) = .search := by
      simpa only [show 3 * (k + 1) = 3 * k + 3 by omega] using hold.2.2
    have hnew := hcycle (3 * (k + 1)) hs
    exact ⟨hs, hnew.1, hnew.2.1⟩

/-- On the existing closed three-state witness, a fixed state-dependent
loss that never increases cannot strictly improve anywhere in the cycle.
In particular, recurrent strict improvement against one stationary law
requires allowing loss increases, changing the state/world, or changing the
comparison. This does not constrain a nonperiodic open-world trajectory. -/
theorem stationary_nonincreasing_loss_flat_on_existing_cycle
    (loss : CGB.RenewableLearning.Cell → ℝ)
    (hmono : ∀ n, loss (trajectory (renewableLanguageContract existingRenewableCarrier) (n + 1)) ≤
      loss (trajectory (renewableLanguageContract existingRenewableCarrier) n)) :
    loss .aSpent = loss .search ∧ loss .aReady = loss .aSpent ∧
      loss .search = loss .aReady := by
  have hcycle := existing_renewable_autonomous_cycle 0
  have hnext := (existing_renewable_autonomous_cycle 1).1
  norm_num at hcycle hnext
  have h0 := hmono 0
  have h1 := hmono 1
  have h2 := hmono 2
  norm_num at h0 h1 h2
  rw [hcycle.1, hcycle.2.1] at h0
  rw [hcycle.2.1, hcycle.2.2] at h1
  rw [hcycle.2.2, hnext] at h2
  exact ⟨le_antisymm h0 (le_trans h2 h1),
    le_antisymm h1 (le_trans h0 h2), le_antisymm h2 (le_trans h1 h0)⟩

/-- The finite witness remains outside vacuum, does not become permanently
passive, and has an action-sensitive boundary in the declared witness world. -/
theorem existing_renewable_cycle_active_and_viable :
    (∀ n, trajectory (renewableLanguageContract existingRenewableCarrier) n ≠ .vacuum) ∧
    (¬ eventuallyPassive (fun n => CGB.RenewableLearning.policy
      (trajectory (renewableLanguageContract existingRenewableCarrier) n)) 0) ∧
    existingRenewableResponse .search 2 ≠ existingRenewableResponse .search 0 := by
  refine ⟨?_, ?_, ?_⟩
  · intro n
    have hv := trajectory_viable (renewableLanguageContract existingRenewableCarrier) n
    rcases hv with ⟨h, he, hnonvacuum⟩
    exact hnonvacuum
  · rintro ⟨t, ht⟩
    have h := ht (3 * t) (by omega)
    change CGB.RenewableLearning.policy
      (trajectory (renewableLanguageContract existingRenewableCarrier) (3 * t)) = 0 at h
    rw [(existing_renewable_autonomous_cycle t).1] at h
    have hp : CGB.RenewableLearning.policy .search = 2 := by decide
    rw [hp] at h
    exact (by decide : (2 : CGB.RenewableLearning.Action) ≠ 0) h
  · decide

/-- When quiet freezes the body, the historical tagged memory law has the
same body marginal after the first observation at every later time. The world
may still evolve; this uses only normalization of its transition kernel. -/
theorem quiet_identity_memory_marginal
    {U Λ A I : Type*} [Fintype U] [DecidableEq U] [Fintype Λ] [DecidableEq Λ] [Fintype A]
    (M : CGB.Medium U Λ A I) (W : CGB.World U Λ I)
    (hid : ∀ x y, M.kernel W.clock W.quiet x y = if x = y then 1 else 0)
    (n₀ : ℕ) (x x₀ : U → Λ) (w₀ : W.W) :
    ∀ m, (∑ w, M.memLaw W n₀ (m + 1) x w x₀ w₀) =
      ∑ w, M.memLaw W n₀ 1 x w x₀ w₀ := by
  classical
  have hstep (k : ℕ) (hk : k ≠ 0) :
      (∑ w, M.memLaw W n₀ (k + 1) x w x₀ w₀) =
        ∑ w, M.memLaw W n₀ k x w x₀ w₀ := by
    simp only [CGB.Medium.memLaw, hk, ite_false, hid, mul_ite, mul_one, mul_zero]
    rw [Finset.sum_comm]
    rw [Finset.sum_eq_single x]
    · simp only [ite_true]
      rw [Finset.sum_comm]
      simp only [← Finset.mul_sum, W.moves_sum, mul_one]
    · intro y _ hy
      simp [hy]
    · simp
  intro m
  induction m with
  | zero => rfl
  | succ m ih =>
    rw [show m + 1 + 1 = (m + 1) + 1 by omega, hstep (m + 1) (by omega)]
    exact ih

/-- For a body frozen under quiet, the old indefinitely retained conditional
information requirement reduces exactly to a single positive-information
write. This is a property of that historical criterion, not a claim that
freezing is sufficient for a useful or adaptive memory. -/
theorem quiet_identity_old_memory_iff_one_write
    {U Λ A I : Type*} [Fintype U] [DecidableEq U] [Fintype Λ] [DecidableEq Λ] [Fintype A]
    (M : CGB.Medium U Λ A I) (W : CGB.World U Λ I)
    (hid : ∀ x y, M.kernel W.clock W.quiet x y = if x = y then 1 else 0) :
    M.Remembers W ↔ ∃ n₀, 0 < CGB.Medium.condMutualInfo
      (fun x₀ w₀ x => ∑ w, M.memLaw W n₀ 1 x w x₀ w₀) := by
  constructor
  · rintro ⟨n₀, ε, hε, hm⟩
    exact ⟨n₀, lt_of_lt_of_le hε (by simpa using hm 0)⟩
  · rintro ⟨n₀, h⟩
    refine ⟨n₀, _, h, ?_⟩
    intro m
    have heq : (fun x₀ w₀ x => ∑ w, M.memLaw W n₀ (m + 1) x w x₀ w₀) =
        (fun x₀ w₀ x => ∑ w, M.memLaw W n₀ 1 x w x₀ w₀) := by
      funext x₀ w₀ x
      exact quiet_identity_memory_marginal M W hid n₀ x x₀ w₀ m
    rw [heq]

/-- If calm freezes every body state, the historical `Persists` contract is
impossible: under permanent calm its body-only alive score stays at the
initial value, which normal persistence requires to be above threshold. -/
theorem historical_persistence_requires_calm_change
    {U Λ A I : Type*} [Fintype U] [DecidableEq U] [Fintype Λ] [DecidableEq Λ] [Fintype A]
    (M : CGB.Medium U Λ A I) (W : CGB.World U Λ I)
    (hid : ∀ x y, M.kernel W.clock M.calm x y = if x = y then 1 else 0)
    (δ : ℝ) (T : ℕ) : ¬ M.Persists W δ T := by
  classical
  let calm : ℕ → W.W → U → I := fun _ _ => M.calm
  have hk : ∀ w x, ∑ y, M.aliveInd W y * M.kernel W.clock (calm 0 w) x y =
      M.aliveInd W x := by
    intro w x
    simp [calm, hid]
  have hconstant : ∀ n, M.aliveProb W calm n = M.aliveProb W calm 0 := by
    intro n
    induction n with
    | zero => rfl
    | succ n ih =>
      have hs := CGB.Medium.stepSum (M.aliveInd W) (M.law W calm n)
        (fun x w w' => W.moves w x w')
        (fun w x y => M.kernel W.clock (calm n w) x y)
        (fun x w => W.moves_sum w x) (by simpa [calm] using hk)
      change M.aliveProb W calm (n + 1) = _
      rw [show M.aliveProb W calm (n + 1) = M.aliveProb W calm n by
        simpa only [CGB.Medium.aliveProb, CGB.Medium.law] using hs]
      exact ih
  intro hp
  obtain ⟨_, _, hn, _, _, n₀, hdead⟩ := hp
  have hstart := hn 0
  have hsame : M.aliveProb W (fun _ => W.sense) 0 = M.aliveProb W calm 0 := by
    simp [CGB.Medium.aliveProb, CGB.Medium.law]
  rw [hsame] at hstart
  have h := hdead n₀ le_rfl
  change M.aliveProb W calm n₀ < _ at h
  rw [hconstant] at h
  linarith

/-- The grace witness freezes under calm, so it cannot satisfy the old
long-disconnection requirement, regardless of its other operational proofs. -/
theorem grace_not_historical_persists (δ : ℝ) (T : ℕ) :
    ¬ GraceRecovery.body.Persists GraceRecovery.world δ T := by
  apply historical_persistence_requires_calm_change
  intro x y
  simp [CGB.Medium.kernel, CGB.Medium.jump, GraceRecovery.body,
    RenewableLearning.plant]

/-- The existing sensory-maintenance witness tolerates one missing-sense
tick and loses its body after three, but the historical interrupt selects
its different frozen equilibrium stimulus. Thus those results cannot be
silently substituted into the old `Persists` contract. The separate material
reserve is a property of this witness, not a requirement on information life. -/
theorem sensory_loss_and_historical_interrupt_differ (δ : ℝ) (T : ℕ) :
    (∀ n, Exchange.survival SensoryMaintenance.body SensoryMaintenance.world
      SensoryMaintenance.ec SensoryMaintenance.material false
      (fun _ => SensoryMaintenance.world.sense) n = 1) ∧
    (∀ n, 7 ≤ n → Exchange.survival SensoryMaintenance.body SensoryMaintenance.world
      SensoryMaintenance.ec SensoryMaintenance.material true
      (fun _ => SensoryMaintenance.world.sense) n ≤ 3 / 4) ∧
    SensoryMaintenanceRecovery.ToleratesSensoryLoss 0 1 ∧
    (∀ n₀ duration n, 3 ≤ duration → n₀ + 3 ≤ n →
      Exchange.survival SensoryMaintenance.body SensoryMaintenance.world
        SensoryMaintenance.ec SensoryMaintenance.material false
        (SensoryMaintenance.sensoryProtocol n₀ duration) n = 0) ∧
    (∀ n₀ w, SensoryMaintenance.body.interrupt SensoryMaintenance.world n₀ 1 n₀ w () ≠
      SensoryMaintenance.sensoryProtocol n₀ 1 n₀ w ()) ∧
    ¬ SensoryMaintenance.body.Persists SensoryMaintenance.world δ T := by
  refine ⟨SensoryMaintenanceRecovery.natural_survival, SensoryMaintenanceClosure.cut_survival,
    SensoryMaintenanceRecovery.one_tick_tolerance,
    SensoryMaintenanceLimits.blackout_survival_zero, ?_, ?_⟩
  · intro n₀ w
    simp [CGB.Medium.interrupt, SensoryMaintenance.sensoryProtocol, SensoryMaintenance.body]
  · apply historical_persistence_requires_calm_change
    intro x y
    simp [CGB.Medium.kernel, CGB.Medium.jump, SensoryMaintenance.body]

/-- In the same grace witness, the old memory criterion only tests whether
one observation can add conditional information before quiet freezing. -/
theorem grace_historical_memory_iff_one_write :
    GraceRecovery.body.Remembers GraceRecovery.world ↔ ∃ n₀,
      0 < CGB.Medium.condMutualInfo (fun x₀ w₀ x => ∑ w,
        GraceRecovery.body.memLaw GraceRecovery.world n₀ 1 x w x₀ w₀) := by
  apply quiet_identity_old_memory_iff_one_write
  intro x y
  simp [CGB.Medium.kernel, CGB.Medium.jump, GraceRecovery.body,
    RenewableLearning.plant]

/-- Two already-certified finite paths also occur in the grace world. At tick
five they have the same body but different sensed world states; at tick six
the body and its policy differ. Positive support is not a numerical mutual
information calculation or a claim of language performance. -/
theorem grace_reachable_memory_changes_action :
    let p := renewableObservableFromBirth (fun _ => false)
    let q := renewableObservableFromBirth (fun _ => true)
    0 < RenewableLearningControls.weight (GraceRecovery.process false GraceRecovery.natural) p 6 ∧
    0 < RenewableLearningControls.weight (GraceRecovery.process false GraceRecovery.natural) q 6 ∧
    (p 5).1 = (q 5).1 ∧
    RenewableLearning.sense (p 5).2 () ≠ RenewableLearning.sense (q 5).2 () ∧
    (p 6).1 ≠ (q 6).1 ∧
    RenewableLearning.policy ((p 6).1 ()) ≠ RenewableLearning.policy ((q 6).1 ()) := by
  dsimp only
  have hweight (b : Bool) :
      RenewableLearningControls.weight (GraceRecovery.process false GraceRecovery.natural)
          (renewableObservableFromBirth (fun _ => b)) 6 =
        RenewableLearningControls.weight (RenewableLearning.process true true false)
          (renewableObservableFromBirth (fun _ => b)) 6 := by
    unfold RenewableLearningControls.weight
    apply Finset.prod_congr rfl
    intro t ht
    apply GraceRecoveryAccounting.normal_step_unchanged
    have hlt : t < 6 := Finset.mem_range.mp ht
    have ht' : t = 0 ∨ t = 1 ∨ t = 2 ∨ t = 3 ∨ t = 4 ∨ t = 5 := by omega
    rcases ht' with rfl | rfl | rfl | rfl | rfl | rfl <;> cases b <;>
      simp [renewableObservableFromBirth, renewableObservablePath, renewableObservableBlock,
        RenewableLearning.initial, GraceRecovery.Good, RenewableLearningLife.Good]
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hweight]; exact (renewable_observable_from_birth (fun _ => false)).2.1 6
  · rw [hweight]; exact (renewable_observable_from_birth (fun _ => true)).2.1 6
  · rfl
  · decide
  · decide
  · decide

/-- Two histories from the same initial process reach the same world state
and bodies with equal total content, buffer and credit, but opposite stored
candidates and actions. In that world the existing transition sends one
body to `missA` with probability zero and the other with probability one.
These are supported histories, not an intervention isolating every physical
degree of freedom or a long-term survival comparison. -/
theorem grace_reachable_equal_budget_memory_contrast :
    let p := renewableObservableFromBirth (fun _ => false)
    let q := renewableObservableFromBirth (fun _ => true)
    0 < RenewableLearningControls.weight (GraceRecovery.process false GraceRecovery.natural) p 9 ∧
    0 < RenewableLearningControls.weight (GraceRecovery.process false GraceRecovery.natural) q 9 ∧
    p 9 = ((fun _ => .aReady), .feedA) ∧ q 9 = ((fun _ => .bReady), .feedA) ∧
    RenewableLearning.cost .aReady = RenewableLearning.cost .bReady ∧
    RenewableLearning.buffer .aReady = RenewableLearning.buffer .bReady ∧
    RenewableLearning.credit .aReady = RenewableLearning.credit .bReady ∧
    RenewableLearning.incoming true .feed .aReady = RenewableLearning.incoming true .feed .bReady ∧
    RenewableLearning.outgoing true true .feed .aReady = RenewableLearning.outgoing true true .feed .bReady ∧
    RenewableLearning.memory .aReady ≠ RenewableLearning.memory .bReady ∧
    RenewableLearning.policy .aReady = 0 ∧ RenewableLearning.policy .bReady = 1 ∧
    GraceRecovery.moves .feedA (fun _ => .aReady) .missA = 0 ∧
    GraceRecovery.moves .feedA (fun _ => .bReady) .missA = 1 := by
  dsimp only
  have hweight (b : Bool) :
      RenewableLearningControls.weight (GraceRecovery.process false GraceRecovery.natural)
          (renewableObservableFromBirth (fun _ => b)) 9 =
        RenewableLearningControls.weight (RenewableLearning.process true true false)
          (renewableObservableFromBirth (fun _ => b)) 9 := by
    unfold RenewableLearningControls.weight
    apply Finset.prod_congr rfl
    intro t ht
    apply GraceRecoveryAccounting.normal_step_unchanged
    have hlt : t < 9 := Finset.mem_range.mp ht
    have ht' : t = 0 ∨ t = 1 ∨ t = 2 ∨ t = 3 ∨ t = 4 ∨ t = 5 ∨
        t = 6 ∨ t = 7 ∨ t = 8 := by omega
    rcases ht' with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> cases b <;>
      simp [renewableObservableFromBirth, renewableObservablePath, renewableObservableBlock,
        RenewableLearning.initial, GraceRecovery.Good, RenewableLearningLife.Good]
  have ha : RenewableLearning.policy .aReady = 0 := by decide
  have hb : RenewableLearning.policy .bReady = 1 := by decide
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ha, hb, ?_, ?_⟩
  · rw [hweight]; exact (renewable_observable_from_birth (fun _ => false)).2.1 9
  · rw [hweight]; exact (renewable_observable_from_birth (fun _ => true)).2.1 9
  · rfl
  · rfl
  all_goals norm_num [RenewableLearning.cost, RenewableLearning.buffer, RenewableLearning.credit,
    RenewableLearning.incoming, RenewableLearning.outgoing, RenewableLearning.memory, GraceRecovery.moves,
    RenewableLearning.moves, ha, hb]

/-- Expose the existing grace-world action dependence, using the same live
representatives as the earlier original-world action interface. -/
noncomputable def graceActionWorld (w : RenewableLearning.Env)
    (a : RenewableLearning.Action) (v : RenewableLearning.Env) : ℝ :=
  GraceRecovery.moves w (fun _ => renewableActionRepresentative a) v

theorem grace_action_world_matches (w v : RenewableLearning.Env)
    (q : RenewableLearning.Cell) (hq : q ≠ .vacuum) :
    GraceRecovery.moves w (fun _ => q) v =
      graceActionWorld w (RenewableLearning.policy q) v := by
  have hr := renewable_action_world_contract.1 (RenewableLearning.policy q)
  have hold := renewable_action_world_contract.2.2.1 w v q hq
  change RenewableLearning.moves w (fun _ => q) v = RenewableLearning.moves w
    (fun _ => renewableActionRepresentative (RenewableLearning.policy q)) v at hold
  unfold graceActionWorld
  simp only [GraceRecovery.moves, hr.2, hold]
  simp [hq, hr.1]

/-- Fixed-action intervention on the existing grace-world law. The body
still receives and acts on its actual stimulus; action only changes world
transitions. As usual for interventions, the exposed world kernel is assumed
to remain invariant under the action override. -/
noncomputable def graceActionMachine :
    CGB.Interaction.Machine RenewableLearning.S RenewableLearning.Action (Option Unit) where
  step a z e z' :=
    (if z.1 () = .vacuum then GraceRecovery.moves z.2 z.1 z'.2
      else graceActionWorld z.2 a z'.2) *
    GraceRecovery.body.eventStep GraceRecovery.world.clock
      (GraceRecovery.world.sense z.2) z.1 e z'.1
  nonneg a z e z' := by
    apply mul_nonneg
    · split_ifs
      · exact GraceRecovery.world.moves_nonneg _ _ _
      · simpa [graceActionWorld, GraceRecovery.world, GraceRecovery.worldWith] using
          (GraceRecovery.world.moves_nonneg z.2 (fun _ => renewableActionRepresentative a) z'.2)
    · exact GraceRecovery.body.eventStep_nonneg GraceRecovery.ec _ _ _ _
  sum_one a z := by
    have hw : (∑ v, if z.1 () = .vacuum then GraceRecovery.moves z.2 z.1 v
        else graceActionWorld z.2 a v) = 1 := by
      split_ifs
      · exact GraceRecovery.world.moves_sum _ _
      · simpa [graceActionWorld, GraceRecovery.world, GraceRecovery.worldWith] using
          (GraceRecovery.world.moves_sum z.2 (fun _ => renewableActionRepresentative a))
    simp only [Fintype.sum_prod_type, ← Finset.sum_mul, hw, one_mul]
    exact GraceRecovery.body.eventStep_sum GraceRecovery.world.clock
      (GraceRecovery.world.sense z.2) z.1

/-- Reconnecting the body's policy reproduces every transition of the
registered grace process; overriding action zero from birth reaches vacuum in
two certain steps. Normal operational survival stays one at every tick. -/
theorem grace_action_intervention_contrast :
    (∀ t z e z', graceActionMachine.step (RenewableLearning.policy (z.1 ())) z e z' =
      (GraceRecovery.process false GraceRecovery.natural).step t z e z') ∧
    graceActionMachine.step 0 RenewableLearning.initial (some ())
      ((fun _ => .search), .missA) = 1 ∧
    graceActionMachine.step 0 ((fun _ => .search), .missA) (some ())
      ((fun _ => .vacuum), .probeA) = 1 ∧
    (∀ n, Exchange.survival GraceRecovery.body GraceRecovery.world GraceRecovery.ec
      GraceRecovery.material false GraceRecovery.natural n = 1) := by
  refine ⟨?_, ?_, ?_, GraceRecovery.natural_survival⟩
  · intro t z e z'
    change (if z.1 () = .vacuum then GraceRecovery.moves z.2 z.1 z'.2
        else graceActionWorld z.2 (RenewableLearning.policy (z.1 ())) z'.2) *
      GraceRecovery.body.eventStep GraceRecovery.world.clock
        (GraceRecovery.world.sense z.2) z.1 e z'.1 =
      GraceRecovery.moves z.2 z.1 z'.2 * GraceRecovery.body.eventStep
        GraceRecovery.world.clock (GraceRecovery.world.sense z.2) z.1 e z'.1
    by_cases hz : z.1 () = .vacuum
    · simp [hz]
    · simp only [hz, ↓reduceIte]
      have hx : (fun _ : Unit => z.1 ()) = z.1 := by funext u; cases u; rfl
      have h := grace_action_world_matches z.2 z'.2 (z.1 ()) hz
      rw [hx] at h
      rw [← h]
  · have h0 : RenewableLearning.policy .aReady = 0 := by decide
    norm_num [graceActionMachine, graceActionWorld, renewableActionRepresentative,
      RenewableLearning.initial, GraceRecovery.moves, GraceRecovery.world,
      GraceRecovery.worldWith, RenewableLearning.moves, RenewableLearning.plant,
      RenewableLearning.world, RenewableLearning.sense, RenewableLearning.next,
      CGB.Medium.eventStep, h0, funext_iff]
  · have h0 : RenewableLearning.policy .aReady = 0 := by decide
    norm_num [graceActionMachine, graceActionWorld, renewableActionRepresentative,
      GraceRecovery.moves, GraceRecovery.world, GraceRecovery.worldWith,
      RenewableLearning.moves, RenewableLearning.plant, RenewableLearning.world,
      RenewableLearning.sense, RenewableLearning.next, CGB.Medium.eventStep, h0, funext_iff]

/-- The existing grace witness supplies the local conditions of the stochastic
joint generator: its already-proved good region is invariant under the
original body policy. The selected policy is not asserted to equal that
original policy; equality of the original policy's full law is the separate
`grace_action_intervention_contrast` result. -/
theorem grace_stochastic_joint_generator_exists :
    ∃ policy : (Unit → RenewableLearning.Cell) → RenewableLearning.Action,
      (∀ s w e z', GraceRecovery.Good (s ()) w →
        0 < graceActionMachine.step (policy s) (s, w) e z' →
          GraceRecovery.Good (z'.1 ()) z'.2) ∧
      ∃ process : CGB.Interaction.Machine RenewableLearning.S ℕ (Option Unit),
        (∀ t z e z', process.step t z e z' = graceActionMachine.step (policy z.1) z e z') ∧
        (∀ n z, ¬ GraceRecovery.Good (z.1 ()) z.2 →
          CGB.Energetics.law process RenewableLearning.initial n z = 0) := by
  apply (stochastic_joint_viability_policy_generation graceActionMachine
    (fun z => GraceRecovery.Good (z.1 ()) z.2) RenewableLearning.initial (by rfl)).1
  intro s
  refine ⟨RenewableLearning.policy (s ()), ?_⟩
  intro w e z' hg hp
  have he := grace_action_intervention_contrast.1 0 (s, w) e z'
  rw [he] at hp
  exact GraceRecovery.naturalCertificate.closed 0 (s, w) e z' hg hp

/-- The same finite grace world supplies the strengthened generator's local
conditions, using its existing actual-intake mark and recovery rank. The
chosen policy is safe and has first hit one; it is not asserted to equal the
old policy or to retain its separate loop-cut and learning certificates. -/
theorem grace_joint_recovery_generator_exists :
    let mark := Exchange.intake GraceRecovery.body GraceRecovery.world
      GraceRecovery.material GraceRecovery.natural
    ∃ policy : (Unit → RenewableLearning.Cell) → RenewableLearning.Action,
      ∃ process : CGB.Interaction.Machine RenewableLearning.S ℕ (Option Unit),
        (∀ t z e z', process.step t z e z' = graceActionMachine.step (policy z.1) z e z') ∧
        (∀ n z, ¬ GraceRecovery.Good (z.1 ()) z.2 →
          CGB.Energetics.law process RenewableLearning.initial n z = 0) ∧
        (∀ t z, GraceRecovery.Good (z.1 ()) z.2 → CGB.EventChannels.Kernel.firstHit
          (CGB.Energetics.events process) mark t GraceRecovery.world.horizon z = 1) := by
  dsimp only
  apply stochastic_joint_recovery_policy_generation graceActionMachine
    (fun z => GraceRecovery.Good (z.1 ()) z.2) RenewableLearning.initial (by rfl)
    GraceRecovery.naturalCertificate.rank GraceRecovery.world.horizon
    (Exchange.intake GraceRecovery.body GraceRecovery.world GraceRecovery.material GraceRecovery.natural)
    GraceRecovery.naturalCertificate.bounded
  intro s
  refine ⟨RenewableLearning.policy (s ()), ?_⟩
  intro t w e z' hg hp
  have he := grace_action_intervention_contrast.1 t (s, w) e z'
  rw [he] at hp
  exact ⟨GraceRecovery.naturalCertificate.closed t (s, w) e z' hg hp,
    GraceRecovery.naturalCertificate.progresses t (s, w) e z' hg hp⟩

/-- On an already-supported prefix, one action override at the same joint
state changes the next observation and the following body: the normal probe
gets an answer and writes the B candidate; action zero yields conflict and
vacuum. Later actions are restored to the body's policy. This is a local
causal contrast under the action-kernel invariance assumption, not an
attribution of all survival benefit to one prior memory write. -/
theorem grace_reachable_one_action_consequence :
    let path := renewableObservableFromBirth (fun _ => true)
    0 < RenewableLearningControls.weight (GraceRecovery.process false GraceRecovery.natural) path 6 ∧
    path 6 = ((fun _ => .search), .probeB) ∧
    RenewableLearning.policy .search = 2 ∧
    graceActionMachine.step 2 (path 6) (some ()) ((fun _ => .search), .answerB) = 1 ∧
    graceActionMachine.step 0 (path 6) (some ()) ((fun _ => .search), .missB) = 1 ∧
    graceActionMachine.step 2 ((fun _ => .search), .answerB) (some ())
      ((fun _ => .bSpent), .feedB) = 1 ∧
    graceActionMachine.step 2 ((fun _ => .search), .missB) (some ())
      ((fun _ => .vacuum), .probeB) = 1 ∧
    RenewableLearning.writeCost true .answerB .search = 1 := by
  dsimp only
  have hs : RenewableLearning.policy .search = 2 := by decide
  have h0 : RenewableLearning.policy .aReady = 0 := by decide
  have h2 : RenewableLearning.policy .search = 2 := hs
  refine ⟨grace_reachable_memory_changes_action.2.1, ?_, hs, ?_, ?_, ?_, ?_, ?_⟩
  · rfl
  all_goals norm_num [renewableObservableFromBirth, renewableObservablePath,
    renewableObservableBlock, graceActionMachine, graceActionWorld, renewableActionRepresentative,
    GraceRecovery.moves, GraceRecovery.world, GraceRecovery.worldWith,
    RenewableLearning.moves, RenewableLearning.plant, RenewableLearning.world,
    RenewableLearning.sense, RenewableLearning.next, RenewableLearning.writeCost,
    CGB.Medium.eventStep, hs, h0, h2, funext_iff]

/-- On a supported mismatch, override reopening for one body event only,
then restore the original body and world rules. The normal event reopens
to `search`; the override holds `aHeld`. Their own policies then lead to
answer-and-write versus conflict-and-vacuum. The override also changes the
event's outgoing-material accounting, so this does not isolate information
from all physical effects. -/
theorem grace_one_reopening_intervention_consequence :
    let path := renewableObservableFromBirth (fun _ => true)
    let off := MaterialSource.machine ((RenewableLearning.plant false true).jointEventKernel
      GraceRecovery.world (RenewableLearning.he false true) false GraceRecovery.natural)
    0 < RenewableLearningControls.weight (GraceRecovery.process false GraceRecovery.natural) path 5 ∧
    path 5 = ((fun _ => .aReady), .missB) ∧
    (GraceRecovery.process false GraceRecovery.natural).step 5 (path 5) (some ())
      ((fun _ => .search), .probeB) = 1 ∧
    off.step 5 (path 5) (some ()) ((fun _ => .aHeld), .probeB) = 1 ∧
    RenewableLearning.policy .search = 2 ∧ RenewableLearning.policy .aHeld = 0 ∧
    graceActionMachine.step 2 ((fun _ => .search), .probeB) (some ())
      ((fun _ => .search), .answerB) = 1 ∧
    graceActionMachine.step 0 ((fun _ => .aHeld), .probeB) (some ())
      ((fun _ => .aHeld), .missB) = 1 ∧
    graceActionMachine.step 2 ((fun _ => .search), .answerB) (some ())
      ((fun _ => .bSpent), .feedB) = 1 ∧
    graceActionMachine.step 0 ((fun _ => .aHeld), .missB) (some ())
      ((fun _ => .vacuum), .probeB) = 1 ∧
    RenewableLearning.outgoing true true .miss .aReady = 2 ∧
    RenewableLearning.outgoing false true .miss .aReady = 1 := by
  dsimp only
  have h6 := grace_reachable_memory_changes_action.2.1
  have h5 : 0 < RenewableLearningControls.weight (GraceRecovery.process false GraceRecovery.natural)
      (renewableObservableFromBirth (fun _ => true)) 5 := by
    unfold RenewableLearningControls.weight at h6 ⊢
    rw [show 6 = 5 + 1 by rfl, Finset.prod_range_succ] at h6
    exact pos_of_mul_pos_left h6 ((GraceRecovery.process false GraceRecovery.natural).nonneg 5
      (renewableObservableFromBirth (fun _ => true) 5)
      (RenewableLearningControls.tag (renewableObservableFromBirth (fun _ => true) 5))
      (renewableObservableFromBirth (fun _ => true) 6))
  have hs : RenewableLearning.policy .search = 2 := by decide
  have ha : RenewableLearning.policy .aHeld = 0 := by decide
  have hr : RenewableLearning.policy .aReady = 0 := by decide
  refine ⟨h5, ?_, ?_, ?_, hs, ha, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rfl
  all_goals norm_num [renewableObservableFromBirth, renewableObservablePath,
    renewableObservableBlock, graceActionMachine, graceActionWorld, renewableActionRepresentative,
    GraceRecovery.process, GraceRecovery.natural, GraceRecovery.moves, GraceRecovery.body,
    GraceRecovery.world, GraceRecovery.worldWith, MaterialSource.machine,
    CGB.Medium.jointEventKernel, CGB.Medium.eventWorldStep, CGB.Medium.eventStep,
    RenewableLearning.moves, RenewableLearning.plant, RenewableLearning.world,
    RenewableLearning.sense, RenewableLearning.next, RenewableLearning.outgoing,
    RenewableLearning.cost, RenewableLearning.buffer, RenewableLearning.credit, hs, ha, hr, funext_iff]

/-- The explicit action override uses the same stochastic machine at every
tick. It is separate from the marginal-ghost loop cut. -/
noncomputable def graceActionAlways (a : RenewableLearning.Action) :
    CGB.Interaction.Machine RenewableLearning.S ℕ (Option Unit) where
  step _ := graceActionMachine.step a
  nonneg _ := graceActionMachine.nonneg a
  sum_one _ := graceActionMachine.sum_one a

theorem grace_action_zero_vacuum_probability :
    CGB.Energetics.law (graceActionAlways 0) RenewableLearning.initial 2
      ((fun _ => .vacuum), .probeA) = 1 := by
  let P := graceActionAlways 0
  let init := RenewableLearning.initial
  let middle : RenewableLearning.S := ((fun _ => .search), .missA)
  let dead : RenewableLearning.S := ((fun _ => .vacuum), .probeA)
  have hstep (t : ℕ) (x y : RenewableLearning.S) (e : Option Unit) :
      CGB.Energetics.law P init t x * P.step t x e y ≤
        CGB.Energetics.law P init (t + 1) y := by
    have hn := (CGB.Energetics.law_probability P init t).1
    calc
      _ ≤ CGB.Energetics.law P init t x * (∑ e, P.step t x e y) := by
        apply mul_le_mul_of_nonneg_left _ (hn x)
        exact Finset.single_le_sum (fun e _ => P.nonneg t x e y) (Finset.mem_univ _)
      _ ≤ _ := by
        apply Finset.single_le_sum _ (Finset.mem_univ x)
        intro z _
        exact mul_nonneg (hn z) (Finset.sum_nonneg (fun e _ => P.nonneg t z e y))
  have hstart : CGB.Energetics.law P init 0 init = 1 := by
    simp [CGB.Energetics.law, CGB.EventChannels.Kernel.stateLaw]
  have hfirst : P.step 0 init (some ()) middle = 1 := grace_action_intervention_contrast.2.1
  have hsecond : P.step 1 middle (some ()) dead = 1 := grace_action_intervention_contrast.2.2.1
  have hfirstlaw : 1 ≤ CGB.Energetics.law P init 1 middle := by
    simpa [hstart, hfirst] using hstep 0 init middle (some ())
  have hlower : 1 ≤ CGB.Energetics.law P init 2 dead := by
    have h := hstep 1 middle dead (some ())
    rw [hsecond, mul_one] at h
    exact hfirstlaw.trans h
  have hupper : CGB.Energetics.law P init 2 dead ≤ 1 := by
    calc
      _ ≤ ∑ z, CGB.Energetics.law P init 2 z :=
        Finset.single_le_sum (fun z _ => (CGB.Energetics.law_probability P init 2).1 z)
          (Finset.mem_univ dead)
      _ = 1 := (CGB.Energetics.law_probability P init 2).2
  exact le_antisymm hupper hlower

/-- The same grace-world process has operational persistence, one-tick
interruption tolerance and a loop-cut maintenance advantage, while supporting
arbitrarily many paid rewrites on positive-probability finite prefixes.
The finite-prefix statement is not an almost-sure infinite learning result;
this material witness is not an instance of the full `Medium.Lives` contract. -/
theorem grace_recovery_learning_joint :
    Exchange.Persists GraceRecovery.body GraceRecovery.world GraceRecovery.ec GraceRecovery.material 0 ∧
    Exchange.ToleratesSensoryInterruption GraceRecovery.body GraceRecovery.world
      GraceRecovery.ec GraceRecovery.material 0 1 ∧
    Exchange.SelfMaintains GraceRecovery.body GraceRecovery.world GraceRecovery.ec GraceRecovery.material ∧
    (∀ k, 0 < RenewableLearningControls.weight
      (GraceRecovery.process false GraceRecovery.natural)
      RenewableLearningControls.path (3 + 10 * k) ∧
      (∑ t ∈ Finset.range (3 + 10 * k), RenewableLearning.writeCost true
        (RenewableLearning.sense (RenewableLearningControls.path t).2 ())
        ((RenewableLearningControls.path t).1 ())) = 1 + 4 * k ∧
      (∑ t ∈ Finset.range (3 + 10 * k), RenewableLearning.topup true
        (RenewableLearning.sense (RenewableLearningControls.path t).2 ())
        ((RenewableLearningControls.path t).1 ())) = 2 + 4 * k) := by
  exact ⟨GraceRecoveryClosure.maintenance.1, GraceRecoveryClosure.maintenance.2.1,
    GraceRecoveryClosure.maintenance.2.2, GraceRecoveryAccounting.arbitrary_relearning⟩

/-- A concrete policy and its actual process simultaneously satisfy the
stochastic generator's one-step closure and ranked recovery conditions, the
original loop-cut maintenance and one-tick interruption contracts, and the
existing arbitrarily long positive-probability paid-learning prefixes. No
property is transferred to an unrelated policy selected by choice. -/
theorem grace_original_policy_joint_certificate :
    let policy := fun s : Unit → RenewableLearning.Cell => RenewableLearning.policy (s ())
    let process := GraceRecovery.process false GraceRecovery.natural
    let mark := Exchange.intake GraceRecovery.body GraceRecovery.world
      GraceRecovery.material GraceRecovery.natural
    (∀ t z e z', process.step t z e z' = graceActionMachine.step (policy z.1) z e z') ∧
    (∀ t z e z', GraceRecovery.Good (z.1 ()) z.2 → 0 < process.step t z e z' →
      GraceRecovery.Good (z'.1 ()) z'.2) ∧
    (∀ n z, ¬ GraceRecovery.Good (z.1 ()) z.2 →
      CGB.Energetics.law process RenewableLearning.initial n z = 0) ∧
    (∀ t z, GraceRecovery.Good (z.1 ()) z.2 → CGB.EventChannels.Kernel.firstHit
      (CGB.Energetics.events process) mark t GraceRecovery.world.horizon z = 1) ∧
    Exchange.Persists GraceRecovery.body GraceRecovery.world GraceRecovery.ec GraceRecovery.material 0 ∧
    Exchange.ToleratesSensoryInterruption GraceRecovery.body GraceRecovery.world
      GraceRecovery.ec GraceRecovery.material 0 1 ∧
    Exchange.SelfMaintains GraceRecovery.body GraceRecovery.world GraceRecovery.ec GraceRecovery.material ∧
    (∀ k, 0 < RenewableLearningControls.weight process RenewableLearningControls.path (3 + 10 * k) ∧
      (∑ t ∈ Finset.range (3 + 10 * k), RenewableLearning.writeCost true
        (RenewableLearning.sense (RenewableLearningControls.path t).2 ())
        ((RenewableLearningControls.path t).1 ())) = 1 + 4 * k) := by
  dsimp only
  refine ⟨?_, ?_, ?_, ?_, GraceRecoveryClosure.maintenance.1,
    GraceRecoveryClosure.maintenance.2.1, GraceRecoveryClosure.maintenance.2.2, ?_⟩
  · intro t z e z'
    exact (grace_action_intervention_contrast.1 t z e z').symm
  · intro t z e z' hg hp
    exact GraceRecovery.naturalCertificate.closed t z e z' hg hp
  · exact closed_region_law_zero (GraceRecovery.process false GraceRecovery.natural)
      (fun z => GraceRecovery.Good (z.1 ()) z.2) RenewableLearning.initial (by rfl)
      (fun t z e z' hg hp => GraceRecovery.naturalCertificate.closed t z e z' hg hp)
  · intro t z hg
    change CGB.EventChannels.Kernel.firstHit
      (GraceRecovery.body.jointEventKernel GraceRecovery.world GraceRecovery.ec false GraceRecovery.natural)
      (Exchange.intake GraceRecovery.body GraceRecovery.world GraceRecovery.material GraceRecovery.natural)
      t GraceRecovery.world.horizon z = 1
    exact CGB.RecoveryCriteria.ranked_firstHit GraceRecovery.naturalCertificate t
      GraceRecovery.world.horizon z hg (GraceRecovery.naturalCertificate.bounded z hg)
  · intro k
    exact ⟨(GraceRecoveryAccounting.arbitrary_relearning k).1,
      (GraceRecoveryAccounting.arbitrary_relearning k).2.1⟩

/-- On every infinite supported path of the original grace process, actual
intake occurs in every three-tick window. This strengthens finite-prefix
support for intake only, not for paid learning writes. -/
theorem grace_supported_paths_recurrently_intake
    (path : ℕ → RenewableLearning.S) (event : ℕ → Option Unit)
    (hstart : path 0 = RenewableLearning.initial)
    (hsupport : ∀ t, 0 < (GraceRecovery.process false GraceRecovery.natural).step t
      (path t) (event t) (path (t + 1))) :
    ∀ start, ∃ t, start ≤ t ∧ t < start + 3 ∧
      Exchange.intake GraceRecovery.body GraceRecovery.world GraceRecovery.material
        GraceRecovery.natural t (path t) (event t) (path (t + 1)) = true := by
  have hseed : GraceRecovery.naturalCertificate.region (path 0) := by rw [hstart]; rfl
  have h := ranked_supported_path_recurrent
    (GraceRecovery.body.jointEventKernel GraceRecovery.world GraceRecovery.ec false GraceRecovery.natural)
    (Exchange.intake GraceRecovery.body GraceRecovery.world GraceRecovery.material GraceRecovery.natural)
    GraceRecovery.world.horizon GraceRecovery.naturalCertificate path event hseed hsupport
  simpa [GraceRecovery.world, GraceRecovery.worldWith] using h

/-- The original process has at least one infinite supported path, and each
three-tick window on that path contains actual intake. -/
theorem grace_exists_recurrently_intaking_path :
    ∃ path : ℕ → RenewableLearning.S, ∃ event : ℕ → Option Unit,
      path 0 = RenewableLearning.initial ∧
      (∀ t, 0 < (GraceRecovery.process false GraceRecovery.natural).step t
        (path t) (event t) (path (t + 1))) ∧
      (∀ start, ∃ t, start ≤ t ∧ t < start + 3 ∧
        Exchange.intake GraceRecovery.body GraceRecovery.world GraceRecovery.material
          GraceRecovery.natural t (path t) (event t) (path (t + 1)) = true) := by
  have h := ranked_exists_recurrently_marked_path
    (GraceRecovery.body.jointEventKernel GraceRecovery.world GraceRecovery.ec false GraceRecovery.natural)
    (Exchange.intake GraceRecovery.body GraceRecovery.world GraceRecovery.material GraceRecovery.natural)
    GraceRecovery.world.horizon GraceRecovery.naturalCertificate RenewableLearning.initial (by rfl)
  obtain ⟨path, event, hzero, hsteps, hrecur⟩ := h
  refine ⟨path, event, hzero, ?_, ?_⟩
  · intro t
    exact hsteps t
  · simpa [GraceRecovery.world, GraceRecovery.worldWith] using hrecur

/-- The already registered paid-relearning path is itself an infinite
supported path of the same grace process. It has unbounded cumulative paid
writes and recurrent actual intake. This is one supported path, not a claim
that any particular infinite path has positive probability or that learning
continues almost surely. -/
theorem grace_infinite_supported_learning_and_intake :
    let path := RenewableLearningControls.path
    let event := fun t => RenewableLearningControls.tag (path t)
    path 0 = RenewableLearning.initial ∧
    (∀ t, 0 < (GraceRecovery.process false GraceRecovery.natural).step t
      (path t) (event t) (path (t + 1))) ∧
    (∀ k, (∑ t ∈ Finset.range (3 + 10 * k), RenewableLearning.writeCost true
      (RenewableLearning.sense (path t).2 ()) ((path t).1 ())) = 1 + 4 * k) ∧
    (∀ B : ℕ, ∃ T, B < ∑ t ∈ Finset.range T, RenewableLearning.writeCost true
      (RenewableLearning.sense (path t).2 ()) ((path t).1 ())) ∧
    (∀ start, ∃ t, start ≤ t ∧ t < start + 3 ∧
      Exchange.intake GraceRecovery.body GraceRecovery.world GraceRecovery.material
        GraceRecovery.natural t (path t) (event t) (path (t + 1)) = true) := by
  dsimp only
  have hstart : RenewableLearningControls.path 0 = RenewableLearning.initial := rfl
  have hsupport : ∀ t, 0 < (GraceRecovery.process false GraceRecovery.natural).step t
      (RenewableLearningControls.path t) (RenewableLearningControls.tag (RenewableLearningControls.path t))
      (RenewableLearningControls.path (t + 1)) := by
    intro t
    have hprod := (GraceRecoveryAccounting.arbitrary_relearning (t + 1)).1
    unfold RenewableLearningControls.weight at hprod
    have hne := (Finset.prod_ne_zero_iff.mp (ne_of_gt hprod)) t
      (Finset.mem_range.mpr (by omega : t < 3 + 10 * (t + 1)))
    have hnonneg := (GraceRecovery.process false GraceRecovery.natural).nonneg t
      (RenewableLearningControls.path t) (RenewableLearningControls.tag (RenewableLearningControls.path t))
      (RenewableLearningControls.path (t + 1))
    exact lt_of_le_of_ne hnonneg (Ne.symm hne)
  refine ⟨hstart, hsupport, ?_, ?_, ?_⟩
  · intro k; exact (GraceRecoveryAccounting.arbitrary_relearning k).2.1
  · intro B
    refine ⟨3 + 10 * (B + 1), ?_⟩
    rw [(GraceRecoveryAccounting.arbitrary_relearning (B + 1)).2.1]
    omega
  · exact grace_supported_paths_recurrently_intake RenewableLearningControls.path
      (fun t => RenewableLearningControls.tag (RenewableLearningControls.path t)) hstart hsupport

/-- The same grace process also admits an infinite supported path that stops
paid writing after its initial phase. Thus bounded-gap ranked recurrence of
paid writes cannot hold on every supported path from the seed, even though a
different supported path has unbounded writing. This does not decide the
probability of eventual stopping under an infinite-path measure. -/
theorem grace_supported_path_eventually_no_writes :
    let path := renewableObservableFromBirth (fun _ => false)
    let event := fun t => RenewableLearningControls.tag (path t)
    path 0 = RenewableLearning.initial ∧
    (∀ t, 0 < (GraceRecovery.process false GraceRecovery.natural).step t
      (path t) (event t) (path (t + 1))) ∧
    (∀ t, 3 ≤ t → RenewableLearning.writeCost true
      (RenewableLearning.sense (path t).2 ()) ((path t).1 ()) = 0) ∧
    ¬ ∃ (horizon : ℕ) (C : CGB.RecoveryCriteria.Ranked
      (GraceRecovery.body.jointEventKernel GraceRecovery.world GraceRecovery.ec false GraceRecovery.natural)
      (fun _ z e _ => decide (e = some () ∧ 0 < RenewableLearning.writeCost true
        (RenewableLearning.sense z.2 ()) (z.1 ()))) horizon), C.region (path 0) := by
  classical
  dsimp only
  let path := renewableObservableFromBirth (fun _ => false)
  let event := fun t => RenewableLearningControls.tag (path t)
  have htail (t : ℕ) (ht : 3 ≤ t) : path t = ((fun _ => .aReady), .feedA) := by
    obtain ⟨n, rfl⟩ := Nat.exists_eq_add_of_le ht
    simp [path, renewableObservableFromBirth, renewableObservablePath, renewableObservableBlock,
      Nat.add_comm]
  have hgood (t : ℕ) : GraceRecovery.Good ((path t).1 ()) (path t).2 := by
    rcases t with _ | _ | _ | t
    all_goals simp [path, renewableObservableFromBirth, renewableObservablePath,
      renewableObservableBlock, RenewableLearning.initial, GraceRecovery.Good,
      RenewableLearningLife.Good]
  have hstart : path 0 = RenewableLearning.initial := rfl
  have hsupport : ∀ t, 0 < (GraceRecovery.process false GraceRecovery.natural).step t
      (path t) (event t) (path (t + 1)) := by
    intro t
    rw [GraceRecoveryAccounting.normal_step_unchanged t (path t) (path (t + 1)) (event t) (hgood t)]
    have hprod := (renewable_observable_from_birth (fun _ => false)).2.1 (t + 1)
    unfold RenewableLearningControls.weight at hprod
    have hne := (Finset.prod_ne_zero_iff.mp (ne_of_gt hprod)) t
      (Finset.mem_range.mpr (by omega : t < t + 1))
    have hnonneg := (RenewableLearning.process true true false).nonneg t
      (path t) (event t) (path (t + 1))
    exact lt_of_le_of_ne hnonneg (Ne.symm hne)
  have hzero : ∀ t, 3 ≤ t → RenewableLearning.writeCost true
      (RenewableLearning.sense (path t).2 ()) ((path t).1 ()) = 0 := by
    intro t ht
    rw [htail t ht]
    decide
  refine ⟨hstart, hsupport, hzero, ?_⟩
  rintro ⟨horizon, C, hseed⟩
  obtain ⟨t, ht, _, hm⟩ := ranked_supported_path_recurrent
    (GraceRecovery.body.jointEventKernel GraceRecovery.world GraceRecovery.ec false GraceRecovery.natural)
    (fun _ z e _ => decide (e = some () ∧ 0 < RenewableLearning.writeCost true
      (RenewableLearning.sense z.2 ()) (z.1 ()))) horizon C path event hseed hsupport 3
  have hz := hzero t ht
  simp [hz] at hm

/-- Minimal probabilistic upgrade for persistent learning: if the probability
of no useful learning event after `n` completed blocks is `q n`, and every
block has conditional success at least `p`, then failure is bounded by
`(1-p)^n`. This is an abstract finite-prefix inequality; a real infinite
process must separately prove the conditional minorization premise. -/
theorem uniform_block_learning_failure_decay
    (p : ℝ) (hp1 : p ≤ 1)
    (q : ℕ → ℝ) (hq0 : q 0 ≤ 1)
    (hrec : ∀ n, q (n + 1) ≤ (1 - p) * q n) :
    ∀ n, q n ≤ (1 - p) ^ n := by
  intro n
  induction n with
  | zero => simpa using hq0
  | succ n ih =>
    calc
      q (n + 1) ≤ (1 - p) * q n := hrec n
      _ ≤ (1 - p) * ((1 - p) ^ n) :=
        mul_le_mul_of_nonneg_left ih (sub_nonneg.mpr hp1)
      _ = (1 - p) ^ (n + 1) := by rw [pow_succ]; ring

/-! ### One theorem for the current language-life contract

The preceding lemmas deliberately keep representation, admission, active
closure, predictive payment, and probabilistic learning separate.  The next
theorem is only their assembly: it does not add a hidden resource, a new
state variable, or an optimizer.  Its purpose is to give the generator and
the implementation layer one named Lean gate.
-/

/-- The current candidate specification and the conditional infinite-stream
guarantees hold for one and the same encoded trajectory.  The first conjunct
is the complete specification currently defined in this file; the remaining
conjuncts add the explicit alignment, predictive-payment, and block-learning
conditions needed for active continual operation. -/
theorem language_conditional_joint_specification
    {Hist Model Ctx Act Obs Byte Answer : Type*}
    [Fintype Model] [DecidableEq Model]
    [Fintype Byte] [Nonempty Byte]
    (L : LanguageSelfProductionCandidate Hist Model Ctx Act Obs Byte Answer)
    (hseed : L.base.encode (L.base.trajectory 0) = L.base.encode L.seed)
    (hstep : ∀ n, L.base.encode (L.base.trajectory (n + 1)) =
      L.base.next (L.base.encode (L.base.trajectory n))
        (L.world_response (L.base.encode (L.base.trajectory n))
          (L.base.policy (L.base.encode (L.base.trajectory n)))))
    (m : ℝ)
    (hce : ∀ h, PredictiveHarvest.crossEntropy (L.base.stream h)
      (L.base.allocation (L.base.encode h)) <
      Real.log (Fintype.card Byte) - m)
    (p : ℝ) (hp1 : p ≤ 1) (q : ℕ → ℝ) (hq0 : q 0 ≤ 1)
    (hrec : ∀ n, q (n + 1) ≤ (1 - p) * q n) :
    language_infinite_stream_life_spec L ∧
    LanguageLifeAdmission.joint_admission_spec L.base ∧
    (∀ n, language_viable L (trajectory (language_contract L) n)) ∧
    ¬ eventuallyPassive (fun n => L.base.policy
      (trajectory (language_contract L) n)) L.base.passive ∧
    (∀ h, 0 < PredictiveHarvest.growth (L.base.stream h)
      (L.base.allocation (L.base.encode h)) - m) ∧
    (∀ n, q n ≤ (1 - p) ^ n) := by
  have hspec := language_infinite_stream_life_complete L
  have hj := language_conditional_joint_existence L hseed hstep m hce p hp1 q hq0 hrec
  exact ⟨hspec, hj.1, hj.2.1, hj.2.2.1, hj.2.2.2.1, hj.2.2.2.2⟩

/-- Minimal-generator form of the total gate.  History factorization supplies
the finite causal carrier; the displayed hypotheses are exactly the remaining
world alignment, predictive payment, and block-learning obligations. -/
theorem minimal_generator_conditional_joint_specification
    {Hist Model Ctx Act Obs Byte Answer : Type*}
    [Fintype Model] [DecidableEq Model]
    [Fintype Byte] [Nonempty Byte]
    (G : MinimalLanguageSelfProductionGenerator Hist Model Ctx Act Obs Byte Answer)
    (hseed : (materializeLanguageSelfProductionGenerator G).base.encode
      ((materializeLanguageSelfProductionGenerator G).base.trajectory 0) =
      (materializeLanguageSelfProductionGenerator G).base.encode
        (materializeLanguageSelfProductionGenerator G).seed)
    (hstep : ∀ n, (materializeLanguageSelfProductionGenerator G).base.encode
      ((materializeLanguageSelfProductionGenerator G).base.trajectory (n + 1)) =
      (materializeLanguageSelfProductionGenerator G).base.next
        ((materializeLanguageSelfProductionGenerator G).base.encode
          ((materializeLanguageSelfProductionGenerator G).base.trajectory n))
        ((materializeLanguageSelfProductionGenerator G).world_response
          ((materializeLanguageSelfProductionGenerator G).base.encode
            ((materializeLanguageSelfProductionGenerator G).base.trajectory n))
          ((materializeLanguageSelfProductionGenerator G).base.policy
            ((materializeLanguageSelfProductionGenerator G).base.encode
              ((materializeLanguageSelfProductionGenerator G).base.trajectory n)))))
    (m : ℝ)
    (hce : ∀ h, PredictiveHarvest.crossEntropy
      ((materializeLanguageSelfProductionGenerator G).base.stream h)
      ((materializeLanguageSelfProductionGenerator G).base.allocation
        ((materializeLanguageSelfProductionGenerator G).base.encode h)) <
      Real.log (Fintype.card Byte) - m)
    (p : ℝ) (hp1 : p ≤ 1) (q : ℕ → ℝ) (hq0 : q 0 ≤ 1)
    (hrec : ∀ n, q (n + 1) ≤ (1 - p) * q n) :
    language_infinite_stream_life_spec
        (materializeLanguageSelfProductionGenerator G) ∧
    LanguageLifeAdmission.joint_admission_spec
        (materializeLanguageSelfProductionGenerator G).base ∧
    (∀ n, language_viable (materializeLanguageSelfProductionGenerator G)
      (trajectory (language_contract
        (materializeLanguageSelfProductionGenerator G)) n)) ∧
    ¬ eventuallyPassive
      (fun n => (materializeLanguageSelfProductionGenerator G).base.policy
        (trajectory (language_contract
          (materializeLanguageSelfProductionGenerator G)) n))
      (materializeLanguageSelfProductionGenerator G).base.passive ∧
    (∀ h, 0 < PredictiveHarvest.growth
      ((materializeLanguageSelfProductionGenerator G).base.stream h)
      ((materializeLanguageSelfProductionGenerator G).base.allocation
        ((materializeLanguageSelfProductionGenerator G).base.encode h)) - m) ∧
    (∀ n, q n ≤ (1 - p) ^ n) := by
  exact language_conditional_joint_specification
    (materializeLanguageSelfProductionGenerator G) hseed hstep m hce p hp1 q hq0 hrec

/-- Existential form of the total gate.  Once a minimal generator and its
world conditions are supplied, materialization produces one concrete
language candidate satisfying the current life specification, active
self-production, predictive payment, and block-learning bound together. -/
theorem exists_conditionally_operating_language_life_of_minimal_generator
    {Hist Model Ctx Act Obs Byte Answer : Type*}
    [Fintype Model] [DecidableEq Model]
    [Fintype Byte] [Nonempty Byte]
    (G : MinimalLanguageSelfProductionGenerator Hist Model Ctx Act Obs Byte Answer)
    (hseed : (materializeLanguageSelfProductionGenerator G).base.encode
      ((materializeLanguageSelfProductionGenerator G).base.trajectory 0) =
      (materializeLanguageSelfProductionGenerator G).base.encode
        (materializeLanguageSelfProductionGenerator G).seed)
    (hstep : ∀ n, (materializeLanguageSelfProductionGenerator G).base.encode
      ((materializeLanguageSelfProductionGenerator G).base.trajectory (n + 1)) =
      (materializeLanguageSelfProductionGenerator G).base.next
        ((materializeLanguageSelfProductionGenerator G).base.encode
          ((materializeLanguageSelfProductionGenerator G).base.trajectory n))
        ((materializeLanguageSelfProductionGenerator G).world_response
          ((materializeLanguageSelfProductionGenerator G).base.encode
            ((materializeLanguageSelfProductionGenerator G).base.trajectory n))
          ((materializeLanguageSelfProductionGenerator G).base.policy
            ((materializeLanguageSelfProductionGenerator G).base.encode
              ((materializeLanguageSelfProductionGenerator G).base.trajectory n)))))
    (m : ℝ)
    (hce : ∀ h, PredictiveHarvest.crossEntropy
      ((materializeLanguageSelfProductionGenerator G).base.stream h)
      ((materializeLanguageSelfProductionGenerator G).base.allocation
        ((materializeLanguageSelfProductionGenerator G).base.encode h)) <
      Real.log (Fintype.card Byte) - m)
    (p : ℝ) (hp1 : p ≤ 1) (q : ℕ → ℝ) (hq0 : q 0 ≤ 1)
    (hrec : ∀ n, q (n + 1) ≤ (1 - p) * q n) :
    ∃ L : LanguageSelfProductionCandidate Hist Model Ctx Act Obs Byte Answer,
      language_infinite_stream_life_spec L ∧
      LanguageLifeAdmission.joint_admission_spec L.base ∧
      (∀ n, language_viable L (trajectory (language_contract L) n)) ∧
      ¬ eventuallyPassive (fun n => L.base.policy
        (trajectory (language_contract L) n)) L.base.passive ∧
      (∀ h, 0 < PredictiveHarvest.growth (L.base.stream h)
        (L.base.allocation (L.base.encode h)) - m) ∧
      (∀ n, q n ≤ (1 - p) ^ n) := by
  refine ⟨materializeLanguageSelfProductionGenerator G, ?_⟩
  exact minimal_generator_conditional_joint_specification
    G hseed hstep m hce p hp1 q hq0 hrec

end CGB.SelfProduction
