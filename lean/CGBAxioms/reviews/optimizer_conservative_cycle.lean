module

public import Mathlib.Basic.Real.Basic
public import Mathlib.Algebra.Order.BigOperators.Group.Finset
public import Mathlib.Algebra.BigOperators.Ring.Finset
public import Mathlib.Algebra.Order.Archimedean.Real.Basic
public import Mathlib.Algebra.Order.Floor.Ring
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring
import Lean.Elab.Tactic.Omega

public section

/-! Conditional generative construction using double-budget geometry and a G20 pool.
This is a local conservative-cycle kernel, not a trained task or a complete
optimizer. The affine coordinate follows AFTER choosing the 2x2 double budget.
Choosing squared allocation as payoff gives 1+m^2. That payoff, its recurrent
gradient dynamics, clipping, demand law and cue normalization are hypotheses H,
not consequences of the six clauses. Conditional generation is intentional:
interaction laws are premises; latching and switching are derived conclusions.
Their usefulness must be assessed independently of the internal phase.
The SAME two-state recursion below handles contraction, regeneration, sensing,
resource use/recovery and cue-driven switching. Gradient-derived cues and
smooth-objective parameter steps are checked below. General-graph/runtime
refinement and beneficial historical function remain separate obligations.
-/
namespace CGB.ConservativeCycleReview

noncomputable def allocation (m : ℝ) : Fin 2 → Fin 2 → ℝ := fun i j =>
  if i = j then (1 + m) / 2 else (1 - m) / 2

noncomputable def clip (x : ℝ) : ℝ := max (-1) (min 1 x)

noncomputable def step (q coupling cueScale refill useRate m reserve cue noise : ℝ) : ℝ × ℝ :=
  (clip ((q + coupling * reserve) * m + cueScale * reserve * cue + noise),
   (reserve + refill * (1 - reserve) - useRate * |m| * reserve))

private theorem clip_at_one_iff (x : ℝ) : clip x = 1 ↔ 1 ≤ x := by
  constructor
  · intro h
    by_contra hn
    have hx : x < 1 := lt_of_not_ge hn
    have hh : max (-1 : ℝ) (min 1 x) < 1 :=
      max_lt (by norm_num) ((min_le_right _ _).trans_lt hx)
    change max (-1 : ℝ) (min 1 x) = 1 at h
    linarith
  · intro h
    simp [clip, min_eq_left h]

private theorem clip_at_neg_one_iff (x : ℝ) : clip (-x) = -1 ↔ 1 ≤ x := by
  constructor
  · intro h
    by_contra hn
    have hx : x < 1 := lt_of_not_ge hn
    have hh : (-1 : ℝ) < min 1 (-x) := lt_min (by norm_num) (by linarith)
    have hh' := hh.trans_le (le_max_right (-1 : ℝ) (min 1 (-x)))
    change max (-1 : ℝ) (min 1 (-x)) = -1 at h
    linarith
  · intro h
    exact max_eq_left ((min_le_right _ _).trans (by linarith))

/-- Generative threshold of the DECLARED law, not an assumed retention score.
At a saturated mode the resource balance fixes r=s/(s+u). Both opposite modes
are fixed precisely when feedback through that resource compensates passive
loss. The uncommitted mode remains fixed too. This classifies equilibria only;
attraction, stochastic escape, useful content and optimality do not follow. -/
theorem regeneration_threshold (q coupling cueScale refill useRate : ℝ)
    (hs : 0 < refill) (hu : 0 ≤ useRate) :
    let r := refill / (refill + useRate)
    step q coupling cueScale refill useRate 0 1 0 0 = (0, 1) ∧
    (step q coupling cueScale refill useRate 1 r 0 0 = (1, r) ↔
      (1 - q) * (refill + useRate) ≤ coupling * refill) ∧
    (step q coupling cueScale refill useRate (-1) r 0 0 = (-1, r) ↔
      (1 - q) * (refill + useRate) ≤ coupling * refill) := by
  dsimp only
  have hd : 0 < refill + useRate := by linarith
  have hr : refill / (refill + useRate) * (refill + useRate) = refill :=
    div_mul_cancel₀ _ (ne_of_gt hd)
  have hbalance : refill / (refill + useRate) +
      refill * (1 - refill / (refill + useRate)) -
      useRate * (refill / (refill + useRate)) = refill / (refill + useRate) := by
    nlinarith
  have hcoef : 1 ≤ q + coupling * (refill / (refill + useRate)) ↔
      (1 - q) * (refill + useRate) ≤ coupling * refill := by
    have heq : q + coupling * (refill / (refill + useRate)) =
        (q * (refill + useRate) + coupling * refill) / (refill + useRate) := by
      field_simp
    rw [heq, le_div_iff₀ hd]
    constructor <;> intro h <;> nlinarith
  refine ⟨by norm_num [step, clip], ?_, ?_⟩
  · simpa only [step, mul_one, mul_zero, add_zero, abs_one, hbalance,
      Prod.mk.injEq, and_true, clip_at_one_iff] using hcoef
  · simpa only [step, mul_neg_one, mul_zero, add_zero, abs_neg, abs_one, mul_one,
      hbalance, Prod.mk.injEq, and_true, clip_at_neg_one_iff] using hcoef

private theorem clip_bounds (x : ℝ) : -1 ≤ clip x ∧ clip x ≤ 1 := by
  constructor
  · exact le_max_left _ _
  · exact max_le (by norm_num) (min_le_left _ _)

private theorem clip_lower {x y : ℝ} (h : y ≤ x) : min 1 y ≤ clip x := by
  exact (min_le_min_left 1 h).trans (le_max_right _ _)

/-- The local allocation always preserves BOTH budgets; its positive-feedback
payoff is exactly 1+m^2. Differentiate this quadratic to obtain the linear
regenerative force; no post-hoc Sinkhorn modification of a parameter step. -/
theorem cycle_geometry (m : ℝ) (hm0 : -1 ≤ m) (hm1 : m ≤ 1) :
    (∀ i j, 0 ≤ allocation m i j) ∧
    (∀ i, allocation m i 0 + allocation m i 1 = 1) ∧
    (∀ j, allocation m 0 j + allocation m 1 j = 1) ∧
    (allocation m 0 0) ^ 2 + (allocation m 0 1) ^ 2 +
      (allocation m 1 0) ^ 2 + (allocation m 1 1) ^ 2 = 1 + m ^ 2 := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · intro i j
    unfold allocation
    split_ifs <;> linarith
  · intro i
    refine Fin.cases ?_ (fun j => ?_) i
    · norm_num [allocation]
      ring
    · have hj : j = 0 := Subsingleton.elim _ _
      subst j
      norm_num [allocation]
      ring
  · intro j
    refine Fin.cases ?_ (fun i => ?_) j
    · norm_num [allocation]
      ring
    · have hi : i = 0 := Subsingleton.elim _ _
      subst i
      norm_num [allocation]
      ring
  · norm_num [allocation]
    ring

/-- In the SAME recursion, the state stays legal and reserve stays above a
strictly positive chosen floor whenever supply supports the declared worst
use. The last identity is the actual replenishment/use ledger. -/
theorem joint_step_invariant (q coupling cueScale refill useRate m reserve cue noise floor : ℝ)
    (_hs : 0 ≤ refill) (hu : 0 ≤ useRate) (hsu : refill + useRate ≤ 1)
    (hf0 : 0 ≤ floor) (hfr : floor ≤ reserve) (hr1 : reserve ≤ 1)
    (hfuel : (refill + useRate) * floor ≤ refill)
    (hm0 : -1 ≤ m) (hm1 : m ≤ 1) :
    let z := step q coupling cueScale refill useRate m reserve cue noise
    (-1 ≤ z.1 ∧ z.1 ≤ 1) ∧ (floor ≤ z.2 ∧ z.2 ≤ 1) ∧
      z.2 + useRate * |m| * reserve = reserve + refill * (1 - reserve) := by
  dsimp only [step]
  have hmabs : |m| ≤ 1 := abs_le.mpr ⟨by linarith, hm1⟩
  have hr0 : 0 ≤ reserve := hf0.trans hfr
  have hd0 : 0 ≤ useRate * |m| := mul_nonneg hu (abs_nonneg m)
  have hd1 : useRate * |m| ≤ useRate := by nlinarith
  have hremain : 0 ≤ 1 - refill - useRate := by linarith
  have hbase := mul_le_mul_of_nonneg_left hfr hremain
  have huse := mul_le_mul_of_nonneg_right hd1 hr0
  have hup := mul_nonneg (show 0 ≤ 1 - refill by linarith) (sub_nonneg.mpr hr1)
  have hcharge := mul_nonneg hd0 hr0
  refine ⟨clip_bounds _, ?_, ?_⟩
  · constructor <;> nlinarith
  · ring

/-- Within the positive memory basin, bounded contrary input and bounded
noise still give a uniform restoring margin. This is active regeneration
against q<1, not a zero-update gate. The reserve lower bound is maintained by
joint_step_invariant, rather than assumed afresh at each time step. -/
theorem quiet_restoration (q coupling cueScale refill useRate m reserve cue noise
    floor threshold cueBound noiseBound margin : ℝ)
    (hk : 0 ≤ coupling) (he : 0 ≤ cueScale)
    (hfloor : floor ≤ reserve) (hr0 : 0 ≤ reserve) (hr1 : reserve ≤ 1)
    (hth : 0 ≤ threshold) (hm : threshold ≤ m)
    (hg : 1 ≤ q + coupling * floor)
    (hcue0 : 0 ≤ cueBound) (hcue : |cue| ≤ cueBound)
    (hnoise : |noise| ≤ noiseBound)
    (hmargin : margin ≤ (q + coupling * floor - 1) * threshold -
      cueScale * cueBound - noiseBound) :
    min 1 (m + margin) ≤
      (step q coupling cueScale refill useRate m reserve cue noise).1 := by
  dsimp only [step]
  apply clip_lower
  have hm0 : 0 ≤ m := hth.trans hm
  have h1 := mul_le_mul_of_nonneg_left hfloor hk
  have h2 := mul_le_mul_of_nonneg_right h1 hm0
  have h3 := mul_le_mul_of_nonneg_left hm (show 0 ≤ q + coupling * floor - 1 by linarith)
  have hc := (abs_le.mp hcue).1
  have hn := (abs_le.mp hnoise).1
  have h4 := mul_le_mul_of_nonneg_left hc (mul_nonneg he hr0)
  have h5 := mul_nonneg (mul_nonneg he hcue0) (sub_nonneg.mpr hr1)
  nlinarith

/-- A sufficiently contrary supplied cue overcomes the SAME recurrent force.
No separate release bit or duration counter is added. Repeated such steps
cross the old basin once the accumulated margin exceeds its width. -/
theorem strong_cue_release (q coupling cueScale refill useRate m reserve cue noise
    floor cueSize noiseBound margin : ℝ)
    (hk : 0 ≤ coupling) (he : 0 ≤ cueScale)
    (hfloor : floor ≤ reserve) (hr0 : 0 ≤ reserve)
    (hm1 : m ≤ 1) (hg : 1 ≤ q + coupling * floor)
    (hsize : coupling ≤ cueScale * cueSize) (hcue : cue ≤ -cueSize)
    (hnoise : |noise| ≤ noiseBound)
    (hmargin : margin ≤ 1 - q + (cueScale * cueSize - coupling) * floor - noiseBound) :
    (step q coupling cueScale refill useRate m reserve cue noise).1 ≤
      max (-1) (m - margin) := by
  dsimp only [step]
  apply max_le_max_left
  apply (min_le_right 1 _).trans
  have h1 := mul_le_mul_of_nonneg_left hfloor hk
  have hgain : 0 ≤ q + coupling * reserve - 1 := by linarith
  have h2 := mul_le_mul_of_nonneg_left hm1 hgain
  have hc := mul_le_mul_of_nonneg_left hcue (mul_nonneg he hr0)
  have hn := (abs_le.mp hnoise).2
  have h3 := mul_le_mul_of_nonneg_left hfloor (show 0 ≤ cueScale * cueSize - coupling by linarith)
  nlinarith

end CGB.ConservativeCycleReview

namespace CGB.ConservativeCycleReview

private theorem clip_neg (x : ℝ) : clip (-x) = -clip x := by
  simp only [clip, min_def, max_def]
  split_ifs <;> linarith

/-- One explicit admissible normalization, not a fitted hyperparameter set.
The rational margins leave room for bounded rounding/input disturbances.
This is the same kernel in all branches; no hold/release state machine. -/
noncomputable def normalizedStep (m reserve cue noise : ℝ) : ℝ × ℝ :=
  step (1 / 2) 2 1 (1 / 4) (1 / 4) m reserve cue noise

theorem normalized_joint_contract (m reserve cue noise : ℝ)
    (hm0 : -1 ≤ m) (hm1 : m ≤ 1)
    (hr0 : (1 / 2 : ℝ) ≤ reserve) (hr1 : reserve ≤ 1)
    (hn : |noise| ≤ (1 / 32 : ℝ)) :
    let z := normalizedStep m reserve cue noise
    ((-1 ≤ z.1 ∧ z.1 ≤ 1) ∧ ((1 / 2 : ℝ) ≤ z.2 ∧ z.2 ≤ 1)) ∧
    ((1 / 4 : ℝ) ≤ m → |cue| ≤ (1 / 32 : ℝ) → min 1 (m + 1 / 16) ≤ z.1) ∧
    (m ≤ -(1 / 4 : ℝ) → |cue| ≤ (1 / 32 : ℝ) → z.1 ≤ max (-1) (m - 1 / 16)) ∧
    (cue ≤ -2 → z.1 ≤ max (-1) (m - 1 / 4)) := by
  dsimp only [normalizedStep]
  have hi := joint_step_invariant (1 / 2) 2 1 (1 / 4) (1 / 4)
    m reserve cue noise (1 / 2) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) hr0 hr1 (by norm_num) hm0 hm1
  refine ⟨⟨hi.1, hi.2.1⟩, ?_, ?_, ?_⟩
  · intro hm hc
    exact quiet_restoration (1 / 2) 2 1 (1 / 4) (1 / 4)
      m reserve cue noise (1 / 2) (1 / 4) (1 / 32) (1 / 32) (1 / 16)
      (by norm_num) (by norm_num) hr0 (by linarith) hr1
      (by norm_num) hm (by norm_num) (by norm_num) hc hn (by norm_num)
  · intro hm hc
    have hp := quiet_restoration (1 / 2) 2 1 (1 / 4) (1 / 4)
      (-m) reserve (-cue) (-noise) (1 / 2) (1 / 4) (1 / 32) (1 / 32) (1 / 16)
      (by norm_num) (by norm_num) hr0 (by linarith) hr1
      (by norm_num) (by linarith) (by norm_num) (by norm_num)
      (by simpa using hc) (by simpa using hn) (by norm_num)
    dsimp only [step] at hp ⊢
    have heq : ((1 / 2 : ℝ) + 2 * reserve) * -m + 1 * reserve * -cue + -noise =
        -(((1 / 2 : ℝ) + 2 * reserve) * m + 1 * reserve * cue + noise) := by ring
    rw [heq, clip_neg] at hp
    have hmin : -min (1 : ℝ) (-m + 1 / 16) = max (-1) (m - 1 / 16) := by
      simp only [min_def, max_def]
      split_ifs <;> linarith
    linarith
  · intro hc
    exact strong_cue_release (1 / 2) 2 1 (1 / 4) (1 / 4)
      m reserve cue noise (1 / 2) 2 (1 / 32) (1 / 4)
      (by norm_num) (by norm_num) hr0 (by linarith) hm1
      (by norm_num) (by norm_num) hc hn (by norm_num)

/-- Quantitative finite-time bridge: repeated restoring progress reaches the
capacity endpoint by any N whose margin covers the initial deficit. This
does not assume that a trajectory merely visits the endpoint eventually. -/
theorem restoration_by_budget (x : ℕ → ℝ) (margin : ℝ) (hm : 0 ≤ margin)
    (hbound : ∀ t, x t ≤ 1)
    (hprogress : ∀ t, min 1 (x t + margin) ≤ x (t + 1))
    (N : ℕ) (hN : 1 ≤ x 0 + (N : ℝ) * margin) : x N = 1 := by
  have lower : ∀ n : ℕ, min 1 (x 0 + (n : ℝ) * margin) ≤ x n := by
    intro n
    induction n with
    | zero => simp
    | succ n ih =>
      have hs := hprogress n
      by_cases h : 1 ≤ x 0 + (n : ℝ) * margin
      · have hx : 1 ≤ x n := by simpa [min_eq_left h] using ih
        have hx' : 1 ≤ x n + margin := by linarith
        have hn : 1 ≤ x (n + 1) := by simpa [min_eq_left hx'] using hs
        exact (min_le_left _ _).trans hn
      · have hle : x 0 + (n : ℝ) * margin ≤ 1 := le_of_lt (lt_of_not_ge h)
        have hx : x 0 + (n : ℝ) * margin ≤ x n := by
          simpa [min_eq_right hle] using ih
        have hz : x 0 + ((n + 1 : ℕ) : ℝ) * margin ≤ x n + margin := by
          push_cast
          linarith
        exact (min_le_min_left 1 hz).trans hs
  have hn := lower N
  rw [min_eq_left hN] at hn
  exact le_antisymm (hbound N) hn

noncomputable def trajectory (cue noise : ℕ → ℝ) (initial : ℝ × ℝ) : ℕ → ℝ × ℝ
  | 0 => initial
  | n + 1 =>
    let old := trajectory cue noise initial n
    normalizedStep old.1 old.2 (cue n) (noise n)

/-- Closed trajectory certificate for one actual recursion and all input
sequences in the declared disturbance class. Quiet-basin perturbations are
repaired within 12 steps; persistent strong contrary evidence switches even
the most deeply held state within 8 steps. Budgets/resources remain legal
throughout. These are local kernel bounds, not language quality evidence. -/
theorem normalized_trajectory_contract (cue noise : ℕ → ℝ) (m₀ r₀ : ℝ)
    (hm0 : -1 ≤ m₀) (hm1 : m₀ ≤ 1)
    (hr0 : (1 / 2 : ℝ) ≤ r₀) (hr1 : r₀ ≤ 1)
    (hnoise : ∀ t, |noise t| ≤ (1 / 32 : ℝ)) :
    let z := trajectory cue noise (m₀, r₀)
    (∀ t, ((-1 ≤ (z t).1 ∧ (z t).1 ≤ 1) ∧
      ((1 / 2 : ℝ) ≤ (z t).2 ∧ (z t).2 ≤ 1))) ∧
    (((1 / 4 : ℝ) ≤ m₀ ∧ ∀ t, |cue t| ≤ (1 / 32 : ℝ)) → (z 12).1 = 1) ∧
    ((∀ t, cue t ≤ -2) → (z 8).1 = -1) := by
  let z := trajectory cue noise (m₀, r₀)
  have hz : ∀ t, z (t + 1) = normalizedStep (z t).1 (z t).2 (cue t) (noise t) :=
    fun _ => rfl
  have legal : ∀ t, ((-1 ≤ (z t).1 ∧ (z t).1 ≤ 1) ∧
      ((1 / 2 : ℝ) ≤ (z t).2 ∧ (z t).2 ≤ 1)) := by
    intro t
    induction t with
    | zero => exact ⟨⟨hm0, hm1⟩, hr0, hr1⟩
    | succ t ih =>
      rw [hz]
      exact (normalized_joint_contract _ _ _ _ ih.1.1 ih.1.2 ih.2.1 ih.2.2 (hnoise t)).1
  refine ⟨legal, ?_, ?_⟩
  · rintro ⟨hstart, hquiet⟩
    have keeps : ∀ t, (1 / 4 : ℝ) ≤ (z t).1 := by
      intro t
      induction t with
      | zero => exact hstart
      | succ t ih =>
        have hl := legal t
        have hp := (normalized_joint_contract _ _ _ _ hl.1.1 hl.1.2
          hl.2.1 hl.2.2 (hnoise t)).2.1 ih (hquiet t)
        have hm : (1 / 4 : ℝ) ≤ min 1 ((z t).1 + 1 / 16) :=
          le_min (by norm_num) (by linarith)
        rw [hz]
        exact hm.trans hp
    have progress : ∀ t, min 1 ((z t).1 + 1 / 16) ≤ (z (t + 1)).1 := by
      intro t
      rw [hz]
      have hl := legal t
      exact (normalized_joint_contract _ _ _ _ hl.1.1 hl.1.2
        hl.2.1 hl.2.2 (hnoise t)).2.1 (keeps t) (hquiet t)
    exact restoration_by_budget (fun t => (z t).1) (1 / 16)
      (by norm_num) (fun t => (legal t).1.2) progress 12
      (by norm_num [z, trajectory]; linarith)
  · intro hcontrary
    have progress : ∀ t, min 1 (-(z t).1 + 1 / 4) ≤ -(z (t + 1)).1 := by
      intro t
      have hl := legal t
      have hp := (normalized_joint_contract _ _ _ _ hl.1.1 hl.1.2
        hl.2.1 hl.2.2 (hnoise t)).2.2.2 (hcontrary t)
      have hdual : -max (-1 : ℝ) ((z t).1 - 1 / 4) =
          min 1 (-(z t).1 + 1 / 4) := by
        simp only [min_def, max_def]
        split_ifs <;> linarith
      rw [hz]
      rw [← hdual]
      linarith
    have heq := restoration_by_budget (fun t => -(z t).1) (1 / 4)
      (by norm_num) (fun t => by have h := (legal t).1.1; linarith)
      progress 8 (by norm_num [z, trajectory]; linarith)
    linarith

end CGB.ConservativeCycleReview

namespace CGB.ConservativeCycleReview

noncomputable def gradientWork (m : ℝ) (g : Fin 2 → Fin 2 → ℝ) : ℝ :=
  allocation m 0 0 * (g 0 0) ^ 2 + allocation m 0 1 * (g 0 1) ^ 2 +
  allocation m 1 0 * (g 1 0) ^ 2 + allocation m 1 1 * (g 1 1) ^ 2

/-- The supplied gradient determines the EXACT finite response of the
allocated first-order descent work to a cycle edit. Thus the cue direction
has an objective-derived source, rather than being an unrelated trigger.
Its statistical reliability and normalization are not assumed proved here. -/
theorem gradient_cycle_response (m change : ℝ) (g : Fin 2 → Fin 2 → ℝ) :
    gradientWork (m + change) g - gradientWork m g = change / 2 *
      ((g 0 0) ^ 2 + (g 1 1) ^ 2 - (g 0 1) ^ 2 - (g 1 0) ^ 2) := by
  norm_num [gradientWork, allocation]
  ring

private theorem scalar_readout_safe (a t L g : ℝ) (ha0 : 0 ≤ a) (ha1 : a ≤ 1)
    (ht : 0 ≤ t) (hL : 0 ≤ L) (hstep : L * t ≤ 1) :
    t / 2 * a * g ^ 2 ≤ g * (t * a * g) - L / 2 * (t * a * g) ^ 2 := by
  have hlt : 0 ≤ L * t := mul_nonneg hL ht
  have hla := mul_le_mul_of_nonneg_left ha1 hlt
  have hm : 0 ≤ 1 - L * t * a := by linarith
  have hp := mul_nonneg (mul_nonneg (mul_nonneg ht ha0) (sq_nonneg g)) hm
  nlinarith

/-- The parameter readout is derived WITH its resource mobility, not
renormalized afterward. For u_ij=t*A_ij*g_ij and an L-smooth objective's
quadratic upper bound, t*L<=1 leaves at least half the first-order gain.
Together with normalized_trajectory_contract this covers the actual step
for every evolving legal allocation, rather than only the raw proposal. -/
theorem allocated_readout_safe (m t L : ℝ) (g : Fin 2 → Fin 2 → ℝ)
    (hm0 : -1 ≤ m) (hm1 : m ≤ 1) (ht : 0 ≤ t) (hL : 0 ≤ L)
    (hstep : L * t ≤ 1) :
    let u := fun i j => t * allocation m i j * g i j
    t / 2 * gradientWork m g ≤
      (g 0 0 * u 0 0 + g 0 1 * u 0 1 + g 1 0 * u 1 0 + g 1 1 * u 1 1) -
      L / 2 * ((u 0 0) ^ 2 + (u 0 1) ^ 2 + (u 1 0) ^ 2 + (u 1 1) ^ 2) := by
  have lo := (cycle_geometry m hm0 hm1).1
  have upper : ∀ i j, allocation m i j ≤ 1 := by
    intro i j
    dsimp [allocation]
    split_ifs <;> linarith
  have h00 := scalar_readout_safe _ _ _ (g 0 0) (lo 0 0) (upper 0 0) ht hL hstep
  have h01 := scalar_readout_safe _ _ _ (g 0 1) (lo 0 1) (upper 0 1) ht hL hstep
  have h10 := scalar_readout_safe _ _ _ (g 1 0) (lo 1 0) (upper 1 0) ht hL hstep
  have h11 := scalar_readout_safe _ _ _ (g 1 1) (lo 1 1) (upper 1 1) ht hL hstep
  dsimp [gradientWork]
  nlinarith

end CGB.ConservativeCycleReview

namespace CGB.ConservativeCycleReview

/-- Coupling and bounded disturbances do not allow a stale phase to suppress
almost all of the CURRENT descent opportunity once the cue is the relative
work contrast. No positive per-coordinate floor is imposed. -/
theorem relative_contrast_coverage (m reserve contrast noise : ℝ)
    (hm0 : -1 ≤ m) (hm1 : m ≤ 1)
    (hr0 : (1 / 2 : ℝ) ≤ reserve) (hr1 : reserve ≤ 1)
    (hc0 : -1 ≤ contrast) (hc1 : contrast ≤ 1)
    (hn : |noise| ≤ (1 / 32 : ℝ)) :
    (1 / 16 : ℝ) ≤
      (1 + (normalizedStep m reserve (2 * contrast) noise).1 * contrast) / 2 := by
  let raw := ((1 / 2 : ℝ) + 2 * reserve) * m + 1 * reserve * (2 * contrast) + noise
  change (1 / 16 : ℝ) ≤ (1 + clip raw * contrast) / 2
  have hz := clip_bounds raw
  have hn0 := (abs_le.mp hn).1
  have hn1 := (abs_le.mp hn).2
  have hg : 0 ≤ (1 / 2 : ℝ) + 2 * reserve := by linarith
  have hr : 0 ≤ reserve := by linarith
  by_cases hpos : (7 / 8 : ℝ) ≤ contrast
  · have hm := mul_le_mul_of_nonneg_left hm0 hg
    have hc := mul_le_mul_of_nonneg_left hpos (show 0 ≤ 2 * reserve by linarith)
    have hlo : -(25 / 32 : ℝ) ≤ raw := by dsimp [raw]; nlinarith
    have hzlo : -(25 / 32 : ℝ) ≤ clip raw := by
      have h := clip_lower hlo
      norm_num at h
      exact h
    have hp := mul_nonneg (show 0 ≤ clip raw + 25 / 32 by linarith)
      (show 0 ≤ contrast by linarith)
    nlinarith
  · by_cases hneg : contrast ≤ -(7 / 8 : ℝ)
    · have hm := mul_le_mul_of_nonneg_left hm1 hg
      have hc := mul_le_mul_of_nonneg_left hneg (show 0 ≤ 2 * reserve by linarith)
      have hhi : raw ≤ (25 / 32 : ℝ) := by dsimp [raw]; nlinarith
      have hzhi : clip raw ≤ (25 / 32 : ℝ) := by
        exact max_le (by norm_num) ((min_le_right 1 raw).trans hhi)
      have hp := mul_nonneg (show 0 ≤ 25 / 32 - clip raw by linarith)
        (show 0 ≤ -contrast by linarith)
      nlinarith
    · by_cases hsign : 0 ≤ contrast
      · have hp := mul_nonneg (show 0 ≤ clip raw + 1 by linarith) hsign
        nlinarith
      · have hp := mul_nonneg (show 0 ≤ 1 - clip raw by linarith)
          (show 0 ≤ -contrast by linarith)
        nlinarith

noncomputable def relativeGradientCue (g : Fin 2 → Fin 2 → ℝ) : ℝ :=
  let diagonal := (g 0 0) ^ 2 + (g 1 1) ^ 2
  let cross := (g 0 1) ^ 2 + (g 1 0) ^ 2
  if diagonal + cross = 0 then 0 else 2 * ((diagonal - cross) / (diagonal + cross))

/-- Exact mathematical encoder, with nearest-grid ties toward positive
infinity. A finite/floating implementation still needs its own refinement.
The encoder is not an additional persistent field or a learned threshold. -/
noncomputable def encodedGradientCue (g : Fin 2 → Fin 2 → ℝ) : ℤ :=
  Int.floor (32 * relativeGradientCue g + 1 / 2)

theorem encoded_gradient_cue_contract (g : Fin 2 → Fin 2 → ℝ) :
    (-64 ≤ encodedGradientCue g ∧ encodedGradientCue g ≤ 64) ∧
    |(encodedGradientCue g : ℝ) / 32 - relativeGradientCue g| ≤ (1 / 64 : ℝ) := by
  let d := (g 0 0) ^ 2 + (g 1 1) ^ 2
  let c := (g 0 1) ^ 2 + (g 1 0) ^ 2
  have hd : 0 ≤ d := by dsimp [d]; positivity
  have hc : 0 ≤ c := by dsimp [c]; positivity
  have hb : -2 ≤ relativeGradientCue g ∧ relativeGradientCue g ≤ 2 := by
    change (-2 ≤ if d + c = 0 then 0 else 2 * ((d - c) / (d + c))) ∧
      (if d + c = 0 then 0 else 2 * ((d - c) / (d + c))) ≤ 2
    split_ifs with hz
    · norm_num
    · have hs : 0 < d + c := by positivity
      have hlo : -1 ≤ (d - c) / (d + c) := by rw [le_div_iff₀ hs]; linarith
      have hhi : (d - c) / (d + c) ≤ 1 := by rw [div_le_iff₀ hs]; linarith
      constructor <;> linarith
  have hlo : (-64 : ℤ) ≤ encodedGradientCue g := by
    apply Int.le_floor.mpr
    norm_num
    linarith [hb.1]
  have hhi : encodedGradientCue g < (65 : ℤ) := by
    apply Int.floor_lt.mpr
    norm_num
    linarith [hb.2]
  have hf0 := Int.floor_le (32 * relativeGradientCue g + (1 / 2 : ℝ))
  have hf1 := Int.lt_floor_add_one (32 * relativeGradientCue g + (1 / 2 : ℝ))
  refine ⟨⟨hlo, by omega⟩, ?_⟩
  dsimp only [encodedGradientCue]
  exact abs_le.mpr ⟨by linarith, by linarith⟩

/-- Exact-gradient interface for the same regenerative step: at least 1/16
of total squared-gradient work remains available, regardless of the old
phase. This addresses useful current participation, not future retention
or a proof that each coordinate must update on every step. -/
theorem relative_gradient_coverage (m reserve noise : ℝ) (g : Fin 2 → Fin 2 → ℝ)
    (hm0 : -1 ≤ m) (hm1 : m ≤ 1)
    (hr0 : (1 / 2 : ℝ) ≤ reserve) (hr1 : reserve ≤ 1)
    (hn : |noise| ≤ (1 / 32 : ℝ)) :
    let next := (normalizedStep m reserve (relativeGradientCue g) noise).1
    ((g 0 0) ^ 2 + (g 1 1) ^ 2 + (g 0 1) ^ 2 + (g 1 0) ^ 2) / 16 ≤
      gradientWork next g := by
  dsimp only
  let d := (g 0 0) ^ 2 + (g 1 1) ^ 2
  let c := (g 0 1) ^ 2 + (g 1 0) ^ 2
  have hsum : (g 0 0) ^ 2 + (g 1 1) ^ 2 + (g 0 1) ^ 2 + (g 1 0) ^ 2 = d + c := by
    dsimp [d, c]
    ring
  rw [hsum]
  have hd : 0 ≤ d := by dsimp [d]; positivity
  have hc : 0 ≤ c := by dsimp [c]; positivity
  have hw : ∀ x, gradientWork x g = ((d + c) + x * (d - c)) / 2 := by
    intro x
    norm_num [gradientWork, allocation, d, c]
    ring
  by_cases hz : d + c = 0
  · have hd0 : d = 0 := by linarith
    have hc0 : c = 0 := by linarith
    rw [hw, hd0, hc0]
    norm_num
  · have hs : 0 < d + c := by positivity
    let contrast := (d - c) / (d + c)
    have hlow : -1 ≤ contrast := by
      dsimp [contrast]
      rw [le_div_iff₀ hs]
      linarith
    have hhigh : contrast ≤ 1 := by
      dsimp [contrast]
      rw [div_le_iff₀ hs]
      linarith
    have heq : relativeGradientCue g = 2 * contrast := by
      dsimp [relativeGradientCue, contrast, d, c] at *
      rw [ite_eq_right hz]
    let next := (normalizedStep m reserve (2 * contrast) noise).1
    have hh := relative_contrast_coverage m reserve contrast noise hm0 hm1 hr0 hr1 hlow hhigh hn
    have hm := mul_le_mul_of_nonneg_right hh hs.le
    have hratio : contrast * (d + c) = d - c := div_mul_cancel₀ _ hz
    have hprod := congrArg (fun x => next * x) hratio
    rw [heq, hw]
    change (d + c) / 16 ≤ ((d + c) + next * (d - c)) / 2
    change (1 / 16 : ℝ) * (d + c) ≤ ((1 + next * contrast) / 2) * (d + c) at hm
    nlinarith

/-- A complete exact-gradient parameter step, including the OLD reserve,
NEW regenerated allocation and actual final readout. Its quadratic decrease
is bounded below uniformly over the old phase; the state update simultaneously
preserves both resource budgets and the reserve invariant. This does not
ascribe weight restoration to a multiplicative mobility. -/
theorem exact_optimizer_step (m reserve noise η L : ℝ) (g : Fin 2 → Fin 2 → ℝ)
    (hm0 : -1 ≤ m) (hm1 : m ≤ 1)
    (hr0 : (1 / 2 : ℝ) ≤ reserve) (hr1 : reserve ≤ 1)
    (hn : |noise| ≤ (1 / 32 : ℝ))
    (hη : 0 ≤ η) (hL : 0 ≤ L) (hηL : L * η ≤ 1) :
    let z := normalizedStep m reserve (relativeGradientCue g) noise
    let u := fun i j => η * reserve * allocation z.1 i j * g i j
    ((-1 ≤ z.1 ∧ z.1 ≤ 1) ∧ ((1 / 2 : ℝ) ≤ z.2 ∧ z.2 ≤ 1)) ∧
    η / 64 * ((g 0 0) ^ 2 + (g 1 1) ^ 2 + (g 0 1) ^ 2 + (g 1 0) ^ 2) ≤
      (g 0 0 * u 0 0 + g 0 1 * u 0 1 + g 1 0 * u 1 0 + g 1 1 * u 1 1) -
      L / 2 * ((u 0 0) ^ 2 + (u 0 1) ^ 2 + (u 1 0) ^ 2 + (u 1 1) ^ 2) := by
  dsimp only
  let z := normalizedStep m reserve (relativeGradientCue g) noise
  have hi := (normalized_joint_contract m reserve (relativeGradientCue g) noise
    hm0 hm1 hr0 hr1 hn).1
  have hc := relative_gradient_coverage m reserve noise g hm0 hm1 hr0 hr1 hn
  have hr : 0 ≤ reserve := by linarith
  have ht : 0 ≤ η * reserve := mul_nonneg hη hr
  have hlt : L * (η * reserve) ≤ 1 := by
    have h := mul_le_mul_of_nonneg_left hr1 (mul_nonneg hL hη)
    nlinarith
  have hd := allocated_readout_safe z.1 (η * reserve) L g hi.1.1 hi.1.2 ht hL hlt
  have hc' := mul_le_mul_of_nonneg_left hc (show 0 ≤ η * reserve / 2 by positivity)
  have hs : 0 ≤ (g 0 0) ^ 2 + (g 1 1) ^ 2 + (g 0 1) ^ 2 + (g 1 0) ^ 2 :=
    by positivity
  have hfloor := mul_nonneg (mul_nonneg hη (show 0 ≤ reserve - 1 / 2 by linarith)) hs
  exact ⟨hi, by dsimp [z] at *; nlinarith⟩

end CGB.ConservativeCycleReview

namespace CGB.ConservativeCycleReview

private theorem perturbed_scalar_readout_safe (a t L estimate actual : ℝ)
    (ha0 : 0 ≤ a) (ha1 : a ≤ 1) (ht : 0 ≤ t) (hL : 0 ≤ L) (hstep : L * t ≤ 1) :
    t / 4 * a * estimate ^ 2 - t * (actual - estimate) ^ 2 ≤
      actual * (t * a * estimate) - L / 2 * (t * a * estimate) ^ 2 := by
  have hbase := scalar_readout_safe a t L estimate ha0 ha1 ht hL hstep
  have hcross := mul_nonneg (mul_nonneg ht ha0)
    (sq_nonneg (actual - estimate + estimate / 2))
  have herror := mul_nonneg (mul_nonneg ht (sub_nonneg.mpr ha1))
    (sq_nonneg (actual - estimate))
  nlinarith

/-- Conditional estimated-gradient guarantee for the FINAL step. The cue is
formed from the supplied estimate, while actual may be a different gradient.
The relative error and the smooth upper model are explicit obligations; no
claim that a batch, variance estimate or trust controller satisfies them.
The same state/resource recursion still applies. -/
theorem noisy_optimizer_step (m reserve noise η L before after : ℝ)
    (estimate actual : Fin 2 → Fin 2 → ℝ)
    (hm0 : -1 ≤ m) (hm1 : m ≤ 1)
    (hr0 : (1 / 2 : ℝ) ≤ reserve) (hr1 : reserve ≤ 1)
    (hn : |noise| ≤ (1 / 32 : ℝ))
    (hη : 0 ≤ η) (hL : 0 ≤ L) (hηL : L * η ≤ 1)
    (herror : (actual 0 0 - estimate 0 0) ^ 2 + (actual 0 1 - estimate 0 1) ^ 2 +
      (actual 1 0 - estimate 1 0) ^ 2 + (actual 1 1 - estimate 1 1) ^ 2 ≤
      ((estimate 0 0) ^ 2 + (estimate 1 1) ^ 2 +
        (estimate 0 1) ^ 2 + (estimate 1 0) ^ 2) / 128)
    (hmodel :
      let z := normalizedStep m reserve (relativeGradientCue estimate) noise
      let u := fun i j => η * reserve * allocation z.1 i j * estimate i j
      after ≤ before -
        (actual 0 0 * u 0 0 + actual 0 1 * u 0 1 +
          actual 1 0 * u 1 0 + actual 1 1 * u 1 1) +
        L / 2 * ((u 0 0) ^ 2 + (u 0 1) ^ 2 + (u 1 0) ^ 2 + (u 1 1) ^ 2)) :
    let z := normalizedStep m reserve (relativeGradientCue estimate) noise
    ((-1 ≤ z.1 ∧ z.1 ≤ 1) ∧ ((1 / 2 : ℝ) ≤ z.2 ∧ z.2 ≤ 1)) ∧
    after + η / 256 * ((estimate 0 0) ^ 2 + (estimate 1 1) ^ 2 +
      (estimate 0 1) ^ 2 + (estimate 1 0) ^ 2) ≤ before := by
  dsimp only at *
  let z := normalizedStep m reserve (relativeGradientCue estimate) noise
  have hi := (normalized_joint_contract m reserve (relativeGradientCue estimate) noise
    hm0 hm1 hr0 hr1 hn).1
  have hc := relative_gradient_coverage m reserve noise estimate hm0 hm1 hr0 hr1 hn
  have hr : 0 ≤ reserve := by linarith
  have ht : 0 ≤ η * reserve := mul_nonneg hη hr
  have hlt : L * (η * reserve) ≤ 1 := by
    have h := mul_le_mul_of_nonneg_left hr1 (mul_nonneg hL hη)
    nlinarith
  have lo := (cycle_geometry z.1 hi.1.1 hi.1.2).1
  have upper : ∀ i j, allocation z.1 i j ≤ 1 := by
    intro i j
    dsimp [allocation]
    split_ifs <;> linarith [hi.1.1, hi.1.2]
  have h00 := perturbed_scalar_readout_safe _ _ _ (estimate 0 0) (actual 0 0)
    (lo 0 0) (upper 0 0) ht hL hlt
  have h01 := perturbed_scalar_readout_safe _ _ _ (estimate 0 1) (actual 0 1)
    (lo 0 1) (upper 0 1) ht hL hlt
  have h10 := perturbed_scalar_readout_safe _ _ _ (estimate 1 0) (actual 1 0)
    (lo 1 0) (upper 1 0) ht hL hlt
  have h11 := perturbed_scalar_readout_safe _ _ _ (estimate 1 1) (actual 1 1)
    (lo 1 1) (upper 1 1) ht hL hlt
  have hwork := mul_le_mul_of_nonneg_left hc (show 0 ≤ η * reserve / 4 by positivity)
  have he := mul_le_mul_of_nonneg_left herror ht
  have hs : 0 ≤ (estimate 0 0) ^ 2 + (estimate 1 1) ^ 2 +
      (estimate 0 1) ^ 2 + (estimate 1 0) ^ 2 := by positivity
  have hf := mul_nonneg (mul_nonneg hη (show 0 ≤ reserve - 1 / 2 by linarith)) hs
  refine ⟨hi, ?_⟩
  dsimp only [gradientWork] at hwork
  nlinarith only [h00, h01, h10, h11, hwork, he, hf, hmodel]

end CGB.ConservativeCycleReview

namespace CGB.ConservativeCycleReview

/-- Bounded integer realization, phase K/32 and reserve R/16. The cue is C/32.
Rounding is part of the declared algorithm, not lossless real arithmetic.
Int denotes mathematical integers; the bounds below certify small storage
and intermediates instead of selecting a machine INT64 representation. -/
def integerCycleStep (K R C : ℤ) : ℤ × ℤ :=
  (max (-32) (min 32 ((2 * (4 + R) * K + R * C + 8) / 16)),
   R + (19 - R) / 4 - R * |K| / 128)

/-- Intervention on the existing coupling, not a new memory carrier. The
passive arm keeps sensing, phase-dependent allocation, use/recovery and the
same rounding. Only the regenerative 2*R*K term is absent. -/
def controlledCycleStep (feedback : Bool) (K R C : ℤ) : ℤ × ℤ :=
  if feedback then integerCycleStep K R C else
    (max (-32) (min 32 ((8 * K + R * C + 8) / 16)),
     (integerCycleStep K R C).2)

/-- Actual integer state, supply/use ledger, packing bound and arithmetic
range in ONE step. Ceil supply and floor use keep R in [8,16]; no floor on
individual parameter updates is added. The 585 possible state pairs fit in
10 information bits (two ordinary bytes); all intermediates fit INT16. -/
theorem integer_cycle_contract (K R C : ℤ)
    (hk0 : -32 ≤ K) (hk1 : K ≤ 32)
    (hr0 : 8 ≤ R) (hr1 : R ≤ 16) (hc0 : -64 ≤ C) (hc1 : C ≤ 64) :
    let z := integerCycleStep K R C
    (-32 ≤ z.1 ∧ z.1 ≤ 32) ∧ (8 ≤ z.2 ∧ z.2 ≤ 16) ∧
    z.2 + R * |K| / 128 = R + (19 - R) / 4 ∧
    (0 ≤ (z.1 + 32) + 65 * (z.2 - 8) ∧ (z.1 + 32) + 65 * (z.2 - 8) < 585) ∧
    (-2304 ≤ 2 * (4 + R) * K + R * C ∧
      2 * (4 + R) * K + R * C ≤ 2304) ∧
    (0 ≤ R * |K| ∧ R * |K| ≤ 512) := by
  dsimp only [integerCycleStep]
  have hkabs : |K| ≤ 32 := abs_le.mpr ⟨by omega, hk1⟩
  have hp0 : 0 ≤ R * |K| := mul_nonneg (by omega) (abs_nonneg K)
  have hp1 : R * |K| ≤ 32 * R := by
    have h := mul_le_mul_of_nonneg_left hkabs (show 0 ≤ R by omega)
    nlinarith
  have hpool : 8 ≤ R + (19 - R) / 4 - R * |K| / 128 ∧
      R + (19 - R) / 4 - R * |K| / 128 ≤ 16 := by omega
  have hphase : -32 ≤ max (-32) (min 32 ((2 * (4 + R) * K + R * C + 8) / 16)) ∧
      max (-32) (min 32 ((2 * (4 + R) * K + R * C + 8) / 16)) ≤ 32 := by
    exact ⟨le_max_left _ _, max_le (by norm_num) (min_le_left _ _)⟩
  have hscale : 0 ≤ 2 * (4 + R) := by omega
  have hn0 := mul_le_mul_of_nonneg_left hk0 hscale
  have hn1 := mul_le_mul_of_nonneg_left hk1 hscale
  have hn2 := mul_le_mul_of_nonneg_left hc0 (show 0 ≤ R by omega)
  have hn3 := mul_le_mul_of_nonneg_left hc1 (show 0 ≤ R by omega)
  refine ⟨hphase, hpool, ?_, ?_, ?_, ?_⟩
  · omega
  · omega
  · constructor <;> nlinarith
  · constructor <;> nlinarith

private theorem integer_clip_cast (q : ℤ) :
    ((max (-32) (min 32 q) : ℤ) : ℝ) / 32 = clip ((q : ℝ) / 32) := by
  by_cases hlo : q ≤ -32
  · have hhi : q ≤ 32 := by omega
    have hcast : (q : ℝ) ≤ -32 := by exact_mod_cast hlo
    rw [min_eq_right hhi, max_eq_left hlo]
    rw [clip, min_eq_right (by linarith), max_eq_left (by linarith)]
    norm_num
  · by_cases hhi : 32 ≤ q
    · have hcast : (32 : ℝ) ≤ q := by exact_mod_cast hhi
      rw [min_eq_left hhi]
      rw [clip, min_eq_left (by linarith)]
      norm_num
    · have hl : -32 ≤ q := by omega
      have hh : q ≤ 32 := by omega
      have hlr : (-32 : ℝ) ≤ q := by exact_mod_cast hl
      have hhr : (q : ℝ) ≤ 32 := by exact_mod_cast hh
      rw [min_eq_right hh, max_eq_right hl]
      rw [clip, min_eq_right (by linarith), max_eq_right (by linarith)]

/-- The integer phase update REFINES the proved real phase update with a
bounded disturbance. Half of the 1/32 budget pays for phase rounding and
half pays for cue quantization; no claim of lossless quantization is made.
The integer reserve law has its own exact ledger above, rather than being
silently identified with the unrounded G20 recurrence. -/
theorem integer_phase_refines (K R C : ℤ) (cue : ℝ)
    (hr0 : 8 ≤ R) (hr1 : R ≤ 16)
    (hcue : |(C : ℝ) / 32 - cue| ≤ (1 / 64 : ℝ)) :
    ∃ noise : ℝ, |noise| ≤ (1 / 32 : ℝ) ∧
      ((integerCycleStep K R C).1 : ℝ) / 32 =
        (normalizedStep ((K : ℝ) / 32) ((R : ℝ) / 16) cue noise).1 := by
  let N := 2 * (4 + R) * K + R * C
  let q := (N + 8) / 16
  have hdiv0 : 16 * q ≤ N + 8 := by dsimp [q]; omega
  have hdiv1 : N + 8 < 16 * (q + 1) := by dsimp [q]; omega
  have hd0 : 16 * (q : ℝ) ≤ (N : ℝ) + 8 := by exact_mod_cast hdiv0
  have hd1 : (N : ℝ) + 8 < 16 * ((q : ℝ) + 1) := by exact_mod_cast hdiv1
  let rounding := (q : ℝ) / 32 - (N : ℝ) / 512
  have he0 : -(1 / 64 : ℝ) ≤ rounding := by dsimp [rounding]; linarith
  have he1 : rounding ≤ (1 / 64 : ℝ) := by dsimp [rounding]; linarith
  have hr : (0 : ℝ) ≤ (R : ℝ) / 16 := by
    have h : (0 : ℝ) ≤ R := by exact_mod_cast (show 0 ≤ R by omega)
    linarith
  have hrmax : (R : ℝ) / 16 ≤ 1 := by
    have h : (R : ℝ) ≤ 16 := by exact_mod_cast hr1
    linarith
  have hc0 := mul_le_mul_of_nonneg_left (abs_le.mp hcue).1 hr
  have hc1 := mul_le_mul_of_nonneg_left (abs_le.mp hcue).2 hr
  let noise := rounding + (R : ℝ) / 16 * ((C : ℝ) / 32 - cue)
  refine ⟨noise, abs_le.mpr ⟨?_, ?_⟩, ?_⟩
  · dsimp only [noise]
    nlinarith
  · dsimp only [noise]
    nlinarith
  · change ((max (-32) (min 32 q) : ℤ) : ℝ) / 32 =
      clip (((1 / 2 : ℝ) + 2 * ((R : ℝ) / 16)) * ((K : ℝ) / 32) +
        1 * ((R : ℝ) / 16) * cue + noise)
    rw [integer_clip_cast]
    congr 1
    dsimp [noise, rounding, N]
    push_cast
    ring

/-- Representation proof only: both shared fields are recovered exactly.
The compact code has 585 legal values; this is a sufficient encoding, not
a claim that ten bits is necessary for every functional optimizer. -/
theorem integer_code_roundtrip (K R : ℤ) (hk0 : -32 ≤ K) (hk1 : K ≤ 32)
    (hr0 : 8 ≤ R) (hr1 : R ≤ 16) :
    let code := K + 32 + 65 * (R - 8)
    (0 ≤ code ∧ code < 585) ∧ code % 65 - 32 = K ∧ code / 65 + 8 = R := by
  dsimp only
  omega

/-- Joint integer optimizer certificate: the SAME finite recursion provides
legal state, restorative phase retention, contrary-evidence release and a
safe final parameter step. No separate witness is substituted for one of
these properties. Historical FUNCTION preservation and task evidence are
not inferred from phase retention. -/
theorem integer_joint_optimizer_step (K R C : ℤ) (η L : ℝ)
    (g : Fin 2 → Fin 2 → ℝ)
    (hk0 : -32 ≤ K) (hk1 : K ≤ 32)
    (hr0 : 8 ≤ R) (hr1 : R ≤ 16) (hc0 : -64 ≤ C) (hc1 : C ≤ 64)
    (hcue : |(C : ℝ) / 32 - relativeGradientCue g| ≤ (1 / 64 : ℝ))
    (hη : 0 ≤ η) (hL : 0 ≤ L) (hηL : L * η ≤ 1) :
    let z := integerCycleStep K R C
    let next := (z.1 : ℝ) / 32
    let u := fun i j => η * ((R : ℝ) / 16) * allocation next i j * g i j
    ((-32 ≤ z.1 ∧ z.1 ≤ 32) ∧ (8 ≤ z.2 ∧ z.2 ≤ 16)) ∧
    (((1 / 4 : ℝ) ≤ (K : ℝ) / 32 → |relativeGradientCue g| ≤ (1 / 32 : ℝ) →
      min 1 ((K : ℝ) / 32 + 1 / 16) ≤ next)) ∧
    (((K : ℝ) / 32 ≤ -(1 / 4 : ℝ) → |relativeGradientCue g| ≤ (1 / 32 : ℝ) →
      next ≤ max (-1) ((K : ℝ) / 32 - 1 / 16))) ∧
    ((relativeGradientCue g ≤ -2 → next ≤ max (-1) ((K : ℝ) / 32 - 1 / 4))) ∧
    ((2 ≤ relativeGradientCue g → min 1 ((K : ℝ) / 32 + 1 / 4) ≤ next)) ∧
    η / 64 * ((g 0 0) ^ 2 + (g 1 1) ^ 2 + (g 0 1) ^ 2 + (g 1 0) ^ 2) ≤
      (g 0 0 * u 0 0 + g 0 1 * u 0 1 + g 1 0 * u 1 0 + g 1 1 * u 1 1) -
      L / 2 * ((u 0 0) ^ 2 + (u 0 1) ^ 2 + (u 1 0) ^ 2 + (u 1 1) ^ 2) := by
  dsimp only
  have hm0 : (-1 : ℝ) ≤ (K : ℝ) / 32 := by
    have h : (-32 : ℝ) ≤ K := by exact_mod_cast hk0
    linarith
  have hm1 : (K : ℝ) / 32 ≤ 1 := by
    have h : (K : ℝ) ≤ 32 := by exact_mod_cast hk1
    linarith
  have hrlo : (1 / 2 : ℝ) ≤ (R : ℝ) / 16 := by
    have h : (8 : ℝ) ≤ R := by exact_mod_cast hr0
    linarith
  have hrhi : (R : ℝ) / 16 ≤ 1 := by
    have h : (R : ℝ) ≤ 16 := by exact_mod_cast hr1
    linarith
  obtain ⟨noise, hn, heq⟩ := integer_phase_refines K R C (relativeGradientCue g) hr0 hr1 hcue
  have hi := integer_cycle_contract K R C hk0 hk1 hr0 hr1 hc0 hc1
  have hc := normalized_joint_contract ((K : ℝ) / 32) ((R : ℝ) / 16)
    (relativeGradientCue g) noise hm0 hm1 hrlo hrhi hn
  have hd := exact_optimizer_step ((K : ℝ) / 32) ((R : ℝ) / 16) noise η L g
    hm0 hm1 hrlo hrhi hn hη hL hηL
  refine ⟨⟨hi.1, hi.2.1⟩, ?_, ?_, ?_, ?_, ?_⟩
  · intro hm hq
    rw [heq]
    exact hc.2.1 hm hq
  · intro hm hq
    rw [heq]
    exact hc.2.2.1 hm hq
  · intro hq
    rw [heq]
    exact hc.2.2.2 hq
  · intro hq
    rw [heq]
    change min 1 ((K : ℝ) / 32 + 1 / 4) ≤
      clip (((1 / 2 : ℝ) + 2 * ((R : ℝ) / 16)) * ((K : ℝ) / 32) +
        1 * ((R : ℝ) / 16) * relativeGradientCue g + noise)
    apply clip_lower
    have hg : 0 ≤ (1 / 2 : ℝ) + 2 * ((R : ℝ) / 16) - 1 := by linarith
    have hm := mul_le_mul_of_nonneg_left hm0 hg
    have hq' := mul_le_mul_of_nonneg_left hq (show 0 ≤ (R : ℝ) / 16 by linarith)
    have hn0 := (abs_le.mp hn).1
    nlinarith
  · dsimp only at hd
    rw [← heq] at hd
    exact hd.2

end CGB.ConservativeCycleReview

namespace CGB.ConservativeCycleReview

open scoped BigOperators

abbrev Parameters (n : ℕ) := Fin n → Fin 2 → Fin 2 → ℝ

noncomputable def normFour (g : Fin 2 → Fin 2 → ℝ) : ℝ :=
  (g 0 0) ^ 2 + (g 0 1) ^ 2 + (g 1 0) ^ 2 + (g 1 1) ^ 2

noncomputable def dotFour (g u : Fin 2 → Fin 2 → ℝ) : ℝ :=
  g 0 0 * u 0 0 + g 0 1 * u 0 1 + g 1 0 * u 1 0 + g 1 1 * u 1 1

noncomputable def integerReadout {n : ℕ} (K R C : Fin n → ℤ) (η : ℝ)
    (g : Parameters n) : Parameters n := fun b i j =>
  η * ((R b : ℝ) / 16) * allocation
    ((integerCycleStep (K b) (R b) (C b)).1 / 32) i j * g b i j

/-- Composition over any finite number of DISJOINT coordinate blocks. The
smooth upper model is for the whole objective, so off-block curvature is
not silently set to zero. This proves an actual parameter update bound;
the tensor layout must separately realize these coordinates without loss.
Local state restoration/release remains supplied by the very same integer
step, rather than by an independently chosen block witness. -/
theorem finite_block_optimizer_step {n : ℕ} (K R C : Fin n → ℤ) (η L : ℝ)
    (F : Parameters n → ℝ) (θ g : Parameters n)
    (hlegal : ∀ b,
      (-32 ≤ K b ∧ K b ≤ 32) ∧ (8 ≤ R b ∧ R b ≤ 16) ∧
      (-64 ≤ C b ∧ C b ≤ 64) ∧
      |(C b : ℝ) / 32 - relativeGradientCue (g b)| ≤ (1 / 64 : ℝ))
    (hη : 0 ≤ η) (hL : 0 ≤ L) (hηL : L * η ≤ 1)
    (hmodel :
      F (θ - integerReadout K R C η g) ≤ F θ -
        (∑ b, dotFour (g b) (integerReadout K R C η g b)) +
        L / 2 * ∑ b, normFour (integerReadout K R C η g b)) :
    (∀ b, let z := integerCycleStep (K b) (R b) (C b)
      (-32 ≤ z.1 ∧ z.1 ≤ 32) ∧ (8 ≤ z.2 ∧ z.2 ≤ 16)) ∧
    F (θ - integerReadout K R C η g) + η / 64 * (∑ b, normFour (g b)) ≤ F θ := by
  have localCert := fun b => integer_joint_optimizer_step (K b) (R b) (C b) η L (g b)
    (hlegal b).1.1 (hlegal b).1.2 (hlegal b).2.1.1 (hlegal b).2.1.2
    (hlegal b).2.2.1.1 (hlegal b).2.2.1.2 (hlegal b).2.2.2 hη hL hηL
  refine ⟨fun b => (localCert b).1, ?_⟩
  have hsum : (∑ b, η / 64 * normFour (g b)) ≤
      ∑ b, (dotFour (g b) (integerReadout K R C η g b) -
        L / 2 * normFour (integerReadout K R C η g b)) := by
    apply Finset.sum_le_sum
    intro b _
    have hd := (localCert b).2.2.2.2.2
    dsimp only [normFour, dotFour, integerReadout]
    nlinarith only [hd]
  rw [← Finset.mul_sum, Finset.sum_sub_distrib, ← Finset.mul_sum] at hsum
  linarith

structure OptimizerState (n : ℕ) where
  parameter : Parameters n
  phase : Fin n → ℤ
  reserve : Fin n → ℤ

/-- Internal finite-state encoding imposes NO alphabet/range on parameters.
Every supplied real vector is a legal initial parameter vector, unchanged.
This is an interface contract, not a new dynamical mechanism. -/
theorem unrestricted_parameter_initialization {n : ℕ} (θ : Parameters n) :
    ∃ s : OptimizerState n, s.parameter=θ ∧
      (∀ b, s.phase b=0 ∧ s.reserve b=16) ∧
      (∀ b, (-32 ≤ s.phase b ∧ s.phase b ≤ 32) ∧
        (8 ≤ s.reserve b ∧ s.reserve b ≤ 16)) := by
  refine ⟨⟨θ,fun _ => 0,fun _ => 16⟩,rfl,?_,?_⟩ <;> intro b <;> norm_num

noncomputable def advanceOptimizer {n : ℕ} (η : ℝ) (s : OptimizerState n)
    (g : Parameters n) (C : Fin n → ℤ) : OptimizerState n where
  parameter := s.parameter - integerReadout s.phase s.reserve C η g
  phase := fun b => (integerCycleStep (s.phase b) (s.reserve b) (C b)).1
  reserve := fun b => (integerCycleStep (s.phase b) (s.reserve b) (C b)).2

noncomputable def optimizerPath {n : ℕ} (η : ℝ)
    (gradient : ℕ → Parameters n → Parameters n)
    (encodeCue : (Fin 2 → Fin 2 → ℝ) → ℤ) (initial : OptimizerState n) : ℕ → OptimizerState n
  | 0 => initial
  | t + 1 =>
    let s := optimizerPath η gradient encodeCue initial t
    let g := gradient t s.parameter
    advanceOptimizer η s g (fun b => encodeCue (g b))

/-- Actual same-trace FUNCTION recovery/tracking under declared smooth and
PL-type evidence conditions. The loss feedback supplies parameter correction;
the optimizer's own phase regeneration is a distinct property. No old weight
copy or external target is stored in the state. With drift=0, functional error
contracts geometrically. With drift, the bound exposes the tracking cost.
This is a sufficient task class, not a guarantee for every neural loss or a
claim that preserving phase alone preserves an unobserved past function. -/
theorem functional_recovery_tracking {n : ℕ} (η L μ drift : ℝ)
    (gradient : ℕ → Parameters n → Parameters n)
    (encodeCue : (Fin 2 → Fin 2 → ℝ) → ℤ) (initial : OptimizerState n)
    (F : ℕ → Parameters n → ℝ) (floor : ℕ → ℝ)
    (hη : 0 < η) (hμ : 0 < μ) (hμL : μ ≤ L) (hηL : L * η ≤ 1)
    (hinit : ∀ b, (-32 ≤ initial.phase b ∧ initial.phase b ≤ 32) ∧
      (8 ≤ initial.reserve b ∧ initial.reserve b ≤ 16))
    (hcode : ∀ g, (-64 ≤ encodeCue g ∧ encodeCue g ≤ 64) ∧
      |(encodeCue g : ℝ) / 32 - relativeGradientCue g| ≤ (1 / 64 : ℝ))
    (hlower : ∀ t, floor t ≤ F t (optimizerPath η gradient encodeCue initial t).parameter)
    (hmodel : ∀ t,
      let s := optimizerPath η gradient encodeCue initial t
      let g := gradient t s.parameter
      let u := integerReadout s.phase s.reserve (fun b => encodeCue (g b)) η g
      F t (s.parameter - u) ≤ F t s.parameter - (∑ b, dotFour (g b) (u b)) +
        L / 2 * ∑ b, normFour (u b))
    (hpl : ∀ t,
      let s := optimizerPath η gradient encodeCue initial t
      2 * μ * (F t s.parameter - floor t) ≤ ∑ b, normFour (gradient t s.parameter b))
    (hshift : ∀ t,
      let s := (optimizerPath η gradient encodeCue initial (t + 1)).parameter
      F (t + 1) s - floor (t + 1) ≤ F t s - floor t + drift) :
    let z := optimizerPath η gradient encodeCue initial
    let ρ := 1 - η * μ / 32
    (0 ≤ ρ ∧ ρ < 1) ∧
    (∀ t b, (-32 ≤ (z t).phase b ∧ (z t).phase b ≤ 32) ∧
      (8 ≤ (z t).reserve b ∧ (z t).reserve b ≤ 16)) ∧
    (∀ t, 0 ≤ F t (z t).parameter - floor t ∧
      F t (z t).parameter - floor t ≤
        ρ ^ t * (F 0 initial.parameter - floor 0) +
          (1 - ρ ^ t) * (32 * drift / (η * μ))) := by
  let z := optimizerPath η gradient encodeCue initial
  let ρ := 1 - η * μ / 32
  have hz : ∀ t, z (t + 1) = advanceOptimizer η (z t)
      (gradient t (z t).parameter) (fun b => encodeCue (gradient t (z t).parameter b)) :=
    fun _ => rfl
  have legal : ∀ t b, (-32 ≤ (z t).phase b ∧ (z t).phase b ≤ 32) ∧
      (8 ≤ (z t).reserve b ∧ (z t).reserve b ≤ 16) := by
    intro t
    induction t with
    | zero => exact hinit
    | succ t ih =>
      intro b
      have hc := hcode (gradient t (z t).parameter b)
      have hi := integer_cycle_contract _ _ _ (ih b).1.1 (ih b).1.2
        (ih b).2.1 (ih b).2.2 hc.1.1 hc.1.2
      rw [hz]
      exact ⟨hi.1, hi.2.1⟩
  have hp : 0 < η * μ := mul_pos hη hμ
  have hupper : η * μ ≤ 1 := by nlinarith
  have hr : 0 ≤ ρ ∧ ρ < 1 := by dsimp [ρ]; constructor <;> linarith
  have descent : ∀ t,
      F (t + 1) (z (t + 1)).parameter - floor (t + 1) ≤
        ρ * (F t (z t).parameter - floor t) + drift := by
    intro t
    have hc : ∀ b,
        (-32 ≤ (z t).phase b ∧ (z t).phase b ≤ 32) ∧
        (8 ≤ (z t).reserve b ∧ (z t).reserve b ≤ 16) ∧
        (-64 ≤ encodeCue (gradient t (z t).parameter b) ∧
          encodeCue (gradient t (z t).parameter b) ≤ 64) ∧
        |(encodeCue (gradient t (z t).parameter b) : ℝ) / 32 -
          relativeGradientCue (gradient t (z t).parameter b)| ≤ (1 / 64 : ℝ) := by
      intro b
      exact ⟨(legal t b).1, (legal t b).2, hcode _⟩
    have hd := (finite_block_optimizer_step (z t).phase (z t).reserve
      (fun b => encodeCue (gradient t (z t).parameter b)) η L (F t)
      (z t).parameter (gradient t (z t).parameter) hc hη.le (by linarith) hηL (hmodel t)).2
    have hpl' := mul_le_mul_of_nonneg_left (hpl t) (show 0 ≤ η / 64 by positivity)
    have hs := hshift t
    change F t (z (t + 1)).parameter +
      η / 64 * (∑ b, normFour (gradient t (z t).parameter b)) ≤ F t (z t).parameter at hd
    dsimp only at hs
    change η / 64 * (2 * μ * (F t (z t).parameter - floor t)) ≤
      η / 64 * ∑ b, normFour (gradient t (z t).parameter b) at hpl'
    dsimp only [ρ]
    nlinarith only [hd, hpl', hs]
  have balance : (η * μ) * (32 * drift / (η * μ)) = 32 * drift := by
    exact mul_div_cancel₀ _ (ne_of_gt hp)
  have bound : ∀ t, F t (z t).parameter - floor t ≤
      ρ ^ t * (F 0 initial.parameter - floor 0) +
        (1 - ρ ^ t) * (32 * drift / (η * μ)) := by
    intro t
    induction t with
    | zero => simp [z, optimizerPath]
    | succ t ih =>
      have hi := mul_le_mul_of_nonneg_left ih hr.1
      have hs := descent t
      rw [pow_succ]
      dsimp only [ρ] at *
      nlinarith only [hi, hs, balance]
  exact ⟨hr, legal, fun t => ⟨sub_nonneg.mpr (hlower t), bound t⟩⟩

end CGB.ConservativeCycleReview

namespace CGB.ConservativeCycleReview

/-- A causal direction estimate, distinct from the allocation phase. Gains may
depend on the entire observed past. Roundoff is explicit; no independent-noise
assumption or steady-state/bias-correction shortcut is used. -/
noncomputable def filterEstimate (gain observed rounding : ℕ → ℝ) (initial : ℝ) : ℕ → ℝ
  | 0 => initial
  | t + 1 => (1 - gain t) * filterEstimate gain observed rounding initial t +
      gain t * observed (t + 1) + rounding t

/-- The actual recursive estimate supplies its own error envelope. A zero
gain is allowed: its envelope must then pay for drift and rounding. This is
a pathwise world contract, not a claim that pooled empirical variances are
known error bounds. It also covers gains chosen from the current observation. -/
theorem filter_error_envelope (gain observed rounding actual radius drift noise quant : ℕ → ℝ)
    (initial : ℝ) (hinit : |initial - actual 0| ≤ radius 0)
    (hg : ∀ t, 0 ≤ gain t ∧ gain t ≤ 1)
    (hd : ∀ t, |actual (t + 1) - actual t| ≤ drift t)
    (hn : ∀ t, |observed (t + 1) - actual (t + 1)| ≤ noise t)
    (hq : ∀ t, |rounding t| ≤ quant t)
    (hb : ∀ t, (1 - gain t) * (radius t + drift t) +
      gain t * noise t + quant t ≤ radius (t + 1)) (t : ℕ) :
    |filterEstimate gain observed rounding initial t - actual t| ≤ radius t := by
  induction t with
  | zero => exact hinit
  | succ t ih =>
    have hprior : |filterEstimate gain observed rounding initial t - actual (t + 1)| ≤
        radius t + drift t := by
      calc
        _ = |(filterEstimate gain observed rounding initial t - actual t) +
            (actual t - actual (t + 1))| := by congr 1; ring
        _ ≤ |filterEstimate gain observed rounding initial t - actual t| +
            |actual t - actual (t + 1)| := abs_add_le _ _
        _ ≤ radius t + drift t := add_le_add ih (by simpa [abs_sub_comm] using hd t)
    have hdecomp : filterEstimate gain observed rounding initial (t + 1) - actual (t + 1) =
        ((1 - gain t) * (filterEstimate gain observed rounding initial t - actual (t + 1)) +
        gain t * (observed (t + 1) - actual (t + 1))) + rounding t := by
      rw [filterEstimate]; ring
    rw [hdecomp]
    calc
      _ ≤ |(1 - gain t) * (filterEstimate gain observed rounding initial t - actual (t + 1)) +
          gain t * (observed (t + 1) - actual (t + 1))| + |rounding t| := abs_add_le _ _
      _ ≤ |(1 - gain t) * (filterEstimate gain observed rounding initial t - actual (t + 1))| +
          |gain t * (observed (t + 1) - actual (t + 1))| + |rounding t| := by
        have htriangle := abs_add_le ((1 - gain t) *
            (filterEstimate gain observed rounding initial t - actual (t + 1)))
            (gain t * (observed (t + 1) - actual (t + 1)))
        linarith only [htriangle]
      _ = (1 - gain t) * |filterEstimate gain observed rounding initial t - actual (t + 1)| +
          gain t * |observed (t + 1) - actual (t + 1)| + |rounding t| := by
        rw [abs_mul, abs_mul, abs_of_nonneg (by linarith [(hg t).2]), abs_of_nonneg (hg t).1]
      _ ≤ (1 - gain t) * (radius t + drift t) + gain t * noise t + quant t :=
        add_le_add (add_le_add
          (mul_le_mul_of_nonneg_left hprior (by linarith [(hg t).2]))
          (mul_le_mul_of_nonneg_left (hn t) (hg t).1)) (hq t)
      _ ≤ radius (t + 1) := hb t

/-- Estimation, diagonal scaling, regenerated allocation and the FINAL
continuous parameter step share one certificate. Scale is the square root
of the preconditioning denominator: theta changes by estimate / scale^2,
whereas the cue measures work estimate^2 / scale^2. Multiplying an old cue
by a new preconditioner afterward would not implement this law.

The upper model is for the entire step and must include cross-coordinate
curvature. Data-dependent scale and filter/phase correlation are allowed;
the scaled estimation-error condition and upper model must actually hold.
Scale is not asserted to be the Hessian or an accurate online estimate. -/
theorem preconditioned_filter_step (m reserve noise η L before after : ℝ)
    (estimate actual scale : Fin 2 → Fin 2 → ℝ)
    (hm0 : -1 ≤ m) (hm1 : m ≤ 1)
    (hr0 : (1 / 2 : ℝ) ≤ reserve) (hr1 : reserve ≤ 1)
    (hn : |noise| ≤ (1 / 32 : ℝ))
    (hη : 0 ≤ η) (hL : 0 ≤ L) (hηL : L * η ≤ 1)
    (hs : ∀ i j, 0 < scale i j)
    (herror : normFour (fun i j => (actual i j - estimate i j) / scale i j) ≤
      normFour (fun i j => estimate i j / scale i j) / 128)
    (hmodel :
      let e := fun i j => estimate i j / scale i j
      let z := normalizedStep m reserve (relativeGradientCue e) noise
      let v := fun i j => η * reserve * allocation z.1 i j * e i j
      let u := fun i j => v i j / scale i j
      after ≤ before - dotFour actual u + L / 2 * normFour (fun i j => scale i j * u i j)) :
    let e := fun i j => estimate i j / scale i j
    let z := normalizedStep m reserve (relativeGradientCue e) noise
    ((-1 ≤ z.1 ∧ z.1 ≤ 1) ∧ ((1 / 2 : ℝ) ≤ z.2 ∧ z.2 ≤ 1)) ∧
    after + η / 256 * normFour e ≤ before := by
  let e := fun i j => estimate i j / scale i j
  let a := fun i j => actual i j / scale i j
  have he : normFour (fun i j => a i j - e i j) ≤ normFour e / 128 := by
    simpa only [a, e, ← sub_div] using herror
  have hm :
      let z := normalizedStep m reserve (relativeGradientCue e) noise
      let v := fun i j => η * reserve * allocation z.1 i j * e i j
      after ≤ before - dotFour a v + L / 2 * normFour v := by
    have hd : ∀ i j x, scale i j * (x / scale i j) = x :=
      fun i j x => mul_div_cancel₀ x (ne_of_gt (hs i j))
    have ha : ∀ i j x, actual i j * (x / scale i j) = a i j * x := by
      intro i j x
      dsimp [a]
      ring
    dsimp only at hmodel ⊢
    simpa only [dotFour, normFour, hd, ha] using hmodel
  have h := noisy_optimizer_step m reserve noise η L before after e a
    hm0 hm1 hr0 hr1 hn hη hL hηL
    (by simpa only [normFour, add_comm, add_left_comm, add_assoc] using he)
    (by simpa only [dotFour, normFour] using hm)
  simpa only [normFour, add_comm, add_left_comm, add_assoc] using h

/-! The implementation keeps the existing finite phase/reserve. This bridge
uses its proved cue/phase rounding envelope; real-valued parameters and the
signed estimate are not interpreted as finite phase codes. -/
theorem integer_preconditioned_step (K R C : ℤ) (η L before after : ℝ)
    (estimate actual scale : Fin 2 → Fin 2 → ℝ)
    (hk0 : -32 ≤ K) (hk1 : K ≤ 32)
    (hr0 : 8 ≤ R) (hr1 : R ≤ 16) (hc0 : -64 ≤ C) (hc1 : C ≤ 64)
    (hcue : |(C : ℝ) / 32 - relativeGradientCue (fun i j => estimate i j / scale i j)| ≤
      (1 / 64 : ℝ))
    (hη : 0 ≤ η) (hL : 0 ≤ L) (hηL : L * η ≤ 1)
    (hs : ∀ i j, 0 < scale i j)
    (herror : normFour (fun i j => (actual i j - estimate i j) / scale i j) ≤
      normFour (fun i j => estimate i j / scale i j) / 128)
    (hmodel :
      let next := ((integerCycleStep K R C).1 : ℝ) / 32
      let u := fun i j => η * ((R : ℝ) / 16) * allocation next i j *
        (estimate i j / scale i j) / scale i j
      after ≤ before - dotFour actual u + L / 2 * normFour (fun i j => scale i j * u i j)) :
    let z := integerCycleStep K R C
    ((-32 ≤ z.1 ∧ z.1 ≤ 32) ∧ (8 ≤ z.2 ∧ z.2 ≤ 16)) ∧
    after + η / 256 * normFour (fun i j => estimate i j / scale i j) ≤ before := by
  have hm0 : (-1 : ℝ) ≤ (K : ℝ) / 32 := by
    have h : (-32 : ℝ) ≤ K := by exact_mod_cast hk0
    linarith
  have hm1 : (K : ℝ) / 32 ≤ 1 := by
    have h : (K : ℝ) ≤ 32 := by exact_mod_cast hk1
    linarith
  have hrlo : (1 / 2 : ℝ) ≤ (R : ℝ) / 16 := by
    have h : (8 : ℝ) ≤ R := by exact_mod_cast hr0
    linarith
  have hrhi : (R : ℝ) / 16 ≤ 1 := by
    have h : (R : ℝ) ≤ 16 := by exact_mod_cast hr1
    linarith
  obtain ⟨noise, hn, heq⟩ := integer_phase_refines K R C _ hr0 hr1 hcue
  have h := preconditioned_filter_step ((K : ℝ) / 32) ((R : ℝ) / 16) noise
    η L before after estimate actual scale hm0 hm1 hrlo hrhi hn hη hL hηL hs herror
    (by dsimp only; rw [← heq]; exact hmodel)
  have hi := integer_cycle_contract K R C hk0 hk1 hr0 hr1 hc0 hc1
  exact ⟨⟨hi.1, hi.2.1⟩, h.2⟩

/-- Executable rational arithmetic oracle for the SAME filtered/scaled
integer step. It supplies gain/scale explicitly; online estimation of either
is not smuggled into this numerical reference or declared proved here. -/
def filteredCycleOracle (K R : ℤ) (gain rate : ℚ)
    (old observed scale : Fin 4 → ℚ) (feedback : Bool := true) : ℤ × (Fin 4 → ℚ) × (Fin 4 → ℚ) :=
  let mean := fun i => (1 - gain) * old i + gain * observed i
  let e := fun i => mean i / scale i
  let diagonal := (e 0)^2 + (e 3)^2
  let cross := (e 1)^2 + (e 2)^2
  let cue := if diagonal + cross = 0 then 0 else 2 * (diagonal - cross) / (diagonal + cross)
  let rounded := 32 * cue + (1 / 2 : ℚ)
  let C := rounded.num / (rounded.den : ℤ)
  let z := controlledCycleStep feedback K R C
  let update := fun i => mean i / scale i *
    (if i = 0 ∨ i = 3 then (32 + (z.1 : ℚ)) / 64 else (32 - (z.1 : ℚ)) / 64) *
    (rate * (R : ℚ) / 16) / scale i
  (z.1 + 32 + 65 * (z.2 - 8), mean, update)

/-- Raw observations reach the existing filter at a resource-derived rate.
The world condition is a bounded observation error about a fixed value, not
an assumption that the computed cue already requests release. No counter is
stored. Availability may depend on the entire joint phase/resource history. -/
theorem resource_filtered_tracking (rate target error initial : ℝ)
    (available mass observed : ℕ → ℝ)
    (hr : 0 < rate) (hr1 : rate ≤ 1) (he : 0 ≤ error)
    (ha : ∀ t, (1 / 2 : ℝ) ≤ available t ∧ available t ≤ 1)
    (hm0 : 0 ≤ mass 0) (hm1 : mass 0 ≤ 1)
    (hm : ∀ t, mass (t + 1) = (1 - rate * available t) * mass t + rate * available t)
    (ho : ∀ t, |observed (t + 1) - target| ≤ error) :
    let gain := fun t => rate * available t / mass (t + 1)
    (∀ t, (0 ≤ mass t ∧ mass t ≤ 1)) ∧
    (∀ t, rate / 2 ≤ gain t ∧ gain t ≤ 1) ∧
    (∀ t, |filterEstimate gain observed (fun _ => 0) initial t - target| ≤
      (1 - rate / 2)^t * |initial - target| + error) := by
  dsimp only
  have legal : ∀ t, 0 ≤ mass t ∧ mass t ≤ 1 := by
    intro t
    induction t with
    | zero => exact ⟨hm0, hm1⟩
    | succ t ih =>
      rw [hm]
      have hα0 : 0 ≤ rate * available t := mul_nonneg hr.le (by linarith [(ha t).1])
      have hα1 : rate * available t ≤ 1 := by nlinarith [(ha t).2]
      constructor
      · exact add_nonneg (mul_nonneg (by linarith) ih.1) hα0
      · nlinarith [mul_nonneg (show 0 ≤ 1 - rate * available t by linarith)
          (show 0 ≤ 1 - mass t by linarith [ih.2])]
  have gain_bounds : ∀ t, rate / 2 ≤ rate * available t / mass (t + 1) ∧
      rate * available t / mass (t + 1) ≤ 1 := by
    intro t
    have hα : 0 < rate * available t := mul_pos hr (by linarith [(ha t).1])
    have hα1 : rate * available t ≤ 1 := by nlinarith [(ha t).2]
    have hb : 0 ≤ (1 - rate * available t) * mass t :=
      mul_nonneg (by linarith) (legal t).1
    have hn : 0 < mass (t + 1) := by rw [hm]; linarith
    constructor
    · apply (le_div_iff₀ hn).mpr
      nlinarith [(legal (t + 1)).2, (ha t).1]
    · apply (div_le_one hn).mpr
      rw [hm]; linarith
  refine ⟨legal, gain_bounds, ?_⟩
  intro t
  apply filter_error_envelope (fun t => rate * available t / mass (t + 1))
    observed (fun _ => 0) (fun _ => target)
    (fun t => (1 - rate / 2)^t * |initial - target| + error)
    (fun _ => 0) (fun _ => error) (fun _ => 0) initial
  · simp only [pow_zero, one_mul]; linarith
  · intro t; exact ⟨by linarith [(gain_bounds t).1], (gain_bounds t).2⟩
  · intro t; simp
  · exact ho
  · intro t; simp
  · intro t
    have hpower : 0 ≤ (1 - rate / 2)^t * |initial - target| :=
      mul_nonneg (pow_nonneg (by linarith) _) (abs_nonneg _)
    have hx := mul_nonneg (show 0 ≤ rate * available t / mass (t + 1) - rate / 2
      by linarith [(gain_bounds t).1]) hpower
    simp only [add_zero, pow_succ]
    nlinarith

/-- Rounding supplies a finite release threshold even when an old filter
component never becomes exactly zero. Strictness matters: equality gives
code -63 because ties round upward. This is the EXISTING cue encoder. -/
theorem encoded_contrary_of_energy_ratio (g : Fin 2 → Fin 2 → ℝ)
    (h : 255 * ((g 0 0)^2 + (g 1 1)^2) < (g 0 1)^2 + (g 1 0)^2) :
    encodedGradientCue g = -64 := by
  let d := (g 0 0)^2 + (g 1 1)^2
  let c := (g 0 1)^2 + (g 1 0)^2
  have hd : 0 ≤ d := by dsimp [d]; positivity
  have hc : 255 * d < c := h
  have hs : 0 < d + c := by linarith
  have hq : relativeGradientCue g < -(127 / 64 : ℝ) := by
    change (if d + c = 0 then 0 else 2 * ((d - c) / (d + c))) < _
    rw [ite_eq_right (ne_of_gt hs)]
    have hratio : (d - c) / (d + c) < -(127 / 128 : ℝ) := by
      apply (div_lt_iff₀ hs).mpr
      linarith
    linarith
  have hlo := (encoded_gradient_cue_contract g).1.1
  have hhi : encodedGradientCue g < (-63 : ℤ) := by
    apply Int.floor_lt.mpr
    norm_num
    linarith
  omega

set_option maxHeartbeats 800000 in
/-- Joint raw-stream bridge. The world supplies a constant cross-mode gradient
of amplitude A; it does NOT supply a desired phase or an encoded release cue.
Every coordinate uses the actual generated observation gain, and the same
integer phase/resource transition consumes the resulting filtered evidence.
Any legal old phase is released in a finite time. A common positive RMS
scale cancels in the cue; floating rounding is a separate implementation check.
This proves learned allocation memory, not restoration of arbitrary weights. -/
theorem raw_stream_forces_release (rate A B : ℝ) (available mass : ℕ → ℝ)
    (initial : Fin 2 → Fin 2 → ℝ) (z : ℕ → ℤ × ℤ)
    (hr : 0 < rate) (hr1 : rate ≤ 1) (hA : 0 < A) (hB : 0 ≤ B)
    (ha : ∀ t, (1 / 2 : ℝ) ≤ available t ∧ available t ≤ 1)
    (hm0 : 0 ≤ mass 0) (hm1 : mass 0 ≤ 1)
    (hm : ∀ t, mass (t + 1) = (1 - rate * available t) * mass t + rate * available t)
    (hinit : ∀ i j, |initial i j - (if i = j then 0 else A)| ≤ B)
    (hz0 : (-32 ≤ (z 0).1 ∧ (z 0).1 ≤ 32) ∧ (8 ≤ (z 0).2 ∧ (z 0).2 ≤ 16))
    (hz : ∀ t, z (t + 1) = integerCycleStep (z t).1 (z t).2
      (encodedGradientCue (fun i j => filterEstimate
        (fun t => rate * available t / mass (t + 1))
        (fun _ => if i = j then 0 else A) (fun _ => 0) (initial i j) (t + 1)))) :
    (∀ t, (-32 ≤ (z t).1 ∧ (z t).1 ≤ 32) ∧ (8 ≤ (z t).2 ∧ (z t).2 ≤ 16)) ∧
    ∃ N : ℕ, (1 - rate / 2)^N * B ≤ A / 32 ∧ ∀ t, (z (N + 8 + t)).1 = -32 := by
  let gain := fun t => rate * available t / mass (t + 1)
  let mean := fun t i j => filterEstimate gain (fun _ => if i = j then 0 else A)
    (fun _ => 0) (initial i j) t
  have legal : ∀ t, (-32 ≤ (z t).1 ∧ (z t).1 ≤ 32) ∧
      (8 ≤ (z t).2 ∧ (z t).2 ≤ 16) := by
    intro t
    induction t with
    | zero => exact hz0
    | succ t ih =>
      rw [hz]
      have hc := (encoded_gradient_cue_contract (mean (t + 1))).1
      have hl := integer_cycle_contract _ _ _ ih.1.1 ih.1.2 ih.2.1 ih.2.2 hc.1 hc.2
      exact ⟨hl.1, hl.2.1⟩
  have track : ∀ t i j, |mean t i j - (if i = j then 0 else A)| ≤
      (1 - rate / 2)^t * B := by
    intro t i j
    have hf := (resource_filtered_tracking rate (if i = j then 0 else A) 0 (initial i j)
      available mass (fun _ => if i = j then 0 else A) hr hr1 (by norm_num)
      ha hm0 hm1 hm (by intro t; simp)).2.2 t
    dsimp only [mean, gain]
    simp only [add_zero] at hf
    exact hf.trans (mul_le_mul_of_nonneg_left (hinit i j) (pow_nonneg (by linarith) _))
  obtain ⟨N, hN⟩ := exists_pow_lt_of_lt_one
    (show 0 < A / (32 * (B + 1)) by positivity) (show 1 - rate / 2 < 1 by linarith)
  have hq0 : 0 ≤ 1 - rate / 2 := by linarith
  have hbudget : (1 - rate / 2)^N * B ≤ A / 32 := by
    have hh := (lt_div_iff₀ (show 0 < 32 * (B + 1) by positivity)).mp hN
    have hp := pow_nonneg hq0 N
    nlinarith
  have accuracy : ∀ t i j, |mean (N + t) i j - (if i = j then 0 else A)| ≤ A / 32 := by
    intro t i j
    have hp : (1 - rate / 2)^t ≤ 1 := pow_le_one₀ hq0 (by linarith)
    have hnb : 0 ≤ (1 - rate / 2)^N * B := mul_nonneg (pow_nonneg hq0 _) hB
    have hh := mul_le_mul_of_nonneg_left hp hnb
    have ht := track (N + t) i j
    rw [pow_add] at ht
    nlinarith
  have encoded : ∀ t, encodedGradientCue (mean (N + t)) = -64 := by
    intro t
    have h00 := abs_le.mp (accuracy t 0 0)
    have h11 := abs_le.mp (accuracy t 1 1)
    have h01 := abs_le.mp (accuracy t 0 1)
    have h10 := abs_le.mp (accuracy t 1 0)
    norm_num at h00 h11 h01 h10
    have d0 := mul_nonneg (show 0 ≤ A / 32 - mean (N + t) 0 0 by linarith)
      (show 0 ≤ A / 32 + mean (N + t) 0 0 by linarith)
    have d1 := mul_nonneg (show 0 ≤ A / 32 - mean (N + t) 1 1 by linarith)
      (show 0 ≤ A / 32 + mean (N + t) 1 1 by linarith)
    have c0 := mul_nonneg (show 0 ≤ mean (N + t) 0 1 - 31 * A / 32 by linarith)
      (show 0 ≤ mean (N + t) 0 1 + 31 * A / 32 by linarith)
    have c1 := mul_nonneg (show 0 ≤ mean (N + t) 1 0 - 31 * A / 32 by linarith)
      (show 0 ≤ mean (N + t) 1 0 + 31 * A / 32 by linarith)
    apply encoded_contrary_of_energy_ratio
    nlinarith only [d0, d1, c0, c1, sq_pos_of_pos hA]
  refine ⟨legal, N, hbudget, ?_⟩
  intro shift
  let x := fun t => -((z (N + shift + t)).1 : ℝ) / 32
  have progress : ∀ t, min 1 (x t + 1 / 4) ≤ x (t + 1) := by
    intro t
    have hl := legal (N + shift + t)
    have hrlo : (1 / 2 : ℝ) ≤ ((z (N + shift + t)).2 : ℝ) / 16 := by
      have hh : (8 : ℝ) ≤ (z (N + shift + t)).2 := by exact_mod_cast hl.2.1
      linarith
    have hrhi : ((z (N + shift + t)).2 : ℝ) / 16 ≤ 1 := by
      have hh : ((z (N + shift + t)).2 : ℝ) ≤ 16 := by exact_mod_cast hl.2.2
      linarith
    have hlo : (-1 : ℝ) ≤ ((z (N + shift + t)).1 : ℝ) / 32 := by
      have hh : (-32 : ℝ) ≤ (z (N + shift + t)).1 := by exact_mod_cast hl.1.1
      linarith
    have hhi : ((z (N + shift + t)).1 : ℝ) / 32 ≤ 1 := by
      have hh : ((z (N + shift + t)).1 : ℝ) ≤ 32 := by exact_mod_cast hl.1.2
      linarith
    obtain ⟨noise, hn, heq⟩ := integer_phase_refines (z (N + shift + t)).1
      (z (N + shift + t)).2 (-64) (-2) hl.2.1 hl.2.2 (by norm_num)
    have hc := (normalized_joint_contract _ _ (-2) noise hlo hhi hrlo hrhi hn).2.2.2 (by norm_num)
    have hstep : z (N + shift + t + 1) =
        integerCycleStep (z (N + shift + t)).1 (z (N + shift + t)).2 (-64) := by
      rw [hz]
      change integerCycleStep _ _ (encodedGradientCue (mean (N + shift + t + 1))) = _
      rw [show N + shift + t + 1 = N + (shift + t + 1) by omega]
      rw [encoded]
    have hi : ((z (N + shift + t + 1)).1 : ℝ) / 32 ≤
        max (-1) (((z (N + shift + t)).1 : ℝ) / 32 - 1 / 4) := by
      rw [hstep, heq]
      exact hc
    have hdual : -max (-1 : ℝ) (((z (N + shift + t)).1 : ℝ) / 32 - 1 / 4) =
        min 1 (-((z (N + shift + t)).1 : ℝ) / 32 + 1 / 4) := by
      simp only [min_def, max_def]; split_ifs <;> linarith
    dsimp [x]
    rw [show N + shift + (t + 1) = N + shift + t + 1 by omega, ← hdual]
    linarith
  have endpoint := restoration_by_budget x (1 / 4) (by norm_num)
    (fun t => by
      have hh : (-32 : ℝ) ≤ (z (N + shift + t)).1 := by exact_mod_cast (legal _).1.1
      dsimp [x]; linarith) progress 8 (by
        have hh : ((z (N + shift)).1 : ℝ) ≤ 32 := by exact_mod_cast (legal _).1.2
        dsimp [x]; norm_num; linarith)
  dsimp [x] at endpoint
  have hh : ((z (N + shift + 8)).1 : ℝ) = -32 := by linarith
  have hi : (z (N + shift + 8)).1 = -32 := by exact_mod_cast hh
  rw [show N + 8 + shift = N + shift + 8 by omega]
  exact hi

/-- A compartment's shared positive RMS changes physical step size, but not
the exact allocation evidence. This connects the raw-stream result to the
already implemented preconditioned observer without a new cue or state. -/
theorem relative_cue_shared_scale (g : Fin 2 → Fin 2 → ℝ) (s : ℝ) (hs : s ≠ 0) :
    relativeGradientCue (fun i j => g i j / s) = relativeGradientCue g := by
  dsimp only [relativeGradientCue]
  simp only [div_pow, ← add_div, ← sub_div]
  have hs2 : s^2 ≠ 0 := pow_ne_zero _ hs
  by_cases hz : (g 0 0)^2 + (g 1 1)^2 + ((g 0 1)^2 + (g 1 0)^2) = 0
  · simp only [hz, zero_div, ite_true]
  · rw [ite_eq_right (div_ne_zero hz hs2), ite_eq_right hz]
    field_simp

/-- Cold evidence ignites either mode in the SAME integer law. With input
withdrawn, phases within eight codes of either endpoint regenerate to that
endpoint in one tick for EVERY legal reserve. No frozen-state branch exists. -/
theorem integer_ignition_and_retention :
    integerCycleStep 0 16 64 = (32, 16) ∧
    integerCycleStep 0 16 (-64) = (-32, 16) ∧
    ∀ K R : ℤ, -32 ≤ K → K ≤ 32 → 8 ≤ R → R ≤ 16 →
      (24 ≤ K → (integerCycleStep K R 0).1 = 32) ∧
      (K ≤ -24 → (integerCycleStep K R 0).1 = -32) := by
  refine ⟨by norm_num [integerCycleStep], by norm_num [integerCycleStep], ?_⟩
  intro K R hk0 hk1 hr0 hr1
  have hcoef : 24 ≤ 2 * (4 + R) := by omega
  constructor
  · intro hk
    have hmul := mul_le_mul_of_nonneg_right hcoef (show 0 ≤ K by omega)
    have hraw : 32 ≤ (2 * (4 + R) * K + R * 0 + 8) / 16 := by omega
    simp only [integerCycleStep, min_eq_left hraw, max_eq_right (by norm_num : (-32 : ℤ) ≤ 32)]
  · intro hk
    have hmul := mul_le_mul_of_nonpos_right hcoef (show K ≤ 0 by omega)
    have hraw : (2 * (4 + R) * K + R * 0 + 8) / 16 ≤ -32 := by omega
    have h32 : (2 * (4 + R) * K + R * 0 + 8) / 16 ≤ 32 := by omega
    simp only [integerCycleStep, min_eq_right h32, max_eq_left hraw]

/-- Both interventions use the SAME finite state domain, supply/use ledger
and double-budget allocation. Turning off feedback does not remove phase,
reserve, sensing, or its parameter readout; no inactive padding state is added. -/
theorem controlled_cycle_contract (feedback : Bool) (K R C : ℤ)
    (hk0 : -32 ≤ K) (hk1 : K ≤ 32) (hr0 : 8 ≤ R) (hr1 : R ≤ 16)
    (hc0 : -64 ≤ C) (hc1 : C ≤ 64) :
    let z := controlledCycleStep feedback K R C
    ((-32 ≤ z.1 ∧ z.1 ≤ 32) ∧ (8 ≤ z.2 ∧ z.2 ≤ 16)) ∧
    z.2 + R * |K| / 128 = R + (19 - R) / 4 ∧
    (∀ i, allocation ((z.1 : ℝ) / 32) i 0 + allocation ((z.1 : ℝ) / 32) i 1 = 1) ∧
    (∀ j, allocation ((z.1 : ℝ) / 32) 0 j + allocation ((z.1 : ℝ) / 32) 1 j = 1) := by
  have old := integer_cycle_contract K R C hk0 hk1 hr0 hr1 hc0 hc1
  have bounds : -32 ≤ (controlledCycleStep feedback K R C).1 ∧
      (controlledCycleStep feedback K R C).1 ≤ 32 := by
    cases feedback
    · exact ⟨le_max_left _ _, max_le (by norm_num) (min_le_left _ _)⟩
    · exact old.1
  have reserve : (controlledCycleStep feedback K R C).2 = (integerCycleStep K R C).2 := by
    cases feedback <;> rfl
  have lo : (-1 : ℝ) ≤ ((controlledCycleStep feedback K R C).1 : ℝ) / 32 := by
    have h : (-32 : ℝ) ≤ (controlledCycleStep feedback K R C).1 := by exact_mod_cast bounds.1
    linarith
  have hi : ((controlledCycleStep feedback K R C).1 : ℝ) / 32 ≤ 1 := by
    have h : ((controlledCycleStep feedback K R C).1 : ℝ) ≤ 32 := by exact_mod_cast bounds.2
    linarith
  have budget := cycle_geometry _ lo hi
  exact ⟨⟨bounds, by rw [reserve]; exact old.2.1⟩,
    by rw [reserve]; exact old.2.2.1, budget.2.1, budget.2.2.1⟩

def passiveQuietPhase : ℕ → ℤ → ℤ
  | 0, K => K
  | t + 1, K => passiveQuietPhase t (max (-32) (min 32 ((8 * K + 8) / 16)))

set_option maxRecDepth 4096 in
/-- Do NOT claim exact integer forgetting: upward half-grid rounding leaves
fixed codes 0 and 1. Nevertheless every legal quiet phase reaches this
one-code band within six ticks, unlike the regenerated endpoints ±32.
The finite proof is kernel reduction, not an empirical memory experiment. -/
theorem passive_quiet_quantization_band (K : ℤ) (hk0 : -32 ≤ K) (hk1 : K ≤ 32) :
    (0 ≤ passiveQuietPhase 6 K ∧ passiveQuietPhase 6 K ≤ 1) ∧
    (∀ R : ℤ, (controlledCycleStep false 0 R 0).1 = 0 ∧
      (controlledCycleStep false 1 R 0).1 = 1) := by
  have finite : ∀ k : Fin 65, 0 ≤ passiveQuietPhase 6 ((k.val : ℤ) - 32) ∧
      passiveQuietPhase 6 ((k.val : ℤ) - 32) ≤ 1 := by decide
  let k : Fin 65 := ⟨(K + 32).toNat, by omega⟩
  have heq : (k.val : ℤ) - 32 = K := by dsimp [k]; omega
  have h := finite k
  rw [heq] at h
  exact ⟨h, by intro R; norm_num [controlledCycleStep]⟩

/-- A common whole-readout certificate for BOTH arms, reusing the existing
perturbed scalar lemma. t is the actual physical rate, including OLD resource,
RMS denominator and tail correction. The global quadratic upper model must
include cross-coordinate curvature. Estimation error is paid explicitly;
RMS is not asserted to be a curvature estimate or to ensure this premise. -/
theorem controlled_final_readout (feedback : Bool) (K R C : ℤ)
    (t L before after : ℝ) (estimate actual : Fin 2 → Fin 2 → ℝ)
    (hk0 : -32 ≤ K) (hk1 : K ≤ 32) (hr0 : 8 ≤ R) (hr1 : R ≤ 16)
    (hc0 : -64 ≤ C) (hc1 : C ≤ 64) (ht : 0 ≤ t) (hL : 0 ≤ L) (hst : L * t ≤ 1)
    (hmodel :
      let m := ((controlledCycleStep feedback K R C).1 : ℝ) / 32
      let u := fun i j => t * allocation m i j * estimate i j
      after ≤ before - dotFour actual u + L / 2 * normFour u) :
    let m := ((controlledCycleStep feedback K R C).1 : ℝ) / 32
    after + t / 4 * gradientWork m estimate ≤
      before + t * normFour (fun i j => actual i j - estimate i j) := by
  let m := ((controlledCycleStep feedback K R C).1 : ℝ) / 32
  have bounds := (controlled_cycle_contract feedback K R C hk0 hk1 hr0 hr1 hc0 hc1).1.1
  have lo : -1 ≤ m := by
    have h : (-32 : ℝ) ≤ (controlledCycleStep feedback K R C).1 := by exact_mod_cast bounds.1
    dsimp [m]; linarith
  have hi : m ≤ 1 := by
    have h : ((controlledCycleStep feedback K R C).1 : ℝ) ≤ 32 := by exact_mod_cast bounds.2
    dsimp [m]; linarith
  have lower := (cycle_geometry m lo hi).1
  have upper : ∀ i j, allocation m i j ≤ 1 := by
    intro i j; dsimp [allocation]; split_ifs <;> linarith
  have h00 := perturbed_scalar_readout_safe _ t L (estimate 0 0) (actual 0 0)
    (lower 0 0) (upper 0 0) ht hL hst
  have h01 := perturbed_scalar_readout_safe _ t L (estimate 0 1) (actual 0 1)
    (lower 0 1) (upper 0 1) ht hL hst
  have h10 := perturbed_scalar_readout_safe _ t L (estimate 1 0) (actual 1 0)
    (lower 1 0) (upper 1 0) ht hL hst
  have h11 := perturbed_scalar_readout_safe _ t L (estimate 1 1) (actual 1 1)
    (lower 1 1) (upper 1 1) ht hL hst
  dsimp only [gradientWork, normFour, dotFour] at hmodel ⊢
  change after + t / 4 * gradientWork m estimate ≤
    before + t * normFour (fun i j => actual i j - estimate i j)
  dsimp only [gradientWork, normFour]
  nlinarith only [h00, h01, h10, h11, hmodel]

/-- Executable projection of the existing online state, solely for arithmetic
refinement. Two cycles share a clock and RMS; no model or task is executed. -/
structure RawObserverOracleState where
  cycle : Fin 2 → ℤ × ℤ
  mass : ℚ
  energy : ℚ
  mean : Fin 2 → Fin 4 → ℚ

def rawObserverOracleStep (old : RawObserverOracleState) (rate : ℚ)
    (observed : Fin 2 → Fin 4 → ℚ) (feedback : Bool := true) : RawObserverOracleState × (Fin 2 → Fin 4 → ℚ) :=
  let available := ((old.cycle 0).2 + (old.cycle 1).2 : ℚ) / 32
  let α := rate * available
  let mass := (1 - α) * old.mass + α
  let gain := α / mass
  let z := fun b => filteredCycleOracle (old.cycle b).1 (old.cycle b).2 gain (1 / 8)
    (old.mean b) (observed b) (fun _ => 1) feedback
  let energy := (1 - gain) * old.energy + gain *
    ((List.ofFn fun b => (List.ofFn fun i => (observed b i)^2).sum).sum / 8)
  -- Materialize finite vectors: retaining nested function closures would
  -- repeatedly reevaluate the entire past rather than a bounded state.
  let cycles := (List.ofFn fun b => ((z b).1 % 65 - 32, (z b).1 / 65 + 8)).toArray
  let means := (List.ofFn fun b => (List.ofFn fun i => (z b).2.1 i).toArray).toArray
  let updates := (List.ofFn fun b => (List.ofFn fun i => (z b).2.2 i).toArray).toArray
  (⟨fun b => cycles[b.val]!, mass, energy, fun b i => means[b.val]![i.val]!⟩,
    fun b i => updates[b.val]![i.val]!)

def rawObserverOracleMain (feedback : Bool := true) (tag : String := "rawobserver") : IO Unit := do
  for rate in [(1 / 8 : ℚ), 1 / 2] do
    for scenario in List.range 3 do
      let shape : Fin 2 → Fin 4 → ℚ := fun b i =>
        if (i = 0 ∨ i = 3) = (b = 0) then 1 else 0
      let observed := if scenario = 2 then (fun _ _ => (0 : ℚ)) else shape
      let initialMean := if scenario = 1 then (fun b i => 1 - shape b i) else (fun _ _ => 0)
      let mut state : RawObserverOracleState :=
        ⟨fun b => (if scenario = 0 then 0 else if b = 0 then -32 else 32,
          if scenario = 0 then 16 else 10),
          if scenario = 0 then 0 else 1, if scenario = 1 then 1 / 2 else 0, initialMean⟩
      for t in List.range 80 do
        let (next, numerator) := rawObserverOracleStep state rate observed feedback
        let frac := fun x : ℚ => s!"{x.num}/{x.den}"
        let fields := [frac rate, toString scenario, toString t] ++
          (List.ofFn fun b => toString ((next.cycle b).1 + 32 + 65 * ((next.cycle b).2 - 8))) ++
          [frac next.mass, frac next.energy] ++
          ((List.ofFn fun b => List.ofFn fun i => frac (next.mean b i)).flatten) ++
          ((List.ofFn fun b => List.ofFn fun i => frac (numerator b i)).flatten)
        IO.println (tag ++ "\t" ++ String.intercalate "\t" fields)
        state := next

end CGB.ConservativeCycleReview

#print axioms CGB.ConservativeCycleReview.controlled_cycle_contract
#print axioms CGB.ConservativeCycleReview.passive_quiet_quantization_band
#print axioms CGB.ConservativeCycleReview.controlled_final_readout
#print axioms CGB.ConservativeCycleReview.relative_cue_shared_scale
#print axioms CGB.ConservativeCycleReview.integer_ignition_and_retention
#print axioms CGB.ConservativeCycleReview.resource_filtered_tracking
#print axioms CGB.ConservativeCycleReview.encoded_contrary_of_energy_ratio
#print axioms CGB.ConservativeCycleReview.raw_stream_forces_release
#print axioms CGB.ConservativeCycleReview.integer_preconditioned_step
#print axioms CGB.ConservativeCycleReview.filter_error_envelope
#print axioms CGB.ConservativeCycleReview.preconditioned_filter_step
#print axioms CGB.ConservativeCycleReview.cycle_geometry
#print axioms CGB.ConservativeCycleReview.regeneration_threshold
#print axioms CGB.ConservativeCycleReview.joint_step_invariant
#print axioms CGB.ConservativeCycleReview.quiet_restoration
#print axioms CGB.ConservativeCycleReview.strong_cue_release
#print axioms CGB.ConservativeCycleReview.normalized_joint_contract
#print axioms CGB.ConservativeCycleReview.restoration_by_budget
#print axioms CGB.ConservativeCycleReview.normalized_trajectory_contract
#print axioms CGB.ConservativeCycleReview.gradient_cycle_response
#print axioms CGB.ConservativeCycleReview.allocated_readout_safe
#print axioms CGB.ConservativeCycleReview.relative_contrast_coverage
#print axioms CGB.ConservativeCycleReview.relative_gradient_coverage
#print axioms CGB.ConservativeCycleReview.encoded_gradient_cue_contract
#print axioms CGB.ConservativeCycleReview.exact_optimizer_step
#print axioms CGB.ConservativeCycleReview.noisy_optimizer_step
#print axioms CGB.ConservativeCycleReview.integer_cycle_contract
#print axioms CGB.ConservativeCycleReview.integer_phase_refines
#print axioms CGB.ConservativeCycleReview.integer_code_roundtrip
#print axioms CGB.ConservativeCycleReview.integer_joint_optimizer_step
#print axioms CGB.ConservativeCycleReview.finite_block_optimizer_step
#print axioms CGB.ConservativeCycleReview.unrestricted_parameter_initialization
#print axioms CGB.ConservativeCycleReview.functional_recovery_tracking

/-- Arithmetic-only cross-language oracle; no model, loss or training loop.
Running with `lean --run` emits every legal state/cue triple for exact
comparison with the independent integer implementation. -/
def main (args : List String) : IO Unit := do
  if args.contains "--feedback-control" then
    CGB.ConservativeCycleReview.rawObserverOracleMain true "control\t1"
    CGB.ConservativeCycleReview.rawObserverOracleMain false "control\t0"
    return
  if args.contains "--raw-observer" then
    CGB.ConservativeCycleReview.rawObserverOracleMain
    return
  for ki in List.range 65 do
    for ri in List.range 9 do
      for ci in List.range 129 do
        let K := (ki : ℤ) - 32
        let R := (ri : ℤ) + 8
        let C := (ci : ℤ) - 64
        let z := CGB.ConservativeCycleReview.integerCycleStep K R C
        let oldCode := K + 32 + 65 * (R - 8)
        let newCode := z.1 + 32 + 65 * (z.2 - 8)
        IO.println s!"cycle\t{K}\t{R}\t{C}\t{z.1}\t{z.2}\t{oldCode}\t{newCode}"
  for K in [(-32 : ℤ), 0, 32] do
    for R in [(8 : ℤ), 16] do
      for oldSign in [(-1 : ℤ), 1] do
        for gain in [(0 : ℚ), 1 / 4, 1] do
          let old : Fin 4 → ℚ := fun i => (oldSign : ℚ) * ((i.val : ℚ) + 1) / 2
          let observed : Fin 4 → ℚ := fun i => ([1, 2, -3, -4] : List ℚ)[i.val]!
          let scale : Fin 4 → ℚ := fun i => ([1, 2, 1 / 2, 4] : List ℚ)[i.val]!
          let z := CGB.ConservativeCycleReview.filteredCycleOracle K R gain (1 / 8) old observed scale
          let frac := fun x : ℚ => s!"{x.num}/{x.den}"
          let fields := [toString K, toString R, toString oldSign, frac gain, toString z.1] ++
            (List.ofFn fun i => frac (z.2.1 i)) ++ (List.ofFn fun i => frac (z.2.2 i))
          IO.println ("filtered\t" ++ String.intercalate "\t" fields)
