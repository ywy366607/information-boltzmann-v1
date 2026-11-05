import CGBAxioms.GenerationHoldCorrect
import Mathlib.Analysis.Calculus.Deriv.Comp
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Analysis.Calculus.Deriv.Add

/-!
# Active navigation: sphere slot, branch residuals, sign credit, reactivation (2026-11-04)

Design: `docs/information_boltzmann/CGB_ACTIVE_LATCH_NAVIGATION_20261104.md`. The user asked
that whatever can be proved is proved instead of run. Four groups.

**Sphere slot (continuous attractor).** The radial renewal `r ↦ (fb |r| / |r|) • r` changes the
length of the read by the scalar latch map and leaves its direction exactly unchanged:
`radial_renewal_norm`, `radial_renewal_direction`. So every `GenerationHoldCorrect` statement
holds for the radius, and the direction is a pure hold (`hold_accumulates`: disturbances along
the sphere add up, to be corrected by input, as a head-direction ring is by landmarks).

**Cross-branch residuals.** Adding any vector common to all branches (an internal clock, a
shared drift) leaves every residual `ẑ k - mean ẑ` unchanged: `branch_residual_common_invariant`.
A regularizer that sees only residuals cannot be satisfied by a common component; differences
between branches must come from the actions. This is why the anti-collapse term of the design
is taken across branches, not along time (the fly line's time-axis SIGReg was met by a clock).

**A latch read carries the sign of the write.** For κ > 1: positive content stays positive
(`latch_keeps_sign_pos`, and by oddness negative stays negative), and two writes of the same sign
inside the basin are read the same up to `L ^ n` times their initial distances from `r*`
(`latch_reads_forget_magnitude`). So the read at a later time depends on the write only through
its sign: the credit a later read gives to the write is credit to its sign.

**Reactivation is exact credit for a held write.** If the write `E θ` (an encoder applied to a
stored input) is held linearly to the read `a ^ n * E θ`, the derivative of any later loss
`L (a ^ n * E θ)` is `l * (a ^ n * e)`, computed at read time from the stored input
(`reactivation_exact_held`). For a linear memory the read is a superposition of writes, and the
derivative is the sum of per-write terms (`reactivation_superposition`): re-encoding a retained
subset of writes gives exactly their part; the writes not retained are the only bias
(`reactivation_partial_bias`). Boundaries: the delta rule's erasure term makes the GDN read
nonlinear in past writes; parameter drift between write and read is not modelled (θ is fixed).
-/

namespace CGB.Generation.ActiveNavigation

open CGB.Generation.HoldCorrect Finset

/-! ## Sphere slot -/

section Sphere

variable {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]

/-- Radial renewal of the read `r` with the scalar latch map on its length. -/
noncomputable def radial (a c g : ℝ) (r : V) : V := (fb a c g ‖r‖ / ‖r‖) • r

theorem fb_pos {a c g x : ℝ} (ha : 0 ≤ a) (hc : 0 < c) (hg : 0 < g) (hx : 0 < x) :
    0 < fb a c g x := by
  rw [fb_nonneg_eq hg hx.le]
  have : 0 < g * x / (1 + g * x) := by positivity
  have : 0 ≤ a * x := mul_nonneg ha hx.le
  nlinarith

/-- The length of the renewed read is the scalar latch map of the old length. -/
theorem radial_renewal_norm {a c g : ℝ} (ha : 0 ≤ a) (hc : 0 < c) (hg : 0 < g) {r : V}
    (hr : r ≠ 0) : ‖radial a c g r‖ = fb a c g ‖r‖ := by
  have hn : 0 < ‖r‖ := norm_pos_iff.mpr hr
  have hs : 0 < fb a c g ‖r‖ / ‖r‖ := div_pos (fb_pos ha hc hg hn) hn
  rw [radial, norm_smul, Real.norm_eq_abs, abs_of_pos hs, div_mul_cancel₀ _ hn.ne']

/-- The direction is unchanged: the renewed read is a positive multiple of the old one. -/
theorem radial_renewal_direction {a c g : ℝ} (ha : 0 ≤ a) (hc : 0 < c) (hg : 0 < g) {r : V}
    (hr : r ≠ 0) : ∃ s : ℝ, 0 < s ∧ radial a c g r = s • r :=
  ⟨_, div_pos (fb_pos ha hc hg (norm_pos_iff.mpr hr)) (norm_pos_iff.mpr hr), rfl⟩

end Sphere

/-! ## Cross-branch residuals -/

/-- Mean of `K` branch vectors. -/
noncomputable def branchMean {K d : ℕ} (z : Fin K → Fin d → ℝ) : Fin d → ℝ :=
  fun i => (∑ k, z k i) / K

/-- Residual of branch `k` from the branch mean. -/
noncomputable def residual {K d : ℕ} (z : Fin K → Fin d → ℝ) (k : Fin K) : Fin d → ℝ :=
  fun i => z k i - branchMean z i

/-- **A common component cancels in every residual** (for at least one branch). -/
theorem branch_residual_common_invariant {K d : ℕ} (hK : 0 < K) (z : Fin K → Fin d → ℝ)
    (c : Fin d → ℝ) (k : Fin K) :
    residual (fun j i => z j i + c i) k = residual z k := by
  funext i
  have hK' : (K : ℝ) ≠ 0 := by exact_mod_cast hK.ne'
  simp only [residual, branchMean, sum_add_distrib, sum_const, card_univ, Fintype.card_fin,
    nsmul_eq_mul]
  field_simp
  ring

/-! ## Sign credit of a latch -/

/-- **Positive content stays positive** under the renewal, at every step. -/
theorem latch_keeps_sign_pos {a c g x₀ : ℝ} (ha : 0 ≤ a) (hc : 0 < c) (hg : 0 < g)
    (hx : 0 < x₀) (n : ℕ) : 0 < (fb a c g)^[n] x₀ := by
  induction n with
  | zero => simpa using hx
  | succ n ih =>
    rw [Function.iterate_succ_apply']
    exact fb_pos ha hc hg ih

/-- Negative content stays negative (oddness of the renewal). -/
theorem latch_keeps_sign_neg {a c g x₀ : ℝ} (ha : 0 ≤ a) (hc : 0 < c) (hg : 0 < g)
    (hx : x₀ < 0) (n : ℕ) : (fb a c g)^[n] x₀ < 0 := by
  have hodd : ∀ m : ℕ, (fb a c g)^[m] x₀ = -((fb a c g)^[m] (-x₀)) := by
    intro m
    induction m with
    | zero => simp
    | succ m ih =>
      rw [Function.iterate_succ_apply', Function.iterate_succ_apply', ih, fb_odd]
  rw [hodd n]
  exact neg_neg_of_pos (latch_keeps_sign_pos ha hc hg (neg_pos.mpr hx) n)

/-- **The read forgets the magnitude of the write**: two writes of the same sign inside the
basin (`ρ ≤ x₀, x₀'`) are read within `L ^ n (|x₀ - r*| + |x₀' - r*|)` of each other, with
`L = factor a g ρ < 1` (`latch_converges`). -/
theorem latch_reads_forget_magnitude {a c g ρ x₀ x₀' : ℝ} (ha0 : 0 ≤ a) (ha1 : a < 1)
    (hg : 0 < g) (hρ : 0 < ρ) (hρr : ρ ≤ rstar a c g) (hx : ρ ≤ x₀) (hx' : ρ ≤ x₀') (n : ℕ) :
    |(fb a c g)^[n] x₀ - (fb a c g)^[n] x₀'| ≤
      factor a g ρ ^ n * (|x₀ - rstar a c g| + |x₀' - rstar a c g|) := by
  have h1 := (latch_converges ha0 ha1 hg hρ hρr hx n).2
  have h2 := (latch_converges ha0 ha1 hg hρ hρr hx' n).2
  calc |(fb a c g)^[n] x₀ - (fb a c g)^[n] x₀'|
      = |((fb a c g)^[n] x₀ - rstar a c g) - ((fb a c g)^[n] x₀' - rstar a c g)| := by
        ring_nf
    _ ≤ |(fb a c g)^[n] x₀ - rstar a c g| + |(fb a c g)^[n] x₀' - rstar a c g| := abs_sub _ _
    _ ≤ _ := by rw [mul_add]; exact add_le_add h1 h2

/-! ## Reactivation credit -/

/-- **Reactivation is exact for a held write**: re-encoding the stored input at read time gives
the derivative of the later loss through the hold `a ^ n`. -/
theorem reactivation_exact_held {E L : ℝ → ℝ} {θ e l a : ℝ} (n : ℕ)
    (hE : HasDerivAt E e θ) (hL : HasDerivAt L l (a ^ n * E θ)) :
    HasDerivAt (fun t => L (a ^ n * E t)) (l * (a ^ n * e)) θ :=
  hL.comp θ (hE.const_mul (a ^ n))

/-- **Superposition**: a linear read `∑ j, k j * E j θ` of several writes has the sum of the
per-write re-encoded terms as its derivative. -/
theorem reactivation_superposition {ι : Type*} (s : Finset ι) {E : ι → ℝ → ℝ} {e k : ι → ℝ}
    {θ : ℝ} (hE : ∀ j ∈ s, HasDerivAt (E j) (e j) θ) :
    HasDerivAt (fun t => ∑ j ∈ s, k j * E j t) (∑ j ∈ s, k j * e j) θ := by
  have h := HasDerivAt.sum fun j hj => (hE j hj).const_mul (k j)
  convert h using 1
  funext t
  simp [Finset.sum_apply]

/-- **The only bias is the writes not retained**: the true derivative minus the retained part
is the sum over the dropped writes. -/
theorem reactivation_partial_bias {ι : Type*} [DecidableEq ι] (s r : Finset ι) (hr : r ⊆ s)
    (e k : ι → ℝ) :
    ∑ j ∈ s, k j * e j - ∑ j ∈ r, k j * e j = ∑ j ∈ s \ r, k j * e j := by
  rw [← sum_sdiff hr]; ring

/-! ## A fixed source runs out of progress; the counterfactual of one entry -/

/-- **Progress from one source is used up.** If the excess losses `e t ≥ 0` of a learner on a
source have bounded running sums (what `mixture_regret_le` gives a Bayes mixture on a fixed
source: at most `log (1 / w θ₀)`), then at most `B / ε` steps can carry progress `≥ ε`, however
long the learner stays. Staying passive on one source leaves eventually nothing to learn. -/
theorem progress_steps_bounded (e : ℕ → ℝ) (he : ∀ t, 0 ≤ e t) {B ε : ℝ}
    (hB : ∀ n, ∑ t ∈ range n, e t ≤ B) (n : ℕ) :
    ((range n).filter (fun t => ε ≤ e t)).card * ε ≤ B := by
  calc ((range n).filter (fun t => ε ≤ e t)).card * ε
      = ∑ t ∈ (range n).filter (fun t => ε ≤ e t), ε := by rw [sum_const, nsmul_eq_mul]
    _ ≤ ∑ t ∈ (range n).filter (fun t => ε ≤ e t), e t := sum_le_sum fun t ht => (mem_filter.mp ht).2
    _ ≤ ∑ t ∈ range n, e t := sum_le_sum_of_subset_of_nonneg (filter_subset _ _)
        (fun t _ _ => he t)
    _ ≤ B := hB n

/-- **Removing one entry from a linear mixture** `p = w q + (1 - w) p₋`: the mixture without
it is `(p - w q) / (1 - w)`, so the counterfactual of one entry needs no second forward pass. -/
theorem mixture_without_entry {p q w pm : ℝ} (hw : w ≠ 1) (hp : p = w * q + (1 - w) * pm) :
    pm = (p - w * q) / (1 - w) := by
  have h1 : (1 - w) ≠ 0 := sub_ne_zero.mpr (Ne.symm hw)
  rw [eq_div_iff h1, hp]; ring

end CGB.Generation.ActiveNavigation
