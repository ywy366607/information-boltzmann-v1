import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Basic.Real.Basic
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring

/-! Conditional on one frozen decision and a fixed downstream learning policy.
`q` is the conditional mean of an arrived H-step outcome in common units;
`m` is any pre-action forecast in those units. `d` can be a policy difference
or a tangent. This addresses first-choice credit, not a total body derivative.
-/
namespace CGB.NativeDelayedActionContrast
open Finset

variable {A : Type*} [Fintype A]

def contrast (d q : A → ℝ) : ℝ := ∑ a, d a * q a

/-- Policy differences and tangents sum to zero: common progress is not choice credit. -/
theorem common_mode (d q : A → ℝ) (b : ℝ) (hd : ∑ a, d a = 0) :
    contrast d (fun a => q a + b) = contrast d q := by
  simp only [contrast, mul_add, sum_add_distrib, ← sum_mul, hd, zero_mul, add_zero]

/-- Expected one-sampled-action residual plus the known forecast contrast.
The sum is over legal actions with positive natural sampling probability. -/
theorem arrived_correction (p d m q : A → ℝ) (hp : ∀ a, p a ≠ 0) :
    contrast d m + (∑ a, p a * ((d a / p a) * (q a - m a))) = contrast d q := by
  have h (a : A) : p a * ((d a / p a) * (q a - m a)) = d a * q a - d a * m a := by
    field_simp [hp a]
  simp only [h, sum_sub_distrib, contrast]
  ring

/-- The reported 2587 reuse/silence ranks exclude even a weakly increasing
scalar decoder of those realized payments into those realized language effects. -/
theorem measured_no_monotone_decoder : ¬ ∃ f : ℚ → ℚ, Monotone f ∧
    f (-0.039522) = 0.006092 ∧ f (-0.003685) = -0.008728 := by
  rintro ⟨f, hm, hr, hs⟩
  have h := hm (show (-0.039522 : ℚ) ≤ -0.003685 by norm_num)
  rw [hr, hs] at h
  norm_num at h

/-- All redistributions of action probability see exactly the same credit
iff the conditional means differ only by a boundary-wide baseline. -/
theorem exact_credit_iff (y q : A → ℝ) (anchor : A) :
    (∀ d : A → ℝ, ∑ a, d a = 0 → contrast d y = contrast d q) ↔
      ∃ c : ℝ, ∀ a, y a = q a + c := by
  classical
  constructor
  · intro h
    refine ⟨y anchor - q anchor, fun a => ?_⟩
    let d : A → ℝ := fun i => (if a = i then 1 else 0) - (if anchor = i then 1 else 0)
    have hd : ∑ i, d i = 0 := by simp [d, sum_sub_distrib]
    have eq := h d hd
    simp [contrast, d, sub_mul, sum_sub_distrib, ite_mul] at eq
    linarith
  · rintro ⟨c, hc⟩ d hd
    rw [show y = fun a => q a + c from funext hc]
    exact common_mode d q c hd

omit [Fintype A] in
/-- Deterministic action ranks (including ties) only need a strictly
order-preserving transform on attained values, not equal credit magnitudes. -/
theorem rank_only (q : A → ℝ) (f : ℝ → ℝ) (hf : StrictMonoOn f (Set.range q)) (a b : A) :
    f (q a) < f (q b) ↔ q a < q b := hf.lt_iff_lt ⟨a,rfl⟩ ⟨b,rfl⟩

omit [Fintype A] in
/-- Same available information and opposite true orders rule out every
single readout of that information that claims both strict rankings. -/
theorem indistinguishable_orders {W V : Type*} (view : W → V) (x z : W)
    (same : view x = view z) (a b : A) :
    ¬ ∃ read : V → A → ℝ, read (view x) a < read (view x) b ∧
                              read (view z) b < read (view z) a := by
  rintro ⟨read,h,k⟩
  rw [same] at h
  exact (lt_asymm h) k

/-- Concrete observable: randomly score the endpoint on a common probe law.
Expected endpoint score vs. progress on that law differ only by the pre-score.
`post a x` may itself be an expectation over stochastic downstream life. -/
theorem common_probe_credit {X : Type*} [Fintype X] (w : X → ℝ) (pre : X → ℝ)
    (post : A → X → ℝ) (anchor : A) (d : A → ℝ) (hd : ∑ a, d a = 0) :
    contrast d (fun a => ∑ x, w x * post a x) =
      contrast d (fun a => ∑ x, w x * (post a x - pre x)) := by
  apply ((exact_credit_iff _ _ anchor).mpr ?_) d hd
  refine ⟨∑ x, w x * pre x, fun a => ?_⟩
  simp only [mul_sub, sum_sub_distrib, sub_add_cancel]

/-- Final-head update uses the score eligibility, not the scalar policy-to-policy
contrast as a reward. A common baseline frozen before sampling drops out. -/
theorem sampled_head_credit (p elig q : A → ℝ) (baseline : ℝ)
    (centered : ∑ a, p a * elig a = 0) :
    (∑ a, p a * elig a * (q a - baseline)) = ∑ a, p a * elig a * q a := by
  simp only [mul_sub, sum_sub_distrib, ← sum_mul, centered, zero_mul, sub_zero]

end CGB.NativeDelayedActionContrast
