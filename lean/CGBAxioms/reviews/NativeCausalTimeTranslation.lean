import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Basic.Real.Basic
import Mathlib.Tactic.Linarith

/-! Time translation of a scoped interventional law, not time evolution of a
conserved charge. `t` is an external-event index; the condition includes any
relevant internal clock, world regime and listener state. `a` is a concrete
intervention, not conditioning on an observational action label. -/
namespace CGB.NativeCausalTimeTranslation
open Finset

variable {C A Y : Type*} [Fintype Y]

structure Law (C A Y : Type*) [Fintype Y] where
  mass : ℕ → C → A → Y → ℝ
  nonneg : ∀ t c a y, 0 ≤ mass t c a y
  normalized : ∀ t c a, ∑ y, mass t c a y = 1

def expected (K : Law C A Y) (t : ℕ) (c : C) (a : A) (f : Y → ℝ) :=
  ∑ y, K.mass t c a y * f y

/-- Only compare the same admissible condition, action and outcome chart. -/
def invariant (K : Law C A Y) (valid : ℕ → C → A → Prop) : Prop :=
  ∀ t k c a, valid t c a → valid (t+k) c a → ∀ y, K.mass (t+k) c a y = K.mass t c a y

/-- Exact characterization: all fixed probes commute with a time shift iff
all the conditional outcome laws do. Singleton probes give the converse. -/
theorem invariant_iff_probes (K : Law C A Y) (valid : ℕ → C → A → Prop) :
    invariant K valid ↔ ∀ t k c a, valid t c a → valid (t+k) c a →
      ∀ f : Y → ℝ, expected K (t+k) c a f = expected K t c a f := by
  classical
  constructor
  · intro h t k c a ht hk f
    simp only [expected, h t k c a ht hk]
  · intro h t k c a ht hk y
    simpa [expected, mul_ite] using h t k c a ht hk (fun z => if z = y then 1 else 0)

/-- A weaker causal invariant: at a common condition and fixed readout, all
pairwise action effects are unchanged iff the time drift is action-independent. -/
theorem effect_invariant_iff_common_drift (K : Law C A Y) (t k : ℕ) (c : C)
    (f : Y → ℝ) (legal : A → Prop) (anchor : A) (ha : legal anchor) :
    (∀ a b, legal a → legal b → expected K (t+k) c a f - expected K (t+k) c b f =
                                expected K t c a f - expected K t c b f) ↔
    ∃ drift : ℝ, ∀ a, legal a → expected K (t+k) c a f = expected K t c a f + drift := by
  constructor
  · intro h
    refine ⟨expected K (t+k) c anchor f - expected K t c anchor f, ?_⟩
    intro a hlegal
    linarith [h a anchor hlegal ha]
  · rintro ⟨drift, h⟩ a b ha' hb
    rw [h a ha', h b hb]; linarith

/-- The stationary transition x ↦ !x has no nonconstant pathwise charge. -/
theorem flip_charges_constant {V : Type*} (Q : Bool → V)
    (conserved : ∀ x, Q (!x) = Q x) : ∀ x y, Q x = Q y := by
  have h : Q true = Q false := conserved false
  intro x y
  cases x <;> cases y <;> simp [h]

end CGB.NativeCausalTimeTranslation
