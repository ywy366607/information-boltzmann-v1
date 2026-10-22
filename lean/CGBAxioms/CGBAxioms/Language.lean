import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Tactic.LinearCombination

/-!
# CGB language: finite-state Markov jump processes

The state–event language of `docs/information_boltzmann/CGB_AXIOMS_V0_20261022.md` §1.
This file states no axioms about brains: it only fixes the vocabulary.
Axiom clauses (P1–P3) are *predicates* on a `JumpProcess`, never Lean `axiom`s,
so every theorem lists in its hypotheses exactly the clauses it uses.
-/

open Finset

namespace CGB

/-- A finite-state Markov jump process, given by nonnegative transition rates. -/
structure JumpProcess (S : Type*) where
  rate : S → S → ℝ
  rate_nonneg : ∀ x y, 0 ≤ rate x y

namespace JumpProcess

variable {S : Type*} [Fintype S] [DecidableEq S]

/-- The generator acting on an observable: `(L f) x = ∑ y, k x y * (f y - f x)`.
`∑ x, p x * L f x` is the time derivative of `⟨f⟩` under the distribution `p`. -/
noncomputable def gen (P : JumpProcess S) (f : S → ℝ) (x : S) : ℝ :=
  ∑ y, P.rate x y * (f y - f x)

/-- `⟨f⟩` is constant for every initial distribution iff `L f = 0` pointwise. -/
theorem expect_const_iff (P : JumpProcess S) (f : S → ℝ) :
    (∀ p : S → ℝ, (∀ x, 0 ≤ p x) → ∑ x, p x = 1 → ∑ x, p x * P.gen f x = 0) ↔
      ∀ x, P.gen f x = 0 := by
  constructor
  · intro h x
    have := h (fun y => if y = x then 1 else 0) (fun y => by split_ifs <;> norm_num) (by simp)
    simpa using this
  · intro h p _ _
    simp [h]

end JumpProcess

end CGB
