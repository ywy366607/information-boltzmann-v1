import CGBAxioms.Language

/-!
# P2: symmetry clauses and their first theorems

* P2a (conservation) in pathwise form, and **T-N**: the Baez–Fong Noether
  correspondence (J. Math. Phys. 54, 013301, 2013).
* P2b (time-reversal symmetry) in detailed-balance form, and **T-E**: the energy
  function is a consequence of reversibility, not a primitive.
* Independence witness for P2b: a one-way 3-cycle.
-/

open Finset

namespace CGB

namespace JumpProcess

variable {S : Type*}

/-- **P2a** (pathwise): every event with positive rate preserves `Q`. -/
def Conserved (P : JumpProcess S) (Q : S → ℝ) : Prop :=
  ∀ x y, 0 < P.rate x y → Q x = Q y

/-- **P2b** (detailed-balance form): some positive weight balances every event
against its reverse. The equivalence with the Kolmogorov cycle condition is a
later theorem. -/
def Reversible (P : JumpProcess S) : Prop :=
  ∃ π : S → ℝ, (∀ x, 0 < π x) ∧ ∀ x y, π x * P.rate x y = π y * P.rate y x

lemma gen_sq_sub [Fintype S] (P : JumpProcess S) (Q : S → ℝ) (x : S) :
    P.gen (fun s => Q s ^ 2) x - 2 * Q x * P.gen Q x
      = ∑ y, P.rate x y * (Q y - Q x) ^ 2 := by
  simp only [gen, Finset.mul_sum, ← Finset.sum_sub_distrib]
  exact Finset.sum_congr rfl (fun y _ => by ring)

/-- **T-N** (Noether for Markov processes, pointwise form): `Q` and `Q²` are both
stationary observables iff every event preserves `Q`. With `expect_const_iff` this is
"⟨Q⟩ and ⟨Q²⟩ are constant for every initial distribution iff `Q` is conserved". -/
theorem noether [Fintype S] (P : JumpProcess S) (Q : S → ℝ) :
    ((∀ x, P.gen Q x = 0) ∧ (∀ x, P.gen (fun s => Q s ^ 2) x = 0)) ↔ P.Conserved Q := by
  constructor
  · rintro ⟨h1, h2⟩ x y hxy
    have hsum : ∑ z, P.rate x z * (Q z - Q x) ^ 2 = 0 := by
      rw [← gen_sq_sub, h2 x, h1 x]; ring
    have hterm := (Finset.sum_eq_zero_iff_of_nonneg (fun z _ =>
      mul_nonneg (P.rate_nonneg x z) (sq_nonneg (Q z - Q x)))).1 hsum y (Finset.mem_univ y)
    have hsq : (Q y - Q x) ^ 2 = 0 := by
      rcases mul_eq_zero.1 hterm with h | h
      · exact absurd h (ne_of_gt hxy)
      · exact h
    exact (sub_eq_zero.1 ((pow_eq_zero_iff two_ne_zero).1 hsq)).symm
  · intro hc
    have key : ∀ g : S → ℝ, (∀ x y, 0 < P.rate x y → g x = g y) → ∀ x, P.gen g x = 0 := by
      intro g hg x
      refine Finset.sum_eq_zero (fun y _ => ?_)
      rcases (P.rate_nonneg x y).lt_or_eq with h | h
      · rw [hg x y h]; ring
      · rw [← h]; ring
    exact ⟨key Q hc, key _ (fun x y h => by simp only [hc x y h])⟩

/-- **T-E** (energy from time-reversal symmetry): a reversible process has an energy
`E = -log π` that fixes the ratio of every event to its reverse. -/
theorem energy_of_reversible (P : JumpProcess S) (h : P.Reversible) :
    ∃ E : S → ℝ, ∀ x y, 0 < P.rate y x →
      P.rate x y / P.rate y x = Real.exp (-(E y - E x)) := by
  obtain ⟨π, hpos, hdb⟩ := h
  refine ⟨fun s => -Real.log (π s), fun x y hyx => ?_⟩
  change P.rate x y / P.rate y x = Real.exp (-(-Real.log (π y) - -Real.log (π x)))
  have hx := hpos x
  have hy := hpos y
  rw [show -(-Real.log (π y) - -Real.log (π x)) = Real.log (π y) - Real.log (π x) by ring,
    Real.exp_sub, Real.exp_log hy, Real.exp_log hx, div_eq_iff (ne_of_gt hyx),
    div_mul_eq_mul_div, eq_div_iff (ne_of_gt hx)]
  linear_combination hdb x y

/-- Independence witness for **P2b**: a one-way 3-cycle `0 → 1 → 2 → 0`.
It is local, finite and conserves the particle, yet runs a perpetual directed cycle. -/
noncomputable def ring3 : JumpProcess (Fin 3) where
  rate x y := if y = x + 1 then 1 else 0
  rate_nonneg x y := by split_ifs <;> norm_num

theorem ring3_not_reversible : ¬ ring3.Reversible := by
  rintro ⟨π, hpos, hdb⟩
  have h : π 0 = 0 := by simpa [ring3] using hdb 0 1
  exact (hpos 0).ne' h

end JumpProcess

end CGB
