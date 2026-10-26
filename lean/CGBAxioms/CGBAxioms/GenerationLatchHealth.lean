import CGBAxioms.GenerationLifeGate

/-!
# When is a latch healthy? (generation chain, latch health)

A latch (ignition, G23/G24) is a basin of states held by positive feedback. It is healthy as
long as it breaks none of the essentials of life; it is harmful exactly when it does:

* `trap_irrecoverable` (H1, release): a basin closed under the organism's own events that
  contains no intake-capable state makes every state in it irrecoverable (death by lock-in).
* `trap_insensitive` (H2, sensing): inside a basin where rates and effects ignore the
  stimulus, the one-tick kernel is the same under every stimulus: the organism is blind for
  as long as it stays latched.
* `pair_rate_ratio` (H1, quantitative): in a pair-reaction medium a cue changes the rate of
  any event by exactly `exp((Δreservoir) · ΔC / 2)`. A latch whose exit rates are suppressed by
  a barrier larger than any cue the world can deliver is, in that world, a trap.
-/

namespace CGB

namespace Medium

variable {U Λ A I : Type*} (M : Medium U Λ A I)

/-- **H1 (lock-in is death)**: if the set `B` is closed under the organism's own events and no
state in it can take content in, no state in it is recoverable. -/
theorem trap_irrecoverable [Fintype U] (B : Set (U → Λ)) (hclosed : ∀ x y, x ∈ B → M.OrgStep x y → y ∈ B)
    (hno : ∀ x ∈ B, ¬ M.CanTakeIn x) : ∀ x ∈ B, ¬ M.Recoverable x := by
  rintro x hx ⟨y, hxy, hy⟩
  have : y ∈ B := by
    induction hxy with
    | refl => exact hx
    | tail _ hst ih => exact hclosed _ _ ih hst
  exact hno y this hy

variable [Fintype U] [DecidableEq U] [Fintype Λ] [DecidableEq Λ] [Fintype A]

/-- **H2 (latched means blind)**: where rates and effects do not depend on the stimulus, the
one-tick kernel out of that state is the same under every stimulus. -/
theorem trap_insensitive {R : ℝ} (x : U → Λ)
    (hrate : ∀ a s s', M.rate a s x = M.rate a s' x) (hact : ∀ a s s', M.act a s x = M.act a s' x)
    (s s' : U → I) (y : U → Λ) : M.kernel R s x y = M.kernel R s' x y := by
  have hj : ∀ z, (M.jump s).rate x z = (M.jump s').rate x z := fun z => by
    simp only [jump]
    exact Finset.sum_congr rfl fun a _ => by rw [hact a s s', hrate a s s']
  simp only [kernel, hj]

end Medium

namespace PairCRN

variable {U Λ I R : Type*} [Fintype U] [DecidableEq U] [Fintype R] [DecidableEq R]
  (C : PairCRN U Λ I R)

omit [Fintype R] [DecidableEq R] in
/-- **H1, quantitative**: a cue changes the rate of an event by exactly
`exp((reservoir s - reservoir s') · dC / 2)`. -/
theorem pair_rate_ratio {r : R} {u v : U} (huv : u ≠ v) (s s' : U → I) (x : U → Λ) :
    C.medium.rate (r, u, v) s x =
      C.medium.rate (r, u, v) s' x *
        Real.exp ((C.reservoir s u v - C.reservoir s' u v) * C.dC r u v x / 2) := by
  simp only [medium, huv, ite_false]
  rw [mul_assoc, ← Real.exp_add]
  congr 2
  simp only [dE]
  ring

end PairCRN

end CGB
