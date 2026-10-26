import CGBAxioms.GenerationGrowth

/-!
# A generator of admitted media (generation chain, M1)

Instead of proving the six clauses case by case for each organism, prove once that a
medium built from a local energy satisfies them, so an organism is admitted by giving
its data and small local checks.

* M1a `p2b_of_energy`: if every event has a reverse event and, in the calm environment,
  the rate of an effective event is `κ a * exp (-(E y - E x) / 2)` with `κ` equal on an
  event and its reverse, the dynamics is detailed balanced with weights `exp (-E)`.
* M1b `rate_local_of_energy`: if the energy difference of every event reads only its
  participants, so does such a rate.
-/

namespace CGB.Medium

variable {U Λ A I : Type*} [Fintype U] [Fintype A] [DecidableEq (U → Λ)]
  (M : Medium U Λ A I)

/-- **M1a**: symmetric (Metropolis-type) rates of a reversible event set under an energy
give P2b. -/
theorem p2b_of_energy (E : (U → Λ) → ℝ) (κ : A → ℝ) (rev : A → A) (hinv : Function.Involutive rev)
    (hκ : ∀ a, κ (rev a) = κ a)
    (hback : ∀ a x, M.act a M.calm x ≠ x → M.act (rev a) M.calm (M.act a M.calm x) = x)
    (hrate : ∀ a x, M.act a M.calm x ≠ x →
      M.rate a M.calm x = κ a * Real.exp (-(E (M.act a M.calm x) - E x) / 2)) :
    M.P2b := by
  refine M.p2b_of_micro_weighted (fun x => Real.exp (-E x)) (fun _ => Real.exp_pos _) rev hinv
    fun a x hne => ⟨hback a x hne, ?_⟩
  set y := M.act a M.calm x
  have hy : M.act (rev a) M.calm y ≠ y := by rw [hback a x hne]; exact fun h => hne h.symm
  rw [hrate a x hne, hrate (rev a) y hy, hback a x hne, hκ]
  have e1 : Real.exp (-E x) * (κ a * Real.exp (-(E y - E x) / 2)) =
      κ a * Real.exp (-(E x + E y) / 2) := by
    rw [mul_left_comm, ← Real.exp_add]; congr 2; ring
  have e2 : Real.exp (-E y) * (κ a * Real.exp (-(E x - E y) / 2)) =
      κ a * Real.exp (-(E x + E y) / 2) := by
    rw [mul_left_comm, ← Real.exp_add]; congr 2; ring
  rw [e1, e2]

/-- **M1b**: a rate that is a function of the participants' states through the event's
energy difference reads only the participants, when that difference does. -/
theorem rate_local_of_energy (E : (U → Λ) → ℝ) (κ : A → ℝ) (a : A) (s s' : U → I)
    (x y : U → Λ)
    (hloc : ∀ x y : U → Λ, (∀ u ∈ M.parts a, x u = y u) →
      E (M.act a s x) - E x = E (M.act a s' y) - E y)
    (hrx : M.rate a s x = κ a * Real.exp (-(E (M.act a s x) - E x) / 2))
    (hry : M.rate a s' y = κ a * Real.exp (-(E (M.act a s' y) - E y) / 2))
    (hxy : ∀ u ∈ M.parts a, x u = y u) : M.rate a s x = M.rate a s' y := by
  rw [hrx, hry, hloc x y hxy]

end CGB.Medium
