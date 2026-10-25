import CGBAxioms.Medium

/-!
# Generation, layer 1: what the axioms force on the equilibrium law

The axioms constrain; this file starts deriving. Under P1b an event reads and writes
only its participants, and participants are pairwise in contact. Under P2b the calm
dynamics has a positive detailed-balance weight `π` (`energy_of_p2b`: `E = -log π`).

* `jump_rate_local` (G1): the total rate of a transition depends only on the changed
  set `D`, the new values on `D`, and the old state and stimulus on `D ∪ scope D`, where
  `scope D` is the set of units in contact with every unit of `D`.
* `equilibrium_ratio_local` (G1E): hence every equilibrium ratio `π y / π x` across one
  transition with a positive reverse rate is a function of that same local data.
* `content_fixed_by_closed_single_unit` (G2): under P2a a closed event that changes one
  unit cannot change its content; content moves inside the body only by transfers that
  change at least two units.
* `JumpProcess.reversible_mul_invariant` (G3): multiplying a detailed-balance weight by
  any positive function that is constant along every move keeps detailed balance. Local
  energy differences therefore fix `π` only up to functions of the move invariants; a
  global coupling can enter through them (for example a conserved total under transfers).
* `local_markov_of_single_unit_moves` (G4): if every value of unit `u` is reachable by a
  calm move that changes only `u`, then the conditional ratio
  `π (x with u := λ) / π x` depends only on `x` on `{u} ∪ scope {u}`: the equilibrium law is
  locally Markov at `u` for the contact relation.

Not claimed: that P1b and P2b alone give a decomposition into local potentials (G3 shows
how it fails), the converse of G3 (that move invariants are the only source of non-local
terms), or where single-unit moves come from. Detailed balance fixes rate ratios only;
the symmetric conductance remains free (`ConstraintAudit.reversible_conductance_representation`).
-/

open Finset

namespace CGB

namespace JumpProcess

variable {S : Type*}

/-- **G3**: a positive factor that is constant along every move preserves detailed
balance. -/
theorem reversible_mul_invariant (P : JumpProcess S) {π w : S → ℝ} (hp : ∀ x, 0 < π x)
    (hdb : ∀ x y, π x * P.rate x y = π y * P.rate y x) (hw : ∀ x, 0 < w x)
    (hinv : ∀ x y, 0 < P.rate x y → w x = w y) :
    (∀ x, 0 < π x * w x) ∧ ∀ x y, π x * w x * P.rate x y = π y * w y * P.rate y x := by
  refine ⟨fun x => mul_pos (hp x) (hw x), fun x y => ?_⟩
  rcases (P.rate_nonneg x y).lt_or_eq with hxy | hxy
  · rw [hinv x y hxy]
    linear_combination w y * hdb x y
  · have hyx : P.rate y x = 0 := by
      have := hdb x y
      rw [← hxy, mul_zero] at this
      exact (mul_eq_zero.1 this.symm).resolve_left (hp y).ne'
    rw [← hxy, hyx]
    ring

end JumpProcess

namespace Medium

variable {U Λ A I : Type*} (M : Medium U Λ A I)

/-- Units in contact with every unit of `D`: where an event that changes all of `D` may
read. -/
def scope (D : Set U) : Set U := {v | ∀ u ∈ D, M.near u v}

/-- Under P1b an event that changes all of `D` reads only inside `scope D`. -/
lemma parts_subset_scope (h : M.P1b) {a : A} {D : Set U} (hD : ∀ u ∈ D, u ∈ M.parts a) :
    ∀ v ∈ M.parts a, v ∈ M.scope D :=
  fun v hv u hu => h.2.1 a u (hD u hu) v hv

/-- **G2**: under P2a a closed event that changes a single unit keeps that unit's content,
so content moves inside the body only by events that change at least two units. -/
theorem content_fixed_by_closed_single_unit [Fintype U] (h : M.P2a) {a : A} {s : U → I}
    {x : U → Λ} (u : U) (ha : M.Closed a) (hr : 0 < M.rate a s x)
    (hone : ∀ v, v ≠ u → M.act a s x v = x v) :
    M.content (M.act a s x u) = M.content (x u) := by
  classical
  have ht := h a s x ha hr
  simp only [total] at ht
  rw [← Finset.add_sum_erase _ _ (Finset.mem_univ u),
    ← Finset.add_sum_erase _ _ (Finset.mem_univ u)] at ht
  have hrest : ∑ v ∈ univ.erase u, M.content (M.act a s x v) =
      ∑ v ∈ univ.erase u, M.content (x v) :=
    Finset.sum_congr rfl fun v hv => by rw [hone v (Finset.ne_of_mem_erase hv)]
  omega

variable [Fintype A] [DecidableEq (U → Λ)]

/-- **G1**: the total rate of a transition `x → y` with changed set `D` depends only on the
new values on `D` and on the old state and stimulus on `D ∪ scope D`. -/
theorem jump_rate_local (h : M.P1b) {s s' : U → I} {x y x' y' : U → Λ} (D : Set U)
    (hD : ∀ u, u ∈ D ↔ x u ≠ y u) (hy : ∀ u, u ∉ D → y' u = x' u)
    (hyD : ∀ u ∈ D, y' u = y u) (hx : ∀ u, u ∈ D ∨ u ∈ M.scope D → x' u = x u)
    (hs : ∀ u, u ∈ D ∨ u ∈ M.scope D → s' u = s u) :
    (M.jump s').rate x' y' = (M.jump s).rate x y := by
  classical
  have hne : x' ≠ y' ↔ x ≠ y := by
    constructor
    · rintro hne rfl
      exact hne (funext fun u => by
        by_cases hu : u ∈ D
        · exact absurd rfl ((hD u).1 hu)
        · exact (hy u hu).symm)
    · intro hne heq
      obtain ⟨u, hu⟩ := Function.ne_iff.1 hne
      have huD := (hD u).2 hu
      exact hu (by rw [← hx u (Or.inl huD), ← hyD u huD, heq])
  simp only [jump]
  refine Finset.sum_congr rfl fun a _ => ?_
  by_cases hsub : ∀ u ∈ D, u ∈ M.parts a
  · have hsc := M.parts_subset_scope h hsub
    obtain ⟨hr, hact⟩ := h.2.2.2 a s' s x' x fun u hu =>
      ⟨hx u (Or.inr (hsc u hu)), hs u (Or.inr (hsc u hu))⟩
    have hyy : ∀ u ∈ M.parts a, y' u = y u := fun u hu => by
      by_cases huD : u ∈ D
      · exact hyD u huD
      · rw [hy u huD, hx u (Or.inr (hsc u hu))]
        exact not_not.1 fun hn => huD ((hD u).2 hn)
    have hcond : M.act a s' x' = y' ↔ M.act a s x = y := by
      constructor
      · intro he
        funext u
        by_cases hu : u ∈ M.parts a
        · rw [← hact u hu, he, hyy u hu]
        · rw [h.2.2.1 a s x u hu]
          exact not_not.1 fun hn => hu (hsub u ((hD u).2 hn))
      · intro he
        funext u
        by_cases hu : u ∈ M.parts a
        · rw [hact u hu, he, hyy u hu]
        · rw [h.2.2.1 a s' x' u hu]
          exact (hy u fun huD => hu (hsub u huD)).symm
    simp only [hcond, hne, hr]
  · push Not at hsub
    obtain ⟨u, huD, hu⟩ := hsub
    have h1 : M.act a s x ≠ y := fun he =>
      (hD u).1 huD (by rw [← he, h.2.2.1 a s x u hu])
    have h2 : M.act a s' x' ≠ y' := fun he =>
      (hD u).1 huD (by rw [← hx u (Or.inl huD), ← hyD u huD, ← he, h.2.2.1 a s' x' u hu])
    simp [h1, h2]

/-- **G1E**: across one calm transition with a positive reverse rate, the equilibrium
ratio `π y / π x` depends only on the same local data as G1. -/
theorem equilibrium_ratio_local (h : M.P1b) {π : (U → Λ) → ℝ} (hp : ∀ x, 0 < π x)
    (hdb : ∀ x y, π x * (M.jump M.calm).rate x y = π y * (M.jump M.calm).rate y x)
    {x y x' y' : U → Λ} (D : Set U)
    (hD : ∀ u, u ∈ D ↔ x u ≠ y u) (hy : ∀ u, u ∉ D → y' u = x' u)
    (hyD : ∀ u ∈ D, y' u = y u) (hx : ∀ u, u ∈ D ∨ u ∈ M.scope D → x' u = x u)
    (hpos : 0 < (M.jump M.calm).rate y x) :
    π y' / π x' = π y / π x := by
  classical
  have hfw := M.jump_rate_local (s := M.calm) (s' := M.calm) h D hD hy hyD hx fun _ _ => rfl
  have hbw : (M.jump M.calm).rate y' x' = (M.jump M.calm).rate y x := by
    refine M.jump_rate_local (s := M.calm) (s' := M.calm) h D (fun u => by rw [hD u]; exact ne_comm)
      (fun u hu => (hy u hu).symm) (fun u hu => hx u (Or.inl hu)) (fun u hu => ?_)
      fun _ _ => rfl
    by_cases huD : u ∈ D
    · exact hyD u huD
    · rw [hy u huD, hx u hu]
      exact not_not.1 fun hn => huD ((hD u).2 hn)
  have ratio : ∀ {a b}, 0 < (M.jump M.calm).rate b a →
      π b / π a = (M.jump M.calm).rate a b / (M.jump M.calm).rate b a := fun {a b} hr => by
    rw [div_eq_div_iff (hp a).ne' hr.ne']
    linear_combination (hdb a b).symm
  rw [ratio (hbw ▸ hpos), ratio hpos, hfw, hbw]

/-- **G4**: if every value of unit `u` can be left by a calm move that changes only `u`, the
equilibrium law is locally Markov at `u`: the ratio `π (x with u := l) / π x` depends only
on `x` on `{u} ∪ scope {u}`. -/
theorem local_markov_of_single_unit_moves [DecidableEq U] (h : M.P1b) {π : (U → Λ) → ℝ}
    (hp : ∀ x, 0 < π x)
    (hdb : ∀ x y, π x * (M.jump M.calm).rate x y = π y * (M.jump M.calm).rate y x) (u : U)
    (hmove : ∀ (x : U → Λ) (l : Λ), l ≠ x u →
      0 < (M.jump M.calm).rate (Function.update x u l) x)
    {x x' : U → Λ} (hx : ∀ v, v = u ∨ v ∈ M.scope {u} → x' v = x v) (l : Λ) :
    π (Function.update x' u l) / π x' = π (Function.update x u l) / π x := by
  by_cases hl : l = x u
  · have hl' : l = x' u := by rw [hx u (Or.inl rfl), hl]
    rw [show Function.update x u l = x by rw [hl]; exact Function.update_eq_self u x,
      show Function.update x' u l = x' by rw [hl']; exact Function.update_eq_self u x',
      div_self (hp x).ne', div_self (hp x').ne']
  · refine M.equilibrium_ratio_local h hp hdb {u} (fun v => ?_) (fun v hv => ?_)
      (fun v hv => ?_) (fun v hv => hx v (by simpa using hv)) (hmove x l hl)
    · by_cases hv : v = u
      · subst hv
        simp [Ne.symm hl]
      · simp [hv, Function.update_of_ne hv]
    · exact Function.update_of_ne (by simpa using hv) _ _
    · rw [Set.mem_singleton_iff.1 hv]
      simp

end Medium

end CGB
