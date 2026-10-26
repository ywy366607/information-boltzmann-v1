import CGBAxioms.Medium

/-!
# Nesting: an organism is a part of one medium, its boundary induced by contact

There is one medium. An organism is a set `O` of its units; the world is the rest (and may hold
other organisms). Nothing is declared about the interface: the boundary is whatever touches
the other side, and interaction is the medium's ordinary events with participants on both
sides. The theorems below are consequences of the medium's own axioms (P1b, P2a, P2b); they
hold for any set of units, so they recurse (a mitochondrion in a cell in a body in a collective)
and they hold for a set chosen from the current state (an organism defined by its states, which
can move through the medium).

* `touches`, `outer`: units outside `S` in contact with `S` (contact taken in either direction).
* **T-N1** `blanket_of_contact`: any event with a participant in `S` has rate and effect on `S`
  that read nothing outside `S ∪ outer S`; events with no participant in `S` leave `S` unchanged
  (`untouched_unchanged`). Applied to the organism `O` this is the organism's half of the
  Markov blanket; applied to the complement it is the world's half, which `Life.World` had to
  assume (`ReadsBoundary`). Both halves come from P1b.
* **T-N2** `exchange_conserves`: a closed event (no unit on the medium's own outer boundary)
  moves content between `O` and the rest without creating any: the gain on one side is the loss
  on the other.
* **T-N3** `freezeOutside_balance`: in the calm environment, with the rest frozen, the
  organism's own moves satisfy detailed balance with the joint weight (G13 for the split into
  organism and world).
-/

open Finset

namespace CGB

namespace Medium

variable {U Λ A I : Type*} (M : Medium U Λ A I)

/-- `u` is in contact with some unit of `S`, in either direction. -/
def touches (S : Set U) (u : U) : Prop := ∃ v ∈ S, M.near u v ∨ M.near v u

/-- The units outside `S` that touch it: the other side of `S`'s boundary. -/
def outer (S : Set U) : Set U := {u | u ∉ S ∧ M.touches S u}

/-- `S`'s own boundary: the units of `S` touching the outside. It is `outer` of the
complement. -/
theorem boundary_eq_outer_compl (S : Set U) :
    {u | u ∈ S ∧ M.touches Sᶜ u} = M.outer Sᶜ := by
  ext u; simp [outer]

/-- A participant of an event that also has a participant in `S` is in `S` or touches it. -/
theorem participant_near (h : M.P1b) {S : Set U} {a : A} {u : U} (hu : u ∈ M.parts a)
    (huS : u ∈ S) {v : U} (hv : v ∈ M.parts a) : v ∈ S ∨ v ∈ M.outer S := by
  by_cases hvS : v ∈ S
  · exact Or.inl hvS
  · exact Or.inr ⟨hvS, u, huS, Or.inr (h.1 a u hu v hv)⟩

/-- **T-N1 (the blanket from contact)**: two states that agree on `S` and on the units touching
it give every event with a participant in `S` the same rate and the same effect on `S`. -/
theorem blanket_of_contact (h : M.P1b) (S : Set U) {x y : U → Λ} (s : U → I)
    (hag : ∀ u, u ∈ S ∨ u ∈ M.outer S → x u = y u) {a : A} (ha : ∃ u ∈ M.parts a, u ∈ S) :
    M.rate a s x = M.rate a s y ∧ ∀ u ∈ S, M.act a s x u = M.act a s y u := by
  obtain ⟨w, hw, hwS⟩ := ha
  have hpart : ∀ u ∈ M.parts a, x u = y u ∧ s u = s u :=
    fun u hu => ⟨hag u (M.participant_near h hw hwS hu), rfl⟩
  obtain ⟨hr, hact⟩ := h.2.2 a s s x y hpart
  refine ⟨hr, fun u huS => ?_⟩
  by_cases hu : u ∈ M.parts a
  · exact hact u hu
  · rw [h.2.1 a s x u hu, h.2.1 a s y u hu]
    exact hag u (Or.inl huS)

/-- Events with no participant in `S` leave `S` unchanged. -/
theorem untouched_unchanged (h : M.P1b) (S : Set U) {a : A} (ha : ∀ u ∈ M.parts a, u ∉ S)
    (s : U → I) (x : U → Λ) : ∀ u ∈ S, M.act a s x u = x u :=
  fun u huS => h.2.1 a s x u (fun hu => ha u hu huS)

/-- The organism's half: what happens to `O` reads only `O` and the units touching it. -/
theorem organism_blanket (h : M.P1b) (O : Set U) {x y : U → Λ} (s : U → I)
    (hag : ∀ u, u ∈ O ∨ u ∈ M.outer O → x u = y u) {a : A} (ha : ∃ u ∈ M.parts a, u ∈ O) :
    M.rate a s x = M.rate a s y ∧ ∀ u ∈ O, M.act a s x u = M.act a s y u :=
  M.blanket_of_contact h O s hag ha

/-- The world's half (formerly the assumption `ReadsBoundary`): what happens to the rest reads
only the rest and the organism's own boundary units. -/
theorem world_blanket (h : M.P1b) (O : Set U) {x y : U → Λ} (s : U → I)
    (hag : ∀ u, u ∉ O ∨ (u ∈ O ∧ M.touches Oᶜ u) → x u = y u) {a : A}
    (ha : ∃ u ∈ M.parts a, u ∉ O) :
    M.rate a s x = M.rate a s y ∧ ∀ u, u ∉ O → M.act a s x u = M.act a s y u := by
  have hag' : ∀ u, u ∈ Oᶜ ∨ u ∈ M.outer Oᶜ → x u = y u := by
    intro u hu
    rcases hu with hu | ⟨hu, ht⟩
    · exact hag u (Or.inl hu)
    · exact hag u (Or.inr ⟨by simpa using hu, ht⟩)
  exact M.blanket_of_contact h Oᶜ s hag' ha

/-- **T-N2 (exchange conserves)**: a closed event with positive rate changes the content of `O`
by exactly minus the change of the rest. -/
theorem exchange_conserves [Fintype U] [DecidableEq U] (h : M.P2a) (O : Finset U) {a : A} {s : U → I}
    {x : U → Λ} (hc : M.Closed a) (hr : 0 < M.rate a s x) :
    ((∑ u ∈ O, M.content (M.act a s x u) : ℕ) : ℤ) - ∑ u ∈ O, M.content (x u) =
      -(((∑ u ∈ Oᶜ, M.content (M.act a s x u) : ℕ) : ℤ) - ∑ u ∈ Oᶜ, M.content (x u)) := by
  have htot := h a s x hc hr
  unfold total at htot
  rw [← Finset.sum_add_sum_compl O, ← Finset.sum_add_sum_compl O (fun u => M.content (x u))]
    at htot
  have : ((∑ u ∈ O, M.content (M.act a s x u) + ∑ u ∈ Oᶜ, M.content (M.act a s x u) : ℕ) : ℤ) =
      ((∑ u ∈ O, M.content (x u) + ∑ u ∈ Oᶜ, M.content (x u) : ℕ) : ℤ) := by
    exact_mod_cast htot
  push_cast at this ⊢
  linarith

end Medium

namespace JumpProcess

variable {S : Type*}

open Classical in
/-- The dynamics with the rest of the medium frozen: only moves that keep the outside of `O`
survive. -/
noncomputable def freezeOutside {U Λ : Type*} (K : JumpProcess (U → Λ)) (O : Set U) :
    JumpProcess (U → Λ) where
  rate x y := if ∀ u, u ∉ O → x u = y u then K.rate x y else 0
  rate_nonneg x y := by split_ifs <;> simp [K.rate_nonneg]

/-- **T-N3 (the organism's own moves are balanced with the world frozen)**: if the joint calm
dynamics has detailed-balance weight `π`, the dynamics with the outside of `O` frozen has the
same detailed balance. (G13 for the split into organism and world.) -/
theorem freezeOutside_balance {U Λ : Type*} (K : JumpProcess (U → Λ)) (O : Set U)
    {π : (U → Λ) → ℝ} (hdb : ∀ x y, π x * K.rate x y = π y * K.rate y x) (x y : U → Λ) :
    π x * (K.freezeOutside O).rate x y = π y * (K.freezeOutside O).rate y x := by
  by_cases hxy : ∀ u, u ∉ O → x u = y u
  · have hyx : ∀ u, u ∉ O → y u = x u := fun u hu => (hxy u hu).symm
    simp only [freezeOutside, eq_true hxy, eq_true hyx, ite_true]
    exact hdb x y
  · have hyx : ¬ ∀ u, u ∉ O → y u = x u := fun h' => hxy fun u hu => (h' u hu).symm
    simp only [freezeOutside, eq_false hxy, eq_false hyx, ite_false, mul_zero]

end JumpProcess

end CGB
