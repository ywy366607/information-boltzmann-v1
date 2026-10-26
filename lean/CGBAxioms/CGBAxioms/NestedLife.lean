import CGBAxioms.Nesting

/-!
# Life of a nested organism

Once the organism is part of one medium with its world (`Nesting`), it has no outside stimulus:
what it senses is the state of the world next to it, and what it takes in comes out of the world.
The life notions are restated for a *part* of the medium, given by the organism's own state
(wherever it is, so it may move) and the content it holds. They are the old `Life` notions with
the stimulus replaced by the rest of the medium.

* `Part.Senses`: from the same own state, the same event changes the organism at a different
  rate when only the rest of the medium differs (old `Exposed`).
* `Part.Metabolizes`: some event with positive rate changes the content the organism holds
  (old `Lives.metabolizes`).
* `Part.Recoverable`: along events of the joint medium the organism can still reach a state
  where it takes content in; losing this is irreversible (`irrecoverable_absorbing`), as in
  the old `Recoverable`, now with the world's own events included.
* `no_intake_of_closed`: an organism that never exchanges content cannot take anything in.
-/

namespace CGB

/-- A part of a medium: the organism's own state and the content it holds. -/
structure Part (U Λ B : Type*) where
  own : (U → Λ) → B
  mass : (U → Λ) → ℕ

namespace Part

variable {U Λ A I B : Type*} (M : Medium U Λ A I) (O : Part U Λ B)

/-- **Nested sensing**: the same event, from the same own state, changes the organism at a
different rate when only the rest of the medium differs. -/
def Senses : Prop :=
  ∃ a s x x', O.own x = O.own x' ∧ O.own (M.act a s x) ≠ O.own x ∧ M.rate a s x ≠ M.rate a s x'

/-- **Nested metabolism**: some event with positive rate changes the organism's content. -/
def Metabolizes : Prop := ∃ a s x, 0 < M.rate a s x ∧ O.mass (M.act a s x) ≠ O.mass x

/-- One event of the joint medium. -/
def Step (x y : U → Λ) : Prop := ∃ a s, 0 < M.rate a s x ∧ M.act a s x = y

/-- The organism can take content in now. -/
def CanTakeIn (x : U → Λ) : Prop := ∃ a s, 0 < M.rate a s x ∧ O.mass x < O.mass (M.act a s x)

/-- **Nested recoverability**: along events of the joint medium the organism can still take
content in. -/
def Recoverable (x : U → Λ) : Prop :=
  ∃ y, Relation.ReflTransGen (Step M) x y ∧ O.CanTakeIn M y

theorem irrecoverable_absorbing {x y : U → Λ} (hd : ¬ O.Recoverable M x) (h : Step M x y) :
    ¬ O.Recoverable M y :=
  fun ⟨z, hz, hi⟩ => hd ⟨z, Relation.ReflTransGen.head h hz, hi⟩

/-- An organism that never exchanges content cannot take anything in, so nothing is
recoverable for it: metabolism is necessary for nested life. -/
theorem no_intake_of_closed (h : ¬ O.Metabolizes M) (x : U → Λ) : ¬ O.Recoverable M x := by
  rintro ⟨y, -, a, s, hr, hlt⟩
  exact h ⟨a, s, y, hr, (ne_of_lt hlt).symm⟩

end Part

end CGB
