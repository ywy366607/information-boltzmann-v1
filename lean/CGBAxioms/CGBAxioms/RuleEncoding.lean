import CGBAxioms.Foundation
import Mathlib.Tactic.DeriveEncodable
import Mathlib.Data.Nat.Size

/-!
# A fixed finite rule language and an honest description budget

This is a representation layer, not a seventh physical axiom. A book denotes a
finite rational-rate medium; membership in P1--P3 must still be proved separately.
The reference language is deliberately tabular, not a claim of compression.
Every port, rate, initial state, identity and contact is charged. The interpreter
below is fixed across books and contains no data-dependent external lookup.
-/

namespace CGB.Description

/-- One participant's required state/stimulus and proposed next state. -/
structure Port where
  site : ℕ
  before : ℕ
  stimulus : ℕ
  after : ℕ
  deriving Encodable, DecidableEq

/-- A guarded local event; its exact rate is numerator / (denominator + 1). -/
structure Event where
  ports : List Port
  numerator : ℕ
  denominator : ℕ
  deriving Encodable, DecidableEq

/-- Complete instance data. State/stimulus alphabet sizes are successor-valued,
so even an empty book has a total interpreter. No adult adjacency is free. -/
structure Book where
  sites : ℕ
  states : ℕ
  stimuli : ℕ
  contacts : List (ℕ × ℕ)
  identities : List ℕ
  boundary : List ℕ
  contents : List ℕ
  initial : List ℕ
  calm : List ℕ
  events : List Event
  deriving Encodable, DecidableEq

/-- Binary length in the registered Encodable convention, with one envelope bit.
This is a fixed-code length, never a computable Kolmogorov minimum. The surrounding
record is encoded once, so field boundaries and list lengths are also charged. -/
def bits {D : Type*} [Encodable D] (d : D) : ℕ := Nat.size (Encodable.encode d) + 1

theorem code_roundtrip (b : Book) : Encodable.decode (Encodable.encode b) = some b :=
  Encodable.encodek b

theorem different_books_different_codes {a b : Book}
    (h : Encodable.encode a = Encodable.encode b) : a = b :=
  Encodable.encode_injective h

/-- At most 2^B distinct descriptions fit in B bits (an intentionally loose bound). -/
theorem bounded_books (B : ℕ) (s : Finset Book) (h : ∀ b ∈ s, bits b ≤ B) :
    s.card ≤ 2 ^ B := by
  have hinj : Set.InjOn (Encodable.encode : Book → ℕ) s :=
    fun _ _ _ _ he => Encodable.encode_injective he
  calc
    s.card = (s.image Encodable.encode).card := (Finset.card_image_of_injOn hinj).symm
    _ ≤ (Finset.range (2 ^ B)).card := Finset.card_le_card (by
      intro n hn
      obtain ⟨b, hb, rfl⟩ := Finset.mem_image.1 hn
      apply Finset.mem_range.2
      apply Nat.size_le.1
      have := h b hb
      unfold bits at this
      omega)
    _ = 2 ^ B := Finset.card_range _

/-- A nonzero comparison encoding and a rational bottleneck parameter. The graph
reference must be registered independently, not selected to inflate this denominator. -/
def WithinBudget (b : Book) (referenceBits numerator denominator : ℕ) : Prop :=
  0 < referenceBits ∧ 0 < denominator ∧ denominator * bits b ≤ numerator * referenceBits

theorem one_percent_budget (b : Book) (G : ℕ) :
    WithinBudget b G 1 100 ↔ 0 < G ∧ 100 * bits b ≤ G := by
  simp [WithinBudget]

/-- A concrete comparison convention: directed unweighted dense wiring on a
registered set of n labelled sites, including diagonal bits. This baseline is
fixed before comparing programs; it is not an arbitrarily large supplied number. -/
structure WiringReference (n : ℕ) where
  edge : Fin n → Fin n → Bool

def WiringReference.payload {n : ℕ} (g : WiringReference n) : List Bool :=
  (List.finRange n).flatMap fun u => (List.finRange n).map fun v => g.edge u v

theorem WiringReference.payload_length {n : ℕ} (g : WiringReference n) :
    g.payload.length = n * n := by
  simp [payload, List.length_flatMap, List.sum_replicate]

/-- With this convention, 1/100 means a genuine total-instance bit comparison.
It is not invariant under choosing another coding language or graph convention. -/
def GenomeBudget (b : Book) (g : WiringReference b.sites) (num den : ℕ) : Prop :=
  WithinBudget b g.payload.length num den

namespace Book

abbrev Site (b : Book) := Fin b.sites
abbrev State (b : Book) := Fin (b.states + 1)
abbrev Stim (b : Book) := Fin (b.stimuli + 1)
abbrev Action (b : Book) := Fin b.events.length

def event (b : Book) (a : b.Action) : Event := b.events[a]

/-- An explicit bounded-work reference scan. Exhaustion is reported, not replaced
by an answer. Fuel counts list-node inspections, NOT machine cycles or bytes. -/
def findWithin {D : Type*} (p : D → Bool) : ℕ → List D → Option (Option D)
  | _, [] => some none
  | 0, _ :: _ => none
  | fuel + 1, x :: xs => if p x then some (some x) else findWithin p fuel xs

theorem findWithin_complete {D : Type*} (p : D → Bool) (xs : List D)
    (fuel : ℕ) (h : xs.length ≤ fuel) : findWithin p fuel xs = some (xs.find? p) := by
  induction xs generalizing fuel with
  | nil => simp [findWithin]
  | cons x xs ih =>
    cases fuel with
    | zero => simp at h
    | succ fuel =>
      have hh : xs.length ≤ fuel := by simpa using h
      simp only [findWithin, List.find?_cons]
      split <;> simp_all [ih fuel hh]

/-- Port matching and updates scan finite lists. Duplicate ports use the first
update; out-of-range sites cannot match. These total defaults are explicit, not
unaccounted parameters. Well-formedness may rule them out in an admitted model. -/
def enabled (b : Book) (a : b.Action) (s : b.Site → b.Stim) (x : b.Site → b.State) : Bool :=
  (b.event a).ports.all fun p =>
    if h : p.site < b.sites then
      (x ⟨p.site, h⟩).val == p.before && (s ⟨p.site, h⟩).val == p.stimulus
    else false

def update (b : Book) (a : b.Action) (x : b.Site → b.State) (u : b.Site) : b.State :=
  match (b.event a).ports.find? (fun p => p.site == u.val) with
  | none => x u
  | some p => ⟨p.after % (b.states + 1), Nat.mod_lt _ (Nat.zero_lt_succ _)⟩

/-- Resource-bounded version of the SAME local update, not a new update rule. -/
def updateWithin (b : Book) (a : b.Action) (x : b.Site → b.State) (u : b.Site)
    (fuel : ℕ) : Option b.State :=
  (findWithin (fun p => p.site == u.val) fuel (b.event a).ports).map fun result =>
    match result with
    | none => x u
    | some p => ⟨p.after % (b.states + 1), Nat.mod_lt _ (Nat.zero_lt_succ _)⟩

theorem updateWithin_complete (b : Book) (a : b.Action) (x : b.Site → b.State)
    (u : b.Site) (fuel : ℕ) (h : (b.event a).ports.length ≤ fuel) :
    b.updateWithin a x u fuel = some (b.update a x u) := by
  simp only [updateWithin, findWithin_complete _ _ fuel h, Option.map_some, update]

def weight (b : Book) (a : b.Action) : ℚ :=
  (b.event a).numerator / ((b.event a).denominator + 1 : ℚ)

theorem weight_nonneg (b : Book) (a : b.Action) : 0 ≤ b.weight a := by
  exact div_nonneg (Nat.cast_nonneg _) (by positivity)

/-- Executable rational event rate. Casting to reals below is only a semantic
embedding of this finite rational representation, not arbitrary real data. -/
def rationalRate (b : Book) (a : b.Action) (s : b.Site → b.Stim) (x : b.Site → b.State) : ℚ :=
  if b.enabled a s x then b.weight a else 0

def apply (b : Book) (a : b.Action) (s : b.Site → b.Stim) (x : b.Site → b.State) :
    b.Site → b.State := if b.enabled a s x then b.update a x else x

noncomputable def medium (b : Book) : Medium b.Site b.State b.Action b.Stim where
  near u v := (u.val, v.val) ∈ b.contacts
  ident u := b.identities[u.val]?.getD 0
  boundary := Finset.univ.filter (fun u => u.val ∈ b.boundary)
  content l := b.contents[l.val]?.getD 0
  parts a := Finset.univ.filter (fun u => ∃ p ∈ (b.event a).ports, p.site = u.val)
  act := b.apply
  rate a s x := (b.rationalRate a s x : ℝ)
  rate_nonneg a s x := by
    apply Rat.cast_nonneg.2
    unfold rationalRate
    split <;> simp_all [b.weight_nonneg]
  init u := ⟨(b.initial[u.val]?.getD 0) % (b.states + 1), Nat.mod_lt _ (Nat.zero_lt_succ _)⟩
  calm u := ⟨(b.calm[u.val]?.getD 0) % (b.stimuli + 1), Nat.mod_lt _ (Nat.zero_lt_succ _)⟩

/-- Equality of codes entails equality of the whole interpreted instance.
There is no arbitrary Medium carried beside a decorative short code. -/
theorem same_code_same_medium {a b : Book} (h : Encodable.encode a = Encodable.encode b) :
    HEq a.medium b.medium := by
  obtain rfl := different_books_different_codes h
  rfl

/-- Locality of effects is a theorem of the interpreter, not a user-supplied promise. -/
theorem unchanged_outside (b : Book) (a : b.Action) (s : b.Site → b.Stim)
    (x : b.Site → b.State) (u : b.Site) (h : u ∉ b.medium.parts a) :
    b.medium.act a s x u = x u := by
  have hports : ∀ p ∈ (b.event a).ports, p.site ≠ u.val := by
    simpa [medium] using h
  have hfind : (b.event a).ports.find? (fun p => p.site == u.val) = none := by
    exact List.find?_eq_none.2 (by simpa using hports)
  change (b.apply a s x) u = x u
  unfold apply
  split <;> simp [update, hfind]

/-- Finite alphabets and this finite rulebook admit some uniform clock. Whether
it fits a *given* world's clock is still a separate admission obligation. -/
theorem admits_some_clock (b : Book) : ∃ R, b.medium.Clocked R :=
  b.medium.exists_clock_of_finite_stimuli

end Book
end CGB.Description
