import CGBAxioms.ConstraintFreedom
import CGBAxioms.GenerationColumnRefinement
import CGBAxioms.GenerationContextAssociation

set_option linter.unusedSectionVars false
set_option linter.deprecated false
set_option linter.unusedVariables false

/-!
# The context-cell language body as a Medium

`ConstraintFreedom.arbitrary_driven_tables_admitted` shows that one exposed site
satisfies every clause with an arbitrary table, so a Medium instance says nothing
about a body unless the body is actually spread over units.  This file gives the
language body as many units: one sense unit, one context unit, one pool unit and
one unit per cell.

An event names the context and an arrival.  A byte touches the sense, context
and pool units and only the cells that context addresses, so the number of units
an event touches is bounded by the address size, not by the number of cells.  A
prediction reads the same units.  A tick touches only the sense and context
units: the clock lives in the context, and a cell settles elapsed time through
`tally` and `score` when it is next addressed.  Everything else stays as it was.

The Model fields are free functions.  They are the Python methods of
`scripts/cgb_language_candidates.py:ContextCells`, so this file states the shape
of the body and not its capability: which `advance`, `tally`, `learn` and `score`
serve language is an empirical question for the language gate.  P2a holds because
every event has a boundary participant; P2c holds because identities are
distinct.  So the content of this instance is the locality of `region`, stated in
`events_touch_bounded`.
-/

open Finset
open scoped Classical

namespace CGB.Generation.ContextBody

abbrev Byte := Fin 256

/-- `Ctx`, `Pool` and `Count` are the finite states of the three kinds of unit.
`tally` and `score` see every addressed cell and the pool, because replacement compares the
cells of one address against the held total.  `pass` is a tick on the context. -/
structure Model (Ctx Pool Cell Count Tick : Type*) where
  region : Ctx → Finset Cell
  advance : Ctx → Byte → Ctx
  pass : Ctx → Tick → Ctx
  tally : (c : Ctx) → Byte → Pool → ({j // j ∈ region c} → Count) → {j // j ∈ region c} → Count
  learn : (c : Ctx) → Byte → ({j // j ∈ region c} → Count) → Pool → Pool
  score : (c : Ctx) → Pool → ({j // j ∈ region c} → Count) → Byte → ℝ
  birthCtx : Ctx
  birthPool : Pool
  birthCell : Count

inductive Site (Cell : Type*) where
  | sense | context | pool
  | cell (j : Cell)
  deriving DecidableEq

open Site

abbrev Local (Ctx Pool Count : Type*) := Option (Ctx ⊕ Pool ⊕ Count)
abbrev Arrival (Tick : Type*) := Byte ⊕ Tick

variable {Ctx Pool Cell Count Tick : Type*} (M : Model Ctx Pool Cell Count Tick)

def asCtx : Local Ctx Pool Count → Ctx
  | some (Sum.inl c) => c
  | _ => M.birthCtx

def asPool : Local Ctx Pool Count → Pool
  | some (Sum.inr (Sum.inl p)) => p
  | _ => M.birthPool

def asCount : Local Ctx Pool Count → Count
  | some (Sum.inr (Sum.inr n)) => n
  | _ => M.birthCell

/-- The units a byte touches at one context. -/
def touched [DecidableEq Cell] (c : Ctx) : Finset (Site Cell) :=
  insert sense (insert context (insert pool ((M.region c).image cell)))

/-- The cells of one address, as a local reading of the body. -/
def view (c : Ctx) (x : Site Cell → Local Ctx Pool Count) : {j // j ∈ M.region c} → Count :=
  fun j => asCount M (x (cell j.1))

section Instance

variable [DecidableEq Cell]

/-- The event is on only for its own arrival at the sense unit and its own context. -/
def active (a : Ctx × Arrival Tick) (s : Site Cell → Option (Arrival Tick))
    (x : Site Cell → Local Ctx Pool Count) : Prop :=
  s sense = some a.2 ∧ x context = some (Sum.inl a.1)

/-- A byte: the context advances, the pool learns, addressed cells tally. -/
noncomputable def writeByte (c : Ctx) (b : Byte) (x : Site Cell → Local Ctx Pool Count) :
    Site Cell → Local Ctx Pool Count
  | sense => x sense
  | context => some (Sum.inl (M.advance c b))
  | pool => some (Sum.inr (Sum.inl (M.learn c b (view M c x) (asPool M (x pool)))))
  | cell j => if h : j ∈ M.region c then
      some (Sum.inr (Sum.inr (M.tally c b (asPool M (x pool)) (view M c x) ⟨j, h⟩))) else x (cell j)

/-- A tick moves only the clock held in the context. -/
def writeTick (c : Ctx) (t : Tick) (x : Site Cell → Local Ctx Pool Count) :
    Site Cell → Local Ctx Pool Count
  | context => some (Sum.inl (M.pass c t))
  | u => x u

noncomputable def write (a : Ctx × Arrival Tick) :
    (Site Cell → Local Ctx Pool Count) → Site Cell → Local Ctx Pool Count :=
  match a.2 with
  | Sum.inl b => writeByte M a.1 b
  | Sum.inr t => writeTick M a.1 t

def parts (a : Ctx × Arrival Tick) : Finset (Site Cell) :=
  match a.2 with
  | Sum.inl _ => touched M a.1
  | Sum.inr _ => {sense, context}

/-- The body as a Medium.  Contact is sharing some context's address. -/
noncomputable def body [Fintype (Site Cell)] :
    Medium (Site Cell) (Local Ctx Pool Count) (Ctx × Arrival Tick) (Option (Arrival Tick)) where
  near u v := ∃ c, u ∈ touched M c ∧ v ∈ touched M c
  ident u := (Fintype.equivFin (Site Cell) u).val
  boundary := {sense}
  content := fun l => if l = none then 0 else 1
  parts := parts M
  act a s x := if active a s x then write M a x else x
  rate a s x := if active a s x then 1 else 0
  rate_nonneg a s x := by split <;> norm_num
  init
    | context => some (Sum.inl M.birthCtx)
    | pool => some (Sum.inr (Sum.inl M.birthPool))
    | cell _ => some (Sum.inr (Sum.inr M.birthCell))
    | sense => none
  calm _ := none

variable [Fintype (Site Cell)]

theorem sense_mem (c : Ctx) : (sense : Site Cell) ∈ touched M c := by simp [touched]
theorem context_mem (c : Ctx) : (context : Site Cell) ∈ touched M c := by simp [touched]
theorem pool_mem (c : Ctx) : (pool : Site Cell) ∈ touched M c := by simp [touched]
theorem cell_mem {c : Ctx} {j : Cell} (h : j ∈ M.region c) :
    (cell j : Site Cell) ∈ touched M c := by simp [touched, h]

theorem parts_subset (a : Ctx × Arrival Tick) : parts M a ⊆ touched M a.1 := by
  rcases a with ⟨c, b | t⟩
  · exact Finset.Subset.refl _
  · intro u hu
    simp only [parts, Finset.mem_insert, Finset.mem_singleton] at hu
    rcases hu with rfl | rfl
    · exact sense_mem M c
    · exact context_mem M c

theorem sense_part (a : Ctx × Arrival Tick) : (sense : Site Cell) ∈ parts M a := by
  rcases a with ⟨c, b | t⟩
  · exact sense_mem M c
  · simp [parts]

theorem context_part (a : Ctx × Arrival Tick) : (context : Site Cell) ∈ parts M a := by
  rcases a with ⟨c, b | t⟩
  · exact context_mem M c
  · simp [parts]

theorem body_rigid : (body M).Rigid :=
  Medium.rigid_of_injective_identity _ fun _ _ h =>
    (Fintype.equivFin (Site Cell)).injective (Fin.ext h)

theorem write_outside (a : Ctx × Arrival Tick) (x : Site Cell → Local Ctx Pool Count)
    {u : Site Cell} (hu : u ∉ parts M a) : write M a x u = x u := by
  rcases a with ⟨c, b | t⟩
  · change u ∉ touched M c at hu
    change writeByte M c b x u = x u
    cases u with
    | sense => rfl
    | context => exact absurd (context_mem M c) hu
    | pool => exact absurd (pool_mem M c) hu
    | cell j =>
        have hj : j ∉ M.region c := fun h => hu (cell_mem M h)
        simp [writeByte, hj]
  · change writeTick M c t x u = x u
    cases u with
    | context => exact absurd (context_part M (c, Sum.inr t)) hu
    | sense => rfl
    | pool => rfl
    | cell j => rfl

theorem write_congr (a : Ctx × Arrival Tick) {x y : Site Cell → Local Ctx Pool Count}
    (h : ∀ u ∈ parts M a, x u = y u) {u : Site Cell} (hu : u ∈ parts M a) :
    write M a x u = write M a y u := by
  rcases a with ⟨c, b | t⟩
  · have hview : view M c x = view M c y := funext fun j => by
      simp only [view, h (cell j.1) (cell_mem M j.2)]
    change writeByte M c b x u = writeByte M c b y u
    cases u with
    | sense => exact h sense (sense_mem M c)
    | context => rfl
    | pool => simp only [writeByte, hview, h pool (pool_mem M c)]
    | cell j =>
        by_cases hj : j ∈ M.region c
        · simp only [writeByte, hj, dif_pos, hview, h pool (pool_mem M c)]
        · simp only [writeByte, hj, dif_neg, not_false_eq_true]
          exact h (cell j) hu
  · change writeTick M c t x u = writeTick M c t y u
    cases u with
    | context => rfl
    | sense => exact h _ hu
    | pool => exact h _ hu
    | cell j => exact h _ hu

theorem body_p1b : (body M).P1b := by
  refine ⟨fun a u hu v hv => ⟨a.1, parts_subset M a hu, parts_subset M a hv⟩, ?_, ?_⟩
  · intro a s x u hu
    change (if active a s x then write M a x else x) u = x u
    split
    · exact write_outside M a x hu
    · rfl
  · intro a s s' x y hh
    have hs := (hh sense (sense_part M a)).2
    have hc := (hh context (context_part M a)).1
    have hactive : active a s x ↔ active a s' y := by
      simp only [active, hs, hc]
    constructor
    · change (if active a s x then (1 : ℝ) else 0) = if active a s' y then 1 else 0
      simp only [hactive]
    · intro u hu
      change (if active a s x then write M a x else x) u =
        (if active a s' y then write M a y else y) u
      by_cases ha : active a s' y
      · rw [if_pos (hactive.2 ha), if_pos ha]
        exact write_congr M a (fun v hv => (hh v hv).1) hu
      · rw [if_neg (fun h => ha (hactive.1 h)), if_neg ha]
        exact (hh u hu).1

theorem body_p2a : (body M).P2a :=
  Medium.p2a_of_every_event_open _ fun a hclosed =>
    (Finset.disjoint_left.1 hclosed (sense_part M a)) (Finset.mem_singleton_self _)

theorem calm_inactive (a : Ctx × Arrival Tick) (x : Site Cell → Local Ctx Pool Count) :
    ¬ active a (body M).calm x := by
  intro h
  exact absurd h.1 (by simp [body])

theorem body_p2b [Fintype Ctx] [Fintype Tick] [DecidableEq (Site Cell → Local Ctx Pool Count)] :
    (body M).P2b := by
  refine ⟨fun _ => 1, fun _ => one_pos, fun x y => ?_⟩
  have hzero : ∀ u v : Site Cell → Local Ctx Pool Count,
      ((body M).jump (body M).calm).rate u v = 0 := by
    intro u v
    refine Finset.sum_eq_zero fun a _ => ?_
    have hrate : (body M).rate a (body M).calm u = 0 := by
      change (if active a (body M).calm u then (1 : ℝ) else 0) = 0
      exact if_neg (calm_inactive M a u)
    simp [hrate]
  rw [hzero x y, hzero y x]

theorem body_p3 : (body M).P3 := by
  refine ⟨?_, Medium.initial_invariant_of_rigid _ (body_rigid M)⟩
  intro a s s' x hs
  have hsense : s sense = s' sense := hs sense (Finset.mem_singleton_self _)
  have hactive : active a s x ↔ active a s' x := by simp only [active, hsense]
  constructor
  · change (if active a s x then (1 : ℝ) else 0) = if active a s' x then 1 else 0
    simp only [hactive]
  · change (if active a s x then write M a x else x) = if active a s' x then write M a x else x
    simp only [hactive]

theorem body_p1a [Finite Ctx] [Finite Pool] [Finite Count] : (body M).P1a := by
  refine (Medium.p1a_iff_unique_vacuum _).2 ⟨none, by simp [body], fun l hl => ?_⟩
  by_contra hne
  have : ((body M).content l) = 1 := if_neg hne
  omega

/-- **O01.**  The body with finite unit states satisfies every clause. -/
theorem body_axioms [Finite Ctx] [Finite Pool] [Finite Count]
    [Fintype Ctx] [Fintype Tick] [DecidableEq (Site Cell → Local Ctx Pool Count)] :
    (body M).Axioms :=
  ⟨body_p1a M, body_p1b M, body_p2a M, body_p2b M,
    Medium.p2c_of_rigid _ (body_rigid M), body_p3 M⟩

/-- **O22.**  An event touches at most `B + 3` units, whatever the cell count,
and with more cells than the address reaches, some unit is left untouched. -/
theorem events_touch_bounded (B : ℕ) (hB : ∀ c, (M.region c).card ≤ B)
    (hsize : B + 3 < Fintype.card (Site Cell)) (a : Ctx × Arrival Tick) :
    ((body M).parts a).card ≤ B + 3 ∧ ∃ u, u ∉ (body M).parts a := by
  have htouched : (touched M a.1).card ≤ B + 3 := by
    calc (touched M a.1).card
        ≤ (insert context (insert pool ((M.region a.1).image cell))).card + 1 :=
          Finset.card_insert_le _ _
      _ ≤ (insert pool ((M.region a.1).image (cell : Cell → Site Cell))).card + 1 + 1 :=
          Nat.add_le_add_right (Finset.card_insert_le _ _) 1
      _ ≤ ((M.region a.1).image (cell : Cell → Site Cell)).card + 1 + 1 + 1 :=
          Nat.add_le_add_right (Nat.add_le_add_right (Finset.card_insert_le _ _) 1) 1
      _ ≤ (M.region a.1).card + 1 + 1 + 1 := by
          have := Finset.card_image_le (s := M.region a.1) (f := (cell : Cell → Site Cell))
          omega
      _ ≤ B + 3 := by have := hB a.1; omega
  have hcard : (parts M a).card ≤ B + 3 :=
    (Finset.card_le_card (parts_subset M a)).trans htouched
  refine ⟨hcard, ?_⟩
  by_contra hall
  push_neg at hall
  have huniv : parts M a = Finset.univ := Finset.eq_univ_of_forall hall
  have hall_card : (parts M a).card = Fintype.card (Site Cell) := by
    rw [huniv, Finset.card_univ]
  omega

/-- **O25.**  Plasticity that depends on the stimulus sits at the boundary (G9,
`modulation_needs_boundary`).  The premises are `body_p1b` and the sensing half of
`body_p3`, so this follows from the Medium instance; it is stated so that the gate can name it.
For this body it is close to vacuous: every event already has the sense unit as a part. -/
theorem modulation_at_boundary [Fintype Ctx] [Fintype Tick]
    [DecidableEq (Site Cell → Local Ctx Pool Count)]
    {s s' : Site Cell → Option (Arrival Tick)} {x y : Site Cell → Local Ctx Pool Count}
    (D : Set (Site Cell)) (hD : ∀ u, u ∈ D ↔ x u ≠ y u)
    (hdiff : ((body M).jump s).rate x y ≠ ((body M).jump s').rate x y) :
    ∃ u, (u ∈ D ∨ u ∈ (body M).scope D) ∧ u ∈ (body M).boundary :=
  (body M).modulation_needs_boundary (body_p1b M) (body_p3 M).1 D hD hdiff

end Instance

section Stream

variable [DecidableEq Cell] [Fintype (Site Cell)]

/-- The sense unit receives the arrival; the event follows from the held context. -/
noncomputable def event (x : Site Cell → Local Ctx Pool Count) (b : Arrival Tick) :
    Ctx × Arrival Tick :=
  (asCtx M (x context), b)

def stimulus (b : Arrival Tick) : Site Cell → Option (Arrival Tick) := fun _ => some b

noncomputable def step (x : Site Cell → Local Ctx Pool Count) (b : Arrival Tick) :
    Site Cell → Local Ctx Pool Count :=
  (body M).act (event M x b) (stimulus b) x

noncomputable def encode (h : List (Arrival Tick)) : Site Cell → Local Ctx Pool Count :=
  h.foldl (step M) (body M).init

/-- A prediction reads the context, the pool and only the addressed cells. -/
noncomputable def predict (x : Site Cell → Local Ctx Pool Count) : Byte → ℝ :=
  M.score (asCtx M (x context)) (asPool M (x pool)) (view M (asCtx M (x context)) x)

theorem step_active {x : Site Cell → Local Ctx Pool Count} {c : Ctx}
    (hc : x context = some (Sum.inl c)) (b : Arrival Tick) :
    step M x b = write M (c, b) x := by
  have hevent : event M x b = (c, b) := by simp [event, hc, asCtx]
  have hon : active (c, b) (stimulus b : Site Cell → Option (Arrival Tick)) x := ⟨rfl, hc⟩
  change (if active (event M x b) (stimulus b) x then write M (event M x b) x else x) = _
  rw [hevent, if_pos hon]

/-- **O07.**  The actual write holds order.  If `a` then `b` reaches a cell that
`b` then `a` never reaches, and that tally moves the count, the two states
differ.  This is `context_write_separates_order` for this body. -/
theorem order_separated {x : Site Cell → Local Ctx Pool Count} {c : Ctx} (a b : Byte)
    (hc : x context = some (Sum.inl c)) (j : Cell) {n : Count}
    (hn : x (cell j) = some (Sum.inr (Sum.inr n)))
    (hstart : j ∉ M.region c)
    (ha : j ∈ M.region (M.advance c a)) (hb : j ∉ M.region (M.advance c b))
    (hmove : ∀ p cells, M.tally (M.advance c a) b p cells ⟨j, ha⟩ ≠ n) :
    [Sum.inl a, Sum.inl b].foldl (step M) x ≠ [Sum.inl b, Sum.inl a].foldl (step M) x := by
  intro h
  have hfirst : step M x (Sum.inl a) = writeByte M c a x := step_active M hc _
  have hfirst' : step M x (Sum.inl b) = writeByte M c b x := step_active M hc _
  have hsecond : step M (writeByte M c a x) (Sum.inl b) =
      writeByte M (M.advance c a) b (writeByte M c a x) := step_active M rfl _
  have hsecond' : step M (writeByte M c b x) (Sum.inl a) =
      writeByte M (M.advance c b) a (writeByte M c b x) := step_active M rfl _
  have hcell := congrFun h (cell j)
  simp only [List.foldl_cons, List.foldl_nil, hfirst, hfirst', hsecond, hsecond'] at hcell
  simp only [writeByte, ha, hb, hstart, dif_pos, dif_neg, not_false_eq_true, hn] at hcell
  exact hmove _ _ (by simpa using hcell)

/-- **O02.**  Activity, carrier, contact, trace and demand are readings of one
body, and the causal state is that body.  The instance is deliberately plain:
the refinement contract forbids a separate page, and this body has none. -/
noncomputable def refinement :
    ColumnRefinement (H := Local Ctx Pool Count) (C := Local Ctx Pool Count)
      (S := Local Ctx Pool Count × Local Ctx Pool Count) (History := List (Arrival Tick))
      (Arrival := Arrival Tick) (Readout := Byte → ℝ)
      (Carrier := Site Cell → Local Ctx Pool Count)
      (Trace := Local Ctx Pool Count × Local Ctx Pool Count)
      (Demand := Local Ctx Pool Count) (body M) where
  projections :=
    { activity := fun x u => x u
      carrier := fun x u => x u
      contact := fun x u v => (x u, x v)
      activity_local := fun h => h
      carrier_local := fun h => h
      contact_local := fun h h' => by simp [h, h'] }
  causal :=
    { encode := encode M
      advance := fun h a => h ++ [a]
      stimulus := stimulus
      event := event M
      state := id
      required := fun h => predict M (encode M h)
      read := predict M
      next := step M
      body_step := fun h a => by simp [encode, step]
      read_correct := fun _ => rfl
      next_correct := fun h a => by simp [encode] }
  assemble := id
  carrier_from_local := fun _ => rfl
  trace := fun x u v => (x u, x v)
  trace_read := fun _ s => s
  trace_from_local := fun _ _ _ => rfl
  demand := fun x u => x u
  demand_read := id
  demand_from_local := fun _ _ => rfl

end Stream

section Witness

/-- `Site Cell` is finite exactly because the three fixed units are. -/
def siteEquiv (Cell : Type*) : Option (Option (Option Cell)) ≃ Site Cell where
  toFun
    | none => sense
    | some none => context
    | some (some none) => pool
    | some (some (some j)) => cell j
  invFun
    | sense => none
    | context => some none
    | pool => some (some none)
    | cell j => some (some (some j))
  left_inv u := by rcases u with _ | _ | _ | _ <;> rfl
  right_inv u := by cases u <;> rfl

noncomputable instance [Fintype Cell] : Fintype (Site Cell) :=
  Fintype.ofEquiv _ (siteEquiv Cell)

theorem card_site [Fintype Cell] : Fintype.card (Site Cell) = Fintype.card Cell + 3 := by
  rw [← Fintype.card_congr (siteEquiv Cell)]
  simp

/-- The smallest instance of the shape: each byte addresses its own cell. -/
def suffixOne : Model Byte Unit Byte (Fin 2) Unit where
  region c := {c}
  advance _ b := b
  pass c _ := c
  tally _ _ _ _ _ := 1
  learn _ _ _ p := p
  score _ _ _ _ := 0
  birthCtx := 0
  birthPool := ()
  birthCell := 0

/-- The premises of `events_touch_bounded` are met by a real instance. -/
theorem suffixOne_is_local (a : Byte × Arrival Unit) :
    ((body suffixOne).parts a).card ≤ 4 ∧ ∃ u, u ∉ (body suffixOne).parts a :=
  events_touch_bounded suffixOne 1 (fun _ => by simp [suffixOne])
    (by rw [card_site]; simp) a

/-- The premises of `order_separated` are met from birth by a real instance. -/
theorem suffixOne_holds_order :
    [Sum.inl 1, Sum.inl 2].foldl (step suffixOne) (body suffixOne).init ≠
      [Sum.inl 2, Sum.inl 1].foldl (step suffixOne) (body suffixOne).init :=
  order_separated suffixOne (1 : Byte) 2 (c := 0) rfl 1 (n := 0) rfl
    (by simp [suffixOne]) (by simp [suffixOne]) (by simp [suffixOne])
    (fun _ _ => by simp [suffixOne])

end Witness

end CGB.Generation.ContextBody
