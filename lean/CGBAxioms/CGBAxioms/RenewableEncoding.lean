import CGBAxioms.RenewableLearningCarrier

/-!
# Concrete rule-book compilation and honest candidate boundaries

The original one-site rules are compiled into the EXISTING tabular language.
Forty-two guarded rows replace one state-dependent event; enabled self-events
are retained because they carry real turnover. The world and source stock
are not fake extra wiring sites for the genome-budget denominator.
-/

open Finset

namespace CGB.RenewableEncoding
open RenewableLearning Description

def cell : Fin 8 → Cell
  | ⟨0, _⟩ => .vacuum
  | ⟨1, _⟩ => .search
  | ⟨2, _⟩ => .aSpent
  | ⟨3, _⟩ => .bSpent
  | ⟨4, _⟩ => .aReady
  | ⟨5, _⟩ => .bReady
  | ⟨6, _⟩ => .aHeld
  | _ => .bHeld

def cellCode : Cell → ℕ
  | .vacuum => 0 | .search => 1 | .aSpent => 2 | .bSpent => 3
  | .aReady => 4 | .bReady => 5 | .aHeld => 6 | .bHeld => 7

def signal : Fin 7 → Signal
  | ⟨0, _⟩ => .calm | ⟨1, _⟩ => .answerA | ⟨2, _⟩ => .answerB
  | ⟨3, _⟩ => .feed | ⟨4, _⟩ => .scarce | ⟨5, _⟩ => .miss | _ => .probe

private def before (a : Fin 42) : Fin 8 := ⟨a.val / 6 + 1, by omega⟩
private def stimulus (a : Fin 42) : Fin 7 := ⟨a.val % 6 + 1, by omega⟩

def row (a : Fin 42) : Description.Event where
  ports := [⟨0, (before a).val, (stimulus a).val,
    cellCode (next true true (signal (stimulus a)) (cell (before a)))⟩]
  numerator := 1
  denominator := 0

def book : Description.ExchangeBook where
  base := {
    sites := 1, states := 7, stimuli := 6,
    contacts := [(0, 0)], identities := [0], boundary := [0],
    contents := [0, 2, 1, 1, 4, 4, 3, 3], initial := [1], calm := [0],
    events := (List.finRange 42).map row }
  transfers := (List.finRange 42).map (fun a =>
    (incoming true (signal (stimulus a)) (cell (before a)),
      outgoing true true (signal (stimulus a)) (cell (before a))))

def eventIndex (a : Fin 42) : book.base.Action := ⟨a.val, by simp [book]⟩

theorem code_roundtrip (q : Fin 8) : cellCode (cell q) = q.val := by fin_cases q <;> rfl

theorem book_layout : book.base.sites = 1 ∧ book.base.events.length = 42 ∧
    book.transfers.length = 42 ∧ book.base.states + 1 = 8 ∧ book.base.stimuli + 1 = 7 := by
  norm_num [book]

private theorem row_lookup (a : Fin 42) : book.base.event (eventIndex a) = row a := by
  simp [Book.event, eventIndex, book]

private theorem transfer_lookup (a : Fin 42) : book.transfer (eventIndex a) =
    (incoming true (signal (stimulus a)) (cell (before a)),
      outgoing true true (signal (stimulus a)) (cell (before a))) := by
  simp [ExchangeBook.transfer, eventIndex, book]

set_option maxRecDepth 10000 in
set_option maxHeartbeats 2000000 in
theorem compiled_rows : ∀ (s : Fin 7) (q : Fin 8),
    (∑ a : Fin 42, if book.base.enabled (eventIndex a) (fun _ => s) (fun _ => q)
      then (1 : ℕ) else 0) = (if s.val = 0 ∨ q.val = 0 then 0 else 1) ∧
    (∀ a : Fin 42, book.base.enabled (eventIndex a) (fun _ => s) (fun _ => q) = true →
      (book.base.update (eventIndex a) (fun _ => q) ⟨0, by decide⟩).val =
        cellCode (next true true (signal s) (cell q)) ∧
      book.transfer (eventIndex a) =
        (incoming true (signal s) (cell q), outgoing true true (signal s) (cell q))) := by
  decide

theorem compiled_rates (s : Fin 7) (q : Fin 8) :
    (∑ a : Fin 42, book.base.rationalRate (eventIndex a) (fun _ => s) (fun _ => q)) =
      if s.val = 0 ∨ q.val = 0 then (0 : ℚ) else 1 := by
  have hw (a : Fin 42) : book.base.weight (eventIndex a) = 1 := by
    simp [Book.weight, row_lookup, row]
  calc
    _ = ∑ a : Fin 42, if book.base.enabled (eventIndex a) (fun _ => s) (fun _ => q)
        then (1 : ℚ) else 0 := by simp only [Book.rationalRate, hw]
    _ = _ := by exact_mod_cast (compiled_rows s q).1

private theorem data_balance : ∀ (s : Fin 7) (q : Fin 8) (a : Fin 42),
    book.base.enabled (eventIndex a) (fun _ => s) (fun _ => q) = true →
      book.base.contents[(book.base.update (eventIndex a) (fun _ => q)
        ⟨0, by decide⟩).val]?.getD 0 + (book.transfer (eventIndex a)).2 =
      book.base.contents[q.val]?.getD 0 + (book.transfer (eventIndex a)).1 := by
  decide

theorem accountable : book.Accountable := by
  classical
  letI : Unique book.base.Site := by change Unique (Fin 1); infer_instance
  let zero : book.base.Site := ⟨0, by decide⟩
  have hz : (⟨0, by decide⟩ : book.base.Site) = default := Subsingleton.elim _ _
  have action (a : book.base.Action) : ∃ i : Fin 42, eventIndex i = a := by
    refine ⟨⟨a.val, ?_⟩, ?_⟩
    · simpa only [book, List.length_map, List.length_finRange] using a.isLt
    · apply Fin.ext; rfl
  constructor
  · intro a s x hp
    obtain ⟨i, rfl⟩ := action a
    let sf : Fin 7 := s zero
    let qf : Fin 8 := x zero
    have hs : s = fun _ => sf := by funext u; exact congrArg s (Subsingleton.elim u zero)
    have hx : x = fun _ => qf := by funext u; exact congrArg x (Subsingleton.elim u zero)
    rw [hs, hx] at hp ⊢
    have hen : book.base.enabled (eventIndex i) (fun _ => sf) (fun _ => qf) = true := by
      by_contra hn
      have hf := Bool.eq_false_of_not_eq_true hn
      simp [Book.medium, Book.rationalRate, hf] at hp
    have hd := data_balance sf qf i hen
    simpa [Medium.total, Book.medium, Book.apply, hen, hz] using hd
  · intro a s x hc _
    obtain ⟨i, rfl⟩ := action a
    have hi : zero ∈ book.base.medium.parts (eventIndex i) := by
      change zero ∈ univ.filter (fun u => ∃ p ∈ (book.base.event (eventIndex i)).ports,
        p.site = u.val)
      refine Finset.mem_filter.mpr ⟨Finset.mem_univ _, ?_⟩
      simp [row_lookup, row, zero]
    have hb : zero ∈ book.base.medium.boundary := by
      change zero ∈ univ.filter (fun u => u.val ∈ book.base.boundary)
      refine Finset.mem_filter.mpr ⟨Finset.mem_univ _, ?_⟩
      simp [book, zero]
    exact False.elim (Finset.disjoint_left.mp hc hi hb)

/-- Necessary bit count for EVERY book under the registered convention. This
does not evaluate an enormous Gödel number or change the encoding language. -/
theorem description_positive (b : ExchangeBook) : 1 ≤ bits b := by
  unfold bits
  omega

theorem small_reference_obstruction (b : ExchangeBook) (g : WiringReference b.base.sites)
    (h : b.base.sites * b.base.sites < 100) : ¬ b.GenomeBudget g 1 100 := by
  rw [ExchangeBook.one_percent_budget]
  have hb := description_positive b
  omega

theorem no_one_percent (g : WiringReference book.base.sites) :
    ¬ book.GenomeBudget g 1 100 :=
  small_reference_obstruction book g (by decide)

theorem world_richness_limit (κ : ℕ) (h :
    κ * Fintype.card (Unit → Cell) ≤ Fintype.card world.W) : κ ≤ 1 := by
  have hc : Fintype.card (Unit → Cell) = 8 := by decide
  have hw : Fintype.card world.W = 9 := by decide
  rw [hc, hw] at h
  omega

/-- A full encoded subtarget cannot be obtained by attaching the already
proved persistence/source certificates to this one-site book. -/
theorem no_encoded_candidate (ha : book.Accountable)
    (Wd : World book.base.Site book.base.State book.base.Stim)
    (g : WiringReference book.base.sites) (δ : ℝ) (κ T : ℕ) :
    ¬ Exchange.EncodedCandidate book ha Wd g 1 100 δ κ T := by
  intro h
  exact no_one_percent g h.budget

end CGB.RenewableEncoding
