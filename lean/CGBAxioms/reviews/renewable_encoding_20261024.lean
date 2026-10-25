import CGBAxioms.RenewableEncoding
import CGBAxioms.RenewableCandidateAudit

/- Exact encoding and candidate-boundary checks; no capability experiments. -/

open CGB CGB.Description CGB.RenewableLearning CGB.RenewableEncoding

namespace RenewableEncodingReview20261024

example : book.base.sites = 1 ∧ book.base.events.length = 42 ∧
    book.transfers.length = 42 ∧ book.base.states + 1 = 8 ∧ book.base.stimuli + 1 = 7 :=
  book_layout

example (q : Fin 8) : cellCode (cell q) = q.val := code_roundtrip q

example (s : Fin 7) (q : Fin 8) :
    (∑ a : Fin 42, if book.base.enabled (eventIndex a) (fun _ => s) (fun _ => q)
      then (1 : ℕ) else 0) = (if s.val = 0 ∨ q.val = 0 then 0 else 1) :=
  (compiled_rows s q).1

example (s : Fin 7) (q : Fin 8) :
    (∑ a : Fin 42, book.base.rationalRate (eventIndex a) (fun _ => s) (fun _ => q)) =
      if s.val = 0 ∨ q.val = 0 then (0 : ℚ) else 1 := compiled_rates s q

example : book.Accountable := accountable

example (g : WiringReference book.base.sites) : ¬ book.GenomeBudget g 1 100 := no_one_percent g

example (b : ExchangeBook) (g : WiringReference b.base.sites) (h : b.base.sites ^ 2 < 100) :
    ¬ b.GenomeBudget g 1 100 := small_reference_obstruction b g (by simpa [pow_two] using h)

example (κ : ℕ) (h : κ * Fintype.card (Unit → Cell) ≤ Fintype.card world.W) : κ ≤ 1 :=
  world_richness_limit κ h

example (Wd : World book.base.Site book.base.State book.base.Stim)
    (g : WiringReference book.base.sites) (δ : ℝ) (κ T : ℕ) :
    ¬ Exchange.EncodedCandidate book accountable Wd g 1 100 δ κ T :=
  no_encoded_candidate accountable Wd g δ κ T

example (T : ℕ) (hT : 0 < T) : 1 / 2 ≤
    (plant true true).law world (RenewableCandidateAudit.schedule T) 6
      (fun _ => .aReady) .trap := RenewableCandidateAudit.interrupted_trap_mass T hT

example (T : ℕ) (hT : 0 < T) :
    Exchange.survival (plant true true) world (he true true) (ledger true true) false
      (RenewableCandidateAudit.schedule T) 6 ≤ 1 / 2 :=
  RenewableCandidateAudit.interruption_failure T hT

example (T : ℕ) : ¬ Exchange.ToleratesSensoryInterruption
    (plant true true) world (he true true) (ledger true true) 0 T :=
  RenewableCandidateAudit.no_positive_interruption_tolerance T

example : (plant true true).ReadsBoundary world := by
  intro w x y v hh
  have hx : x = y := by
    funext u
    exact hh u (by simp [plant])
  rw [hx]

end RenewableEncodingReview20261024
