import CGBAxioms.MaterialSource
import CGBAxioms.TurnoverControls

/-!
# Finite supply / exchange / exhaustion control

A mathematical carrier-interpretation witness, not a bacterial/sponge model or
a capability experiment. Two fixed tokens suffice to distinguish external
arrival, genuine replacement at constant body count, and source exhaustion.
The original six-axiom body and its clock are reused unchanged.
-/

open Finset

namespace CGB.MaterialSourceControls

open MaterialSource

/-- 0: supply not yet arrived; 1: accessible; 2: spent. -/
abbrev Phase := Fin 3

def next (p : Phase) : Phase := if p = 0 then 1 else 2

def event (p : Phase) : Option Unit := if p = 1 then some () else none

def owner (p : Phase) (c : Bool) : Place :=
  if p = 0 then (if c then .outside else .body)
  else if p = 1 then (if c then .source else .body)
  else (if c then .body else .sink)

noncomputable def world : World Unit Bool Bool where
  W := Phase
  sense p _ := decide (p = 1)
  moves p _ p' := if next p = p' then 1 else 0
  moves_nonneg _ _ _ := by split_ifs <;> norm_num
  moves_sum _ _ := by simp
  start := 0
  quiet _ := false
  clock := 1
  clock_pos := by norm_num
  horizon := 2
  recovery := 1 / 2
  recovery_pos := by norm_num

noncomputable def physical : Interaction.Machine Phase ℕ (Option Unit) :=
  Interaction.Machine.deterministic (fun _ p => (event p, next p))

def project (p : Phase) : (Unit → Bool) × world.W := ((fun _ => true), p)

noncomputable def coarse (cut : Bool) : Interaction.Machine ((Unit → Bool) × world.W) ℕ (Option Unit) :=
  machine (TurnoverControls.body.jointEventKernel world TurnoverControls.body_eventClock
    cut (fun _ => world.sense))

/-- Match the original declared ledger only on real event occurrences.
An idle slot may supply the exterior reservoir but cannot count as intake. -/
noncomputable def declared (_t : ℕ) (z : (Unit → Bool) × world.W) (e : Option Unit)
    (_z' : (Unit → Bool) × world.W) : ℕ :=
  e.elim 0 (fun a => (TurnoverControls.ledger 1).incoming a (world.sense z.2) z.1)

/-- Explicit source realization: no changed body dynamics, no extra physical
ticks, and no nonempty-body false positive standing in for actual replacement. -/
theorem realized (cut : Bool) : Realization physical (coarse cut) owner project
    (fun z => TurnoverControls.body.total z.1) declared declared world.clock := by
  classical
  have hw (t : ℕ) (z : (Unit → Bool) × world.W) (w' : world.W) :
      TurnoverControls.body.eventWorldStep world cut (fun _ => world.sense) t z w' =
        world.moves z.2 z.1 w' := by
    cases cut
    · rfl
    · simp only [Medium.eventWorldStep, ↓reduceIte, Medium.cutMove]
      have hm : ∀ x', world.moves z.2 x' w' = world.moves z.2 z.1 w' := by intro x'; rfl
      simp_rw [hm]
      rw [← Finset.sum_mul, TurnoverControls.body.lawCut_sum, one_mul]
  refine ⟨⟨by norm_num [world], rfl, ?_⟩, ?_, ?_⟩
  · intro t p e z'
    rcases z' with ⟨x, w⟩
    have hx : x = (fun _ => x ()) := by funext u; cases u; rfl
    rw [hx]
    change (∑ p', if project p' = ((fun _ => x ()), w) then physical.step t p e p' else 0) =
      TurnoverControls.body.eventWorldStep world cut (fun _ => world.sense) t (project p) w *
        TurnoverControls.body.eventStep world.clock (world.sense p) (fun _ => true) e (fun _ => x ())
    rw [hw]
    fin_cases p <;> cases hxv : x () <;> fin_cases w <;> cases e <;>
      simp [physical, Interaction.Machine.deterministic, project, event, next,
        Medium.eventStep, TurnoverControls.body, ConstraintFreedom.openSite, world, funext_iff]
  · intro p
    fin_cases p <;> simp [count, owner, project, Medium.total, TurnoverControls.body,
      ConstraintFreedom.openSite]
  · intro t p e p' hp
    fin_cases p <;> fin_cases p' <;> cases e <;>
      norm_num [physical, Interaction.Machine.deterministic, event, next] at hp
    all_goals
      refine ⟨?_, ?_, ?_⟩
      · intro c; cases c <;> simp [owner]
      · simp [moved, owner, declared, TurnoverControls.ledger]
      · simp [moved, owner, declared, TurnoverControls.ledger]

/-- One arrival, then one equal input/output replacement, then no accessible
source. Body material count stays one throughout. This is not a full life witness. -/
theorem supply_exchange_exhaustion :
    moved (owner 0) (owner 1) .outside .source = 1 ∧
    moved (owner 0) (owner 1) .source .body = 0 ∧
    moved (owner 1) (owner 2) .source .body = 1 ∧
    moved (owner 1) (owner 2) .body .sink = 1 ∧
    count (owner 2) .source = 0 ∧
    (∀ p, count (owner p) .body = 1) := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp [moved, owner]
  · simp [moved, owner]
  · simp [moved, owner]
  · simp [moved, owner]
  · simp [count, owner]
  · intro p
    fin_cases p <;> simp [count, owner]

/-- Reuse the existing refinement/first-hit theorem for the WHOLE future.
No additional scoring implementation is introduced for the physical carrier. -/
theorem no_future_intake_after_exhaustion (cut : Bool) (history : List (Option Unit)) (k : ℕ) :
    (coarse cut).firstHit List.length Option.isSome history k (project 2) = 0 := by
  have hp : ∀ h n, physical.firstHit List.length Option.isSome h n (2 : Phase) = 0 := by
    intro h n
    induction n generalizing h with
    | zero => rfl
    | succ n ih =>
      have hs : ∀ t e p', physical.step t 2 e p' =
          if e = none ∧ p' = 2 then 1 else 0 := by
        intro t e p'
        simp [physical, Interaction.Machine.deterministic, event, next]
      simp only [Interaction.Machine.firstHit, hs, ite_mul, one_mul, zero_mul, ite_and]
      simp [ih]
  have he := Interaction.Refines.firstHit_eq (realized cut).process.representation
    List.length Option.isSome history k (2 : Phase)
  rw [← he]
  exact hp history k

end CGB.MaterialSourceControls
