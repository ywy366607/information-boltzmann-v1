import CGBAxioms.EnergyBudget
import CGBAxioms.ConstraintFreedom
import CGBAxioms.LifeControls

/-!
# Proposed responsive loop and algebraic energy account

The proved results are medium axioms, clock and boundary admission, world-row
normalization, and an expected first-law identity. No SelfMaintains, survival
advantage, Isothermal, or exhaustion theorem is established in this module.

Review 2026-10-24: feeding forgets the B action, so even the natural B branch
enters world state 5. The declared heat port closes an algebraic residual;
it is not a separately calibrated heat bath. Starvation freezes stored energy
with zero load, rather than establishing depletion. See the accompanying review
checks in reviews/active_loop_20261024.lean and the corrected technical report.
-/

open Finset

namespace CGB.ActiveLoop

open Operational FirstHit Energetics

/-! ## State spaces -/

/-- Local states: 0: vacuum, 1: action A, 2: action B, 3: fed. -/
abbrev State := Fin 4

/-- World states:
0: start
1: cue A
2: cue B
3: arena A
4: arena B
5: starve -/
abbrev WorldState := Fin 6

/-- Stimuli: 0: calm, 1: cue A, 2: cue B, 3: feed. -/
abbrev Stim := Fin 4

def cost : State → ℕ
  | 0 => 0
  | 1 => 1
  | 2 => 1
  | 3 => 2

theorem unique_vacuum : ∃! l : State, cost l = 0 := by
  refine ⟨0, rfl, ?_⟩
  intro l hl
  fin_cases l <;> revert hl <;> decide

/-- Organism transition under active stimuli:
- Under cue A (1): take action A (1).
- Under cue B (2): take action B (2).
- Under feed (3): cycle between action and fed (intake and digestion). -/
def advance (_e : Unit) (s : Stim) (l : State) : State :=
  if s = 1 then 1
  else if s = 2 then 2
  else if s = 3 then
    if l = 3 then 1 else 3
  else l

/-- A single-site medium with responsive action. -/
noncomputable def body : Medium Unit State Unit Stim where
  near _ _ := True
  ident _ := 0
  boundary := univ
  content := cost
  parts _ := univ
  act _ s x := fun _ => advance () (s ()) (x ())
  rate _ s x := if s () = 0 then 0 else 1
  rate_nonneg _ s x := by split_ifs <;> norm_num
  init _ := 1
  calm _ := 0

theorem body_clock : body.Clocked 1 := by
  constructor
  · norm_num
  · intro s x
    classical
    simp only [Medium.jump, body, Fintype.sum_unique]
    split_ifs with hs
    · simp
    · rw [Finset.sum_eq_single (fun _ : Unit => advance () (s ()) (x ()))]
      · split_ifs <;> norm_num
      · intro y _ hy
        simp [Ne.symm hy]
      · simp

theorem body_axioms : body.Axioms := by
  have hr : body.Rigid := by
    intro σ _
    apply Equiv.ext
    intro u
    exact Subsingleton.elim _ _
  refine ⟨(body.p1a_iff_unique_vacuum).2 unique_vacuum, ?_, ?_, ?_, body.p2c_of_rigid hr, ?_⟩
  · refine ⟨by intros; trivial, ?_, ?_⟩
    · intro e s x u hu
      simp [body] at hu
    · intro e s s' x y hh
      have hx : x = y := funext fun u => (hh u (by simp [body])).1
      have hst : s = s' := funext fun u => (hh u (by simp [body])).2
      subst y; subst s'
      exact ⟨rfl, fun _ _ => rfl⟩
  · apply body.p2a_of_every_event_open
    intro e
    change ¬ Disjoint (univ : Finset Unit) univ
    decide
  · refine ⟨fun _ => 1, fun _ => one_pos, fun x y => ?_⟩
    simp [Medium.jump, body]
  · exact ⟨body.onlyBoundarySenses_of_all_boundary rfl, body.initial_invariant_of_rigid hr⟩

/-! ## The Environment -/

def worldSense (w : WorldState) (_u : Unit) : Stim :=
  if w = 1 then 1
  else if w = 2 then 2
  else if w = 3 ∨ w = 4 then 3
  else 0

theorem fin6_sum (f : WorldState → ℝ) :
    ∑ w : WorldState, f w = f 0 + f 1 + f 2 + f 3 + f 4 + f 5 := by
  have : (univ : Finset WorldState) = {0, 1, 2, 3, 4, 5} := by decide
  rw [this]
  simp
  ring

noncomputable def worldMoves (w : WorldState) (x : Unit → State) (w' : WorldState) : ℝ :=
  if w = 0 then
    if w' = 1 then 1 / 2
    else if w' = 2 then 1 / 2
    else 0
  else if w = 1 then
    if w' = 3 then 1 else 0
  else if w = 2 then
    if w' = 4 then 1 else 0
  else if w = 3 then
    if x () = 1 ∨ x () = 3 then
      if w' = 3 then 1 else 0
    else
      if w' = 5 then 1 else 0
  else if w = 4 then
    if x () = 2 ∨ x () = 3 then
      if w' = 4 then 1 else 0
    else
      if w' = 5 then 1 else 0
  else
    if w' = 5 then 1 else 0

theorem worldMoves_nonneg (w : WorldState) (x : Unit → State) (w' : WorldState) :
    0 ≤ worldMoves w x w' := by
  fin_cases w <;> fin_cases w' <;> simp [worldMoves] <;> split_ifs <;> norm_num

theorem worldMoves_sum (w : WorldState) (x : Unit → State) :
    ∑ w' : WorldState, worldMoves w x w' = 1 := by
  rw [fin6_sum]
  fin_cases w
  · simp [worldMoves]; ring
  · simp [worldMoves]
  · simp [worldMoves]
  · simp [worldMoves]; split_ifs <;> ring
  · simp [worldMoves]; split_ifs <;> ring
  · simp [worldMoves]

noncomputable def world : World Unit State Stim where
  W := WorldState
  sense := worldSense
  moves := worldMoves
  moves_nonneg := worldMoves_nonneg
  moves_sum := worldMoves_sum
  start := 0
  quiet _ := 0
  clock := 1
  clock_pos := by norm_num
  horizon := 2
  recovery := 1 / 2
  recovery_pos := by norm_num

theorem body_readsBoundary : body.ReadsBoundary world := by
  intro w x y w' h
  have hxy : x = y := by
    funext u
    cases u
    exact h () (by simp [body])
  rw [hxy]

/-! ## The Energy Budget of the Active Loop -/

/-- The combined physical interaction machine of the body in the world. -/
noncomputable def jointMachine (cut : Bool) :
    Interaction.Machine ((Unit → State) × world.W) ℕ Unit where
  step t z _ z' :=
    if cut then
      (body.scheduledCutKernel world body_clock (fun _ => world.sense) t).step z z'
    else
      (body.scheduledKernel world body_clock (fun _ => world.sense) t).step z z'
  nonneg t z _ z' := by
    split_ifs
    · exact (body.scheduledCutKernel world body_clock (fun _ => world.sense) t).nonneg z z'
    · exact (body.scheduledKernel world body_clock (fun _ => world.sense) t).nonneg z z'
  sum_one t z := by
    simp only [Fintype.sum_unique]
    split_ifs
    · exact (body.scheduledCutKernel world body_clock (fun _ => world.sense) t).sum_one z
    · exact (body.scheduledKernel world body_clock (fun _ => world.sense) t).sum_one z

/-- Stored energy is proportional to the material content. -/
def storedEnergy (a : ℝ) (z : (Unit → State) × world.W) : ℝ :=
  (cost (z.1 ())) * a

/-- Work load paid for action: taking Action A or Action B costs work w_load. -/
def actionWork (w_act : ℝ) (z z' : (Unit → State) × world.W) : ℝ :=
  if z.1 () = 3 ∧ (z'.1 () = 1 ∨ z'.1 () = 2) then w_act else 0

/-- Declared matter energy input on action-to-fed transitions. Physical transport
and chemical availability require separate source and entropy accounts. -/
def matterIntakeEnergy (a : ℝ) (z z' : (Unit → State) × world.W) : ℝ :=
  if (z.1 () = 1 ∨ z.1 () = 2) ∧ z'.1 () = 3 then a else 0

/-- Algebraic account with a residual heat port. Its first law alone does not
establish Isothermal admission, a physical reservoir, or a work budget. -/
noncomputable def activeAccount (a w_act : ℝ) (cut : Bool) :
    Account (jointMachine cut) where
  stored _ := storedEnergy a
  drive _ _ _ _ := 0
  load _ z _ z' := actionWork w_act z z'
  matter _ z _ z' := matterIntakeEnergy a z z'
  heat _ z _ z' := storedEnergy a z' + actionWork w_act z z' -
    storedEnergy a z - matterIntakeEnergy a z z'
  balance _ z _ z' _ := by ring

/-- Average first law holds along the actual active process. -/
theorem active_expected_first_law (a w_act : ℝ) (cut : Bool)
    (initial : (Unit → State) × world.W) (t : ℕ) :
    mean (law (jointMachine cut) initial (t + 1)) ((activeAccount a w_act cut).stored (t + 1)) +
        flux (jointMachine cut) initial (activeAccount a w_act cut).load t =
      mean (law (jointMachine cut) initial t) ((activeAccount a w_act cut).stored t) +
        flux (jointMachine cut) initial (activeAccount a w_act cut).drive t +
        flux (jointMachine cut) initial (activeAccount a w_act cut).matter t +
        flux (jointMachine cut) initial (activeAccount a w_act cut).heat t :=
  expected_first_law (jointMachine cut) (activeAccount a w_act cut) initial t

end CGB.ActiveLoop
