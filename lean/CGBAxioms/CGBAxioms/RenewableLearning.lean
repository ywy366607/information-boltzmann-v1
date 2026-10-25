import CGBAxioms.LearningMaintenance

/-!
# Renewable paid learning in a continuously changing finite world

Resource access can fail after an unannounced regime change. A funded conflict
reopens candidates, a subsequent probe identifies the regime, and actual food
replenishes both the safety buffer and write credit. All rules are declared
finite mathematical carrier rules, not empirical organism capability claims.
-/

open Finset

namespace CGB.RenewableLearning

inductive Cell
  | vacuum | search | aSpent | bSpent | aReady | bReady | aHeld | bHeld
  deriving DecidableEq

inductive Signal
  | calm | answerA | answerB | feed | scarce | miss | probe
  deriving DecidableEq

inductive Env
  | feedA | feedB | missA | missB | probeA | probeB | answerA | answerB | trap
  deriving DecidableEq

instance : Fintype Cell where
  elems := {.vacuum, .search, .aSpent, .bSpent, .aReady, .bReady, .aHeld, .bHeld}
  complete q := by cases q <;> decide

instance : Fintype Signal where
  elems := {.calm, .answerA, .answerB, .feed, .scarce, .miss, .probe}
  complete s := by cases s <;> decide

instance : Fintype Env where
  elems := {.feedA, .feedB, .missA, .missB, .probeA, .probeB, .answerA, .answerB, .trap}
  complete w := by cases w <;> decide

abbrev H := LearningMaintenance.H
abbrev Action := LearningMaintenance.Action
abbrev possible := LearningMaintenance.possible
abbrev safe := LearningMaintenance.safe

def credit : Cell → ℕ
  | .search => 1
  | .aReady | .bReady | .aHeld | .bHeld => 2
  | _ => 0

def buffer : Cell → ℕ
  | .vacuum => 0
  | .aReady | .bReady => 2
  | _ => 1

def cost (q : Cell) : ℕ := buffer q + credit q

def memory (q : Cell) : CausalLearning.Memory H 2 :=
  match q with
  | .vacuum => ⟨∅, 0⟩
  | .search => ⟨univ, 1⟩
  | .aSpent => ⟨{0}, 0⟩
  | .bSpent => ⟨{1}, 0⟩
  | .aReady | .aHeld => ⟨{0}, 2⟩
  | .bReady | .bHeld => ⟨{1}, 2⟩

def policy (q : Cell) : Action :=
  CausalLearning.certifiedOr possible safe 2 (memory q) ()

private theorem policy_table (q : Cell) : policy q =
    match q with
    | .aSpent | .aReady | .aHeld => 0
    | .bSpent | .bReady | .bHeld => 1
    | _ => 2 := by
  cases q <;> decide

/-- Contradictory observed consequences can reopen a model family only after
paying for that change. This is a new-epoch hypothesis set, not a claim that the
new family explains the old contradictory observation under a fixed regime. -/
def reconsider (m : CausalLearning.Memory H 2) (a : Action)
    (o : LearningMaintenance.Outcome) : CausalLearning.Memory H 2 :=
  if h : 0 < m.reserve.val ∧ CausalLearning.screen possible m.candidates () a o = ∅ then
    ⟨univ, ⟨m.reserve.val - 1, by have := m.reserve.isLt; omega⟩⟩
  else m

def next (reopen refill : Bool) (s : Signal) (q : Cell) : Cell :=
  if q = .vacuum then .vacuum else
  match s with
  | .calm | .probe => q
  | .scarce => .vacuum
  | .answerA => if q = .search then .aSpent else q
  | .answerB => if q = .search then .bSpent else q
  | .feed => if refill then
      match q with
      | .aSpent | .aHeld => .aReady
      | .bSpent | .bHeld => .bReady
      | _ => q
    else q
  | .miss =>
      match q with
      | .aReady => if reopen then .search else .aHeld
      | .bReady => if reopen then .search else .bHeld
      | _ => .vacuum

def incoming (_refill : Bool) (s : Signal) (q : Cell) : ℕ :=
  if q = .vacuum ∨ s = .calm then 0 else
  match s with
  | .scarce | .miss => 0
  | .feed =>
      match q with
      | .aSpent | .bSpent => 4
      | .aHeld | .bHeld => 2
      | _ => 1
  | _ => 1

def outgoing (reopen refill : Bool) (s : Signal) (q : Cell) : ℕ :=
  if q = .vacuum ∨ s = .calm then 0 else
  match s with
  | .scarce => cost q
  | .miss =>
      match q with
      | .aReady | .bReady => if reopen then 2 else 1
      | _ => cost q
  | .answerA | .answerB => if q = .search then 2 else 1
  | .feed => if refill then 1 else
      match q with
      | .aSpent | .bSpent => 4
      | .aHeld | .bHeld => 2
      | _ => 1
  | _ => 1

def writeCost (reopen : Bool) (s : Signal) (q : Cell) : ℕ :=
  match s, q with
  | .answerA, .search | .answerB, .search => 1
  | .miss, .aReady | .miss, .bReady => if reopen then 1 else 0
  | _, _ => 0

def topup (refill : Bool) (s : Signal) (q : Cell) : ℕ :=
  if refill ∧ s = .feed ∧ (q = .aSpent ∨ q = .bSpent) then 2 else 0

/-- Unused write fuel is destroyed if the physical body is lost. It is counted
separately from completed candidate writes. -/
def discarded (s : Signal) (q : Cell) : ℕ :=
  if s = .scarce ∨ (s = .miss ∧ q ≠ .aReady ∧ q ≠ .bReady) then credit q else 0

theorem local_budget (reopen refill : Bool) (s : Signal) (q : Cell) :
    cost (next reopen refill s q) + outgoing reopen refill s q = cost q + incoming refill s q ∧
    credit (next reopen refill s q) + writeCost reopen s q + discarded s q =
      credit q + topup refill s q := by
  cases reopen <;> cases refill <;> cases s <;> cases q <;> decide

/-- Paid reopening is driven by a contradictory consequence, and the ensuing
probe uses exactly the previously verified candidate-screening update. -/
theorem learning_contract :
    (∀ q ∈ ({Cell.aReady, Cell.bReady} : Finset Cell),
      memory (next true true .miss q) = reconsider (memory q) (policy q) 0) ∧
    (∀ h : H, memory (next true true (if h = 0 then .answerA else .answerB) .search) =
      CausalLearning.advance possible (memory .search) () (policy .search)
        (if h = 0 then 1 else 2)) := by
  constructor
  · intro q hq
    cases q <;> simp at hq <;> decide
  · intro h
    fin_cases h <;> decide

noncomputable def plant (reopen refill : Bool) : Medium Unit Cell Unit Signal where
  near _ _ := True
  ident _ := 0
  boundary := univ
  content := cost
  parts _ := univ
  act _ s x := fun _ => next reopen refill (s ()) (x ())
  rate _ s x := if s () = .calm ∨ x () = .vacuum then 0 else 1
  rate_nonneg _ _ _ := by split_ifs <;> norm_num
  init _ := .search
  calm _ := .calm

theorem admission (reopen refill : Bool) :
    (plant reopen refill).Axioms ∧ (plant reopen refill).EventClocked 1 := by
  let M := plant reopen refill
  have hr : M.Rigid := by
    intro σ _
    apply Equiv.ext
    intro u
    exact Subsingleton.elim _ _
  constructor
  · refine ⟨(M.p1a_iff_unique_vacuum).2 ?_, ?_, ?_, ?_, M.p2c_of_rigid hr, ?_⟩
    · refine ⟨.vacuum, rfl, ?_⟩
      intro q hq
      cases q <;> norm_num [M, plant, cost, buffer, credit] at *
    · refine ⟨?_, by intros; trivial, ?_, ?_⟩
      · intro u v
        exact ⟨Equiv.refl Unit, by intros; rfl, Subsingleton.elim _ _⟩
      · intro e s x u hu
        simp [M, plant] at hu
      · intro e s s' x y hh
        have hx : x = y := funext fun u => (hh u (by simp [M, plant])).1
        have hs : s = s' := funext fun u => (hh u (by simp [M, plant])).2
        subst y; subst s'
        exact ⟨rfl, fun _ _ => rfl⟩
    · apply M.p2a_of_every_event_open
      intro e
      change ¬ Disjoint (univ : Finset Unit) univ
      decide
    · refine ⟨fun _ => 1, fun _ => one_pos, fun x y => ?_⟩
      simp [Medium.jump, M, plant]
    · exact ⟨M.onlyBoundarySenses_of_all_boundary rfl, M.initial_invariant_of_rigid hr⟩
  · refine ⟨by norm_num, ?_⟩
    intro s x
    simp only [plant, Fintype.sum_unique]
    split_ifs <;> norm_num

def sense : Env → Unit → Signal
  | .feedA, _ | .feedB, _ => .feed
  | .missA, _ | .missB, _ => .miss
  | .probeA, _ | .probeB, _ => .probe
  | .answerA, _ => .answerA
  | .answerB, _ => .answerB
  | .trap, _ => .scarce

noncomputable def moves (w : Env) (x : Unit → Cell) (v : Env) : ℝ :=
  if x () = .vacuum then (if v = .trap then 1 else 0) else
  match w with
  | .feedA => if policy (x ()) = 2 then (if v = .answerA then 1 else 0)
      else if policy (x ()) = 0 then (if v = .feedA ∨ v = .feedB then 1 / 2 else 0)
      else if v = .missA then 1 else 0
  | .feedB => if policy (x ()) = 2 then (if v = .answerB then 1 else 0)
      else if policy (x ()) = 1 then (if v = .feedA ∨ v = .feedB then 1 / 2 else 0)
      else if v = .missB then 1 else 0
  | .missA => if v = .probeA then 1 else 0
  | .missB => if v = .probeB then 1 else 0
  | .probeA => if v = (if policy (x ()) = 2 then .answerA else .trap) then 1 else 0
  | .probeB => if v = (if policy (x ()) = 2 then .answerB else .trap) then 1 else 0
  | .answerA => if v = .feedA then 1 else 0
  | .answerB => if v = .feedB then 1 else 0
  | .trap => if v = .trap then 1 else 0

private theorem sum_env (f : Env → ℝ) : ∑ w, f w =
    f .feedA + f .feedB + f .missA + f .missB + f .probeA + f .probeB +
      f .answerA + f .answerB + f .trap := by
  have h : (univ : Finset Env) =
      {.feedA, .feedB, .missA, .missB, .probeA, .probeB, .answerA, .answerB, .trap} := by decide
  rw [h]; simp; ring

set_option maxHeartbeats 1800000 in
-- Normalize all finite world/action rows once for the shared world.
noncomputable abbrev world : World Unit Cell Signal where
  W := Env
  sense := sense
  moves := moves
  moves_nonneg w x v := by cases w <;> unfold moves <;> split_ifs <;> norm_num
  moves_sum w x := by
    rw [sum_env]
    generalize hx : x () = q
    cases w <;> cases q <;> norm_num [moves, policy_table, hx]
  start := .probeA
  quiet _ := .calm
  clock := 1
  clock_pos := by norm_num
  horizon := 2
  recovery := 1 / 2
  recovery_pos := by norm_num

abbrev he (reopen refill : Bool) : (plant reopen refill).EventClocked world.clock :=
  (admission reopen refill).2

noncomputable def ledger (reopen refill : Bool) : (plant reopen refill).MaterialLedger where
  incoming _ s x := incoming refill (s ()) (x ())
  outgoing _ s x := outgoing reopen refill (s ()) (x ())
  balance _ s x _ := (local_budget reopen refill (s ()) (x ())).1
  closed_zero a s x h _ := by
    have hn : ¬ (plant reopen refill).Closed a := by simp [Medium.Closed, plant]
    exact False.elim (hn h)

abbrev S := (Unit → Cell) × Env

noncomputable def process (reopen refill cut : Bool) : Interaction.Machine S ℕ (Option Unit) :=
  MaterialSource.machine ((plant reopen refill).jointEventKernel world (he reopen refill)
    cut (fun _ => world.sense))

noncomputable def initial : S := ((fun _ => .search), .probeA)

theorem actual_law (reopen refill cut : Bool) (n : ℕ) (z : S) :
    Energetics.law (process reopen refill cut) initial n z = if cut then
      (plant reopen refill).lawCut world (fun _ => world.sense) n z.1 z.2 else
      (plant reopen refill).law world (fun _ => world.sense) n z.1 z.2 := by
  have hi : (fun z : S => if z = initial then (1 : ℝ) else 0) =
      (fun z : S => if z.1 = (plant reopen refill).init ∧ z.2 = world.start then 1 else 0) := by
    funext z
    simp [initial, plant, world, Prod.ext_iff]
  change EventChannels.Kernel.stateLaw
    ((plant reopen refill).jointEventKernel world (he reopen refill) cut (fun _ => world.sense))
    (fun z : S => if z = initial then 1 else 0) n z = _
  rw [hi]
  exact (plant reopen refill).jointEventKernel_stateLaw world (he reopen refill)
    cut (fun _ => world.sense) n z

end CGB.RenewableLearning
