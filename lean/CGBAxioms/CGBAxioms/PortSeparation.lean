import CGBAxioms.SensoryMaintenanceCarrier
import CGBAxioms.SensoryMaintenanceClosure

/-!
# Independent sensory and resource ports on the same finite body

Absent sensing with resources supplied is a noninformative maintenance inlet.
Resource withdrawal preserves available observations and paid model changes;
food offers cannot refill the body through a clamped material port.
-/
open Finset

namespace CGB.PortSeparation
open RenewableLearning (Cell Signal Env)
abbrev State := SensoryMaintenance.State
abbrev Input := SensoryMaintenance.Stim × Bool
abbrev empty := SensoryMaintenance.empty
abbrev cost := SensoryMaintenance.cost

/-- With no observations, a supplied inlet carries one unit of turnover but
no regime label. It has the old probe's nonidentifying body effect. -/
def supplied (s : SensoryMaintenance.Stim) : SensoryMaintenance.Stim :=
  some (s.getD .probe)

def starved (s : SensoryMaintenance.Stim) (x : State) : State :=
  if x.1 = .vacuum ∨ x.2.val = 0 then empty else
    let q := match s with
      | none | some .feed => x.1
      | some a => RenewableLearning.next true true a x.1
    if q = .vacuum then empty else (q, ⟨x.2.val - 1, by omega⟩)

def next (u : Input) (x : State) : State :=
  if u.2 then SensoryMaintenance.next (supplied u.1) x else starved u.1 x

def incoming (u : Input) (x : State) : ℕ :=
  if u.2 then SensoryMaintenance.incoming (supplied u.1) x else 0

/-- Structure and reserve are debited separately below; their released units
are actual material output, including the final disassembly. -/
def outgoing (u : Input) (x : State) : ℕ :=
  if u.2 then SensoryMaintenance.outgoing (supplied u.1) x else cost x - cost (starved u.1 x)

def writes (u : Input) (x : State) : ℕ :=
  if u.2 then SensoryMaintenance.writes (supplied u.1) x else
    if x.1 = .vacuum ∨ x.2.val = 0 ∨ u.1 = some .feed then 0 else
      SensoryMaintenance.writes u.1 x

def topups (u : Input) (x : State) : ℕ :=
  if u.2 then SensoryMaintenance.topups (supplied u.1) x else 0

def discarded (u : Input) (x : State) : ℕ :=
  if (next u x).1 = .vacuum then RenewableLearning.credit x.1 else 0

theorem local_budgets (u : Input) (x : State) :
    cost (next u x) + outgoing u x = cost x + incoming u x ∧
    RenewableLearning.credit (next u x).1 + writes u x + discarded u x =
      RenewableLearning.credit x.1 + topups u x ∧ writes u x ≤ outgoing u x := by
  rcases u with ⟨s,b⟩; rcases x with ⟨q,r⟩
  cases b <;> fin_cases r <;> cases s with
  | none => cases q <;> decide
  | some a => cases a <;> cases q <;> decide

def rank (x : State) : ℕ := if x = empty then 0 else x.2.val + 1

theorem starvation_progress (s : SensoryMaintenance.Stim) (x : State) :
    rank x ≤ 3 ∧ (rank x = 0 ↔ x = empty) ∧
    (x ≠ empty → rank (next (s,false) x) < rank x) ∧
    (x ≠ empty → 0 < outgoing (s,false) x) := by
  rcases x with ⟨q,r⟩
  fin_cases r <;> cases s with
  | none => cases q <;> decide
  | some a => cases a <;> cases q <;> decide

theorem next_empty (u : Input) : next u empty = empty := by
  rcases u with ⟨s,b⟩
  cases b <;> cases s with
  | none => decide
  | some a => cases a <;> decide

noncomputable def body : Medium Unit State Unit Input where
  near _ _ := True
  ident _ := 0
  boundary := univ
  content := cost
  parts _ := univ
  act _ s x := fun _ => next (s ()) (x ())
  rate _ s x := if s () = (some .calm,true) ∨ x () = empty then 0 else 1
  rate_nonneg _ _ _ := by split_ifs <;> norm_num
  init _ := (.search,2)
  calm _ := (some .calm,true)

theorem admission : body.Axioms ∧ body.EventClocked 1 := by
  have hr : body.Rigid := by
    intro σ _; apply Equiv.ext; intro u; exact Subsingleton.elim _ _
  constructor
  · refine ⟨(body.p1a_iff_unique_vacuum).2 ?_, ?_, ?_, ?_, body.p2c_of_rigid hr, ?_⟩
    · exact (SensoryMaintenance.admission.1.p1a |> (SensoryMaintenance.body.p1a_iff_unique_vacuum).1)
    · refine ⟨by intros; trivial, ?_, ?_⟩
      · intro e s x u hu; simp [body] at hu
      · intro e s s' x y hh
        have hx : x = y := funext fun u => (hh u (by simp [body])).1
        have hs : s = s' := funext fun u => (hh u (by simp [body])).2
        subst y; subst s'; exact ⟨rfl, fun _ _ => rfl⟩
    · apply body.p2a_of_every_event_open
      intro e; change ¬ Disjoint (univ : Finset Unit) univ; decide
    · refine ⟨fun _ => 1, fun _ => one_pos, fun x y => ?_⟩
      simp [Medium.jump, body]
    · exact ⟨body.onlyBoundarySenses_of_all_boundary rfl, body.initial_invariant_of_rigid hr⟩
  · refine ⟨by norm_num, ?_⟩
    intro s x; simp only [body, Fintype.sum_unique]; split_ifs <;> norm_num

abbrev ec := admission.2
noncomputable def material : body.MaterialLedger where
  incoming _ s x := incoming (s ()) (x ())
  outgoing _ s x := outgoing (s ()) (x ())
  balance _ s x _ := (local_budgets (s ()) (x ())).1
  closed_zero a s x h _ := by
    have hn : ¬ body.Closed a := by simp [Medium.Closed, body]
    exact False.elim (hn h)

noncomputable abbrev world : World Unit State Input := {
  SensoryMaintenance.world with
  sense := fun w _ => (some (RenewableLearning.sense w ()),true)
  quiet := fun _ => (none,true) }

def legacy (s : SensoryMaintenance.Stim) : Input :=
  (s, s.isSome)

theorem legacy_rules (s : SensoryMaintenance.Stim) (x : State) :
    next (legacy s) x = SensoryMaintenance.next s x ∧
    incoming (legacy s) x = SensoryMaintenance.incoming s x ∧
    outgoing (legacy s) x = SensoryMaintenance.outgoing s x := by
  rcases x with ⟨q,r⟩
  fin_cases r <;> cases s with
  | none => cases q <;> decide
  | some a => cases a <;> cases q <;> decide

def protocol (sensing resources : Bool) (n₀ T : ℕ) : ℕ → Env → Unit → Input :=
  fun t w _ => if n₀ ≤ t ∧ t < n₀+T then
    (if sensing then some (RenewableLearning.sense w ()) else none, resources)
    else (some (RenewableLearning.sense w ()),true)

abbrev S := SensoryMaintenance.S
noncomputable def process (cut : Bool) (sched : ℕ → Env → Unit → Input) :=
  MaterialSource.machine (body.jointEventKernel world ec cut sched)
noncomputable abbrev initial := SensoryMaintenance.initial

theorem actual_transition (cut : Bool) (sched : ℕ → Env → Unit → Input)
    (t : ℕ) (z z' : S) (e : Option Unit)
    (hp : 0 < (process cut sched).step t z e z') :
    z'.1 () = next (sched t z.2 ()) (z.1 ()) := by
  have hb : 0 < body.eventStep 1 (sched t z.2) z.1 e z'.1 := by
    by_contra hn
    exact (not_lt_of_ge (mul_nonpos_of_nonneg_of_nonpos
      (body.eventWorldStep_nonneg world ec cut sched t z z'.2) (le_of_not_gt hn))) hp
  cases e with
  | some u =>
    have he : body.act u (sched t z.2) z.1 = z'.1 := by
      by_contra hn; simp [Medium.eventStep, hn] at hb
    exact (congrFun he ()).symm
  | none =>
    have hx : z.1 = z'.1 := by by_contra hn; simp [Medium.eventStep, hn] at hb
    have hc : sched t z.2 () = (some .calm,true) ∨ z.1 () = empty := by
      by_contra hn; simp [Medium.eventStep, body, hn] at hb
    rw [← hx]
    rcases hc with hc | hc
    · simp [next, supplied, SensoryMaintenance.next, hc]
    · simp [hc, next_empty]

end CGB.PortSeparation
