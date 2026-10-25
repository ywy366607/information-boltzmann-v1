import CGBAxioms.GraceRecoveryAccounting
import CGBAxioms.GraceRecoveryClosure

/-!
# Sensory loss does not suspend physical maintenance

`some calm` remains the equilibrium stimulus. `none` means a missing sensory
signal in an active environment. A two-unit physical reserve is part of state,
is replenished only with actual intake, and is spent during sensory loss.
Existing awake turnover already pays maintenance; it is not charged twice.
These are explicit finite carrier rules, not a seventh general axiom.
-/

open Finset

namespace CGB.SensoryMaintenance
open RenewableLearning

abbrev State := Cell × Fin 3
abbrev Stim := Option Signal

def empty : State := (.vacuum, 0)
def cost (x : State) : ℕ := RenewableLearning.cost x.1 + x.2.val

def next (s : Stim) (x : State) : State :=
  if s = some .calm then x else if x.1 = .vacuum then empty else
  match s with
  | none => if x.2.val = 0 then empty else (x.1, ⟨x.2.val - 1, by omega⟩)
  | some a =>
      let q := RenewableLearning.next true true a x.1
      if q = .vacuum then empty else
        (q, if 0 < RenewableLearning.incoming true a x.1 then 2 else x.2)

def incoming (s : Stim) (x : State) : ℕ :=
  if s = some .calm ∨ x.1 = .vacuum then 0 else
  match s with
  | none => 0
  | some a => RenewableLearning.incoming true a x.1 +
      if RenewableLearning.next true true a x.1 ≠ .vacuum ∧
          0 < RenewableLearning.incoming true a x.1 then 2 - x.2.val else 0

def outgoing (s : Stim) (x : State) : ℕ :=
  if s = some .calm then 0 else if x.1 = .vacuum then x.2.val else
  match s with
  | none => if x.2.val = 0 then RenewableLearning.cost x.1 else 1
  | some a => RenewableLearning.outgoing true true a x.1 +
      if RenewableLearning.next true true a x.1 = .vacuum then x.2.val else 0

def writes (s : Stim) (x : State) : ℕ :=
  s.elim 0 (fun a => writeCost true a x.1)

def topups (s : Stim) (x : State) : ℕ :=
  s.elim 0 (fun a => topup true a x.1)

def lostCredit (s : Stim) (x : State) : ℕ :=
  match s with
  | none => if x.2.val = 0 then credit x.1 else 0
  | some a => discarded a x.1

theorem local_budgets (s : Stim) (x : State) :
    cost (next s x) + outgoing s x = cost x + incoming s x ∧
    credit (next s x).1 + writes s x + lostCredit s x = credit x.1 + topups s x := by
  rcases x with ⟨q,r⟩
  fin_cases r <;> cases s with
  | none => cases q <;> decide
  | some a => cases a <;> cases q <;> decide

theorem transfer_bounds (s : Stim) (x : State) :
    cost x ≤ 6 ∧ incoming s x ≤ 6 ∧ outgoing s x ≤ cost x := by
  rcases x with ⟨q,r⟩
  fin_cases r <;> cases s with
  | none => cases q <;> decide
  | some a => cases a <;> cases q <;> decide

theorem active_loss (s : Stim) (x : State) (hs : s ≠ some .calm) (hx : 0 < cost x) :
    0 < outgoing s x := by
  rcases x with ⟨q,r⟩
  fin_cases r <;> cases s with
  | none => cases q <;> norm_num [cost, RenewableLearning.cost, buffer, credit,
      outgoing] at *
  | some a => cases a <;> cases q <;> norm_num [cost, RenewableLearning.cost,
      buffer, credit, outgoing, RenewableLearning.outgoing, RenewableLearning.next] at *

theorem missing_no_intake (x : State) : incoming none x = 0 := by
  simp [incoming]

theorem next_empty (s : Stim) : next s empty = empty := by
  by_cases h : s = some .calm <;> simp [next, empty, h]

noncomputable def body : Medium Unit State Unit Stim where
  near _ _ := True
  ident _ := 0
  boundary := univ
  content := cost
  parts _ := univ
  act _ s x := fun _ => next (s ()) (x ())
  rate _ s x := if s () = some .calm ∨ x () = empty then 0 else 1
  rate_nonneg _ _ _ := by split_ifs <;> norm_num
  init _ := (.search, 2)
  calm _ := some .calm

theorem admission : body.Axioms ∧ body.EventClocked 1 := by
  have hr : body.Rigid := by
    intro σ _; apply Equiv.ext; intro u; exact Subsingleton.elim _ _
  constructor
  · refine ⟨(body.p1a_iff_unique_vacuum).2 ?_, ?_, ?_, ?_, body.p2c_of_rigid hr, ?_⟩
    · refine ⟨empty, rfl, ?_⟩
      rintro ⟨q,r⟩ h
      fin_cases r <;> cases q <;>
        norm_num [cost, RenewableLearning.cost, buffer, credit, empty, body] at *
    · refine ⟨?_, by intros; trivial, ?_, ?_⟩
      · intro u v; exact ⟨Equiv.refl Unit, by intros; rfl, Subsingleton.elim _ _⟩
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

noncomputable abbrev world : World Unit State Stim where
  W := Env
  sense w _ := some (sense w ())
  moves w x v := GraceRecovery.moves w (fun u => (x u).1) v
  moves_nonneg w x v := GraceRecovery.world.moves_nonneg w (fun u => (x u).1) v
  moves_sum w x := GraceRecovery.world.moves_sum w (fun u => (x u).1)
  start := .probeA
  quiet _ := none
  clock := 1
  clock_pos := by norm_num
  horizon := 3
  recovery := 1 / 2
  recovery_pos := by norm_num

/-- The old interrupt selects equilibrium. This protocol masks sensing only,
and leaves the active maintenance transition and the same world clock running. -/
def sensoryProtocol (n₀ T : ℕ) : ℕ → Env → Unit → Stim :=
  fun t w _ => if n₀ ≤ t ∧ t < n₀ + T then none else some (sense w ())

theorem equilibrium_distinct_from_loss : body.calm () ≠ world.quiet () := by decide

abbrev S := (Unit → State) × Env
noncomputable def process (cut : Bool) (sched : ℕ → Env → Unit → Stim) :=
  MaterialSource.machine (body.jointEventKernel world ec cut sched)
noncomputable def initial : S := (body.init, world.start)

theorem actual_transition (cut : Bool) (sched : ℕ → Env → Unit → Stim)
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
    have hc : sched t z.2 () = some .calm ∨ z.1 () = empty := by
      by_contra hn; simp [Medium.eventStep, body, hn] at hb
    rw [← hx]
    rcases hc with hc | hc
    · simp [next, hc]
    · simp [hc, next_empty]

theorem actual_law (cut : Bool) (sched : ℕ → Env → Unit → Stim) (n : ℕ) (z : S) :
    Energetics.law (process cut sched) initial n z = if cut then
      body.lawCut world sched n z.1 z.2 else body.law world sched n z.1 z.2 := by
  have hi : (fun z : S => if z = initial then (1 : ℝ) else 0) =
      (fun z => if z.1 = body.init ∧ z.2 = world.start then 1 else 0) := by
    funext z; simp [initial, Prod.ext_iff]
  change EventChannels.Kernel.stateLaw (body.jointEventKernel world ec cut sched)
    (fun z : S => if z = initial then (1 : ℝ) else 0) n z = _
  rw [hi]
  exact body.jointEventKernel_stateLaw world ec cut sched n z

theorem kernel_table (s : Unit → Stim) (x y : State) :
    body.kernel 1 s (fun _ => x) (fun _ => y) = if y = next (s ()) x then 1 else 0 := by
  have hs : s = fun _ => s () := by funext u; cases u; rfl
  rw [hs, ← body.eventStep_forget, Fintype.sum_option, Fintype.sum_unique]
  by_cases hc : s () = some .calm ∨ x = empty
  · have hn : next (s ()) x = x := by
      rcases hc with h | h
      · simp [next, h]
      · simp [h, next_empty]
    simp [Medium.eventStep, body, hc, hn, funext_iff]
    simp only [eq_comm]
  · simp [Medium.eventStep, body, hc, funext_iff]
    simp only [eq_comm]

def burn (x : State) (n : ℕ) : State := (next none)^[n] x

theorem burn_empty (n : ℕ) : burn empty n = empty := by
  induction n with
  | zero => rfl
  | succ n ih =>
    unfold burn at ih ⊢
    rw [Function.iterate_succ_apply', ih]
    rfl

theorem burn_three (x : State) : burn x 3 = empty := by
  rcases x with ⟨q,r⟩
  fin_cases r <;> cases q <;> decide

theorem burn_all_later (x : State) (n : ℕ) (hn : 3 ≤ n) : burn x n = empty := by
  obtain ⟨k,rfl⟩ := Nat.exists_eq_add_of_le hn
  unfold burn
  rw [Function.iterate_add_apply]
  exact burn_three _

theorem funded_two_tick_bridge (q : Cell) (hq : q ≠ .vacuum) :
    burn (q, 2) 1 = (q, 1) ∧ burn (q, 2) 2 = (q, 0) ∧ burn (q, 2) 3 = empty := by
  cases q <;> simp_all [burn, next, empty]

/-- Exact actual-channel fuel balance for every registered input path.
Physical time does not stop during absent sensing. -/
theorem path_budget (stim : ℕ → Stim) (p : ℕ → State)
    (hp : ∀ t, p (t + 1) = next (stim t) (p t)) (n : ℕ) :
    cost (p n) + ∑ t ∈ range n, outgoing (stim t) (p t) =
      cost (p 0) + ∑ t ∈ range n, incoming (stim t) (p t) := by
  induction n with
  | zero => simp
  | succ n ih =>
    have h := (local_budgets (stim n) (p n)).1
    rw [← hp n] at h
    simp only [Finset.sum_range_succ]
    omega

theorem path_credit_budget (stim : ℕ → Stim) (p : ℕ → State)
    (hp : ∀ t, p (t + 1) = next (stim t) (p t)) (n : ℕ) :
    credit (p n).1 + ∑ t ∈ range n, (writes (stim t) (p t) + lostCredit (stim t) (p t)) =
      credit (p 0).1 + ∑ t ∈ range n, topups (stim t) (p t) := by
  induction n with
  | zero => simp
  | succ n ih =>
    have h := (local_budgets (stim n) (p n)).2
    rw [← hp n] at h
    simp only [Finset.sum_range_succ]
    omega

end CGB.SensoryMaintenance
