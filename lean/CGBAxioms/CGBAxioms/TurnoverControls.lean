import CGBAxioms.MaterialExchange
import CGBAxioms.ConstraintFreedom
import CGBAxioms.LifeControls

/-!
# Constant-content turnover controls

These are mathematical observation controls, not bacterial/sponge models or
capability experiments. A ledger is a declared physical interface that still
needs realization evidence; identical state dynamics can support different
consistent ledgers, so balance alone cannot identify a real flux.
-/

open Finset

namespace CGB.TurnoverControls

noncomputable def body : Medium Unit Bool Unit Bool :=
  ConstraintFreedom.openSite Bool.toNat (fun _ b => b)
    (fun _ b => if b then 1 else 0) (fun _ b => by split_ifs <;> norm_num) true

theorem body_axioms : body.Axioms :=
  ConstraintFreedom.arbitrary_driven_tables_admitted Bool.toNat
    ⟨false, rfl, by decide⟩ (fun _ b => b)
    (fun _ b => if b then 1 else 0) (fun _ b => by split_ifs <;> norm_num) true

theorem body_unchanged (a : Unit) (s : Unit → Bool) (x : Unit → Bool) :
    body.act a s x = x := by
  funext u
  cases u
  simp [body, ConstraintFreedom.openSite]

theorem body_eventClock : body.EventClocked 1 := by
  refine ⟨by norm_num, ?_⟩
  intro s x
  simp [body, ConstraintFreedom.openSite]
  split_ifs <;> norm_num

/-- A declared balanced replacement, or a zero-transfer observation of the same
state dynamics. Neither choice follows from the body-state kernel alone. -/
noncomputable def ledger (q : ℕ) : body.MaterialLedger where
  incoming _ _ _ := q
  outgoing _ _ _ := q
  balance a s x _ := by rw [body_unchanged]
  closed_zero a s x h _ := by
    have hn : ¬ body.Closed a := by simp [Medium.Closed, body, ConstraintFreedom.openSite]
    exact False.elim (hn h)

noncomputable abbrev world : World Unit Bool Bool :=
  { RecoveryControls.world false with start := true }

theorem event_at_full (e : Option Unit) (y : Unit → Bool) :
    body.eventStep 1 (fun _ => true) (fun _ => true) e y =
      if e = some () ∧ y () = true then 1 else 0 := by
  cases e with
  | none => simp [Medium.eventStep, body, ConstraintFreedom.openSite]
  | some a =>
    cases a
    have hy : y = (fun _ => y ()) := by funext u; cases u; rfl
    rw [hy]
    cases h : y () <;> simp [Medium.eventStep, body, ConstraintFreedom.openSite, funext_iff]

theorem old_net_recovery_zero (cut : Bool) (sched : ℕ → world.W → Unit → Bool)
    (t k : ℕ) (x : Unit → Bool) (w : world.W) :
    Operational.recovery body world (body.eventClocked_implies_clocked body_eventClock)
      cut sched t k (x, w) = 0 := by
  apply Operational.recovery_zero_without_intake
  intro y h
  obtain ⟨a, s, hp, hi⟩ := h
  rw [body_unchanged] at hi
  exact lt_irrefl _ hi

/-- A one-world-tick exchange occurs with certainty although net content never
increases. Marking channels detects the positive incoming AND outgoing count. -/
theorem balanced_exchange_recovery_one (t : ℕ) :
    Exchange.recovery body world body_eventClock (ledger 1) false
      (fun _ => world.sense) t 1 ((fun _ => true), true) = 1 := by
  simp only [Exchange.recovery, EventChannels.Kernel.firstHit,
    Medium.jointEventKernel, Fintype.sum_option, Fintype.sum_prod_type]
  simp only [RecoveryControls.sum_functions, Fintype.sum_bool]
  simp [Exchange.intake, ledger, Medium.eventWorldStep, Medium.eventStep, body,
    ConstraintFreedom.openSite, funext_iff]

theorem zero_ledger_recovery_zero (cut : Bool) (sched : ℕ → world.W → Unit → Bool)
    (t k : ℕ) (z : (Unit → Bool) × world.W) :
    Exchange.recovery body world body_eventClock (ledger 0) cut sched t k z = 0 :=
  Exchange.recovery_zero_without_incoming body world body_eventClock (ledger 0)
    (fun _ _ _ => rfl) cut sched t k z

theorem balanced_exchange_alive (t : ℕ) :
    Exchange.alive body world body_eventClock (ledger 1) false
      (fun _ => world.sense) t ((fun _ => true), true) = 1 := by
  classical
  have hr : Exchange.recovery body world body_eventClock (ledger 1) false
      (fun _ => world.sense) t world.horizon ((fun _ => true), true) = 1 := by
    change Exchange.recovery body world body_eventClock (ledger 1) false
      (fun _ => world.sense) t 2 ((fun _ => true), true) = 1
    simp only [Exchange.recovery, EventChannels.Kernel.firstHit,
      Medium.jointEventKernel, Fintype.sum_option, Fintype.sum_prod_type]
    simp only [RecoveryControls.sum_functions, Fintype.sum_bool]
    simp [Exchange.intake, ledger, Medium.eventWorldStep, Medium.eventStep, body,
      ConstraintFreedom.openSite, funext_iff]
  unfold Exchange.alive Viability.alive
  rw [hr]
  norm_num [Medium.total, world, RecoveryControls.world, body, ConstraintFreedom.openSite]

/-- The state process stays at the same full-body/world state forever, even
though the registered replacement channel fires each tick. -/
theorem natural_law (n : ℕ) (x : Unit → Bool) (w : world.W) :
    body.law world (fun _ => world.sense) n x w =
      if x = (fun _ => true) ∧ w = true then 1 else 0 := by
  classical
  have hk : ∀ s x y, body.kernel 1 s x y = if x = y then 1 else 0 := by
    intro s x y
    exact body.kernel_from_frozen_state x (fun a s _ => body_unchanged a s x) 1 s y
  induction n generalizing x w with
  | zero => rfl
  | succ n ih =>
    simp only [Medium.law, ih, RecoveryControls.sum_functions, Fintype.sum_bool]
    simp [hk, funext_iff, ite_and]

theorem natural_survival_one (n : ℕ) :
    Exchange.survival body world body_eventClock (ledger 1) false
      (fun _ => world.sense) n = 1 := by
  classical
  simpa [Exchange.survival, natural_law, mul_ite, ite_and] using balanced_exchange_alive n

/-- A material-throughput subtarget witness, not the complete life or cognitive
candidate: the separate SelfMaintains and cognitive obligations are not assumed. -/
theorem balanced_exchange_persists :
    Exchange.Persists body world body_eventClock (ledger 1) 0 := by
  refine ⟨by norm_num, fun n => ?_⟩
  rw [natural_survival_one]
  norm_num

/-- Same original state process, two different first-intake observations. Thus
net-state conservation cannot identify whether there is gross replacement. -/
theorem old_net_and_balanced_exchange_disagree :
    Operational.recovery body world (body.eventClocked_implies_clocked body_eventClock)
      false (fun _ => world.sense) 0 1 ((fun _ => true), true) = 0 ∧
    Exchange.recovery body world body_eventClock (ledger 1) false
      (fun _ => world.sense) 0 1 ((fun _ => true), true) = 1 :=
  ⟨old_net_recovery_zero _ _ _ _ _ _, balanced_exchange_recovery_one 0⟩

/-- Two algebraically accountable ledgers give different gross observations
on IDENTICAL body/world dynamics. Physical transport must therefore be registered
and validated beyond the net-balance equation. -/
theorem ledger_not_identified_by_state_dynamics :
    Exchange.recovery body world body_eventClock (ledger 0) false
      (fun _ => world.sense) 0 1 ((fun _ => true), true) = 0 ∧
    Exchange.recovery body world body_eventClock (ledger 1) false
      (fun _ => world.sense) 0 1 ((fun _ => true), true) = 1 :=
  ⟨zero_ledger_recovery_zero _ _ _ _ _, balanced_exchange_recovery_one 0⟩

/-- The old exit-rate clock admits this zero-exit process at any positive rate;
resolving more than one self-event per slot would require stronger admission. -/
noncomputable def fastBody : Medium Unit Bool Unit Bool :=
  { body with rate := fun _ _ _ => 2, rate_nonneg := fun _ _ _ => by norm_num }

theorem fastBody_old_clock : fastBody.Clocked 1 := by
  refine ⟨by norm_num, ?_⟩
  intro s x
  simp [Medium.jump, fastBody, body_unchanged]

theorem fastBody_no_event_clock : ¬ fastBody.EventClocked 1 := by
  intro h
  have hh := h.2 (fun _ => true) (fun _ => true)
  norm_num [fastBody] at hh

end CGB.TurnoverControls
