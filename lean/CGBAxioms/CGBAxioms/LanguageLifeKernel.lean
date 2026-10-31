import CGBAxioms.AlgebraicCore

/-!
# Compact algebraic reference kernel for the language-life model

This is a finite exact reference object, not a neural-network capability
claim. Token/event drives are supplied as edge flows. The kernel proves only
the properties needed by a later Torch/CUDA implementation: ternary
incidence, exact transport conservation, a strictly advancing clock, and a
paid reserve update. It deliberately does not introduce K/T/H/D as axioms.
-/

namespace CGB.LanguageLifeKernel

structure TernaryGraph (N E : Nat) where
  source : Fin E → Fin N
  target : Fin E → Fin N

def incidence {N E : Nat} (g : TernaryGraph N E) (e : Fin E) (n : Fin N) : Int :=
  (if n = g.target e then 1 else 0) - (if n = g.source e then 1 else 0)

theorem incidence_ternary {N E : Nat} (g : TernaryGraph N E)
    (e : Fin E) (n : Fin N) :
    incidence g e n = -1 ∨ incidence g e n = 0 ∨ incidence g e n = 1 := by
  unfold incidence
  split_ifs <;> norm_num

theorem incidence_sum_zero {N E : Nat} [NeZero N]
    (g : TernaryGraph N E) (e : Fin E) :
    ∑ n : Fin N, incidence g e n = 0 := by
  classical
  simp [incidence, Finset.sum_sub_distrib]

def chargeUpdate {N E : Nat} (g : TernaryGraph N E)
    (charge : Fin N → Int) (flow : Fin E → Int) (n : Fin N) : Int :=
  charge n + ∑ e, flow e * incidence g e n

theorem charge_total_preserved {N E : Nat} [NeZero N]
    (g : TernaryGraph N E) (charge : Fin N → Int) (flow : Fin E → Int) :
    ∑ n, chargeUpdate g charge flow n = ∑ n, charge n := by
  classical
  simp only [chargeUpdate, Finset.sum_add_distrib]
  rw [Finset.sum_comm]
  simp_rw [← Finset.mul_sum, incidence_sum_zero]
  simp

def reserveUpdate (reserve incoming outgoing : Nat) : Nat :=
  reserve + incoming - outgoing

theorem reserve_balance (reserve incoming outgoing : Nat)
    (h : outgoing ≤ reserve + incoming) :
    reserveUpdate reserve incoming outgoing + outgoing = reserve + incoming := by
  exact Nat.sub_add_cancel h

structure State (N : Nat) where
  tick : Nat
  charge : Fin N → Int
  reserve : Nat

def step {N E : Nat} (g : TernaryGraph N E) (s : State N)
    (flow : Fin E → Int) (incoming outgoing : Nat) : State N :=
  { tick := s.tick + 1
    charge := chargeUpdate g s.charge flow
    reserve := reserveUpdate s.reserve incoming outgoing }

theorem step_clock_advances {N E : Nat} (g : TernaryGraph N E) (s : State N)
    (flow : Fin E → Int) (incoming outgoing : Nat) :
    s.tick < (step g s flow incoming outgoing).tick := by
  exact Nat.lt_succ_self s.tick

theorem step_charge_conserved {N E : Nat} [NeZero N]
    (g : TernaryGraph N E) (s : State N) (flow : Fin E → Int)
    (incoming outgoing : Nat) :
    ∑ n, (step g s flow incoming outgoing).charge n = ∑ n, s.charge n := by
  exact charge_total_preserved g s.charge flow

theorem step_reserve_paid {N E : Nat} (g : TernaryGraph N E) (s : State N)
    (flow : Fin E → Int) (incoming outgoing : Nat)
    (h : outgoing ≤ s.reserve + incoming) :
    (step g s flow incoming outgoing).reserve + outgoing = s.reserve + incoming := by
  exact reserve_balance s.reserve incoming outgoing h

/-!
## Delayed credit: the first necessity result

This is deliberately a representation-independent obstruction.  The two
histories have the same visible state at the time a delayed result arrives,
but different past participation.  If the required credit differs between
those histories, a rule that reads only the current visible state cannot be
correct on both.  The theorem does **not** choose an exponential eligibility
trace (or any other particular carrier); it only proves that some history
carrier or equivalent sufficient statistic is required under these contracts.
-/

def delayedCreditTarget {C : Type} (credit₀ credit₁ : C) : Bool → C
  | false => credit₀
  | true => credit₁

theorem no_visible_only_delayed_credit_rule {X C : Type}
    (rule : X → C) (visible : X) (credit₀ credit₁ : C)
    (different : credit₀ ≠ credit₁) :
    ¬ (rule visible = delayedCreditTarget credit₀ credit₁ false ∧
      rule visible = delayedCreditTarget credit₀ credit₁ true) := by
  rintro ⟨first, second⟩
  exact different (first.symm.trans second)

theorem delayed_credit_requires_history_carrier {X C : Type}
    (rule : X → C) (visible : X) (credit₀ credit₁ : C)
    (different : credit₀ ≠ credit₁) :
    ¬ (rule visible = credit₀ ∧ rule visible = credit₁) := by
  exact no_visible_only_delayed_credit_rule rule visible credit₀ credit₁ different

/-- Delayed action credit needs a state distinction between alternatives whose
later consequences require different credit.  This does not require a separate
action log: an existing action-induced body change, a self-produced utterance
written into the same state, or a world-mediated state change may supply the
needed distinction.  If all are absent, no rule reading only the later state
can recover which action occurred. -/
theorem delayed_action_credit_requires_state_separation {S A C : Type}
    (advance : S → A → S) (rule : S → C) (s : S)
    (action₀ action₁ : A) (credit₀ credit₁ : C)
    (same_state : advance s action₀ = advance s action₁)
    (different_credit : credit₀ ≠ credit₁) :
    ¬ (rule (advance s action₀) = credit₀ ∧
      rule (advance s action₁) = credit₁) := by
  rintro ⟨h₀, h₁⟩
  apply different_credit
  calc
    credit₀ = rule (advance s action₀) := h₀.symm
    _ = rule (advance s action₁) := by rw [same_state]
    _ = credit₁ := h₁

/-!
## What conservation adds

The obstruction above says that a local rule needs a history carrier.  The
following small structure records the stronger physical case in which that
carrier is a typed, locally conserved transfer: one source, one target, and a
named amount.  The balance equations then prove that the amount is recoverable
from the target (or source) state difference.  The structure is an explicit
model condition, not a new general axiom; conservation alone does not provide
the typing or observability needed for this conclusion when several flows can
cancel or when the carrier is hidden from the controller.
-/

structure LocalConservedTransfer where
  sourceBefore : Int
  targetBefore : Int
  sourceAfter : Int
  targetAfter : Int
  amount : Int
  source_balance : sourceAfter + amount = sourceBefore
  target_balance : targetAfter = targetBefore + amount

theorem conserved_transfer_target_trace (t : LocalConservedTransfer) :
    t.targetAfter - t.targetBefore = t.amount := by
  rw [t.target_balance]
  ring

theorem conserved_transfer_source_trace (t : LocalConservedTransfer) :
    t.sourceBefore - t.sourceAfter = t.amount := by
  rw [← t.source_balance]
  ring

theorem conserved_transfer_total_preserved (t : LocalConservedTransfer) :
    t.sourceAfter + t.targetAfter = t.sourceBefore + t.targetBefore := by
  linarith [t.source_balance, t.target_balance]

/-!
## Observability and the limits of conservation

The previous structure supplies the physical transfer.  A delayed learner can
use it only when the relevant carrier is typed and a target readout is still
available when the result arrives.  The first theorem below derives exact
credit from that readout.  The two following results are explicit failure
boundaries: superposed flows expose only their net sum, and leakage exposes
only input minus leakage.  They prevent us from upgrading global conservation
to an unqualified eligibility-trace claim.
-/

structure ObservableConservedTransfer extends LocalConservedTransfer where
  targetReadout : Int
  target_readout : targetReadout = targetAfter - targetBefore

theorem observable_transfer_recovers_amount (t : ObservableConservedTransfer) :
    t.targetReadout = t.amount := by
  exact t.target_readout.trans (conserved_transfer_target_trace t.toLocalConservedTransfer)

theorem superposed_transfer_readout_is_net (targetBefore flow₁ flow₂ : Int) :
    (targetBefore + flow₁ + flow₂) - targetBefore = flow₁ + flow₂ := by
  ring

theorem conservation_does_not_identify_components :
    ∃ (flow₁ flow₂ other₁ other₂ : Int),
      flow₁ + flow₂ = other₁ + other₂ ∧ flow₁ ≠ other₁ := by
  refine ⟨1, 0, 0, 1, by norm_num, by norm_num⟩

structure LeakyConservedTransfer where
  before : Int
  after : Int
  input : Int
  leaked : Int
  balance : after + leaked = before + input

theorem leaky_transfer_readout_is_net (t : LeakyConservedTransfer) :
    t.after - t.before = t.input - t.leaked := by
  linarith [t.balance]

/-!
## A persistent carrier compresses a finite history

This is the discrete form of the physical argument.  A carrier need not keep
an autograd tape: if every tick obeys the local balance law, its current value
contains the telescoped net history.  The theorem is exact for the declared
carrier; it does not claim that an arbitrary neural state has this property.
-/

structure ConservedCarrierPath where
  value : Nat → Int
  input : Nat → Int
  output : Nat → Int
  balance : ∀ n, value (n + 1) + output n = value n + input n

theorem conserved_carrier_path_balance (p : ConservedCarrierPath) (k : Nat) :
    p.value k - p.value 0 =
      Finset.sum (Finset.range k) (fun n => p.input n - p.output n) := by
  induction k with
  | zero => simp
  | succ k ih =>
      rw [Finset.sum_range_succ]
      linarith [p.balance k]

theorem conserved_carrier_path_no_output (p : ConservedCarrierPath) (k : Nat)
    (no_output : ∀ n, p.output n = 0) :
    p.value k - p.value 0 =
      Finset.sum (Finset.range k) p.input := by
  simpa [no_output] using conserved_carrier_path_balance p k

/-!
## Which history can be discarded?

Conservation is an accounting property. Exact task readout additionally needs
the encoding to separate histories whenever their required answers differ.
Recursive use of a compressed state also needs encoded transitions to agree
on every encoding fibre. These results characterize the two requirements;
they neither assume biological tags nor prove that a finite sufficient
encoding exists for an arbitrary task. `fallback` handles unreachable codes.
-/

theorem history_readout_exists_iff {H S C : Type*}
    (encode : H → S) (required : H → C) (fallback : C) :
    (∃ read : S → C, ∀ h, read (encode h) = required h) ↔
      (∀ h h', encode h = encode h' → required h = required h') := by
  let : Nonempty C := ⟨fallback⟩
  simpa only [Function.FactorsThrough, funext_iff, Function.comp_apply, eq_comm] using
    (Function.factorsThrough_iff (f := encode) required).symm

theorem history_transition_exists_iff {H S U : Type*}
    (encode : H → S) (advance : H → U → H) (fallback : S) :
    (∃ next : S → U → S, ∀ h u,
      next (encode h) u = encode (advance h u)) ↔
      (∀ h h', encode h = encode h' → ∀ u,
        encode (advance h u) = encode (advance h' u)) := by
  let : Nonempty (U → S) := ⟨fun _ => fallback⟩
  simpa only [Function.FactorsThrough, funext_iff, Function.comp_apply, eq_comm] using
    (Function.factorsThrough_iff (f := encode) (fun h u => encode (advance h u))).symm

/-!
## Distributed traces: internal state and retrieved external records

The external store is not directly visible to a learner. The retrieval channel
has an explicit input and return value. Exact answers to a family of queries
(which may be specified interventions) exist precisely when the internal state
and actual returned view separate histories requiring different answers.
No premise identifies causal effects from passive records: `required` must be
grounded separately in a world model/identification contract. These are static
readout results; recursive retrieval policy, cost and latency remain separate.
-/

theorem external_causal_readout_exists_iff {H I E A V C : Type*}
    (internal : H → I) (external : H → E) (retrieve : I → E → A → V)
    (required : H → A → C) (fallback : C) :
    (∃ read : I → A → V → C, ∀ h a,
      read (internal h) a (retrieve (internal h) (external h) a) = required h a) ↔
      (∀ h h' a, internal h = internal h' →
        retrieve (internal h) (external h) a =
          retrieve (internal h') (external h') a → required h a = required h' a) := by
  constructor
  · rintro ⟨read, correct⟩ h h' a sameInternal sameView
    rw [← correct h a, ← correct h' a, sameView, sameInternal]
  · intro sufficient
    let encode : H × A → (I × A) × V := fun p =>
      ((internal p.1, p.2), retrieve (internal p.1) (external p.1) p.2)
    have separates : ∀ p p', encode p = encode p' →
        required p.1 p.2 = required p'.1 p'.2 := by
      rintro ⟨h, a⟩ ⟨h', a'⟩ same
      have sameQuery : a = a' := congrArg (fun p => p.1.2) same
      subst a'
      exact sufficient h h' a (congrArg (fun p => p.1.1) same) (congrArg Prod.snd same)
    obtain ⟨read, correct⟩ :=
      (history_readout_exists_iff encode (fun p => required p.1 p.2) fallback).mpr separates
    exact ⟨fun i a v => read ((i, a), v), fun h a => correct (h, a)⟩

theorem external_trace_requires_retrieved_distinction {H I A V C : Type*}
    (internal : H → I) (returned : H → A → V) (required : H → A → C)
    (read : I → A → V → C)
    (correct : ∀ h a, read (internal h) a (returned h a) = required h a)
    (h h' : H) (a : A) (sameInternal : internal h = internal h')
    (different : required h a ≠ required h' a) : returned h a ≠ returned h' a := by
  intro sameView
  apply different
  rw [← correct h a, ← correct h' a, sameView, sameInternal]

/-- Two world equations agree on every recorded A=U trajectory but disagree
under the intervention U=false, A=true. This is an exact logical countermodel,
not a learning/capability experiment or an empirical causal-identification claim. -/
theorem observational_record_does_not_fix_intervention :
    let response : Bool → Bool → Bool → Bool := fun direct u a =>
      if direct then a else u
    (∀ u, response true u u = response false u u) ∧
      response true false true ≠ response false false true := by
  decide

/-!
## A concrete causal record for the existing sensory-maintenance body

The record is derived from the admitted body's existing transition, not a new
learning rule. Five live structural states, each with three reserve levels,
plus vacuum form a closed sixteen-state region. Held states and vacuum with
nonzero reserve lie outside it. The decoder is an exact representation of the
body state, so every existing world response and budget using that state is
preserved. Reachability witnesses are checked separately by the finite export
audit. No hidden world state is read by `write`: only the previous record and
the actual boundary stimulus are used.
-/

namespace SensoryRecord

open RenewableLearning

def admissible (x : SensoryMaintenance.State) : Prop :=
  x.1 ≠ .aHeld ∧ x.1 ≠ .bHeld ∧ (x.1 = .vacuum → x.2.val = 0)

instance (x : SensoryMaintenance.State) : Decidable (admissible x) :=
  inferInstanceAs (Decidable (_ ∧ _ ∧ _))

def encode (x : SensoryMaintenance.State) : Fin 16 :=
  match x.1 with
  | .search => ⟨1 + x.2.val, by have := x.2.isLt; omega⟩
  | .aSpent => ⟨4 + x.2.val, by have := x.2.isLt; omega⟩
  | .bSpent => ⟨7 + x.2.val, by have := x.2.isLt; omega⟩
  | .aReady => ⟨10 + x.2.val, by have := x.2.isLt; omega⟩
  | .bReady => ⟨13 + x.2.val, by have := x.2.isLt; omega⟩
  | _ => 0

def read (r : Fin 16) : SensoryMaintenance.State :=
  if r.val = 0 then SensoryMaintenance.empty else
    let k := r.val - 1
    (if k < 3 then .search else if k < 6 then .aSpent
      else if k < 9 then .bSpent else if k < 12 then .aReady else .bReady,
      ⟨k % 3, Nat.mod_lt _ (by decide)⟩)

def write (s : SensoryMaintenance.Stim) (r : Fin 16) : Fin 16 :=
  encode (SensoryMaintenance.next s (read r))

theorem exact_record_contract :
    (∀ r, encode (read r) = r ∧ admissible (read r)) ∧
    (∀ x, admissible x → read (encode x) = x) ∧
    (∀ s r, read (write s r) = SensoryMaintenance.next s (read r)) := by
  constructor
  · decide
  constructor
  · intro x
    rcases x with ⟨q, r⟩
    cases q <;> fin_cases r <;> decide
  · decide

/-- All sixteen records already have distinct action/content observations.
Any exact representation preserving both must distinguish all sixteen. -/
theorem action_content_distinguishes :
    ∀ r r' : Fin 16,
      policy (read r).1 = policy (read r').1 →
      SensoryMaintenance.cost (read r) = SensoryMaintenance.cost (read r') → r = r' := by
  decide

theorem record_state_lower_bound {R : Type*} [Fintype R]
    (record : Fin 16 → R) (readAction : R → Action) (readContent : R → Nat)
    (action_correct : ∀ r, readAction (record r) = policy (read r).1)
    (content_correct : ∀ r, readContent (record r) = SensoryMaintenance.cost (read r)) :
    16 ≤ Fintype.card R := by
  have injective : Function.Injective record := by
    intro r r' same
    apply action_content_distinguishes r r'
    · rw [← action_correct r, ← action_correct r', same]
    · rw [← content_correct r, ← content_correct r', same]
  simpa using Fintype.card_le_of_injective record injective

/-- The finite record supports arbitrary-length replay by a forward fold,
without consulting the world regime or keeping an autograd history. -/
theorem fold_refines (inputs : List SensoryMaintenance.Stim) (r : Fin 16) :
    read (inputs.foldl (fun code s => write s code) r) =
      inputs.foldl (fun x s => SensoryMaintenance.next s x) (read r) := by
  induction inputs generalizing r with
  | nil => rfl
  | cons s rest ih =>
      simp only [List.foldl_cons]
      rw [ih, exact_record_contract.2.2]

/-- Every positive-probability successor in the original joint body/world
process is exactly representable, for either registered cut and any schedule.
The environment and event label remain outside the body record. -/
theorem joint_successor_reconstruct (cut : Bool)
    (sched : Nat → Env → Unit → SensoryMaintenance.Stim)
    (t : Nat) (r : Fin 16) (w : Env) (z' : SensoryMaintenance.S)
    (e : Option Unit)
    (positive : 0 < (SensoryMaintenance.process cut sched).step t
      ((fun _ => read r), w) e z') :
    read (encode (z'.1 ())) = z'.1 () := by
  have update := SensoryMaintenance.actual_transition cut sched t
    ((fun _ => read r), w) z' e positive
  rw [update]
  exact exact_record_contract.2.2 _ _

end SensoryRecord

/-! The existing world's actuator port factors through the live body's action.
These integer weights are its exact probabilities times two. They describe
one world move, before the body's simultaneous update; they are not the cut
kernel, a new world, or an independently executable action policy. -/
namespace CausalRecord

open RenewableLearning
open Finset

def weight (w : Env) (a : Action) (v : Env) : Nat :=
  match w with
  | .feedA => if a = 2 then (if v = .answerA then 2 else 0)
      else if a = 0 then (if v = .feedA ∨ v = .feedB then 1 else 0)
      else if v = .missA then 2 else 0
  | .feedB => if a = 2 then (if v = .answerB then 2 else 0)
      else if a = 1 then (if v = .feedA ∨ v = .feedB then 1 else 0)
      else if v = .missB then 2 else 0
  | .missA => if v = .probeA then 2 else 0
  | .missB => if v = .probeB then 2 else 0
  | .probeA => if v = (if a = 2 then .answerA else .missA) then 2 else 0
  | .probeB => if v = (if a = 2 then .answerB else .missB) then 2 else 0
  | .answerA => if v = .feedA then 2 else 0
  | .answerB => if v = .feedB then 2 else 0
  | .trap => if v = .trap then 2 else 0

private theorem policy_cases (q : Cell) : policy q = match q with
    | .aSpent | .aReady | .aHeld => 0
    | .bSpent | .bReady | .bHeld => 1
    | _ => 2 := by cases q <;> decide

set_option maxHeartbeats 1200000 in
-- Exhaust the finite world/cell/successor cases to certify the integer law.
theorem weights_exact (w v : Env) (q : Cell) (live : q ≠ .vacuum) :
    GraceRecovery.moves w (fun _ => q) v = (weight w (policy q) v : ℝ) / 2 := by
  cases w <;> cases q <;> cases v <;>
    norm_num [GraceRecovery.moves, RenewableLearning.moves, weight, policy_cases] at *

def observeWeight (w : Env) (a : Action) (o : Signal) : Nat :=
  ∑ v : Env, if sense v () = o then weight w a v else 0

/-- Same present sensor, different consequences of the same actuator setting. -/
theorem sensor_alias_witness :
    sense .feedA () = sense .feedB () ∧
    observeWeight .feedA 0 .feed = 2 ∧ observeWeight .feedB 0 .feed = 0 := by
  decide

/-- Support information only: this is not a posterior probability or a paid
physical learning mechanism. Both the executed action and actual next sensor
are inputs; the hidden true world state is not. -/
def update (prior : Finset Env) (a : Action) (o : Signal) : Finset Env :=
  univ.filter (fun v => sense v () = o ∧ ∃ w ∈ prior, 0 < weight w a v)

theorem update_sound (prior : Finset Env) (w v : Env) (q : Cell)
    (live : q ≠ .vacuum) (known : w ∈ prior)
    (possible : 0 < GraceRecovery.moves w (fun _ => q) v) :
    v ∈ update prior (policy q) (sense v ()) := by
  have hp : 0 < weight w (policy q) v := by
    by_contra hn
    have zero : weight w (policy q) v = 0 := Nat.eq_zero_of_not_pos hn
    rw [weights_exact w v q live, zero] at possible
    norm_num at possible
  simp only [update, mem_filter, mem_univ, true_and]
  exact ⟨w, known, hp⟩

/-- Any candidate set retaining every compatible successor contains this one.
Thus the filter discards exactly the support exclusions licensed by the
declared model and supplied evidence; it does not assert identifiability. -/
theorem update_least (prior posterior : Finset Env) (a : Action) (o : Signal)
    (safe : ∀ w ∈ prior, ∀ v, sense v () = o → 0 < weight w a v → v ∈ posterior) :
    update prior a o ⊆ posterior := by
  intro v hv
  simp only [update, mem_filter, mem_univ, true_and] at hv
  rcases hv with ⟨observed, w, known, possible⟩
  exact safe w known v observed possible

/-- An identical later sensor has different interpretations when the earlier
action differs. Keeping an existing action-bearing record can supply it;
this does not mandate a separate action log or biological eligibility rule. -/
theorem action_feedback_witness :
    update {.feedA, .feedB} 0 .miss = {.missB} ∧
    update {.feedA, .feedB} 1 .miss = {.missA} := by
  decide

end CausalRecord

/-! Paid inference about a FROZEN source state. A delayed receipt must carry
its own action and the corresponding next sensor. Its source is not the world
state at delivery. Costs below use the existing write-credit convention, not
joules; capture/holding/read tariffs are supplied carrier hypotheses. -/
namespace PaidRecord

open RenewableLearning
open Finset

def possible : CausalLearning.Support Env Unit Action Signal :=
  fun w _ a o => decide (0 < CausalRecord.observeWeight w a o)

def prepay {B : Nat} (m : CausalLearning.Memory Env B) (cost : Nat) :
    CausalLearning.Memory Env B :=
  ⟨m.candidates, ⟨m.reserve.val - cost, by have := m.reserve.isLt; omega⟩⟩

/-- Unaffordable overhead admits no read. A funded read can still leave no
write credit, in which case the original candidate-learning rule holds. -/
def settle {B : Nat} (m : CausalLearning.Memory Env B) (a : Action)
    (o : Signal) (overhead : Nat) : CausalLearning.Memory Env B :=
  if overhead ≤ m.reserve.val then
    CausalLearning.advance possible (prepay m overhead) () a o else m

def fee {B : Nat} (m : CausalLearning.Memory Env B) (a : Action)
    (o : Signal) (overhead : Nat) : Nat :=
  if overhead ≤ m.reserve.val then
    overhead + CausalLearning.fee possible (prepay m overhead) () a o else 0

theorem settlement_contract {B : Nat} (m : CausalLearning.Memory Env B)
    (a : Action) (o : Signal) (overhead : Nat) :
    (settle m a o overhead).reserve.val + fee m a o overhead = m.reserve.val ∧
    (∀ w ∈ m.candidates, possible w () a o = true →
      w ∈ (settle m a o overhead).candidates) := by
  by_cases funded : overhead ≤ m.reserve.val
  · have hb := (CausalLearning.write_contract possible (prepay m overhead) () a o).2.1
    constructor
    · simp only [settle, fee, ite_eq_left funded]
      change _ + _ = m.reserve.val - overhead at hb
      omega
    · intro w hw ho
      simp only [settle, ite_eq_left funded]
      exact CausalLearning.retain_consistent possible (prepay m overhead) () a o w hw ho
  · constructor
    · simp [settle, fee, funded]
    · intro w hw _
      simpa [settle, funded] using hw

theorem world_feedback_possible (w v : Env) (q : Cell) (live : q ≠ .vacuum)
    (hp : 0 < GraceRecovery.moves w (fun _ => q) v) :
    possible w () (policy q) (sense v ()) = true := by
  have hpw : 0 < CausalRecord.weight w (policy q) v := by
    by_contra hn
    have hz := Nat.eq_zero_of_not_pos hn
    rw [CausalRecord.weights_exact w v q live, hz] at hp
    norm_num at hp
  have finite_support : ∀ w a v, 0 < CausalRecord.weight w a v →
      possible w () a (sense v ()) = true := by decide
  exact finite_support w (policy q) v hpw

/-- Add the recorder's separately endowed reserve to the ORIGINAL body's
credit ledger. The original writes are not charged twice, and its credit is
not silently reused to fund the added recorder. Capture/hold charges compose
with this identity by the same prepayment subtraction. -/
theorem combined_write_budget {B : Nat} (m : CausalLearning.Memory Env B)
    (s : SensoryMaintenance.Stim) (x : SensoryMaintenance.State)
    (a : Action) (o : Signal) (overhead : Nat) :
    credit (SensoryMaintenance.next s x).1 + (settle m a o overhead).reserve.val +
      SensoryMaintenance.writes s x + SensoryMaintenance.lostCredit s x +
      fee m a o overhead =
    credit x.1 + m.reserve.val + SensoryMaintenance.topups s x := by
  have body_budget := (SensoryMaintenance.local_budgets s x).2
  have recorder_budget := (settlement_contract m a o overhead).1
  omega

/-- Holding a source hypothesis is sound about its origin, not about a moving
world. Also, substituting another action changes which source is compatible. -/
theorem historical_frame_witness :
    CausalRecord.weight .missA 0 .probeA = 2 ∧
    .probeA ∉ ({.missA} : Finset Env) ∧
    possible .feedA () 1 .miss = true ∧ possible .feedA () 0 .miss = false := by
  decide

/-- The ORIGINAL normal joint kernel, with any registered sensory schedule,
licenses a receipt only for its pre-update body action and its successor's
actual sensor. Whether that sensor is delivered is a separate port condition. -/
theorem joint_source_possible (sched : Nat → Env → Unit → SensoryMaintenance.Stim)
    (t : Nat) (z z' : SensoryMaintenance.S) (e : Option Unit)
    (live : (z.1 ()).1 ≠ .vacuum)
    (hp : 0 < (SensoryMaintenance.process false sched).step t z e z') :
    possible z.2 () (policy (z.1 ()).1) (sense z'.2 ()) = true := by
  have hw : 0 < SensoryMaintenance.world.moves z.2 z.1 z'.2 := by
    by_contra hn
    have hb := SensoryMaintenance.body.eventStep_nonneg SensoryMaintenance.ec
      (sched t z.2) z.1 e z'.1
    exact (not_lt_of_ge (mul_nonpos_of_nonpos_of_nonneg (le_of_not_gt hn) hb)) hp
  exact world_feedback_possible z.2 z'.2 (z.1 ()).1 live hw

/-- Deterministic finite recorder attachment. The physical event and its
probability stay those of P, even if the recorder fails or refuses a write. -/
noncomputable def attach {S I O R : Type*} [Fintype S] [Fintype O] [Fintype R]
    [DecidableEq R] (P : Interaction.Machine S I O)
    (advance : I → S → O → S → R → R) : Interaction.Machine (S × R) I O where
  step i z o z' := if z'.2 = advance i z.1 o z'.1 z.2 then P.step i z.1 o z'.1 else 0
  nonneg i z o z' := by split <;> simp_all [P.nonneg]
  sum_one i z := by
    simp only [Fintype.sum_prod_type]
    simpa using P.sum_one i z.1

/-- Projection recovers the whole original kernel, hence its registered
finite-future observations by Interaction.Refines.sameBehavior. This theorem
does not certify a particular recorder's sensor provenance or tariff physics. -/
theorem attach_refines {S I O R : Type*} [Fintype S] [DecidableEq S]
    [Fintype O] [Fintype R] [DecidableEq R] (P : Interaction.Machine S I O)
    (advance : I → S → O → S → R → R) :
    Interaction.Refines (attach P advance) P Prod.fst := by
  intro i z o y
  simp [attach, Fintype.sum_prod_type]

end PaidRecord

namespace LocalEdit

/-- A binary factor edit depends on joint contrast, not just either marginal.
This algebraic identity specifies no history recurrence or world causal law. -/
theorem binary_factor_balance {R : Type*} [CommRing R] (a b c d u v : R) :
    (b + v * d) * (a + b + u * c + u * d) -
      (b + u * d) * (a + b + v * c + v * d) =
        (v - u) * (a * d - b * c) := by ring

/-- Exact integer admission for one fixed-context candidate; not its generator. -/
def accepts {S : Type*} [Fintype S] [DecidableEq S]
    (event : Finset S) (old candidate : S → Nat) : Bool :=
  decide ((∑ s ∈ event, candidate s) * (∑ s, old s) >
    (∑ s ∈ event, old s) * (∑ s, candidate s))

/-- With positive total masses, the executable comparison exactly detects
improvement of the same event; inference and resource contracts are separate. -/
theorem accepts_iff {S : Type*} [Fintype S] [DecidableEq S]
    (event : Finset S) (old candidate : S → Nat)
    (ho : 0 < ∑ s, old s) (hc : 0 < ∑ s, candidate s) :
    accepts event old candidate = true ↔
      (∑ s ∈ event, (old s : ℝ)) / (∑ s, (old s : ℝ)) <
        (∑ s ∈ event, (candidate s : ℝ)) / (∑ s, (candidate s : ℝ)) := by
  have hor : 0 < ∑ s, (old s : ℝ) := by exact_mod_cast ho
  have hcr : 0 < ∑ s, (candidate s : ℝ) := by exact_mod_cast hc
  rw [accepts, decide_eq_true_eq, div_lt_div_iff₀ hor hcr]
  norm_cast

end LocalEdit

namespace FirstHit

/-! Finite-support interface for the candidate independent, constant-rate
trial model H_G. Independence and the identification of a biological contact
with `lifetime` are constitutive hypotheses, not consequences of conservation.
Zero lifetimes are allowed, so the algebra also covers absent opportunities.
The cutoff theorem needs only nonnegative weights; normalization is used only
when interpreting the resulting quantity as a probability. -/

noncomputable def success {I : Type*} [Fintype I]
    (weight : I → ℝ) (lifetime : I → Nat) (r : ℝ) : ℝ :=
  ∑ i, weight i * (1 - (1 - r) ^ lifetime i)

noncomputable def survival {I : Type*} [Fintype I]
    (weight : I → ℝ) (lifetime : I → Nat) (horizon : Nat) : ℝ :=
  ∑ i, if horizon < lifetime i then weight i else 0

/-- Conditional on a fixed lifetime, first-success probabilities telescope.
The equality is algebraic and remains valid at rates zero and one. -/
theorem first_hit_sum (r : ℝ) (t : Nat) :
    1 - (1 - r) ^ t = r * ∑ n ∈ Finset.range t, (1 - r) ^ n := by
  induction t with
  | zero => simp
  | succ t ih =>
    rw [Finset.sum_range_succ, pow_succ]
    nlinarith

/-- Stop unfolding after `horizon` trials. The uncomputed success probability
is at most the probability of both surviving the horizon and failing its
trials. No tail error bound is assumed as a premise. -/
theorem truncation {I : Type*} [Fintype I]
    (weight : I → ℝ) (lifetime : I → Nat) (r : ℝ) (horizon : Nat)
    (hw : ∀ i, 0 ≤ weight i) (hr0 : 0 ≤ r) (hr1 : r ≤ 1) :
    success weight (fun i => min (lifetime i) horizon) r ≤ success weight lifetime r ∧
    success weight lifetime r ≤
      success weight (fun i => min (lifetime i) horizon) r +
        (1 - r) ^ horizon * survival weight lifetime horizon := by
  classical
  have hq0 : 0 ≤ 1 - r := sub_nonneg.mpr hr1
  have hq1 : 1 - r ≤ 1 := by linarith
  constructor
  · apply Finset.sum_le_sum
    intro i _
    exact mul_le_mul_of_nonneg_left
      (sub_le_sub_left (pow_le_pow_of_le_one hq0 hq1 (Nat.min_le_left _ _)) 1) (hw i)
  · have hpoint : ∀ i, weight i * (1 - (1 - r) ^ lifetime i) ≤
        weight i * (1 - (1 - r) ^ min (lifetime i) horizon) +
          (1 - r) ^ horizon * (if horizon < lifetime i then weight i else 0) := by
      intro i
      by_cases hi : horizon < lifetime i
      · simp only [ite_eq_left hi, Nat.min_eq_right (Nat.le_of_lt hi)]
        have hp := mul_nonneg (hw i) (pow_nonneg hq0 (lifetime i))
        nlinarith
      · simp [hi, Nat.min_eq_left (Nat.le_of_not_gt hi)]
    have hs := Finset.sum_le_sum (fun i (_ : i ∈ Finset.univ) => hpoint i)
    simpa [success, survival, Finset.sum_add_distrib, Finset.mul_sum] using hs

/-- A normalized nonnegative finite lifetime distribution gives a genuine
probability, independently of the chosen support representation. -/
theorem probability_bounds {I : Type*} [Fintype I]
    (weight : I → ℝ) (lifetime : I → Nat) (r : ℝ)
    (hw : ∀ i, 0 ≤ weight i) (hs : ∑ i, weight i = 1)
    (hr0 : 0 ≤ r) (hr1 : r ≤ 1) :
    0 ≤ success weight lifetime r ∧ success weight lifetime r ≤ 1 := by
  have hq0 : 0 ≤ 1 - r := sub_nonneg.mpr hr1
  have hq1 : 1 - r ≤ 1 := by linarith
  constructor
  · exact Finset.sum_nonneg fun i _ =>
      mul_nonneg (hw i) (sub_nonneg.mpr (pow_le_one₀ hq0 hq1))
  · rw [← hs]
    apply Finset.sum_le_sum
    intro i _
    simpa using mul_le_mul_of_nonneg_left
      (show 1 - (1 - r) ^ lifetime i ≤ 1 by
        linarith [pow_nonneg hq0 (lifetime i)]) (hw i)

end FirstHit

namespace CausalState

/-! Canonical state for a registered deterministic history-readout contract.
Equality means equal responses to ALL future suffixes. This abstract function
space may be infinite or non-effective; no finite encoding or useful learned
target is assumed. The projection below is exact on reachable fold states.
Randomized bodies need a separate complete-state/random-input contract. -/

variable {U O S : Type*}

def future (required : List U → O) (h : List U) : List U → O :=
  fun suffix => required (h ++ suffix)

abbrev State (required : List U → O) := Set.range (future required)

def encode (required : List U → O) (h : List U) : State required :=
  ⟨future required h, h, rfl⟩

def read (required : List U → O) (s : State required) : O := s.val []

def advance (required : List U → O) (s : State required) (u : U) : State required :=
  ⟨fun suffix => s.val (u :: suffix), by
    obtain ⟨h, same⟩ := s.property
    refine ⟨h ++ [u], ?_⟩
    funext suffix
    rw [← same]
    simp [future, List.append_assoc]⟩

/-- Canonical readout and single-input recursion realize the same contract. -/
theorem realizes (required : List U → O) (h : List U) (u : U) :
    read required (encode required h) = required h ∧
      advance required (encode required h) u = encode required (h ++ [u]) := by
  constructor
  · simp [read, encode, future]
  · apply Subtype.ext
    funext suffix
    simp [advance, encode, future, List.append_assoc]

/-- Arbitrarily long forward recursion; not a bounded-memory implementation. -/
theorem fold_refines (required : List U → O) (h suffix : List U) :
    suffix.foldl (advance required) (encode required h) = encode required (h ++ suffix) := by
  induction suffix generalizing h with
  | nil => simp
  | cons u suffix ih =>
    rw [List.foldl_cons, (realizes required h u).2, ih]
    simp [List.append_assoc]

/-- Every exact deterministic realization projects onto the canonical states. -/
theorem minimal (required : List U → O) (next : S → U → S) (initial : S)
    (out : S → O) (correct : ∀ h, out (h.foldl next initial) = required h) :
    ∃ project : S → State required,
      (∀ h, project (h.foldl next initial) = encode required h) ∧
        Function.Surjective project := by
  have separates : ∀ h h', h.foldl next initial = h'.foldl next initial →
      encode required h = encode required h' := by
    intro h h' same
    apply Subtype.ext
    funext suffix
    change required (h ++ suffix) = required (h' ++ suffix)
    rw [← correct, ← correct, List.foldl_append, List.foldl_append, same]
  obtain ⟨project, exact⟩ :=
    (history_readout_exists_iff (fun h => h.foldl next initial)
      (encode required) (encode required [])).mpr separates
  refine ⟨project, exact, ?_⟩
  rintro ⟨f, h, rfl⟩
  exact ⟨h.foldl next initial, exact h⟩

/-- Finiteness follows from a finite exact realization; it is not assumed. -/
theorem finite_bound [Fintype S] (required : List U → O)
    (next : S → U → S) (initial : S) (out : S → O)
    (correct : ∀ h, out (h.foldl next initial) = required h) :
    Finite (State required) ∧ Nat.card (State required) ≤ Fintype.card S := by
  obtain ⟨project, _, onto⟩ := minimal required next initial out correct
  let : Finite (State required) := Finite.of_surjective project onto
  exact ⟨inferInstance, by
    simpa [Nat.card_eq_fintype_card] using Nat.card_le_card_of_surjective project onto⟩

/-- Finite input and output alphabets alone do not imply finite exact memory. -/
theorem finite_alphabets_unbounded :
    Infinite (State (fun h : List Bool => decide (h.count true = h.count false))) := by
  apply Infinite.of_injective
    (fun n : Nat => encode (fun h : List Bool => decide (h.count true = h.count false))
      (List.replicate n true))
  intro n m same
  have distinguish := congrFun (congrArg Subtype.val same) (List.replicate n false)
  have : m = n := by
    simpa [encode, future, List.count_append, List.count_replicate] using distinguish
  exact this.symm

/-- Uniform precision requires distinct readable codes for separated futures.
Codes precede suffix choice; query-dependent messages need fixed-query instances. -/
theorem separated_histories_capacity {U O I S : Type*} [PseudoMetricSpace O]
    (required : List U → O) (history : I → List U)
    (code : List U → S) (decode : S → List U → O) (legal : I → List U → Prop) (ε : ℝ)
    (accurate : ∀ i suffix, legal i suffix →
      dist (required (history i ++ suffix)) (decode (code (history i)) suffix) ≤ ε)
    (separated : ∀ i j, i ≠ j → ∃ suffix,
      legal i suffix ∧ legal j suffix ∧
        ε + ε < dist (required (history i ++ suffix)) (required (history j ++ suffix))) :
    Function.Injective (code ∘ history) ∧
      ∀ [Fintype I] [Fintype S], Fintype.card I ≤ Fintype.card S := by
  have injective : Function.Injective (code ∘ history) := by
    intro i j same
    by_contra different
    obtain ⟨suffix, hi, hj, apart⟩ := separated i j different
    have first := accurate i suffix hi
    have second := accurate j suffix hj
    change code (history i) = code (history j) at same
    have triangle := dist_triangle_right (required (history i ++ suffix))
      (required (history j ++ suffix)) (decode (code (history i)) suffix)
    rw [same] at first triangle
    linarith
  exact ⟨injective, Fintype.card_le_of_injective _ injective⟩

end CausalState

namespace Approximation

/-! Conditional error certificates for the same arrived input sequence.
The scalar error must measure or bound the registered readout discrepancy.
The one-step premise must cover the full update, including learning; neither
its constants nor a contraction follow from the physical clauses alone. -/

variable {X Z U : Type*}

/-- Full one-step error certificates compose over any finite continuation. -/
theorem fold_error (reference : X → U → X) (reduced : Z → U → Z)
    (error : X → Z → ℝ) (L δ : ℝ) (hL : 0 ≤ L)
    (hstep : ∀ x z u, error (reference x u) (reduced z u) ≤ L * error x z + δ)
    (x : X) (z : Z) (inputs : List U) :
    error (inputs.foldl reference x) (inputs.foldl reduced z) ≤
      L ^ inputs.length * error x z + δ * ∑ k ∈ Finset.range inputs.length, L ^ k := by
  induction inputs using List.reverseRecOn with
  | nil => simp
  | append_singleton inputs u ih =>
    simp only [List.foldl_append, List.foldl_cons, List.foldl_nil,
      List.length_append, List.length_singleton]
    calc
      _ ≤ L * error (inputs.foldl reference x) (inputs.foldl reduced z) + δ := hstep _ _ u
      _ ≤ L * (L ^ inputs.length * error x z +
          δ * ∑ k ∈ Finset.range inputs.length, L ^ k) + δ :=
        add_le_add (mul_le_mul_of_nonneg_left ih hL) le_rfl
      _ = _ := by rw [geom_sum_succ, pow_succ]; ring

/-- A locally closed error budget stays uniform in elapsed sequence length. -/
theorem invariant_error (reference : X → U → X) (reduced : Z → U → Z)
    (error : X → Z → ℝ) (L δ ε : ℝ) (hL : 0 ≤ L)
    (hstep : ∀ x z u, error (reference x u) (reduced z u) ≤ L * error x z + δ)
    (closed : L * ε + δ ≤ ε) (x : X) (z : Z) (initial : error x z ≤ ε)
    (inputs : List U) : error (inputs.foldl reference x) (inputs.foldl reduced z) ≤ ε := by
  induction inputs generalizing x z with
  | nil => exact initial
  | cons u inputs ih =>
    apply ih
    exact (hstep x z u).trans
      ((add_le_add (mul_le_mul_of_nonneg_left initial hL) le_rfl).trans closed)

end Approximation

end CGB.LanguageLifeKernel
